from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.config import load_yaml, resolve_path, save_yaml
from common.logging_utils import build_logger, close_logger
from common.model import build_model


class SegmentationDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, task: str, augment: bool):
        self.rows = rows
        self.root = root
        self.task = task
        self.augment = augment
        self.mask_key = f"{task}_mask_path"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(self.root / row["image_path"]).convert("RGB"), dtype="float32") / 255.0
        mask = np.asarray(Image.open(self.root / row[self.mask_key]), dtype="uint8")
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.flip(image, 1), np.flip(mask, 1)
            if random.random() < 0.5:
                image, mask = np.flip(image, 0), np.flip(mask, 0)
            k = random.randrange(4)
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                image = np.clip(image * factor, 0, 1)
        image = np.ascontiguousarray(np.moveaxis(image, -1, 0))
        mask = np.ascontiguousarray(mask)
        return (
            torch.from_numpy(image),
            torch.from_numpy(mask.astype("int64")),
            row.get("scene", "unknown"),
            row["sample_id"],
            float(row.get("positive_weight", 1.0)),
            float(row.get("negative_weight", 1.0)),
        )


def _read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"数据清单为空: {path}")
    return rows


def _supervision_stats(rows: list[dict], root: Path, task: str) -> dict[str, int]:
    mask_key = f"{task}_mask_path"
    positive_pixels = 0
    negative_pixels = 0
    for row in rows:
        mask = np.asarray(Image.open(root / row[mask_key]), dtype="uint8")
        if float(row.get("positive_weight", 1.0)) > 0:
            positive_pixels += int((mask == 1).sum())
        if float(row.get("negative_weight", 1.0)) > 0:
            negative_pixels += int((mask == 0).sum())
        if positive_pixels > 0 and negative_pixels > 0:
            break
    return {"positive_pixels": positive_pixels, "negative_pixels": negative_pixels}


def _row_has_supervision(row: dict, task: str) -> bool:
    positive = float(row.get(f"{task}_ratio", 0.0)) > 0 and float(row.get("positive_weight", 1.0)) > 0
    negative = float(row.get(f"{task}_negative_ratio", 1.0)) > 0 and float(row.get("negative_weight", 1.0)) > 0
    return positive or negative


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(logits, mask, positive_weight, negative_weight):
    valid = mask != 255
    target = (mask == 1).float()
    positive_weight = positive_weight.to(logits.device).float().view(-1, 1, 1)
    negative_weight = negative_weight.to(logits.device).float().view(-1, 1, 1)
    pixel_weight = torch.where(target > 0.5, positive_weight, negative_weight)
    supervised = valid & (pixel_weight > 0)
    if not supervised.any():
        return logits.sum() * 0.0
    binary_values = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    binary = (binary_values[supervised] * pixel_weight[supervised]).sum() / pixel_weight[supervised].sum().clamp_min(1e-6)
    probability = torch.sigmoid(logits[:, 0])
    intersection = (probability[supervised] * target[supervised] * pixel_weight[supervised]).sum()
    predicted = (probability[supervised] * pixel_weight[supervised]).sum()
    expected = (target[supervised] * pixel_weight[supervised]).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (predicted + expected + 1.0)
    return binary + dice


def _counts(logits, mask, threshold: float) -> tuple[int, int, int, int]:
    valid = mask != 255
    prediction = torch.sigmoid(logits[:, 0]) >= threshold
    target = mask == 1
    tp = int((prediction & target & valid).sum().item())
    fp = int((prediction & ~target & valid).sum().item())
    fn = int((~prediction & target & valid).sum().item())
    tn = int((~prediction & ~target & valid).sum().item())
    return tp, fp, fn, tn


def _metrics(counts: tuple[int, int, int, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
    }


@torch.no_grad()
def _evaluate(model, loader, device, threshold: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_counts = np.zeros(4, dtype="int64")
    scene_counts: dict[str, np.ndarray] = {}
    for image, mask, scenes, _, positive_weight, negative_weight in loader:
        image, mask = image.to(device), mask.to(device)
        logits = model(image)
        total_loss += float(_loss(logits, mask, positive_weight, negative_weight).item()) * image.shape[0]
        total_counts += np.asarray(_counts(logits, mask, threshold))
        for idx, scene in enumerate(scenes):
            scene_counts.setdefault(scene, np.zeros(4, dtype="int64"))
            scene_counts[scene] += np.asarray(_counts(logits[idx:idx + 1], mask[idx:idx + 1], threshold))
    result = {"loss": total_loss / max(len(loader.dataset), 1), **_metrics(tuple(total_counts))}
    result["by_scene"] = {key: _metrics(tuple(value)) for key, value in scene_counts.items()}
    return result


def _atomic_checkpoint(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(config_path: str | Path, emit: Callable[[str], None] = print) -> dict:
    config = load_yaml(config_path)
    root = resolve_path(config.get("project_root", "."))
    manifest_path = resolve_path(config["manifest"], root)
    task = str(config.get("task", "building"))
    if task not in {"building", "road"}:
        raise ValueError("task 必须为 building 或 road")
    seed = int(config.get("seed", 20260919))
    _seed_everything(seed)

    use_cuda = torch.cuda.is_available() and config.get("device", "auto") != "cpu"
    if not use_cuda and not bool(config.get("allow_cpu", False)):
        raise RuntimeError("没有可用 CUDA。正式训练请安装 GPU 版 PyTorch；调试可设置 allow_cpu: true")
    device = torch.device("cuda" if use_cuda else "cpu")
    run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(config.get("runs_dir", "trainModel/runs"), root) / task / str(run_id)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = build_logger(f"trainModel.{task}", run_dir / "train.log")
    emit(f"设备: {device}；运行目录: {run_dir}")

    rows = _read_manifest(manifest_path)
    raw_train_rows = [row for row in rows if row["split"] == "train"]
    raw_val_rows = [row for row in rows if row["split"] == "val"]
    train_rows = [row for row in raw_train_rows if _row_has_supervision(row, task)]
    val_rows = [row for row in raw_val_rows if _row_has_supervision(row, task)]
    emit(
        f"有效监督切片: train={len(train_rows)}/{len(raw_train_rows)}, "
        f"val={len(val_rows)}/{len(raw_val_rows)}"
    )
    allow_bootstrap_validation = bool(config.get("bootstrap_without_reliable_validation", True))
    temporary_validation = False
    if train_rows and not val_rows and allow_bootstrap_validation:
        ordered = sorted(train_rows, key=lambda row: hashlib.sha1(row["sample_id"].encode("utf-8")).hexdigest())
        holdout = max(1, min(len(ordered) // 10, 64))
        if len(ordered) <= holdout:
            raise ValueError("有效监督切片太少，无法建立首轮临时验证子集")
        val_rows = ordered[:holdout]
        train_rows = ordered[holdout:]
        temporary_validation = True
        emit(f"警告：没有可靠验证切片；从训练样本临时留出 {holdout} 块，仅用于首轮运行监控")
    if not train_rows or not val_rows:
        raise ValueError(f"训练/验证有效监督样本不足: train={len(train_rows)}, val={len(val_rows)}")
    validation_supervision = _supervision_stats(val_rows, root, task)
    reliable_validation = not temporary_validation and all(value > 0 for value in validation_supervision.values())
    bootstrap_mode = not reliable_validation and allow_bootstrap_validation
    if not reliable_validation and not bootstrap_mode:
        raise ValueError(
            f"验证集缺少同时可信的正负像元: {validation_supervision}。"
            "请建立人工复核验证区，或仅在首轮引导训练时启用 bootstrap_without_reliable_validation"
        )
    if bootstrap_mode:
        emit(f"警告：验证指标不可靠 {validation_supervision}；首轮按训练损失选择引导权重")
    batch_size = int(config.get("batch_size", 4))
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        SegmentationDataset(train_rows, root, task, True), batch_size=batch_size,
        shuffle=True, num_workers=workers, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_rows, root, task, False), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=use_cuda,
    )

    model_config = config.get("model", {"name": "tiny_unet", "base_channels": 32})
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and bool(config.get("mixed_precision", True)))
    threshold = float(config.get("threshold", 0.5))
    accumulation = max(int(config.get("gradient_accumulation", 1)), 1)
    start_epoch, best_iou, best_selection, patience = 1, -1.0, float("-inf"), 0
    resume = config.get("resume")
    if resume:
        payload = torch.load(resolve_path(resume, root), map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_iou = float(payload.get("best_iou", -1.0))
        best_selection = float(payload.get("best_selection", best_iou))
        emit(f"从 epoch {start_epoch} 恢复")

    save_yaml(config, run_dir / "config.resolved.yaml")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(manifest_path.parent / "dataset.yaml", run_dir / "dataset_snapshot.yaml") if (manifest_path.parent / "dataset.yaml").exists() else None
    history: list[dict] = []
    early_patience = int(config.get("early_stopping_patience", 15))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, (image, mask, _, _, positive_weight, negative_weight) in enumerate(train_loader, 1):
            image, mask = image.to(device), mask.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(image)
                loss = _loss(logits, mask, positive_weight, negative_weight) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item()) * accumulation * image.shape[0]
        scheduler.step()
        val = _evaluate(model, val_loader, device, threshold)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val["loss"], "val_iou": val["iou"], "val_f1": val["f1"],
            "val_precision": val["precision"], "val_recall": val["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        emit(f"epoch {epoch}/{epochs} loss={record['train_loss']:.4f} val_iou={val['iou']:.4f} val_f1={val['f1']:.4f}")
        log.info("epoch=%s metrics=%s scenes=%s", epoch, record, val["by_scene"])
        selection_value = -record["train_loss"] if bootstrap_mode else val["iou"]
        payload = {
            "epoch": epoch,
            "task": task,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": max(best_iou, val["iou"]),
            "best_selection": max(best_selection, selection_value),
            "selection_metric": "train_loss" if bootstrap_mode else "val_iou",
            "validation_reliable": reliable_validation,
            "threshold": threshold,
            "normalization": {"scale": 255.0},
            "dataset_version": manifest_path.parent.name,
            "target_geometry": "ground_base_footprint" if task == "building" else "road_surface",
        }
        _atomic_checkpoint(payload, checkpoint_dir / "last.pt")
        if selection_value > best_selection:
            best_selection = selection_value
            best_iou = max(best_iou, val["iou"])
            patience = 0
            _atomic_checkpoint(payload, checkpoint_dir / "best.pt")
            (run_dir / "evaluation.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            patience += 1

        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if patience >= early_patience:
            emit(f"模型选择指标连续 {patience} 轮未提升，提前停止")
            break

    inference_config = {
        "task": task,
        "target_geometry": "ground_base_footprint" if task == "building" else "road_surface",
        "model": model_config,
        "tile_size": int(config.get("tile_size", 512)),
        "stride": int(config.get("inference_stride", int(config.get("tile_size", 512)) // 2)),
        "threshold": threshold,
        "normalization": {"scale": 255.0},
    }
    save_yaml(inference_config, run_dir / "inference.yaml")
    card = (
        f"# Model card: {task}\n\n- Run: `{run_id}`\n- Dataset: `{manifest_path.parent.name}`\n"
        f"- Best validation IoU: `{best_iou:.6f}`\n- Validation reliable: `{reliable_validation}`\n"
        f"- Selection metric: `{'train_loss (bootstrap)' if bootstrap_mode else 'val_iou'}`\n"
        f"- Device: `{device}`\n- Threshold: `{threshold}`\n"
        "- Intended use: RGB aerial/satellite imagery matching the training GSD.\n"
        "- Limitation: OSM-derived labels require manual review; inspect urban and rural metrics separately.\n"
    )
    (run_dir / "model_card.md").write_text(card, encoding="utf-8")
    emit(f"训练完成，val IoU={best_iou:.4f}，验证可靠={reliable_validation}")
    result = {"run_dir": str(run_dir), "best_checkpoint": str(checkpoint_dir / "best.pt"), "best_iou": best_iou}
    close_logger(log)
    result["validation_reliable"] = reliable_validation
    result["selection_metric"] = "train_loss" if bootstrap_mode else "val_iou"
    return result
