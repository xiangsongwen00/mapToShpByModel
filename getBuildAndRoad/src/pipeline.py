from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import torch
from scipy import ndimage
from rasterio.features import shapes
from rasterio.transform import xy
from rasterio.windows import Window
from shapely.geometry import LineString, shape

from common.config import load_yaml, resolve_path, save_yaml
from common.geo import approximate_gsd, checksum, hann_weight, sliding_windows
from common.logging_utils import build_logger, close_logger
from common.model import build_model


def _model_files(value: str | Path, root: Path) -> tuple[Path, Path]:
    path = resolve_path(value, root)
    if path.is_dir():
        checkpoint = path / "checkpoints" / "best.pt"
        inference = path / "inference.yaml"
    else:
        checkpoint = path
        inference = path.parents[1] / "inference.yaml" if path.parent.name == "checkpoints" else path.with_name("inference.yaml")
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型权重不存在: {checkpoint}")
    if not inference.exists():
        raise FileNotFoundError(f"模型推理配置不存在: {inference}")
    return checkpoint, inference


def _load_model(package, root: Path, device):
    checkpoint_path, inference_path = _model_files(package, root)
    inference = load_yaml(inference_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = payload.get("model_config", inference.get("model", {}))
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, inference, checkpoint_path


@torch.no_grad()
def _predict(src, model, device, tile: int, stride: int, batch_size: int, emit) -> np.ndarray:
    probability_sum = np.zeros((src.height, src.width), dtype="float32")
    weight_sum = np.zeros((src.height, src.width), dtype="float32")
    weight = hann_weight(tile)
    windows = list(sliding_windows(src.width, src.height, tile, stride))
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start:start + batch_size]
        arrays = []
        for window in batch_windows:
            x, y = int(window.col_off), int(window.row_off)
            array = src.read([1, 2, 3], window=Window(x, y, tile, tile), boundless=True, fill_value=0).astype("float32") / 255.0
            arrays.append(array)
        tensor = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = torch.sigmoid(model(tensor)[:, 0]).float().cpu().numpy()
        for window, predicted in zip(batch_windows, values):
            x, y, width, height = int(window.col_off), int(window.row_off), int(window.width), int(window.height)
            probability_sum[y:y + height, x:x + width] += predicted[:height, :width] * weight[:height, :width]
            weight_sum[y:y + height, x:x + width] += weight[:height, :width]
        emit(f"  推理窗口 {min(start + batch_size, len(windows))}/{len(windows)}")
    return probability_sum / np.maximum(weight_sum, 1e-6)


def _pixel_size_metres(src) -> tuple[float, float]:
    if src.crs and src.crs.to_epsg() == 4326:
        latitude = (src.bounds.bottom + src.bounds.top) / 2
        return approximate_gsd(src.transform, latitude)
    return abs(src.transform.a), abs(src.transform.e)


def _postprocess(probability: np.ndarray, threshold: float, min_area_m2: float, close_gap_m: float, gsd: tuple[float, float]) -> np.ndarray:
    mask = (probability >= threshold).astype("uint8")
    mean_gsd = max(sum(gsd) / 2.0, 1e-6)
    if close_gap_m > 0:
        radius = max(1, int(round(close_gap_m / mean_gsd)))
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = (xx * xx + yy * yy) <= radius * radius
        mask = ndimage.binary_closing(mask.astype(bool), structure=kernel).astype("uint8")
    min_pixels = max(1, int(math.ceil(min_area_m2 / max(gsd[0] * gsd[1], 1e-9))))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels].astype("uint8")


def _write_raster(path: Path, array: np.ndarray, src, dtype: str, nodata=None):
    profile = src.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate", tiled=True, nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _polygonize(mask: np.ndarray, probability: np.ndarray, src, task: str, model_name: str) -> gpd.GeoDataFrame:
    records = []
    for geometry, value in shapes(mask, mask=mask.astype(bool), transform=src.transform):
        if int(value) != 1:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        records.append({"object_id": len(records) + 1, "task": task, "model": model_name, "geometry": geom})
    result = gpd.GeoDataFrame(records if records else {"geometry": []}, geometry="geometry", crs=src.crs)
    if not result.empty:
        try:
            metric = result.to_crs(result.estimate_utm_crs())
            result["area_m2"] = metric.area.round(2).to_numpy()
        except Exception:
            result["area_m2"] = None
        result["mean_conf"] = [float(probability[mask == 1].mean())] * len(result)
    return result


def _skeleton_lines(mask: np.ndarray, transform, crs, min_length_pixels: int = 10) -> gpd.GeoDataFrame:
    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError("中心线输出需要 scikit-image；请安装后重试") from exc
    skeleton = skeletonize(mask.astype(bool))
    pixels = {tuple(value) for value in np.argwhere(skeleton)}
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

    def neighbours(point):
        y, x = point
        return {(y + dy, x + dx) for dy, dx in offsets if (y + dy, x + dx) in pixels}

    nodes = {point for point in pixels if len(neighbours(point)) != 2}
    visited: set[frozenset] = set()
    paths: list[list[tuple[int, int]]] = []
    for start in nodes:
        for nxt in neighbours(start):
            edge = frozenset((start, nxt))
            if edge in visited:
                continue
            path = [start, nxt]
            visited.add(edge)
            previous, current = start, nxt
            while current not in nodes:
                candidates = neighbours(current) - {previous}
                if not candidates:
                    break
                future = next(iter(candidates))
                visited.add(frozenset((current, future)))
                path.append(future)
                previous, current = current, future
            if len(path) >= min_length_pixels:
                paths.append(path)
    records = []
    for path in paths:
        coords = [xy(transform, row, col, offset="center") for row, col in path]
        records.append({"line_id": len(records) + 1, "geometry": LineString(coords)})
    return gpd.GeoDataFrame(records if records else {"geometry": []}, geometry="geometry", crs=crs)


def run(config_path: str | Path, emit: Callable[[str], None] = print) -> dict:
    config = load_yaml(config_path)
    root = resolve_path(config.get("project_root", "."))
    input_dir = resolve_path(config["input_dir"], root)
    output_root = resolve_path(config.get("output_dir", "getBuildAndRoad/outputs"), root)
    job_id = config.get("job_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = output_root / str(job_id)
    logger = build_logger("getBuildAndRoad", job_dir / "run.log")
    use_cuda = torch.cuda.is_available() and config.get("device", "auto") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    emit(f"设备: {device}；任务目录: {job_dir}")

    packages = config.get("models", {})
    loaded = {}
    for task in ("building", "road"):
        if packages.get(task):
            loaded[task] = _load_model(packages[task], root, device)
            emit(f"已加载 {task} 模型: {loaded[task][2]}")
    if not loaded:
        raise ValueError("models.building 与 models.road 至少配置一个模型包")

    tiffs = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"没有找到 TIFF: {input_dir}")
    summaries = []
    for index, tif in enumerate(tiffs, 1):
        started = time.time()
        emit(f"[{index}/{len(tiffs)}] {tif.name}")
        result_dir = job_dir / tif.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        source_summary = {"source": str(tif), "source_checksum": checksum(tif), "tasks": {}}
        with rasterio.open(tif) as src:
            if src.count < 3 or src.crs is None:
                raise ValueError(f"影像必须至少 3 波段且包含 CRS: {tif}")
            gsd = _pixel_size_metres(src)
            for task, (model, inference, checkpoint) in loaded.items():
                tile = int(inference.get("tile_size", 512))
                stride = int(inference.get("stride", tile // 2))
                batch = int(config.get("batch_size", 4 if use_cuda else 1))
                probability = _predict(src, model, device, tile, stride, batch, emit)
                task_cfg = config.get("postprocess", {}).get(task, {})
                threshold = float(task_cfg.get("threshold", inference.get("threshold", 0.5)))
                mask = _postprocess(
                    probability, threshold,
                    float(task_cfg.get("min_area_m2", 4.0 if task == "building" else 8.0)),
                    float(task_cfg.get("close_gap_m", 0.0 if task == "building" else 1.0)), gsd,
                )
                output_name = "building_base" if task == "building" else "road"
                _write_raster(result_dir / f"{output_name}_probability.tif", probability, src, "float32")
                _write_raster(result_dir / f"{output_name}_mask.tif", mask, src, "uint8", 0)
                vector_task = "building_base" if task == "building" else "road_surface"
                vectors = _polygonize(mask, probability, src, vector_task, checkpoint.parent.parent.name)
                vector_path = result_dir / ("building_base.gpkg" if task == "building" else "road_surface.gpkg")
                if not vectors.empty:
                    vectors.to_file(vector_path, layer=vector_task, driver="GPKG")
                if task == "road" and bool(task_cfg.get("centerline", False)):
                    lines = _skeleton_lines(mask, src.transform, src.crs, int(task_cfg.get("min_line_pixels", 10)))
                    if not lines.empty:
                        lines.to_file(result_dir / "road_centerline.gpkg", layer="road_centerline", driver="GPKG")
                source_summary["tasks"][task] = {
                    "checkpoint": str(checkpoint), "threshold": threshold,
                    "positive_pixels": int(mask.sum()), "objects": len(vectors),
                }
        source_summary["elapsed_seconds"] = round(time.time() - started, 3)
        (result_dir / "summary.json").write_text(json.dumps(source_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        save_yaml(config, result_dir / "run.resolved.yaml")
        summaries.append(source_summary)
        logger.info("completed %s in %.2fs", tif, source_summary["elapsed_seconds"])
    emit(f"推理完成: {job_dir}")
    result = {"job_dir": str(job_dir), "images": len(summaries), "summaries": summaries}
    close_logger(logger)
    return result
