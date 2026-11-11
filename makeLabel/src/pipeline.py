from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window

from common.config import load_yaml, resolve_path, save_yaml
from common.geo import approximate_gsd, checksum, sliding_windows
from common.logging_utils import build_logger, close_logger
from makeLabel.src.building_geometry import roof_to_base
from makeLabel.src.osm_downloader import run as run_osm_download


def _split_for(source_id: str, split: dict) -> str:
    value = int(hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000 / 10_000
    train = float(split.get("train", 0.7))
    val = float(split.get("val", 0.15))
    if value < train:
        return "train"
    return "val" if value < train + val else "test"


def _read_layers(path: Path, raster=None, building_config: dict | None = None, image_id: str = "") -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if path.suffix.lower() == ".gpkg":
        import fiona

        layers = set(fiona.listlayers(path))
        if "building_base" in layers:
            building = gpd.read_file(path, layer="building_base")
        elif "building" in layers:
            # 向后兼容旧项目；旧 building 图层按底座语义处理。
            building = gpd.read_file(path, layer="building")
        elif "building_roof" in layers:
            if raster is None:
                raise ValueError("读取 building_roof 时需要对应栅格以计算偏移")
            roofs = gpd.read_file(path, layer="building_roof")
            building = roof_to_base(roofs, raster, building_config or {}, image_id)
        else:
            building = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        road_layer = "road_surface" if "road_surface" in layers else "road"
        road = gpd.read_file(path, layer=road_layer) if road_layer in layers else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return building, road
    gdf = gpd.read_file(path)
    if "task" not in gdf.columns:
        raise ValueError(f"非 GPKG 标注文件需要 task 字段: {path}")
    return gdf[gdf.task == "building"].copy(), gdf[gdf.task.isin(["road", "road_surface"])].copy()


def _read_negative_layers(path: Path, crs) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if path.suffix.lower() != ".gpkg":
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        return empty, empty.copy()
    import fiona

    layers = set(fiona.listlayers(path))
    building = gpd.read_file(path, layer="building_negative") if "building_negative" in layers else gpd.GeoDataFrame(geometry=[], crs=crs)
    road = gpd.read_file(path, layer="road_negative") if "road_negative" in layers else gpd.GeoDataFrame(geometry=[], crs=crs)
    return building, road


def _quality_for(config: dict, image_id: str, status: str) -> dict:
    section = config.get("label_quality", {})
    quality = dict(section.get("default", {}))
    quality.update(section.get("per_image", {}).get(str(image_id), {}))
    quality.setdefault("unlabeled_policy", "ignore" if status == "candidate" else "background")
    quality.setdefault("completeness", 0.0 if status == "candidate" else 1.0)
    quality.setdefault("positive_weight", 0.6 if status == "candidate" else 1.0)
    # ignore 区域始终不参与损失；若存在显式 negative 图层，默认允许其作为可信负样本。
    quality.setdefault("negative_weight", 1.0)
    if quality["unlabeled_policy"] not in {"ignore", "background"}:
        raise ValueError(f"unlabeled_policy 必须是 ignore 或 background: {image_id}")
    return quality


def _mask_profile(src) -> dict:
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True)
    return profile


def _rasterize_layer(gdf: gpd.GeoDataFrame, src, all_touched: bool) -> np.ndarray:
    if gdf.empty:
        return np.zeros((src.height, src.width), dtype="uint8")
    if gdf.crs is None:
        raise ValueError("标注矢量缺少 CRS")
    local = gdf.to_crs(src.crs)
    shapes = ((geom, 1) for geom in local.geometry if geom is not None and not geom.is_empty)
    return rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8", all_touched=all_touched)


def _write_mask(path: Path, mask: np.ndarray, src) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_mask_profile(src)) as dst:
        dst.write(mask, 1)


def run(config_path: str | Path, emit: Callable[[str], None] = print) -> dict:
    config = load_yaml(config_path)
    root = resolve_path(config.get("project_root", "."))
    input_dir = resolve_path(config["input_dir"], root)
    labels_dir = resolve_path(config["labels_dir"], root)
    dataset_root = resolve_path(config["dataset_dir"], root)
    version = str(config.get("dataset_version") or datetime.now().strftime("v%Y%m%d_%H%M%S"))
    dataset_dir = dataset_root / version
    log = build_logger("makeLabel", dataset_dir / "make_label.log")
    tiffs = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"没有找到 TIFF: {input_dir}")

    if bool(config.get("download_osm", False)):
        emit("先执行独立 OSM 候选图斑下载……")
        run_osm_download(config_path, emit)

    tile = int(config.get("tile_size", 512))
    stride = int(config.get("stride", 256))
    vector_dir = labels_dir / "vectors"
    candidate_dir = resolve_path(config.get("osm", {}).get("candidate_dir", "data/labels/osm_candidates"), root)
    mask_dir = labels_dir / "masks"
    rows: list[dict] = []
    stats = {"images": len(tiffs), "tiles": 0, "positive_building": 0, "positive_road": 0}

    for number, tif in enumerate(tiffs, 1):
        emit(f"[{number}/{len(tiffs)}] 处理 {tif.name}")
        log.info("processing %s", tif)
        annotation = vector_dir / f"{tif.stem}.gpkg"
        candidate = candidate_dir / f"{tif.stem}.gpkg"
        with rasterio.open(tif) as src:
            if src.crs is None:
                raise ValueError(f"影像缺少 CRS: {tif}")
            if annotation.exists():
                buildings, roads = _read_layers(annotation, src, config.get("building", {}), tif.stem)
                building_negatives, road_negatives = _read_negative_layers(annotation, src.crs)
                label_status = "reviewed"
                emit(f"使用人工标注: {annotation.name}")
            elif candidate.exists():
                buildings, roads = _read_layers(candidate, src, config.get("building", {}), tif.stem)
                building_negatives, road_negatives = _read_negative_layers(candidate, src.crs)
                label_status = "candidate"
                emit(f"使用待复核 OSM 候选: {candidate.name}")
            else:
                if not bool(config.get("allow_empty_labels", False)):
                    raise FileNotFoundError(
                        f"{tif.name} 没有正式标注或 OSM 候选。请先运行 --download-osm，"
                        "或提供人工标注；如仅做流程测试可设置 allow_empty_labels: true"
                    )
                buildings = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                roads = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                building_negatives = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                road_negatives = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                label_status = "empty"
                emit(f"警告：{tif.name} 无标注，生成空标签；请先下载 OSM 候选或提供 {annotation.name}")

            quality = _quality_for(config, tif.stem, label_status)
            fill = 255 if quality["unlabeled_policy"] == "ignore" else 0
            building_mask = np.full((src.height, src.width), fill, dtype="uint8")
            road_mask = np.full((src.height, src.width), fill, dtype="uint8")
            building_positive = _rasterize_layer(buildings, src, bool(config.get("building_all_touched", False)))
            road_positive = _rasterize_layer(roads, src, bool(config.get("road_all_touched", True)))
            building_negative = _rasterize_layer(building_negatives, src, True)
            road_negative = _rasterize_layer(road_negatives, src, True)
            building_mask[building_negative == 1] = 0
            road_mask[road_negative == 1] = 0
            building_mask[building_positive == 1] = 1
            road_mask[road_positive == 1] = 1
            building_full = mask_dir / f"{tif.stem}_building.tif"
            road_full = mask_dir / f"{tif.stem}_road.tif"
            _write_mask(building_full, building_mask, src)
            _write_mask(road_full, road_mask, src)

            source_split = _split_for(tif.stem, config.get("split", {}))
            scene = config.get("scenes", {}).get(tif.stem, "unknown")
            latitude = (src.bounds.bottom + src.bounds.top) / 2
            gsd_x, gsd_y = approximate_gsd(src.transform, latitude) if src.crs.to_epsg() == 4326 else (abs(src.transform.a), abs(src.transform.e))
            for window in sliding_windows(src.width, src.height, tile, stride):
                x, y = int(window.col_off), int(window.row_off)
                sample_id = f"{tif.stem}_{x}_{y}"
                image = src.read([1, 2, 3], window=Window(x, y, tile, tile), boundless=True, fill_value=0)
                bm = np.full((tile, tile), 255, dtype="uint8")
                rm = np.full((tile, tile), 255, dtype="uint8")
                height, width = int(window.height), int(window.width)
                bm[:height, :width] = building_mask[y:y + height, x:x + width]
                rm[:height, :width] = road_mask[y:y + height, x:x + width]
                image_path = dataset_dir / "images" / f"{sample_id}.png"
                building_path = dataset_dir / "masks_building" / f"{sample_id}.png"
                road_path = dataset_dir / "masks_road" / f"{sample_id}.png"
                for path in (image_path, building_path, road_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.moveaxis(image, 0, -1)).save(image_path)
                Image.fromarray(bm).save(building_path)
                Image.fromarray(rm).save(road_path)
                bounds = rasterio.windows.bounds(window, src.transform)
                building_ratio = float((bm == 1).sum() / max((bm != 255).sum(), 1))
                road_ratio = float((rm == 1).sum() / max((rm != 255).sum(), 1))
                building_negative_ratio = float((bm == 0).sum() / max((bm != 255).sum(), 1))
                road_negative_ratio = float((rm == 0).sum() / max((rm != 255).sum(), 1))
                rows.append({
                    "sample_id": sample_id,
                    "image_path": str(image_path.relative_to(root)),
                    "building_mask_path": str(building_path.relative_to(root)),
                    "road_mask_path": str(road_path.relative_to(root)),
                    "source_image_id": tif.stem,
                    "scene": scene,
                    "split": source_split,
                    "x_off": x, "y_off": y, "width": width, "height": height,
                    "crs": str(src.crs), "bounds": json.dumps(bounds),
                    "gsd_x": round(gsd_x, 4), "gsd_y": round(gsd_y, 4),
                    "building_ratio": round(building_ratio, 8), "road_ratio": round(road_ratio, 8),
                    "building_negative_ratio": round(building_negative_ratio, 8),
                    "road_negative_ratio": round(road_negative_ratio, 8),
                    "label_status": label_status,
                    "building_geometry": "base",
                    "label_completeness": float(quality["completeness"]),
                    "unlabeled_policy": quality["unlabeled_policy"],
                    "positive_weight": float(quality["positive_weight"]),
                    "negative_weight": float(quality["negative_weight"]),
                    "label_version": version,
                    "checksum": checksum(image_path),
                })
                stats["tiles"] += 1
                stats["positive_building"] += int(building_ratio > 0)
                stats["positive_road"] += int(road_ratio > 0)

    manifest = dataset_dir / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": {"background": 0, "target": 1, "ignore": 255},
        "building_target": "ground_base_footprint",
        "tile_size": tile,
        "stride": stride,
        "source_images": [path.name for path in tiffs],
        "stats": stats,
    }
    save_yaml(metadata, dataset_dir / "dataset.yaml")
    (dataset_dir / "split.json").write_text(
        json.dumps({row["source_image_id"]: row["split"] for row in rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit(f"数据集完成: {dataset_dir}，样本 {len(rows)} 个")
    result = {"dataset_dir": str(dataset_dir), "manifest": str(manifest), **stats}
    close_logger(log)
    return result
