from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import geopandas as gpd
import pandas as pd
import requests
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

from common.config import load_yaml, resolve_path, save_yaml
from common.logging_utils import build_logger, close_logger


def split_bounds(bounds: tuple[float, float, float, float], max_span_deg: float) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Split WGS84 bounds into deterministic cells small enough for Overpass."""
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"非法 EPSG:4326 范围: {bounds}")
    if max_span_deg <= 0:
        raise ValueError("max_span_deg 必须大于 0")
    columns = max(1, math.ceil((east - west) / max_span_deg))
    rows = max(1, math.ceil((north - south) / max_span_deg))
    dx, dy = (east - west) / columns, (north - south) / rows
    result = []
    for row in range(rows):
        cell_south = south + row * dy
        cell_north = north if row == rows - 1 else south + (row + 1) * dy
        for column in range(columns):
            cell_west = west + column * dx
            cell_east = east if column == columns - 1 else west + (column + 1) * dx
            result.append((row, column, (cell_west, cell_south, cell_east, cell_north)))
    return result


def build_query(kind: str, bounds: tuple[float, float, float, float], timeout: int) -> str:
    west, south, east, north = bounds
    bbox = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    if kind == "building":
        selectors = f'way["building"]({bbox});relation["building"]({bbox});'
    elif kind == "road":
        selectors = f'way["highway"]({bbox});'
    else:
        raise ValueError(f"未知 OSM 类型: {kind}")
    return f"[out:json][timeout:{timeout}];({selectors});out tags geom;"


def download_cell(
    kind: str,
    bounds: tuple[float, float, float, float],
    endpoints: list[str],
    timeout: int,
    retries: int,
    retry_delay: float,
    user_agent: str,
    session: requests.Session | None = None,
) -> dict:
    client = session or requests.Session()
    query = build_query(kind, bounds, timeout)
    errors: list[str] = []
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            response = client.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=timeout + 30,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("elements"), list):
                raise ValueError("响应缺少 elements 数组")
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{endpoint}: {exc}")
            if attempt + 1 < retries:
                time.sleep(retry_delay * (2**attempt))
    raise RuntimeError(f"OSM {kind} 下载失败 {bounds}: " + " | ".join(errors))


def _properties(element: dict) -> dict:
    tags = element.get("tags", {})
    def numeric(value):
        if value is None:
            return None
        try:
            return float(str(value).lower().replace("meters", "").replace("meter", "").replace("m", "").strip())
        except ValueError:
            return None
    return {
        "osm_type": element.get("type", "way"),
        "osm_id": int(element.get("id", 0)),
        "kind": tags.get("building") or tags.get("highway") or "unknown",
        "name": tags.get("name"),
        "surface": tags.get("surface"),
        "lanes": tags.get("lanes"),
        "width": tags.get("width"),
        "layer": tags.get("layer"),
        "bridge": tags.get("bridge"),
        "tunnel": tags.get("tunnel"),
        "height_m": numeric(tags.get("height")),
        "building_levels": numeric(tags.get("building:levels")),
    }


def _empty_osm_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osm_type": [], "osm_id": [], "kind": [], "name": [], "surface": [],
            "lanes": [], "width": [], "layer": [], "bridge": [], "tunnel": [],
            "height_m": [], "building_levels": [], "geometry": [],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _coordinates(geometry: Iterable[dict]) -> list[tuple[float, float]]:
    return [(float(point["lon"]), float(point["lat"])) for point in geometry if "lon" in point and "lat" in point]


def _relation_polygon(element: dict):
    outers: list[LineString] = []
    inners: list[LineString] = []
    for member in element.get("members", []):
        coords = _coordinates(member.get("geometry", []))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        (inners if member.get("role") == "inner" else outers).append(line)
    outer_polygons = list(polygonize(unary_union(outers))) if outers else []
    if not outer_polygons:
        return None
    geometry = unary_union(outer_polygons)
    if inners:
        holes = list(polygonize(unary_union(inners)))
        if holes:
            geometry = geometry.difference(unary_union(holes))
    return geometry


def parse_payloads(
    building_payloads: Iterable[dict],
    road_payloads: Iterable[dict],
    excluded_highways: set[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    excluded = excluded_highways or set()
    buildings: dict[tuple[str, int], dict] = {}
    roads: dict[tuple[str, int], dict] = {}
    for payload in building_payloads:
        for element in payload.get("elements", []):
            tags = element.get("tags", {})
            if "building" not in tags:
                continue
            key = (element.get("type", "way"), int(element.get("id", 0)))
            try:
                if element.get("type") == "relation":
                    geometry = _relation_polygon(element)
                else:
                    coords = _coordinates(element.get("geometry", []))
                    geometry = Polygon(coords) if len(coords) >= 4 else None
                if geometry is None:
                    continue
                if not geometry.is_valid:
                    geometry = geometry.buffer(0)
                if not geometry.is_empty:
                    buildings[key] = {**_properties(element), "geometry": geometry}
            except Exception:
                continue
    for payload in road_payloads:
        for element in payload.get("elements", []):
            tags = element.get("tags", {})
            highway = str(tags.get("highway", ""))
            if not highway or highway in excluded:
                continue
            coords = _coordinates(element.get("geometry", []))
            if len(coords) < 2:
                continue
            key = (element.get("type", "way"), int(element.get("id", 0)))
            roads[key] = {**_properties(element), "geometry": LineString(coords)}
    building_gdf = gpd.GeoDataFrame(
        list(buildings.values()), geometry="geometry", crs="EPSG:4326"
    ) if buildings else _empty_osm_frame()
    road_gdf = gpd.GeoDataFrame(
        list(roads.values()), geometry="geometry", crs="EPSG:4326"
    ) if roads else _empty_osm_frame()
    return building_gdf, road_gdf


def make_road_surfaces(roads: gpd.GeoDataFrame, widths: dict[str, float], default_width: float) -> gpd.GeoDataFrame:
    if roads.empty:
        result = roads.copy()
        result["width_m"] = []
        return result
    metric_crs = roads.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("无法为道路数据确定本地米制坐标系")
    projected = roads.to_crs(metric_crs).copy()
    projected["width_m"] = [float(widths.get(str(kind), default_width)) for kind in projected["kind"]]
    projected.geometry = [
        geometry.buffer(width / 2.0, cap_style="flat", join_style="round")
        for geometry, width in zip(projected.geometry, projected["width_m"])
    ]
    return projected.to_crs(roads.crs)


def _clip(gdf: gpd.GeoDataFrame, bounds4326: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    clipped = gpd.clip(gdf, box(*bounds4326), keep_geom_type=True)
    return clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].reset_index(drop=True)


def _write_gpkg(path: Path, buildings: gpd.GeoDataFrame, road_lines: gpd.GeoDataFrame, road_surfaces: gpd.GeoDataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".downloading.gpkg")
    if temporary.exists():
        temporary.unlink()
    layers = (("building_base", buildings), ("road_centerline", road_lines), ("road_surface", road_surfaces))
    for index, (layer, gdf) in enumerate(layers):
        gdf.to_file(temporary, layer=layer, driver="GPKG", mode="a" if index else "w")
    if path.exists():
        path.unlink()
    temporary.replace(path)


def _valid_vector_cache(path: Path, kind: str) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        import fiona

        layers = set(fiona.listlayers(path))
        expected = {"building_base"} if kind == "building" else {"road_centerline", "road_surface"}
        return expected.issubset(layers)
    except Exception:
        return False


def _write_vector_cache(
    path: Path,
    kind: str,
    payload: dict,
    excluded: set[str],
    widths: dict[str, float],
    default_width: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    buildings, roads = parse_payloads(
        [payload] if kind == "building" else [],
        [payload] if kind == "road" else [],
        excluded,
    )
    surfaces = make_road_surfaces(roads, widths, default_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".converting.gpkg")
    if temporary.exists():
        temporary.unlink()
    if kind == "building":
        buildings.to_file(temporary, layer="building_base", driver="GPKG")
    else:
        roads.to_file(temporary, layer="road_centerline", driver="GPKG")
        surfaces.to_file(temporary, layer="road_surface", driver="GPKG", mode="a")
    if path.exists():
        path.unlink()
    temporary.replace(path)
    if not _valid_vector_cache(path, kind):
        raise RuntimeError(f"GeoPackage 分块校验失败: {path}")
    return buildings, roads


def _read_vector_cache(path: Path, kind: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if kind == "building":
        return gpd.read_file(path, layer="building_base"), _empty_osm_frame()
    return _empty_osm_frame(), gpd.read_file(path, layer="road_centerline")


def _merge_frames(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return _empty_osm_frame()
    result = gpd.GeoDataFrame(pd.concat(usable, ignore_index=True), geometry="geometry", crs=usable[0].crs)
    identity = [column for column in ("osm_type", "osm_id") if column in result.columns]
    if identity:
        result = result.drop_duplicates(subset=identity, keep="last")
    return result.reset_index(drop=True)


def _valid_candidate(path: Path) -> bool:
    try:
        import fiona

        return {"building_base", "road_centerline", "road_surface"}.issubset(set(fiona.listlayers(path)))
    except Exception:
        return False


def convert_legacy_json(config_path: str | Path, emit: Callable[[str], None] = print) -> dict:
    """Convert old Overpass JSON caches to editable GeoPackages, then remove JSON after verification."""
    config = load_yaml(config_path)
    root = resolve_path(config.get("project_root", "."))
    input_dir = resolve_path(config["input_dir"], root)
    osm = config.get("osm", config)
    if str(osm.get("vector_format", "GPKG")).upper() != "GPKG":
        raise ValueError("OSM 标准化阶段统一使用 GeoPackage；vector_format 必须为 GPKG")
    legacy_dir = resolve_path(osm.get("legacy_json_dir", "data/osm_raw"), root)
    cache_dir = resolve_path(osm.get("cache_dir", "data/osm_cache"), root)
    candidate_dir = resolve_path(osm.get("candidate_dir", "data/labels/osm_candidates"), root)
    excluded = set(osm.get("excluded_highways", []))
    widths = osm.get("road_width_m", config.get("road_width_m", {}))
    default_width = float(osm.get("default_road_width_m", config.get("default_road_width_m", 5.0)))
    remove_json = bool(osm.get("remove_legacy_json_after_conversion", True))
    json_files = sorted(legacy_dir.glob("*/*.json")) if legacy_dir.exists() else []
    converted: dict[str, list[Path]] = {}

    for number, source in enumerate(json_files, 1):
        kind = "building" if source.name.startswith("building_") else "road" if source.name.startswith("road_") else None
        if kind is None:
            continue
        target = cache_dir / source.parent.name / f"{source.stem}.gpkg"
        emit(f"[{number}/{len(json_files)}] 转换 {source.relative_to(root)} → {target.relative_to(root)}")
        if not _valid_vector_cache(target, kind):
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload.get("elements"), list):
                raise ValueError(f"旧缓存不是有效 Overpass JSON: {source}")
            _write_vector_cache(target, kind, payload, excluded, widths, default_width)
        converted.setdefault(source.parent.name, []).append(target)

    tiffs = {path.stem: path for path in list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))}
    summaries = {}
    for image_id in sorted(converted):
        tif = tiffs.get(image_id)
        if tif is None:
            raise FileNotFoundError(f"找不到旧 OSM 缓存对应的 TIFF: {image_id}")
        building_frames, road_frames = [], []
        for cache in sorted((cache_dir / image_id).glob("*.gpkg")):
            kind = "building" if cache.name.startswith("building_") else "road"
            buildings, roads = _read_vector_cache(cache, kind)
            building_frames.append(buildings)
            road_frames.append(roads)
        with rasterio.open(tif) as src:
            bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        buildings = _clip(_merge_frames(building_frames), bounds4326)
        road_lines = _clip(_merge_frames(road_frames), bounds4326)
        road_surfaces = _clip(make_road_surfaces(road_lines, widths, default_width), bounds4326)
        candidate = candidate_dir / f"{image_id}.gpkg"
        _write_gpkg(candidate, buildings, road_lines, road_surfaces)
        if not _valid_candidate(candidate):
            raise RuntimeError(f"汇总 GeoPackage 校验失败: {candidate}")
        summary = {
            "source_image": str(tif.resolve()), "converted_at": datetime.now(timezone.utc).isoformat(),
            "building_count": len(buildings), "road_count": len(road_lines),
            "candidate": str(candidate.resolve()), "review_status": "candidate",
            "vector_format": "GPKG",
        }
        save_yaml(summary, candidate.with_suffix(".download.yaml"))
        summaries[image_id] = summary
        emit(f"  汇总完成：{candidate.name}，建筑 {len(buildings)}，道路 {len(road_lines)}")

    if remove_json:
        for source in json_files:
            kind = "building" if source.name.startswith("building_") else "road" if source.name.startswith("road_") else None
            target = cache_dir / source.parent.name / f"{source.stem}.gpkg"
            if kind and _valid_vector_cache(target, kind) and _valid_candidate(candidate_dir / f"{source.parent.name}.gpkg"):
                source.unlink()
        for old_summary in candidate_dir.glob("*.download.json"):
            yaml_summary = old_summary.with_suffix(".yaml")
            if yaml_summary.exists():
                old_summary.unlink()

    result = {
        "json_found": len(json_files), "images": len(summaries),
        "cache_dir": str(cache_dir), "candidate_dir": str(candidate_dir),
        "legacy_json_removed": remove_json,
    }
    emit(f"旧 OSM 数据标准化完成: {result}")
    return result


def run(config_path: str | Path, emit: Callable[[str], None] = print) -> dict:
    config = load_yaml(config_path)
    root = resolve_path(config.get("project_root", "."))
    input_dir = resolve_path(config["input_dir"], root)
    osm = config.get("osm", config)
    if str(osm.get("vector_format", "GPKG")).upper() != "GPKG":
        raise ValueError("OSM 下载阶段统一使用 GeoPackage；vector_format 必须为 GPKG")
    raw_dir = resolve_path(osm.get("cache_dir", "data/osm_cache"), root)
    candidate_dir = resolve_path(osm.get("candidate_dir", "data/labels/osm_candidates"), root)
    log = build_logger("makeLabel.osm", raw_dir / "osm_download.log")
    tiffs = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"没有找到 TIFF: {input_dir}")
    endpoints = list(osm.get("endpoints") or ["https://overpass-api.de/api/interpreter"])
    max_span = float(osm.get("max_span_deg", 0.015))
    timeout = int(osm.get("query_timeout", 180))
    retries = int(osm.get("retries", 4))
    retry_delay = float(osm.get("retry_delay_seconds", 2.0))
    request_delay = float(osm.get("request_delay_seconds", 0.5))
    resume = bool(osm.get("resume", True))
    overwrite = bool(osm.get("overwrite_candidates", False))
    excluded = set(osm.get("excluded_highways", []))
    widths = osm.get("road_width_m", config.get("road_width_m", {}))
    default_width = float(osm.get("default_road_width_m", config.get("default_road_width_m", 5.0)))
    session = requests.Session()
    totals = {"images": len(tiffs), "downloaded": 0, "skipped": 0, "buildings": 0, "roads": 0}

    try:
        for image_number, tif in enumerate(tiffs, 1):
            candidate_path = candidate_dir / f"{tif.stem}.gpkg"
            if _valid_candidate(candidate_path) and not overwrite:
                emit(f"[{image_number}/{len(tiffs)}] 跳过已有候选: {candidate_path.name}")
                totals["skipped"] += 1
                continue
            with rasterio.open(tif) as src:
                if src.crs is None:
                    raise ValueError(f"影像缺少 CRS: {tif}")
                bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            cells = split_bounds(bounds4326, max_span)
            emit(f"[{image_number}/{len(tiffs)}] {tif.name}: {len(cells)} 个下载网格")
            building_frames: list[gpd.GeoDataFrame] = []
            road_frames: list[gpd.GeoDataFrame] = []
            for cell_number, (row, column, cell_bounds) in enumerate(cells, 1):
                emit(f"  网格 {cell_number}/{len(cells)} 行{row}列{column}")
                cell_dir = raw_dir / tif.stem
                for kind in ("building", "road"):
                    cache = cell_dir / f"{kind}_r{row:03d}_c{column:03d}.gpkg"
                    was_cached = resume and _valid_vector_cache(cache, kind)
                    if was_cached:
                        cell_buildings, cell_roads = _read_vector_cache(cache, kind)
                    else:
                        payload = download_cell(
                            kind, cell_bounds, endpoints, timeout, retries, retry_delay,
                            osm.get("user_agent", "BuildAndRoadModel/1.0 (local research)"), session,
                        )
                        cell_buildings, cell_roads = _write_vector_cache(
                            cache, kind, payload, excluded, widths, default_width,
                        )
                    building_frames.append(cell_buildings)
                    road_frames.append(cell_roads)
                    if not was_cached and request_delay > 0:
                        time.sleep(request_delay)
            buildings = _merge_frames(building_frames)
            road_lines = _merge_frames(road_frames)
            buildings = _clip(buildings, bounds4326)
            road_lines = _clip(road_lines, bounds4326)
            road_surfaces = _clip(make_road_surfaces(road_lines, widths, default_width), bounds4326)
            _write_gpkg(candidate_path, buildings, road_lines, road_surfaces)
            summary = {
                "source_image": str(tif.resolve()),
                "source_bounds_epsg4326": bounds4326,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "grid_count": len(cells),
                "building_count": len(buildings),
                "road_count": len(road_lines),
                "candidate": str(candidate_path.resolve()),
                "review_status": "candidate",
            }
            save_yaml(summary, candidate_path.with_suffix(".download.yaml"))
            totals["downloaded"] += 1
            totals["buildings"] += len(buildings)
            totals["roads"] += len(road_lines)
            emit(f"  完成：建筑 {len(buildings)}，道路 {len(road_lines)} → {candidate_path.name}")
            log.info("completed %s summary=%s", tif, summary)
    finally:
        session.close()
        close_logger(log)
    emit(f"OSM 下载完成: {totals}")
    return {"candidate_dir": str(candidate_dir), **totals}
