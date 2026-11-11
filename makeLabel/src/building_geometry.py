from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.affinity import translate


def _image_rule(config: dict, image_id: str) -> dict:
    base = dict(config.get("roof_to_base", {}))
    override = base.pop("per_image", {}).get(str(image_id), {})
    base.update(override)
    return base


def _pixel_offsets(transform, dx_pixels: float, dy_pixels: float) -> tuple[float, float]:
    """Convert image-column/image-row movement to CRS movement."""
    x_offset = dx_pixels * transform.a + dy_pixels * transform.b
    y_offset = dx_pixels * transform.d + dy_pixels * transform.e
    return x_offset, y_offset


def roof_to_base(
    roofs: gpd.GeoDataFrame,
    raster,
    config: dict,
    image_id: str,
) -> gpd.GeoDataFrame:
    """Translate visible roof polygons to estimated ground-base polygons.

    Azimuth is the direction in which the roof must move to reach the base,
    clockwise from geographic north. It is deliberately not called satellite
    azimuth because vendor metadata conventions differ.
    """
    if roofs.empty:
        return roofs.copy()
    if roofs.crs is None:
        raise ValueError("building_roof 图层缺少 CRS")
    rule = _image_rule(config, image_id)
    mode = str(rule.get("mode", "none"))
    local = roofs.to_crs(raster.crs).copy()

    if mode == "fixed_pixels":
        xoff, yoff = _pixel_offsets(
            raster.transform,
            float(rule.get("dx_pixels", 0.0)),
            float(rule.get("dy_pixels", 0.0)),
        )
        local.geometry = [translate(geometry, xoff=xoff, yoff=yoff) for geometry in local.geometry]
        local["base_shift_method"] = "fixed_pixels"
        local["base_dx_px"] = float(rule.get("dx_pixels", 0.0))
        local["base_dy_px"] = float(rule.get("dy_pixels", 0.0))
        return local

    if mode == "feature_pixels":
        x_field = str(rule.get("dx_field", "base_dx_px"))
        y_field = str(rule.get("dy_field", "base_dy_px"))
        if x_field not in local.columns or y_field not in local.columns:
            raise ValueError(f"building_roof 缺少逐建筑偏移字段: {x_field}, {y_field}")
        shifted = []
        for _, row in local.iterrows():
            xoff, yoff = _pixel_offsets(raster.transform, float(row[x_field]), float(row[y_field]))
            shifted.append(translate(row.geometry, xoff=xoff, yoff=yoff))
        local.geometry = shifted
        local["base_shift_method"] = "feature_pixels"
        return local

    raster_crs = CRS.from_user_input(raster.crs)
    uses_metres = (
        raster_crs.is_projected
        and raster_crs.axis_info
        and abs(float(raster_crs.axis_info[0].unit_conversion_factor) - 1.0) < 1e-9
    )
    metric_crs = raster.crs if uses_metres else local.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("无法确定屋顶偏移所需的本地米制坐标系")
    metric = local.to_crs(metric_crs).copy()

    if mode == "fixed_metres":
        east = float(rule.get("east_m", 0.0))
        north = float(rule.get("north_m", 0.0))
        metric.geometry = [translate(geometry, xoff=east, yoff=north) for geometry in metric.geometry]
        metric["base_shift_method"] = "fixed_metres"
        metric["base_east_m"] = east
        metric["base_north_m"] = north
        return metric.to_crs(raster.crs)

    if mode == "angle_height":
        angle = math.radians(float(rule["off_nadir_deg"]))
        azimuth = math.radians(float(rule["correction_azimuth_deg"]))
        default_height = float(rule.get("default_height_m", 12.0))
        level_height = float(rule.get("level_height_m", 3.0))
        height_field = str(rule.get("height_field", "height_m"))
        levels_field = str(rule.get("levels_field", "building_levels"))
        shifted, distances, heights = [], [], []
        for _, row in metric.iterrows():
            height = row.get(height_field)
            if height is None or pd.isna(height):
                levels = row.get(levels_field)
                height = float(levels) * level_height if levels is not None and not pd.isna(levels) else default_height
            height = float(height)
            distance = height * math.tan(angle)
            east = distance * math.sin(azimuth)
            north = distance * math.cos(azimuth)
            shifted.append(translate(row.geometry, xoff=east, yoff=north))
            distances.append(distance)
            heights.append(height)
        metric.geometry = shifted
        metric["base_shift_method"] = "angle_height"
        metric["base_height_m"] = heights
        metric["base_shift_m"] = distances
        return metric.to_crs(raster.crs)

    raise ValueError(
        f"存在 building_roof，但 roof_to_base.mode={mode!r}。"
        "必须选择 fixed_pixels、feature_pixels、fixed_metres 或 angle_height，不能把屋顶直接作为底座。"
    )
