from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window


def raster_summary(path: str | Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "path": str(Path(path).resolve()),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
            "resolution": tuple(src.res),
        }


def checksum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sliding_windows(width: int, height: int, tile: int, stride: int) -> Iterator[Window]:
    if tile <= 0 or stride <= 0:
        raise ValueError("tile 和 stride 必须大于 0")
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    last_x = max(width - tile, 0)
    last_y = max(height - tile, 0)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    for y in ys:
        for x in xs:
            yield Window(x, y, min(tile, width - x), min(tile, height - y))


def hann_weight(size: int, minimum: float = 0.05) -> np.ndarray:
    axis = np.hanning(size).astype("float32")
    weight = np.outer(axis, axis)
    return np.maximum(weight, minimum)


def approximate_gsd(transform, latitude: float) -> tuple[float, float]:
    metres_per_degree_lat = 111_132.0
    metres_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude))
    return abs(transform.a) * metres_per_degree_lon, abs(transform.e) * metres_per_degree_lat

