"""Shared infrastructure for the three independent pipeline stages."""

# Conda's GDAL packages keep schema/projection files under Library/share.
# Some Python wheels do not discover them automatically when an entry script is
# launched directly, so configure the conventional paths before importing GIS libs.
import os
import sys
from pathlib import Path

_prefix = Path(sys.prefix)
_gdal_data = _prefix / "Library" / "share" / "gdal"
_proj_data = _prefix / "Library" / "share" / "proj"
if _gdal_data.is_dir():
    os.environ.setdefault("GDAL_DATA", str(_gdal_data))
if _proj_data.is_dir():
    os.environ.setdefault("PROJ_DATA", str(_proj_data))
