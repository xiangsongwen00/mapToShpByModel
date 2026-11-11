from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    with file_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件根节点必须是对象: {file_path}")
    config = deep_merge(defaults or {}, loaded)
    config["_config_path"] = str(file_path)
    return config


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return ((Path(root) if root else project_root()) / path).resolve()


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in data.items() if not key.startswith("_")}
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(clean, stream, allow_unicode=True, sort_keys=False)

