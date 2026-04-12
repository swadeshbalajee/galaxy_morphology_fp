
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_ENV = "APP_CONFIG_PATH"
DEFAULT_CONFIG_FILE = "config.yaml"


@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.getenv(DEFAULT_CONFIG_ENV) or DEFAULT_CONFIG_FILE).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_meta"] = {
        "config_path": str(path),
        "project_root": str(path.parent.resolve()),
    }
    return config


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def project_root(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    return Path(cfg["_meta"]["project_root"])


def resolve_path(config: dict[str, Any], dotted_key: str) -> Path:
    raw_value = get_config_value(config, dotted_key)
    if raw_value is None:
        raise KeyError(f"Missing config path for key: {dotted_key}")
    path = Path(str(raw_value))
    if path.is_absolute():
        return path
    return project_root(config) / path
