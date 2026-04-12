
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(folder: str | Path, allowed_suffixes: list[str] | set[str] | None = None) -> list[Path]:
    folder = Path(folder)
    allowed = {s.lower() for s in (allowed_suffixes or ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])}
    return sorted([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in allowed])


def class_dirs(source_dir: str | Path) -> list[Path]:
    source = Path(source_dir)
    if not source.exists():
        return []
    return sorted([p for p in source.iterdir() if p.is_dir()])


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))
