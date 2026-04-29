from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config import get_config_value, load_config
from src.common.io_utils import class_dirs, list_image_files


@dataclass
class DatasetValidationResult:
    is_valid: bool
    class_count: int
    total_images: int
    issues: list[str]
    per_class_counts: dict[str, int]


def validate_dataset_layout(
    source_dir: str | Path, expected_classes: list[str] | None = None
) -> DatasetValidationResult:
    config = load_config()
    source_dir = Path(source_dir)
    issues: list[str] = []
    expected_classes = expected_classes or get_config_value(config, "data.classes", [])
    allowed_suffixes = get_config_value(
        config, "data.allowed_suffixes", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )

    if not source_dir.exists():
        return DatasetValidationResult(
            False, 0, 0, [f"Source directory not found: {source_dir}"], {}
        )

    classes = class_dirs(source_dir)
    per_class_counts: dict[str, int] = {}
    total_images = 0

    if expected_classes:
        missing = sorted(set(expected_classes) - {c.name for c in classes})
        if missing:
            issues.append(f"Missing expected class folders: {', '.join(missing)}")

    for class_dir in classes:
        images = list_image_files(class_dir, allowed_suffixes=allowed_suffixes)
        per_class_counts[class_dir.name] = len(images)
        total_images += len(images)
        if len(images) == 0:
            issues.append(
                f"Class folder {class_dir.name!r} contains no supported image files."
            )

    if total_images == 0:
        issues.append("No images detected in dataset.")

    min_required_class_count = int(
        get_config_value(config, "data.min_required_class_count", 2)
    )
    if len(classes) < min_required_class_count:
        issues.append(f"Expected at least {min_required_class_count} class folders.")

    return DatasetValidationResult(
        is_valid=len(issues) == 0,
        class_count=len(classes),
        total_images=total_images,
        issues=issues,
        per_class_counts=per_class_counts,
    )
