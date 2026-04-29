from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image

from src.common.io_utils import list_image_files


def compute_image_baseline(folder: str | Path) -> dict:
    files = list_image_files(folder)
    brightness_values: list[float] = []
    widths: list[int] = []
    heights: list[int] = []

    for path in files:
        with Image.open(path).convert("RGB") as image:
            widths.append(image.width)
            heights.append(image.height)
            pixels = list(image.getdata())
            pixel_mean = sum(sum(pixel) / 3 for pixel in pixels) / len(pixels)
            brightness_values.append(pixel_mean)

    if not brightness_values:
        return {
            "count": 0,
            "brightness_mean": 0.0,
            "brightness_std": 0.0,
            "width_mean": 0.0,
            "height_mean": 0.0,
        }

    return {
        "count": len(brightness_values),
        "brightness_mean": mean(brightness_values),
        "brightness_std": (
            pstdev(brightness_values) if len(brightness_values) > 1 else 0.0
        ),
        "width_mean": mean(widths),
        "height_mean": mean(heights),
    }


def save_baseline(folder: str | Path, output_path: str | Path) -> dict:
    baseline = compute_image_baseline(folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline
