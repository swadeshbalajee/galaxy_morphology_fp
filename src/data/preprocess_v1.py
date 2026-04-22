
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from src.common.artifact_store import store_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import class_dirs, ensure_dir, list_image_files, reset_dir
from src.common.logging_utils import configure_logging
from src.data.validate import validate_dataset_layout

LOGGER = configure_logging("preprocess_v1")


def preprocess_v1(source_dir: str | Path, output_dir: str | Path) -> dict:
    config = load_config()
    source_dir = Path(source_dir)
    output_dir = reset_dir(output_dir)
    validation = validate_dataset_layout(source_dir)
    if not validation.is_valid:
        raise ValueError("Raw dataset validation failed: " + "; ".join(validation.issues))

    resize_to = int(get_config_value(config, "data.preprocess_v1_size", 256))
    start = time.time()
    summary = {
        "stage": "preprocess_v1",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "resize_to": resize_to,
        "classes": {},
        "total_images": 0,
    }

    for class_dir in class_dirs(source_dir):
        destination_class = ensure_dir(output_dir / class_dir.name)
        written = 0
        for image_path in list_image_files(class_dir):
            with Image.open(image_path) as image:
                processed = image.convert("RGB").resize((resize_to, resize_to))
                destination = destination_class / f"{image_path.stem}.jpg"
                processed.save(destination, format="JPEG", quality=95)
            written += 1
        summary["classes"][class_dir.name] = written
        summary["total_images"] += written
        LOGGER.info("Preprocess v1 complete for class=%s count=%s", class_dir.name, written)

    summary["duration_seconds"] = round(time.time() - start, 3)
    store_pipeline_artifact("processed_v1_summary", summary, config=config)
    LOGGER.info("Preprocess v1 finished: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the first processed dataset version from raw images.")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    source = args.source_dir or resolve_path(cfg, "paths.raw_dataset_dir")
    output = args.output_dir or resolve_path(cfg, "paths.processed_v1_dir")
    preprocess_v1(source, output)
