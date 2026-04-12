
from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import class_dirs, ensure_dir, list_image_files, reset_dir, write_json
from src.common.logging_utils import configure_logging
from src.data.validate import validate_dataset_layout
from src.features.baseline import save_baseline

LOGGER = configure_logging("preprocess_final")


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def build_training_ready_dataset(source_dir: str | Path, output_dir: str | Path, seed: int | None = None) -> dict:
    config = load_config()
    source_dir = Path(source_dir)
    output_dir = reset_dir(output_dir)
    validation = validate_dataset_layout(source_dir)
    if not validation.is_valid:
        raise ValueError("Processed v1 validation failed: " + "; ".join(validation.issues))

    seed = int(seed if seed is not None else get_config_value(config, "project.random_seed", 42))
    random.seed(seed)
    train_ratio = float(get_config_value(config, "data.train_split", 0.70))
    val_ratio = float(get_config_value(config, "data.val_split", 0.15))
    start = time.time()

    for split in ["train", "val", "test"]:
        ensure_dir(output_dir / split)

    manifest_rows: list[dict[str, str]] = []
    split_summary: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for class_dir in class_dirs(source_dir):
        files = list_image_files(class_dir)
        random.shuffle(files)
        train_count, val_count, test_count = split_counts(len(files), train_ratio, val_ratio)
        chunk_map = {
            "train": files[:train_count],
            "val": files[train_count:train_count + val_count],
            "test": files[train_count + val_count:train_count + val_count + test_count],
        }

        for split_name, split_files in chunk_map.items():
            destination_class_dir = ensure_dir(output_dir / split_name / class_dir.name)
            split_summary[split_name][class_dir.name] = len(split_files)
            for path in split_files:
                destination = destination_class_dir / path.name
                shutil.copy2(path, destination)
                manifest_rows.append({
                    "source_path": str(path),
                    "destination_path": str(destination),
                    "split": split_name,
                    "label": class_dir.name,
                })
        LOGGER.info(
            "Final preprocessing complete for class=%s train=%s val=%s test=%s",
            class_dir.name,
            train_count,
            val_count,
            test_count,
        )

    artifacts_dir = resolve_path(config, "paths.artifacts_dir")
    manifest_path = artifacts_dir / "data_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "destination_path", "split", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    baseline = save_baseline(output_dir / "train", resolve_path(config, "paths.drift_baseline_path"))
    summary = {
        "stage": "preprocess_final",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "splits": split_summary,
        "manifest_path": str(manifest_path),
        "baseline": baseline,
        "duration_seconds": round(time.time() - start, 3),
    }
    write_json(resolve_path(config, "paths.processed_final_summary_path"), summary)
    LOGGER.info("Final preprocessing summary: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final training-ready dataset version.")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    source = args.source_dir or resolve_path(cfg, "paths.processed_v1_dir")
    output = args.output_dir or resolve_path(cfg, "paths.processed_final_dir")
    build_training_ready_dataset(source, output, seed=args.seed)
