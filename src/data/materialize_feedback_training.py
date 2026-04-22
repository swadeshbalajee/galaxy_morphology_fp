from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from psycopg.rows import dict_row

from src.common.artifact_store import store_pipeline_artifact
from src.common.config import get_config_value, load_config, project_root, resolve_path
from src.common.io_utils import ensure_dir, reset_dir
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database

LOGGER = configure_logging("feedback_materializer")

MANIFEST_COLUMNS = [
    "prediction_id",
    "original_filename",
    "corrected_label",
    "model_version",
    "feedback_created_at",
    "image_path",
]
CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "feedback_image"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _resolve_suffix(original_filename: str, content_type: str | None, allowed_suffixes: set[str]) -> str:
    suffix = Path(original_filename).suffix.lower()
    if suffix in allowed_suffixes:
        return suffix
    guessed = CONTENT_TYPE_SUFFIXES.get((content_type or "").lower(), ".jpg")
    return guessed if guessed in allowed_suffixes else ".jpg"


def _fetch_feedback_rows() -> list[dict]:
    initialize_database()
    with get_db_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    c.prediction_id,
                    c.corrected_label,
                    c.created_at AS feedback_created_at,
                    p.original_filename,
                    p.content_type,
                    p.image_bytes,
                    p.model_version
                FROM feedback_corrections c
                JOIN predictions p ON p.prediction_id = c.prediction_id
                ORDER BY c.created_at ASC
                """,
            )
            return list(cur.fetchall())


def initialize_feedback_training_snapshot(
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict:
    config = load_config()
    output_root = Path(output_root) if output_root else resolve_path(config, "paths.feedback_training_dir")
    manifest_path = Path(manifest_path) if manifest_path else resolve_path(config, "paths.feedback_training_manifest_path")
    summary_path = Path(summary_path) if summary_path else resolve_path(config, "paths.feedback_training_summary_path")

    reset_dir(output_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

    summary = {
        "stage": "materialize_feedback_training",
        "status": "initialized",
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "source_feedback_count": 0,
        "materialized_feedback_count": 0,
        "class_counts": {},
        "skipped_invalid_labels": {},
    }
    store_pipeline_artifact("feedback_training_summary", summary, config=config)
    LOGGER.info("Initialized empty feedback training snapshot at %s", output_root)
    return summary


def materialize_feedback_training_dataset(
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict:
    config = load_config()
    root = project_root(config)
    output_root = Path(output_root) if output_root else resolve_path(config, "paths.feedback_training_dir")
    manifest_path = Path(manifest_path) if manifest_path else resolve_path(config, "paths.feedback_training_manifest_path")
    summary_path = Path(summary_path) if summary_path else resolve_path(config, "paths.feedback_training_summary_path")

    allowed_labels = {str(label).strip().lower() for label in get_config_value(config, "data.classes", [])}
    allowed_suffixes = {str(suffix).lower() for suffix in get_config_value(config, "data.allowed_suffixes", [])}
    output_root = reset_dir(output_root)

    rows = _fetch_feedback_rows()
    manifest_rows: list[dict[str, str]] = []
    class_counts = {label: 0 for label in sorted(allowed_labels)}
    skipped_invalid_labels: dict[str, int] = {}

    for row in rows:
        corrected_label = str(row["corrected_label"]).strip().lower()
        if corrected_label not in allowed_labels:
            skipped_invalid_labels[corrected_label] = skipped_invalid_labels.get(corrected_label, 0) + 1
            LOGGER.warning("Skipping feedback row for unsupported label=%s prediction_id=%s", corrected_label, row["prediction_id"])
            continue

        suffix = _resolve_suffix(
            original_filename=str(row["original_filename"]),
            content_type=str(row.get("content_type") or ""),
            allowed_suffixes=allowed_suffixes,
        )
        filename = f"{row['prediction_id']}__{_safe_stem(str(row['original_filename']))}{suffix}"
        destination = ensure_dir(output_root / corrected_label) / filename
        destination.write_bytes(bytes(row["image_bytes"]))

        manifest_rows.append(
            {
                "prediction_id": str(row["prediction_id"]),
                "original_filename": str(row["original_filename"]),
                "corrected_label": corrected_label,
                "model_version": str(row["model_version"]),
                "feedback_created_at": row["feedback_created_at"].isoformat(),
                "image_path": destination.relative_to(root).as_posix(),
            }
        )
        class_counts[corrected_label] = class_counts.get(corrected_label, 0) + 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "stage": "materialize_feedback_training",
        "status": "success",
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "source_feedback_count": len(rows),
        "materialized_feedback_count": len(manifest_rows),
        "class_counts": class_counts,
        "skipped_invalid_labels": skipped_invalid_labels,
    }
    store_pipeline_artifact("feedback_training_summary", summary, config=config)
    LOGGER.info("Feedback training snapshot materialized: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize accepted feedback into a DVC-tracked training dataset snapshot.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kwargs = {
        "output_root": Path(args.output_root) if args.output_root else None,
        "manifest_path": Path(args.manifest_path) if args.manifest_path else None,
        "summary_path": Path(args.summary_path) if args.summary_path else None,
    }
    if args.initialize_only:
        initialize_feedback_training_snapshot(**kwargs)
    else:
        materialize_feedback_training_dataset(**kwargs)
