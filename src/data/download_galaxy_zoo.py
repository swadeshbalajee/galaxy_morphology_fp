
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import ensure_dir, reset_dir, write_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("download_data")


@dataclass
class DatasetColumns:
    objid: str
    smooth: str
    features: str
    edgeon_yes: str | None
    edgeon_no: str | None
    spiral_yes: str | None
    no_spiral: str | None
    merger: str | None
    irregular: str | None
    disturbed: str | None
    bulge_obvious: str | None
    bulge_dominant: str | None


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Download skipped, already present: %s", destination)
        return destination

    LOGGER.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    LOGGER.info("Download complete: %s", destination)
    return destination


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def pick_column(columns: Iterable[str], required_terms: list[str], preferred_terms: list[str] | None = None) -> str | None:
    preferred_terms = preferred_terms or []
    candidates: list[tuple[int, str]] = []
    for column in columns:
        column_l = column.lower()
        if all(term.lower() in column_l for term in required_terms):
            score = sum(term.lower() in column_l for term in preferred_terms)
            candidates.append((score, column))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def detect_columns(df: pd.DataFrame) -> DatasetColumns:
    columns = list(df.columns)
    objid = next((c for c in ["dr7objid", "objid", "dr7_objid"] if c in df.columns), None)
    if objid is None:
        raise ValueError("Could not find objid column in labels table")
    preferred = ["debiased", "weighted_fraction", "wt_fraction", "fraction"]
    smooth = pick_column(columns, ["smooth_or_features", "smooth"], preferred)
    features = pick_column(columns, ["smooth_or_features", "features_or_disk"], preferred)
    return DatasetColumns(
        objid=objid,
        smooth=smooth,
        features=features,
        edgeon_yes=pick_column(columns, ["edgeon", "yes"], preferred),
        edgeon_no=pick_column(columns, ["edgeon", "no"], preferred),
        spiral_yes=pick_column(columns, ["spiral", "spiral"], preferred),
        no_spiral=pick_column(columns, ["spiral", "no_spiral"], preferred),
        merger=pick_column(columns, ["odd", "merger"], preferred),
        irregular=pick_column(columns, ["odd", "irregular"], preferred),
        disturbed=pick_column(columns, ["odd", "disturbed"], preferred),
        bulge_obvious=pick_column(columns, ["bulge", "obvious"], preferred),
        bulge_dominant=pick_column(columns, ["bulge", "dominant"], preferred),
    )


def score(row: pd.Series, column: str | None) -> float:
    if not column:
        return 0.0
    value = row.get(column, 0.0)
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def assign_label(row: pd.Series, cols: DatasetColumns, threshold: float, merger_threshold: float, bulge_threshold: float) -> str | None:
    merger = score(row, cols.merger)
    irregular = max(score(row, cols.irregular), score(row, cols.disturbed))
    smooth = score(row, cols.smooth)
    features = score(row, cols.features)
    edgeon_no = score(row, cols.edgeon_no)
    spiral_yes = score(row, cols.spiral_yes)
    no_spiral = score(row, cols.no_spiral)
    bulge_prom = max(score(row, cols.bulge_obvious), score(row, cols.bulge_dominant))

    if merger >= merger_threshold:
        return "merger"
    if irregular >= merger_threshold:
        return "irregular"
    if smooth >= threshold and smooth >= features:
        return "elliptical"
    if features >= threshold and edgeon_no >= threshold and spiral_yes >= threshold:
        return "spiral"
    if features >= threshold and edgeon_no >= threshold and no_spiral >= threshold and bulge_prom >= bulge_threshold:
        return "lenticular"
    return None


def sample_per_class(df: pd.DataFrame, label_col: str, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_per_class is None:
        return df
    chunks = []
    for _, group in df.groupby(label_col):
        chunks.append(group if len(group) <= max_per_class else group.sample(n=max_per_class, random_state=seed))
    return pd.concat(chunks, ignore_index=True) if chunks else df.head(0)


def write_images_from_zip(selected: pd.DataFrame, zip_path: Path, output_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_members = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        exact_index = {name: name for name in zip_members}
        basename_index = {Path(name).name: name for name in zip_members}
        stem_index = {Path(name).stem: name for name in zip_members}
        for _, row in selected.iterrows():
            label = str(row["label"])
            asset_id = str(row["asset_id"])
            candidate_names = [
                str(row.get(col)).strip()
                for col in ["filename", "file_name", "image_filename", "image_name", "path"]
                if col in row and pd.notna(row[col]) and str(row.get(col)).strip()
            ]
            candidate_names.extend([f"{asset_id}.jpg", f"{asset_id}.jpeg", f"{asset_id}.png", asset_id])
            archive_name = None
            for candidate in candidate_names:
                if candidate in exact_index:
                    archive_name = exact_index[candidate]
                    break
                base = Path(candidate).name
                if base in basename_index:
                    archive_name = basename_index[base]
                    break
                stem = Path(candidate).stem
                if stem in stem_index:
                    archive_name = stem_index[stem]
                    break
            if archive_name is None:
                continue
            destination_dir = ensure_dir(output_root / label)
            destination = destination_dir / Path(archive_name).name
            if destination.exists():
                counts[label] = counts.get(label, 0) + 1
                continue
            with zf.open(archive_name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[label] = counts.get(label, 0) + 1
    return counts


def choose_thresholds(merged: pd.DataFrame, cols: DatasetColumns, classes: list[str], config: dict) -> tuple[pd.DataFrame, dict]:
    target_per_class = int(get_config_value(config, "data.target_images_per_class", 1000))
    max_per_class = int(get_config_value(config, "data.max_images_per_class", 1000))
    seed = int(get_config_value(config, "data.sampling_seed", 42))
    threshold_start = float(get_config_value(config, "data.threshold_start", 0.8))
    threshold_min = float(get_config_value(config, "data.threshold_min", 0.35))
    threshold_step = float(get_config_value(config, "data.threshold_step", 0.05))
    merger_start = float(get_config_value(config, "data.merger_threshold_start", 0.6))
    merger_min = float(get_config_value(config, "data.merger_threshold_min", 0.3))
    bulge_start = float(get_config_value(config, "data.lenticular_bulge_threshold_start", 0.25))
    bulge_min = float(get_config_value(config, "data.lenticular_bulge_threshold_min", 0.05))

    best_df = merged.head(0).copy()
    best_meta: dict = {"class_counts": {cls: 0 for cls in classes}}

    threshold = threshold_start
    while threshold >= threshold_min - 1e-9:
        merger_threshold = max(merger_min, merger_start - (threshold_start - threshold))
        bulge_threshold = max(bulge_min, bulge_start - (threshold_start - threshold))
        labeled = merged.copy()
        labeled["label"] = labeled.apply(
            lambda row: assign_label(row, cols, threshold, merger_threshold, bulge_threshold),
            axis=1,
        )
        labeled = labeled[labeled["label"].notna()].copy()
        sampled = sample_per_class(labeled, "label", max_per_class=max_per_class, seed=seed)
        class_counts = {label: int(sampled[sampled['label'] == label].shape[0]) for label in classes}
        LOGGER.info(
            "Threshold search candidate threshold=%.2f merger=%.2f bulge=%.2f counts=%s",
            threshold,
            merger_threshold,
            bulge_threshold,
            class_counts,
        )
        if min(class_counts.values()) > min(best_meta.get("class_counts", {c: 0 for c in classes}).values()):
            best_df = sampled
            best_meta = {
                "threshold": threshold,
                "merger_threshold": merger_threshold,
                "bulge_threshold": bulge_threshold,
                "class_counts": class_counts,
            }
        if all(class_counts.get(label, 0) >= target_per_class for label in classes):
            return sampled, best_meta
        threshold = round(threshold - threshold_step, 4)
    LOGGER.warning("Target per-class count not reached for all classes. Using best candidate=%s", best_meta)
    return best_df, best_meta


def build_dataset(output_root: Path, cache_dir: Path) -> dict:
    config = load_config()
    output_root = reset_dir(output_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    labels_url = get_config_value(config, "data.source.labels_url")
    mapping_url = get_config_value(config, "data.source.mapping_url")
    images_url = get_config_value(config, "data.source.images_url")
    classes = get_config_value(config, "data.classes", [])
    seed = int(get_config_value(config, "data.sampling_seed", 42))

    labels_path = download_file(labels_url, cache_dir / "gz2_hart16.csv.gz")
    mapping_path = download_file(mapping_url, cache_dir / "gz2_filename_mapping.csv")
    images_zip = download_file(images_url, cache_dir / "images_gz2.zip")

    labels = normalize_columns(pd.read_csv(labels_path, compression="gzip", low_memory=False))
    mapping = normalize_columns(pd.read_csv(mapping_path, low_memory=False))
    cols = detect_columns(labels)

    mapping["objid"] = mapping["objid"].astype(str)
    labels[cols.objid] = labels[cols.objid].astype(str)
    merged = mapping.merge(labels, left_on="objid", right_on=cols.objid, how="inner")
    if "sample" in merged.columns:
        merged = merged[merged["sample"].astype(str).str.lower() == "original"].copy()

    selected, chosen = choose_thresholds(merged, cols, classes, config)
    counts = write_images_from_zip(selected, images_zip, output_root)
    actual_counts = {label: len(list((output_root / label).glob("*"))) for label in classes}
    summary = {
        "stage": "fetch_raw",
        "status": "success",
        "layout": get_config_value(config, "data.layout"),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "selected_rows": int(len(selected)),
        "requested_max_per_class": int(get_config_value(config, "data.max_images_per_class", 1000)),
        "threshold_search": chosen,
        "written_counts": counts,
        "actual_counts": actual_counts,
        "resolved_columns": asdict(cols),
        "seed": seed,
    }
    write_json(resolve_path(config, "paths.raw_summary_path"), summary)
    write_json(cache_dir / "download_summary_project5.json", summary)
    LOGGER.info("Raw dataset build complete: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and materialize the Galaxy Zoo dataset subset for this project.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    output = Path(args.output_root) if args.output_root else resolve_path(cfg, "paths.raw_dataset_dir")
    cache = Path(args.cache_dir) if args.cache_dir else resolve_path(cfg, "paths.raw_cache_dir")
    try:
        result = build_dataset(output, cache)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
