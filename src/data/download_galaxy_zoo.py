from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.common.artifact_store import store_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import ensure_dir, reset_dir
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("download_data")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
FILENAME_COLUMNS = ("filename", "file_name", "image_filename", "image_name", "path")


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


@dataclass(frozen=True)
class ScoreArrays:
    merger: np.ndarray
    irregular: np.ndarray
    smooth: np.ndarray
    features: np.ndarray
    edgeon_no: np.ndarray
    spiral_yes: np.ndarray
    no_spiral: np.ndarray
    bulge_prom: np.ndarray


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


def required_dataset_columns(cols: DatasetColumns) -> list[str]:
    return list(dict.fromkeys(value for value in asdict(cols).values() if value))


def pick_column(
    columns: Iterable[str],
    required_terms: list[str],
    preferred_terms: list[str] | None = None,
) -> str | None:
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
    objid = next(
        (c for c in ["dr7objid", "objid", "dr7_objid"] if c in df.columns), None
    )
    if objid is None:
        raise ValueError("Could not find objid column in labels table")
    preferred = ["debiased", "weighted_fraction", "wt_fraction", "fraction"]
    smooth = pick_column(columns, ["smooth_or_features", "smooth"], preferred)
    features = pick_column(
        columns, ["smooth_or_features", "features_or_disk"], preferred
    )
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


def read_labels_table(labels_path: Path) -> tuple[pd.DataFrame, DatasetColumns]:
    header = pd.read_csv(labels_path, compression="gzip", nrows=0)
    normalized_header = normalize_columns(header)
    cols = detect_columns(normalized_header)
    raw_columns_by_name = {
        str(column).strip().lower(): column for column in header.columns
    }
    required_columns = required_dataset_columns(cols)
    labels = pd.read_csv(
        labels_path,
        compression="gzip",
        usecols=[raw_columns_by_name[column] for column in required_columns],
        dtype={
            raw_columns_by_name[column]: (
                "int64" if column == cols.objid else "float32"
            )
            for column in required_columns
        },
        low_memory=False,
    )
    return normalize_columns(labels), cols


def read_mapping_table(mapping_path: Path) -> pd.DataFrame:
    mapping = normalize_columns(
        pd.read_csv(
            mapping_path,
            usecols=["objid", "sample", "asset_id"],
            dtype={"objid": "int64", "asset_id": "int64", "sample": "category"},
            low_memory=False,
        )
    )
    if "sample" in mapping.columns:
        sample_mask = mapping["sample"].astype("string").str.lower().eq("original")
        if sample_mask.any():
            mapping = mapping.loc[sample_mask, ["objid", "asset_id"]].copy()
        else:
            mapping = mapping[["objid", "asset_id"]].copy()
    else:
        mapping = mapping[["objid", "asset_id"]].copy()
    return mapping.drop_duplicates(subset=["objid", "asset_id"]).reset_index(drop=True)


def sample_per_class(
    df: pd.DataFrame, label_col: str, max_per_class: int | None, seed: int
) -> pd.DataFrame:
    if max_per_class is None:
        return df
    chunks = []
    for _, group in df.groupby(label_col, sort=False):
        chunks.append(
            group
            if len(group) <= max_per_class
            else group.sample(n=max_per_class, random_state=seed)
        )
    return pd.concat(chunks, ignore_index=True) if chunks else df.head(0)


def build_zip_member_index(
    zf: zipfile.ZipFile, include_filename_indexes: bool
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    exact_index: dict[str, str] = {}
    basename_index: dict[str, str] = {}
    stem_index: dict[str, str] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        archive_name = info.filename
        if not archive_name.lower().endswith(IMAGE_SUFFIXES):
            continue
        stem_index.setdefault(Path(archive_name).stem, archive_name)
        if include_filename_indexes:
            exact_index.setdefault(archive_name, archive_name)
            basename_index.setdefault(Path(archive_name).name, archive_name)
    return exact_index, basename_index, stem_index


def write_images_from_zip(
    selected: pd.DataFrame, zip_path: Path, output_root: Path
) -> dict[str, int]:
    counts: dict[str, int] = {}
    filename_columns = [
        column for column in FILENAME_COLUMNS if column in selected.columns
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        exact_index, basename_index, stem_index = build_zip_member_index(
            zf,
            include_filename_indexes=bool(filename_columns),
        )
        for _, row in selected.iterrows():
            label = str(row["label"])
            asset_id = str(row["asset_id"])
            candidate_names = [
                str(row.get(col)).strip()
                for col in filename_columns
                if col in row and pd.notna(row[col]) and str(row.get(col)).strip()
            ]
            candidate_names.extend(
                [f"{asset_id}.jpg", f"{asset_id}.jpeg", f"{asset_id}.png", asset_id]
            )
            archive_name = None
            for candidate in candidate_names:
                if exact_index and candidate in exact_index:
                    archive_name = exact_index[candidate]
                    break
                base = Path(candidate).name
                if basename_index and base in basename_index:
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


def score_values(df: pd.DataFrame, column: str | None) -> np.ndarray:
    if not column or column not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[column].fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def prepare_score_arrays(df: pd.DataFrame, cols: DatasetColumns) -> ScoreArrays:
    return ScoreArrays(
        merger=score_values(df, cols.merger),
        irregular=np.maximum(
            score_values(df, cols.irregular), score_values(df, cols.disturbed)
        ),
        smooth=score_values(df, cols.smooth),
        features=score_values(df, cols.features),
        edgeon_no=score_values(df, cols.edgeon_no),
        spiral_yes=score_values(df, cols.spiral_yes),
        no_spiral=score_values(df, cols.no_spiral),
        bulge_prom=np.maximum(
            score_values(df, cols.bulge_obvious), score_values(df, cols.bulge_dominant)
        ),
    )


def assign_labels_vectorized(
    scores: ScoreArrays,
    threshold: float,
    merger_threshold: float,
    bulge_threshold: float,
) -> np.ndarray:
    labels = np.full(scores.smooth.shape[0], "", dtype=object)
    assigned = np.zeros(scores.smooth.shape[0], dtype=bool)

    merger_mask = scores.merger >= merger_threshold
    labels[merger_mask] = "merger"
    assigned |= merger_mask

    irregular_mask = (~assigned) & (scores.irregular >= merger_threshold)
    labels[irregular_mask] = "irregular"
    assigned |= irregular_mask

    elliptical_mask = (
        (~assigned) & (scores.smooth >= threshold) & (scores.smooth >= scores.features)
    )
    labels[elliptical_mask] = "elliptical"
    assigned |= elliptical_mask

    spiral_mask = (
        (~assigned)
        & (scores.features >= threshold)
        & (scores.edgeon_no >= threshold)
        & (scores.spiral_yes >= threshold)
    )
    labels[spiral_mask] = "spiral"
    assigned |= spiral_mask

    lenticular_mask = (
        (~assigned)
        & (scores.features >= threshold)
        & (scores.edgeon_no >= threshold)
        & (scores.no_spiral >= threshold)
        & (scores.bulge_prom >= bulge_threshold)
    )
    labels[lenticular_mask] = "lenticular"
    return labels


def summarize_class_counts(
    labels: np.ndarray, classes: list[str], max_per_class: int | None
) -> tuple[dict[str, int], dict[str, int]]:
    eligible_counts = {
        label: int(np.count_nonzero(labels == label)) for label in classes
    }
    sampled_counts = {
        label: (min(count, max_per_class) if max_per_class is not None else count)
        for label, count in eligible_counts.items()
    }
    return eligible_counts, sampled_counts


def materialize_selection(
    merged: pd.DataFrame, labels: np.ndarray, max_per_class: int | None, seed: int
) -> pd.DataFrame:
    selected_mask = labels != ""
    if not np.any(selected_mask):
        return merged.head(0).assign(label=pd.Series(dtype="object"))
    selected = merged.loc[selected_mask].copy()
    selected["label"] = labels[selected_mask]
    return sample_per_class(selected, "label", max_per_class=max_per_class, seed=seed)


def choose_thresholds(
    merged: pd.DataFrame, cols: DatasetColumns, classes: list[str], config: dict
) -> tuple[pd.DataFrame, dict]:
    target_per_class = int(
        get_config_value(config, "data.target_images_per_class", 1000)
    )
    max_per_class = int(get_config_value(config, "data.max_images_per_class", 1000))
    seed = int(get_config_value(config, "data.sampling_seed", 42))
    threshold_start = float(get_config_value(config, "data.threshold_start", 0.8))
    threshold_min = float(get_config_value(config, "data.threshold_min", 0.35))
    threshold_step = float(get_config_value(config, "data.threshold_step", 0.05))
    merger_start = float(get_config_value(config, "data.merger_threshold_start", 0.6))
    merger_min = float(get_config_value(config, "data.merger_threshold_min", 0.3))
    bulge_start = float(
        get_config_value(config, "data.lenticular_bulge_threshold_start", 0.25)
    )
    bulge_min = float(
        get_config_value(config, "data.lenticular_bulge_threshold_min", 0.05)
    )

    scores = prepare_score_arrays(merged, cols)
    best_labels = np.full(len(merged), "", dtype=object)
    best_meta: dict = {
        "threshold": None,
        "merger_threshold": None,
        "bulge_threshold": None,
        "class_counts": {cls: 0 for cls in classes},
        "eligible_class_counts": {cls: 0 for cls in classes},
    }

    threshold = threshold_start
    while threshold >= threshold_min - 1e-9:
        merger_threshold = max(merger_min, merger_start - (threshold_start - threshold))
        bulge_threshold = max(bulge_min, bulge_start - (threshold_start - threshold))
        labels = assign_labels_vectorized(
            scores, threshold, merger_threshold, bulge_threshold
        )
        eligible_counts, class_counts = summarize_class_counts(
            labels, classes, max_per_class=max_per_class
        )
        LOGGER.info(
            "Threshold search candidate threshold=%.2f merger=%.2f bulge=%.2f eligible=%s sampled=%s",
            threshold,
            merger_threshold,
            bulge_threshold,
            eligible_counts,
            class_counts,
        )
        if min(class_counts.values()) > min(
            best_meta.get("class_counts", {c: 0 for c in classes}).values()
        ):
            best_labels = labels.copy()
            best_meta = {
                "threshold": threshold,
                "merger_threshold": merger_threshold,
                "bulge_threshold": bulge_threshold,
                "class_counts": class_counts,
                "eligible_class_counts": eligible_counts,
            }
        if all(class_counts.get(label, 0) >= target_per_class for label in classes):
            return materialize_selection(
                merged, labels, max_per_class=max_per_class, seed=seed
            ), {
                "threshold": threshold,
                "merger_threshold": merger_threshold,
                "bulge_threshold": bulge_threshold,
                "class_counts": class_counts,
                "eligible_class_counts": eligible_counts,
            }
        threshold = round(threshold - threshold_step, 4)
    LOGGER.warning(
        "Target per-class count not reached for all classes. Using best candidate=%s",
        best_meta,
    )
    return (
        materialize_selection(
            merged, best_labels, max_per_class=max_per_class, seed=seed
        ),
        best_meta,
    )


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

    labels, cols = read_labels_table(labels_path)
    mapping = read_mapping_table(mapping_path)
    merged = mapping.merge(labels, left_on="objid", right_on=cols.objid, how="inner")
    if cols.objid != "objid" and cols.objid in merged.columns:
        merged = merged.drop(columns=[cols.objid])

    selected, chosen = choose_thresholds(merged, cols, classes, config)
    counts = write_images_from_zip(selected, images_zip, output_root)
    actual_counts = {
        label: len(list((output_root / label).glob("*"))) for label in classes
    }
    summary = {
        "stage": "fetch_raw",
        "status": "success",
        "layout": get_config_value(config, "data.layout"),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "selected_rows": int(len(selected)),
        "requested_max_per_class": int(
            get_config_value(config, "data.max_images_per_class", 1000)
        ),
        "threshold_search": chosen,
        "written_counts": counts,
        "actual_counts": actual_counts,
        "resolved_columns": asdict(cols),
        "seed": seed,
    }
    store_pipeline_artifact("raw_summary", summary, config=config)
    LOGGER.info("Raw dataset build complete: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and materialize the Galaxy Zoo dataset subset for this project."
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
    output = (
        Path(args.output_root)
        if args.output_root
        else resolve_path(cfg, "paths.raw_dataset_dir")
    )
    cache = (
        Path(args.cache_dir)
        if args.cache_dir
        else resolve_path(cfg, "paths.raw_cache_dir")
    )
    try:
        result = build_dataset(output, cache)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
