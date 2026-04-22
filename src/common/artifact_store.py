from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_json
from src.common.logging_utils import configure_logging
from src.common.postgres import ensure_pipeline_artifact_partition, get_db_connection, initialize_database

LOGGER = configure_logging('artifact_store')
_MISSING = object()


@dataclass(frozen=True)
class ArtifactSpec:
    stage_name: str
    path_key: str | None = None
    keep_local: bool = False


ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    'raw_summary': ArtifactSpec(stage_name='fetch_raw', path_key='paths.raw_summary_path'),
    'processed_v1_summary': ArtifactSpec(stage_name='preprocess_v1', path_key='paths.processed_v1_summary_path'),
    'processed_final_summary': ArtifactSpec(stage_name='preprocess_final', path_key='paths.processed_final_summary_path'),
    'drift_baseline': ArtifactSpec(stage_name='preprocess_final', path_key='paths.drift_baseline_path', keep_local=True),
    'feedback_training_summary': ArtifactSpec(stage_name='materialize_feedback_training', path_key='paths.feedback_training_summary_path'),
    'train_metrics': ArtifactSpec(stage_name='train', path_key='paths.train_metrics_path'),
    'validation_metrics': ArtifactSpec(stage_name='train', path_key='paths.validation_metrics_path'),
    'classification_report': ArtifactSpec(stage_name='train', path_key='paths.classification_report_path'),
    'pipeline_runtime_summary': ArtifactSpec(stage_name='train', path_key='paths.pipeline_runtime_summary_path'),
    'test_metrics': ArtifactSpec(stage_name='evaluate', path_key='paths.test_metrics_path'),
    'live_metrics': ArtifactSpec(stage_name='evaluate', path_key='paths.live_metrics_path'),
    'registry_status': ArtifactSpec(stage_name='register_best_model', path_key='paths.registry_status_path'),
}


def _normalize_recorded_at(recorded_at: str | datetime | None = None) -> datetime:
    if recorded_at is None:
        return datetime.now(timezone.utc)
    if isinstance(recorded_at, datetime):
        return recorded_at.astimezone(timezone.utc) if recorded_at.tzinfo else recorded_at.replace(tzinfo=timezone.utc)
    normalized = str(recorded_at).strip().replace('Z', '+00:00')
    parsed = datetime.fromisoformat(normalized)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _artifact_spec(artifact_key: str) -> ArtifactSpec:
    try:
        return ARTIFACT_SPECS[artifact_key]
    except KeyError as exc:  # noqa: PERF203
        raise KeyError(f'Unknown pipeline artifact key: {artifact_key}') from exc


def _artifact_path(config: dict, artifact_key: str) -> Path | None:
    spec = _artifact_spec(artifact_key)
    if not spec.path_key:
        return None
    return resolve_path(config, spec.path_key)


def _write_local_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def store_pipeline_artifact(
    artifact_key: str,
    payload: Any,
    *,
    config: dict | None = None,
    stage_name: str | None = None,
    run_id: str | None = None,
    recorded_at: str | datetime | None = None,
    keep_local: bool | None = None,
) -> Any:
    cfg = config or load_config()
    spec = _artifact_spec(artifact_key)
    artifact_path = _artifact_path(cfg, artifact_key)
    should_keep_local = spec.keep_local if keep_local is None else keep_local
    recorded_dt = _normalize_recorded_at(recorded_at)

    initialize_database()
    with get_db_connection(row_factory=dict_row) as conn:
        ensure_pipeline_artifact_partition(conn, recorded_dt)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_artifact_snapshots (
                    artifact_key,
                    stage_name,
                    run_id,
                    source_path,
                    payload,
                    recorded_at,
                    recorded_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    artifact_key,
                    stage_name or spec.stage_name,
                    run_id,
                    str(artifact_path) if artifact_path else None,
                    Jsonb(payload),
                    recorded_dt,
                    recorded_dt.date(),
                ),
            )

    if should_keep_local and artifact_path is not None:
        _write_local_json(artifact_path, payload)

    return payload


def load_pipeline_artifact(
    artifact_key: str,
    *,
    config: dict | None = None,
    default: Any = None,
    fallback_to_local: bool = True,
) -> Any:
    cfg = config or load_config()

    try:
        initialize_database()
        with get_db_connection(row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload
                    FROM latest_pipeline_artifact_snapshots
                    WHERE artifact_key = %s
                    """,
                    (artifact_key,),
                )
                row = cur.fetchone()
        if row and row.get('payload', _MISSING) is not _MISSING:
            return row['payload']
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Unable to read pipeline artifact=%s from Postgres: %s', artifact_key, exc)

    if not fallback_to_local:
        return default

    artifact_path = _artifact_path(cfg, artifact_key)
    if artifact_path is None or not artifact_path.exists():
        return default

    payload = read_json(artifact_path, _MISSING)
    if payload is _MISSING:
        return default

    try:
        store_pipeline_artifact(artifact_key, payload, config=cfg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Unable to bootstrap artifact=%s from local JSON into Postgres: %s', artifact_key, exc)

    return payload


def merge_pipeline_artifact(
    artifact_key: str,
    updates: dict[str, Any],
    *,
    config: dict | None = None,
    stage_name: str | None = None,
    run_id: str | None = None,
    recorded_at: str | datetime | None = None,
    keep_local: bool | None = None,
) -> dict[str, Any]:
    current = load_pipeline_artifact(artifact_key, config=config, default={})
    if not isinstance(current, dict):
        current = {}
    merged = {**current, **updates}
    store_pipeline_artifact(
        artifact_key,
        merged,
        config=config,
        stage_name=stage_name,
        run_id=run_id,
        recorded_at=recorded_at,
        keep_local=keep_local,
    )
    return merged
