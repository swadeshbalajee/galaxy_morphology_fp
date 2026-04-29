from __future__ import annotations

from datetime import date, datetime, timedelta
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from src.common.config import get_config_value, load_config
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("postgres")


def get_database_url(config: dict | None = None) -> str:
    cfg = config or load_config()
    env_name = get_config_value(cfg, "database.app_url_env", "DATABASE_URL")
    url = os.getenv(env_name) or get_config_value(cfg, "database.app_url")
    if not url:
        raise RuntimeError(
            f"Database URL is not configured. Expected env var {env_name} or config database.app_url."
        )
    return str(url)


@contextmanager
def get_db_connection(
    row_factory=dict_row, autocommit: bool = True
) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(
        get_database_url(), row_factory=row_factory, autocommit=autocommit
    )
    try:
        yield conn
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_batches (
    batch_id UUID PRIMARY KEY,
    source_filename TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('single_image', 'zip_archive')),
    total_files INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    batch_id UUID REFERENCES prediction_batches(batch_id) ON DELETE SET NULL,
    original_filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    image_bytes BYTEA NOT NULL,
    predicted_label TEXT NOT NULL,
    top_k JSONB NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    brightness_zscore DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_label ON predictions (predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions (batch_id);

CREATE TABLE IF NOT EXISTS feedback_uploads (
    upload_id UUID PRIMARY KEY,
    source_filename TEXT NOT NULL,
    raw_csv BYTEA NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_uploads_created_at ON feedback_uploads (created_at DESC);

CREATE TABLE IF NOT EXISTS feedback_corrections (
    correction_id UUID PRIMARY KEY,
    upload_id UUID REFERENCES feedback_uploads(upload_id) ON DELETE SET NULL,
    prediction_id UUID NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    corrected_label TEXT NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    prediction_created_at TIMESTAMPTZ NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_feedback_corrections_prediction UNIQUE (prediction_id)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_feedback_corrections_diff'
          AND conrelid = 'feedback_corrections'::regclass
    ) THEN
        ALTER TABLE feedback_corrections
            ADD CONSTRAINT chk_feedback_corrections_diff
            CHECK (predicted_label <> corrected_label) NOT VALID;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_feedback_corrections_created_at ON feedback_corrections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_corrections_prediction_created_at ON feedback_corrections (prediction_created_at DESC);

CREATE TABLE IF NOT EXISTS control_plane_state (
    state_id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (state_id = 1),
    last_feedback_snapshot_count INTEGER NOT NULL DEFAULT 0,
    last_feedback_snapshot_at TIMESTAMPTZ,
    last_report_sent_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE control_plane_state
    ADD COLUMN IF NOT EXISTS last_pipeline_config_fingerprint TEXT;

ALTER TABLE control_plane_state
    ADD COLUMN IF NOT EXISTS last_pipeline_config_updated_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS pipeline_artifact_snapshots (
    artifact_id BIGINT GENERATED ALWAYS AS IDENTITY,
    artifact_key TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    run_id TEXT,
    source_path TEXT,
    payload JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    recorded_date DATE NOT NULL DEFAULT CURRENT_DATE,
    payload_version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (recorded_date, artifact_id)
) PARTITION BY RANGE (recorded_date);

CREATE INDEX IF NOT EXISTS idx_pipeline_artifacts_key_recorded_at
    ON pipeline_artifact_snapshots (artifact_key, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_artifacts_stage_recorded_at
    ON pipeline_artifact_snapshots (stage_name, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_artifacts_recorded_date
    ON pipeline_artifact_snapshots (recorded_date DESC);

CREATE TABLE IF NOT EXISTS service_logs (
    log_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    service TEXT NOT NULL,
    component TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    exception TEXT,
    pathname TEXT,
    lineno INTEGER,
    func_name TEXT,
    process_id INTEGER,
    thread_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_service_logs_created_at
    ON service_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_logs_service_created_at
    ON service_logs (service, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_logs_level_created_at
    ON service_logs (level, created_at DESC);

CREATE OR REPLACE VIEW latest_pipeline_artifact_snapshots AS
SELECT DISTINCT ON (artifact_key)
    artifact_key,
    stage_name,
    run_id,
    source_path,
    payload,
    recorded_at,
    recorded_date,
    payload_version
FROM pipeline_artifact_snapshots
ORDER BY artifact_key, recorded_at DESC, artifact_id DESC;
"""


def initialize_database() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute("ANALYZE prediction_batches;")
            cur.execute("ANALYZE predictions;")
            cur.execute("ANALYZE feedback_uploads;")
            cur.execute("ANALYZE feedback_corrections;")
            cur.execute("ANALYZE control_plane_state;")
            cur.execute("ANALYZE pipeline_artifact_snapshots;")
            cur.execute("ANALYZE service_logs;")
        ensure_pipeline_artifact_partition(conn, date.today())
        ensure_pipeline_artifact_partition(conn, date.today() + timedelta(days=31))
    LOGGER.info("Postgres schema initialized successfully.")


def cluster_table(table_name: str, index_name: str) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"CLUSTER {table_name} USING {index_name};")
            cur.execute(f"ANALYZE {table_name};")
    LOGGER.info("Clustered table=%s using index=%s", table_name, index_name)


def ensure_pipeline_artifact_partition(
    conn: psycopg.Connection, for_date: date | datetime | None = None
) -> str:
    target = (
        for_date.date()
        if isinstance(for_date, datetime)
        else (for_date or date.today())
    )
    month_start = target.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    partition_name = f"pipeline_artifact_snapshots_{month_start:%Y%m}"
    with conn.cursor() as cur:
        statement = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF pipeline_artifact_snapshots
            FOR VALUES FROM ({month_start}) TO ({next_month})
            """).format(
            partition_name=sql.Identifier(partition_name),
            month_start=sql.Literal(month_start),
            next_month=sql.Literal(next_month),
        )
        cur.execute(statement)
    return partition_name


def _serialize_timestamp(value: datetime | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.isoformat().replace("+00:00", "Z")


def _normalize_control_plane_state(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "last_feedback_snapshot_count": 0,
            "last_feedback_snapshot_at": None,
            "last_report_sent_at": None,
            "last_pipeline_config_fingerprint": None,
            "last_pipeline_config_updated_at": None,
            "updated_at": None,
        }
    return {
        "last_feedback_snapshot_count": int(
            row.get("last_feedback_snapshot_count") or 0
        ),
        "last_feedback_snapshot_at": _serialize_timestamp(
            row.get("last_feedback_snapshot_at")
        ),
        "last_report_sent_at": _serialize_timestamp(row.get("last_report_sent_at")),
        "last_pipeline_config_fingerprint": row.get("last_pipeline_config_fingerprint"),
        "last_pipeline_config_updated_at": _serialize_timestamp(
            row.get("last_pipeline_config_updated_at")
        ),
        "updated_at": _serialize_timestamp(row.get("updated_at")),
    }


def get_control_plane_state(
    default_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    initial_state = default_state or {}
    initial_count = int(initial_state.get("last_feedback_snapshot_count") or 0)
    initial_snapshot_at = initial_state.get("last_feedback_snapshot_at")
    initial_report_sent_at = initial_state.get("last_report_sent_at")
    initial_pipeline_config_fingerprint = initial_state.get(
        "last_pipeline_config_fingerprint"
    )
    initial_pipeline_config_updated_at = initial_state.get(
        "last_pipeline_config_updated_at"
    )

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO control_plane_state (
                    state_id,
                    last_feedback_snapshot_count,
                    last_feedback_snapshot_at,
                    last_report_sent_at,
                    last_pipeline_config_fingerprint,
                    last_pipeline_config_updated_at
                )
                VALUES (1, %s, %s, %s, %s, %s)
                ON CONFLICT (state_id) DO NOTHING
                """,
                (
                    initial_count,
                    initial_snapshot_at,
                    initial_report_sent_at,
                    initial_pipeline_config_fingerprint,
                    initial_pipeline_config_updated_at,
                ),
            )
            cur.execute("""
                SELECT
                    last_feedback_snapshot_count,
                    last_feedback_snapshot_at,
                    last_report_sent_at,
                    last_pipeline_config_fingerprint,
                    last_pipeline_config_updated_at,
                    updated_at
                FROM control_plane_state
                WHERE state_id = 1
                """)
            row = cur.fetchone()
    return _normalize_control_plane_state(row)


def update_control_plane_state(
    *,
    last_feedback_snapshot_count: int | None = None,
    last_feedback_snapshot_at: str | datetime | None = None,
    last_report_sent_at: str | datetime | None = None,
    last_pipeline_config_fingerprint: str | None = None,
    last_pipeline_config_updated_at: str | datetime | None = None,
) -> dict[str, Any]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO control_plane_state (state_id)
                VALUES (1)
                ON CONFLICT (state_id) DO NOTHING
                """)
            cur.execute(
                """
                UPDATE control_plane_state
                SET
                    last_feedback_snapshot_count = COALESCE(%s, last_feedback_snapshot_count),
                    last_feedback_snapshot_at = COALESCE(%s, last_feedback_snapshot_at),
                    last_report_sent_at = COALESCE(%s, last_report_sent_at),
                    last_pipeline_config_fingerprint = COALESCE(%s, last_pipeline_config_fingerprint),
                    last_pipeline_config_updated_at = COALESCE(%s, last_pipeline_config_updated_at),
                    updated_at = NOW()
                WHERE state_id = 1
                RETURNING
                    last_feedback_snapshot_count,
                    last_feedback_snapshot_at,
                    last_report_sent_at,
                    last_pipeline_config_fingerprint,
                    last_pipeline_config_updated_at,
                    updated_at
                """,
                (
                    last_feedback_snapshot_count,
                    last_feedback_snapshot_at,
                    last_report_sent_at,
                    last_pipeline_config_fingerprint,
                    last_pipeline_config_updated_at,
                ),
            )
            row = cur.fetchone()
    return _normalize_control_plane_state(row)
