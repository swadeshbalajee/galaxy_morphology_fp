from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from src.common.config import get_config_value, load_config

LOGGER = logging.getLogger("postgres")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def get_database_url(config: dict | None = None) -> str:
    cfg = config or load_config()
    env_name = get_config_value(cfg, 'database.app_url_env', 'DATABASE_URL')
    url = os.getenv(env_name) or get_config_value(cfg, 'database.app_url')
    if not url:
        raise RuntimeError(f'Database URL is not configured. Expected env var {env_name} or config database.app_url.')
    return str(url)


@contextmanager
def get_db_connection(row_factory=dict_row, autocommit: bool = True) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(get_database_url(), row_factory=row_factory, autocommit=autocommit)
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
    CONSTRAINT uq_feedback_corrections_prediction UNIQUE (prediction_id),
    CONSTRAINT chk_feedback_corrections_diff CHECK (predicted_label <> corrected_label)
);

CREATE INDEX IF NOT EXISTS idx_feedback_corrections_created_at ON feedback_corrections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_corrections_prediction_created_at ON feedback_corrections (prediction_created_at DESC);
"""


def initialize_database() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute('ANALYZE prediction_batches;')
            cur.execute('ANALYZE predictions;')
            cur.execute('ANALYZE feedback_uploads;')
            cur.execute('ANALYZE feedback_corrections;')
    LOGGER.info('Postgres schema initialized successfully.')


def cluster_table(table_name: str, index_name: str) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'CLUSTER {table_name} USING {index_name};')
            cur.execute(f'ANALYZE {table_name};')
    LOGGER.info('Clustered table=%s using index=%s', table_name, index_name)
