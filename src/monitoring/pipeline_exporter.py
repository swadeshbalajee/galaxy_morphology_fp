from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
from psycopg.rows import dict_row

from src.common.artifact_store import load_pipeline_artifact
from src.common.config import load_config
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database

LOGGER = configure_logging("pipeline_exporter")
app = FastAPI(title="Galaxy Pipeline Exporter", version="1.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "pipeline-exporter"}


@app.get("/metrics")
def metrics():
    config = load_config()
    registry = CollectorRegistry()

    g_test_accuracy = Gauge(
        "galaxy_pipeline_test_accuracy",
        "Latest offline test accuracy.",
        registry=registry,
    )
    g_test_macro_f1 = Gauge(
        "galaxy_pipeline_test_macro_f1",
        "Latest offline test macro f1.",
        registry=registry,
    )
    g_live_accuracy = Gauge(
        "galaxy_live_feedback_accuracy",
        "Latest live-feedback accuracy.",
        registry=registry,
    )
    g_live_macro_f1 = Gauge(
        "galaxy_live_feedback_macro_f1",
        "Latest live-feedback macro f1.",
        registry=registry,
    )
    g_feedback_count = Gauge(
        "galaxy_live_feedback_count",
        "Latest-model predictions with received feedback.",
        registry=registry,
    )
    g_live_prediction_count = Gauge(
        "galaxy_live_prediction_count",
        "Latest-model predictions used for live accuracy.",
        registry=registry,
    )
    g_assumed_correct_count = Gauge(
        "galaxy_live_assumed_correct_count",
        "Latest-model predictions assumed correct because no correction feedback was received.",
        registry=registry,
    )
    g_train_duration = Gauge(
        "galaxy_pipeline_train_duration_seconds",
        "Latest train duration.",
        registry=registry,
    )
    g_raw_count = Gauge(
        "galaxy_raw_images_total",
        "Raw materialized images by class.",
        ["label"],
        registry=registry,
    )
    g_service_logs = Gauge(
        "galaxy_service_logs_total",
        "Application log records stored in Postgres.",
        ["service", "component", "level"],
        registry=registry,
    )
    g_prediction_db_count = Gauge(
        "galaxy_db_prediction_count",
        "Total predictions stored in Postgres.",
        registry=registry,
    )
    g_correction_db_count = Gauge(
        "galaxy_db_correction_count",
        "Total corrections stored in Postgres.",
        registry=registry,
    )

    test_metrics = load_pipeline_artifact("test_metrics", config=config, default={})
    live_metrics = load_pipeline_artifact("live_metrics", config=config, default={})
    raw_summary = load_pipeline_artifact("raw_summary", config=config, default={})
    runtime_summary = load_pipeline_artifact(
        "pipeline_runtime_summary", config=config, default={}
    )

    if isinstance(test_metrics.get("accuracy"), (int, float)):
        g_test_accuracy.set(float(test_metrics["accuracy"]))
    if isinstance(test_metrics.get("macro_f1"), (int, float)):
        g_test_macro_f1.set(float(test_metrics["macro_f1"]))
    if isinstance(live_metrics.get("accuracy"), (int, float)):
        g_live_accuracy.set(float(live_metrics["accuracy"]))
    if isinstance(live_metrics.get("macro_f1"), (int, float)):
        g_live_macro_f1.set(float(live_metrics["macro_f1"]))
    g_feedback_count.set(float(live_metrics.get("feedback_count", 0) or 0))
    g_live_prediction_count.set(float(live_metrics.get("prediction_count", 0) or 0))
    g_assumed_correct_count.set(
        float(live_metrics.get("assumed_correct_count", 0) or 0)
    )
    if isinstance(runtime_summary.get("train_duration_seconds"), (int, float)):
        g_train_duration.set(float(runtime_summary["train_duration_seconds"]))

    for label, count in (raw_summary.get("actual_counts") or {}).items():
        g_raw_count.labels(label=label).set(float(count))

    try:
        initialize_database()
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT COUNT(*) AS prediction_count FROM predictions")
                g_prediction_db_count.set(float(cur.fetchone()["prediction_count"]))
                cur.execute(
                    "SELECT COUNT(*) AS correction_count FROM feedback_corrections"
                )
                g_correction_db_count.set(float(cur.fetchone()["correction_count"]))
                cur.execute("""
                    SELECT service, component, level, COUNT(*) AS log_count
                    FROM service_logs
                    GROUP BY service, component, level
                    """)
                for row in cur.fetchall():
                    g_service_logs.labels(
                        service=row["service"],
                        component=row["component"],
                        level=row["level"],
                    ).set(float(row["log_count"]))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to collect Postgres-backed exporter metrics: %s", exc)

    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
