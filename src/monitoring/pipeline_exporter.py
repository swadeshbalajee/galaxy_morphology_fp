from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from psycopg.rows import dict_row

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_json
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database

LOGGER = configure_logging('pipeline_exporter')
app = FastAPI(title='Galaxy Pipeline Exporter', version='1.1.0')


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'pipeline-exporter'}


@app.get('/metrics')
def metrics():
    config = load_config()
    registry = CollectorRegistry()

    g_test_accuracy = Gauge('galaxy_pipeline_test_accuracy', 'Latest offline test accuracy.', registry=registry)
    g_test_macro_f1 = Gauge('galaxy_pipeline_test_macro_f1', 'Latest offline test macro f1.', registry=registry)
    g_live_accuracy = Gauge('galaxy_live_feedback_accuracy', 'Latest live-feedback accuracy.', registry=registry)
    g_live_macro_f1 = Gauge('galaxy_live_feedback_macro_f1', 'Latest live-feedback macro f1.', registry=registry)
    g_feedback_count = Gauge('galaxy_live_feedback_count', 'Total live feedback rows.', registry=registry)
    g_train_duration = Gauge('galaxy_pipeline_train_duration_seconds', 'Latest train duration.', registry=registry)
    g_raw_count = Gauge('galaxy_raw_images_total', 'Raw materialized images by class.', ['label'], registry=registry)
    g_log_size = Gauge('galaxy_log_file_bytes', 'Tracked log file size.', ['log_file'], registry=registry)
    g_prediction_db_count = Gauge('galaxy_db_prediction_count', 'Total predictions stored in Postgres.', registry=registry)
    g_correction_db_count = Gauge('galaxy_db_correction_count', 'Total corrections stored in Postgres.', registry=registry)

    test_metrics = read_json(resolve_path(config, 'paths.test_metrics_path'), {})
    live_metrics = read_json(resolve_path(config, 'paths.live_metrics_path'), {})
    raw_summary = read_json(resolve_path(config, 'paths.raw_summary_path'), {})
    runtime_summary = read_json(resolve_path(config, 'paths.pipeline_runtime_summary_path'), {})

    if isinstance(test_metrics.get('accuracy'), (int, float)):
        g_test_accuracy.set(float(test_metrics['accuracy']))
    if isinstance(test_metrics.get('macro_f1'), (int, float)):
        g_test_macro_f1.set(float(test_metrics['macro_f1']))
    if isinstance(live_metrics.get('accuracy'), (int, float)):
        g_live_accuracy.set(float(live_metrics['accuracy']))
    if isinstance(live_metrics.get('macro_f1'), (int, float)):
        g_live_macro_f1.set(float(live_metrics['macro_f1']))
    g_feedback_count.set(float(live_metrics.get('feedback_count', 0) or 0))
    if isinstance(runtime_summary.get('train_duration_seconds'), (int, float)):
        g_train_duration.set(float(runtime_summary['train_duration_seconds']))

    for label, count in (raw_summary.get('actual_counts') or {}).items():
        g_raw_count.labels(label=label).set(float(count))

    logs_dir = resolve_path(config, 'paths.logs_dir')
    for path in logs_dir.glob('*.log'):
        g_log_size.labels(log_file=path.name).set(float(path.stat().st_size))

    airflow_logs_dir = resolve_path(config, 'paths.artifacts_dir').parent / 'airflow' / 'logs'
    if airflow_logs_dir.exists():
        total_size = sum(p.stat().st_size for p in airflow_logs_dir.rglob('*') if p.is_file())
        g_log_size.labels(log_file='airflow_total').set(float(total_size))

    try:
        initialize_database()
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute('SELECT COUNT(*) AS prediction_count FROM predictions')
                g_prediction_db_count.set(float(cur.fetchone()['prediction_count']))
                cur.execute('SELECT COUNT(*) AS correction_count FROM feedback_corrections')
                g_correction_db_count.set(float(cur.fetchone()['correction_count']))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Unable to collect Postgres-backed exporter metrics: %s', exc)

    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
