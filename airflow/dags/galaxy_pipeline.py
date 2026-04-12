
from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import requests
from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from src.common.config import get_config_value, load_config
from src.common.email_utils import send_email_report
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('airflow_control_plane')
CONFIG = load_config('/opt/airflow/project/config.yaml')
PROJECT_ROOT = Path('/opt/airflow/project')
CONTROL_STATE_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.control_plane_state_path', 'artifacts/control_plane_state.json')
PREDICTIONS_DB_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.predictions_db_path', 'artifacts/predictions.db')
REPORT_MD_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.latest_report_md_path', 'artifacts/reports/latest_report.md')
REPORT_HTML_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.latest_report_html_path', 'artifacts/reports/latest_report.html')
TEST_METRICS_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.test_metrics_path', 'artifacts/test_metrics.json')
LIVE_METRICS_PATH = PROJECT_ROOT / get_config_value(CONFIG, 'paths.live_metrics_path', 'artifacts/live_metrics.json')
RAW_DATASET_DIR = PROJECT_ROOT / get_config_value(CONFIG, 'paths.raw_dataset_dir', 'data/raw/galaxy_dataset')
MODELS_DIR = PROJECT_ROOT / get_config_value(CONFIG, 'paths.models_dir', 'models/latest')

DEFAULT_ARGS = {'owner': 'mlops', 'retries': 1}


def _run_command(command: list[str]) -> None:
    LOGGER.info('Running command: %s', ' '.join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if result.stdout:
        LOGGER.info('STDOUT\n%s', result.stdout)
    if result.stderr:
        LOGGER.warning('STDERR\n%s', result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def _feedback_count() -> int:
    if not PREDICTIONS_DB_PATH.exists():
        return 0
    with sqlite3.connect(PREDICTIONS_DB_PATH) as conn:
        return int(conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0])


def inspect_runtime_state() -> dict:
    state = read_json(CONTROL_STATE_PATH, {}) or {}
    test_metrics = read_json(TEST_METRICS_PATH, {}) or {}
    live_metrics = read_json(LIVE_METRICS_PATH, {}) or {}
    feedback_count = _feedback_count()
    previous_feedback_count = int(state.get('last_feedback_count', 0))
    new_feedback = max(feedback_count - previous_feedback_count, 0)
    raw_exists = RAW_DATASET_DIR.exists() and any(RAW_DATASET_DIR.rglob('*.jpg'))
    model_exists = MODELS_DIR.exists() and (MODELS_DIR / 'class_names.json').exists()

    accuracy_threshold = float(get_config_value(CONFIG, 'continuous_improvement.accuracy_threshold', 0.85))
    macro_f1_threshold = float(get_config_value(CONFIG, 'continuous_improvement.macro_f1_threshold', 0.80))
    min_new_feedback = int(get_config_value(CONFIG, 'continuous_improvement.min_new_feedback_samples', 5))
    min_total_feedback = int(get_config_value(CONFIG, 'continuous_improvement.min_total_feedback_samples', 10))

    offline_accuracy = test_metrics.get('accuracy')
    offline_macro_f1 = test_metrics.get('macro_f1')
    live_accuracy = live_metrics.get('accuracy')
    live_macro_f1 = live_metrics.get('macro_f1')

    model_degraded = False
    if isinstance(live_accuracy, (int, float)) and feedback_count >= min_total_feedback and live_accuracy < accuracy_threshold:
        model_degraded = True
    elif isinstance(live_macro_f1, (int, float)) and feedback_count >= min_total_feedback and live_macro_f1 < macro_f1_threshold:
        model_degraded = True
    elif isinstance(offline_accuracy, (int, float)) and offline_accuracy < accuracy_threshold:
        model_degraded = True
    elif isinstance(offline_macro_f1, (int, float)) and offline_macro_f1 < macro_f1_threshold:
        model_degraded = True

    runtime_state = {
        'checked_at': datetime.utcnow().isoformat() + 'Z',
        'raw_exists': raw_exists,
        'model_exists': model_exists,
        'feedback_count': feedback_count,
        'previous_feedback_count': previous_feedback_count,
        'new_feedback_count': new_feedback,
        'offline_accuracy': offline_accuracy,
        'offline_macro_f1': offline_macro_f1,
        'live_accuracy': live_accuracy,
        'live_macro_f1': live_macro_f1,
        'model_degraded': model_degraded,
        'should_run_pipeline': (not raw_exists) or (not model_exists) or model_degraded or (new_feedback >= min_new_feedback),
    }
    write_json(CONTROL_STATE_PATH, {**state, **runtime_state})
    LOGGER.info('Runtime inspection: %s', runtime_state)
    return runtime_state


def branch_runtime_decision() -> str:
    state = inspect_runtime_state()
    return 'run_dvc_pipeline' if state['should_run_pipeline'] else 'skip_retraining'


def run_dvc_pipeline() -> None:
    _run_command(['dvc', 'repro', 'report'])


def reload_model_service() -> None:
    response = requests.post(get_config_value(CONFIG, 'services.model_service_url', 'http://model-service:8001') + '/reload', timeout=60)
    response.raise_for_status()
    LOGGER.info('Model service reloaded successfully: %s', response.text)


def send_latest_report() -> None:
    report_text = REPORT_MD_PATH.read_text(encoding='utf-8') if REPORT_MD_PATH.exists() else 'No report generated yet.'
    report_html = REPORT_HTML_PATH.read_text(encoding='utf-8') if REPORT_HTML_PATH.exists() else None
    subject = f"{get_config_value(CONFIG, 'email.subject_prefix', '[Galaxy MLOps]')} Latest pipeline report"
    send_email_report(CONFIG, subject=subject, body_text=report_text, body_html=report_html, attachments=[REPORT_MD_PATH, REPORT_HTML_PATH])
    state = read_json(CONTROL_STATE_PATH, {}) or {}
    state['last_feedback_count'] = _feedback_count()
    state['last_report_sent_at'] = datetime.utcnow().isoformat() + 'Z'
    write_json(CONTROL_STATE_PATH, state)


with DAG(
    dag_id='galaxy_morphology_control_plane',
    description='Airflow control plane for the DVC-driven galaxy pipeline',
    start_date=datetime(2026, 1, 1),
    schedule=get_config_value(CONFIG, 'continuous_improvement.monitor_schedule', '*/30 * * * *'),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=['mlops', 'galaxy', 'control-plane'],
) as dag:
    decide = BranchPythonOperator(task_id='inspect_and_branch', python_callable=branch_runtime_decision)
    run_pipeline = PythonOperator(task_id='run_dvc_pipeline', python_callable=run_dvc_pipeline)
    skip_retraining = EmptyOperator(task_id='skip_retraining')
    reload_service = PythonOperator(task_id='reload_model_service', python_callable=reload_model_service)
    send_email = PythonOperator(task_id='send_report_email', python_callable=send_latest_report, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    finish = EmptyOperator(task_id='finish', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    decide >> run_pipeline >> reload_service >> send_email >> finish
    decide >> skip_retraining >> send_email >> finish
