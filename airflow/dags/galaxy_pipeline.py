from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.task.trigger_rule import TriggerRule

PROJECT_ROOT = Path('/opt/airflow/project')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database
from src.common.email_utils import send_email_report

CONFIG = load_config(PROJECT_ROOT / 'config.yaml')
LOGGER = configure_logging('airflow_control_plane')
TRAINING_VENV = Path('/opt/venvs/training')
TRAINING_BIN = str(TRAINING_VENV / 'bin')
TRAINING_PYTHON = str(TRAINING_VENV / 'bin' / 'python')
CONTROL_STATE_PATH = resolve_path(CONFIG, 'paths.control_plane_state_path')
RAW_DATASET_DIR = resolve_path(CONFIG, 'paths.raw_dataset_dir')
MODELS_DIR = resolve_path(CONFIG, 'paths.models_dir')
TEST_METRICS_PATH = resolve_path(CONFIG, 'paths.test_metrics_path')
LIVE_METRICS_PATH = resolve_path(CONFIG, 'paths.live_metrics_path')
FEEDBACK_TRAINING_DIR = resolve_path(CONFIG, 'paths.feedback_training_dir')
FEEDBACK_TRAINING_MANIFEST_PATH = resolve_path(CONFIG, 'paths.feedback_training_manifest_path')
FEEDBACK_TRAINING_SUMMARY_PATH = resolve_path(CONFIG, 'paths.feedback_training_summary_path')
REPORT_MD_PATH = resolve_path(CONFIG, 'paths.latest_report_md_path')
REPORT_HTML_PATH = resolve_path(CONFIG, 'paths.latest_report_html_path')
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
DEFAULT_ARGS = {'owner': 'airflow', 'retries': 1, 'retry_delay': timedelta(minutes=5)}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace('+00:00', 'Z')


def _build_env(use_training_venv: bool = False) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env['APP_CONFIG_PATH'] = str(CONFIG_PATH)
    env['PROJECT_ROOT'] = str(PROJECT_ROOT)
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    if use_training_venv:
        env['PATH'] = TRAINING_BIN + os.pathsep + env.get('PATH', '')
        env['VIRTUAL_ENV'] = str(TRAINING_VENV)
        env['TRAINING_PYTHON'] = TRAINING_PYTHON
    return env


def _run_command(command: list[str], use_training_venv: bool = False) -> None:
    LOGGER.info('Running command: %s', ' '.join(command))
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=_build_env(use_training_venv=use_training_venv),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        LOGGER.info('STDOUT\n%s', result.stdout)
    if result.stderr:
        if result.returncode == 0:
            LOGGER.info('STDERR (non-fatal)\n%s', result.stderr)
        else:
            LOGGER.warning('STDERR\n%s', result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _feedback_count() -> int:
    try:
        initialize_database()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) AS feedback_count FROM feedback_corrections')
                row = cur.fetchone()
                return int(row['feedback_count'] if isinstance(row, dict) else row[0])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Unable to read feedback count from Postgres: %s', exc)
        return 0


def _feedback_snapshot_count(state: dict | None = None) -> int:
    current_state = state if state is not None else (read_json(CONTROL_STATE_PATH, {}) or {})
    return int(current_state.get('last_feedback_snapshot_count', 0))


def sync_feedback_training_snapshot() -> dict:
    from src.data.materialize_feedback_training import (
        initialize_feedback_training_snapshot,
        materialize_feedback_training_dataset,
    )

    state = read_json(CONTROL_STATE_PATH, {}) or {}
    feedback_count = _feedback_count()
    last_snapshot_count = _feedback_snapshot_count(state)
    min_new_feedback = int(get_config_value(CONFIG, 'continuous_improvement.min_new_feedback_samples', 5))
    new_feedback = max(feedback_count - last_snapshot_count, 0)

    if not (
        FEEDBACK_TRAINING_DIR.exists()
        and FEEDBACK_TRAINING_MANIFEST_PATH.exists()
        and FEEDBACK_TRAINING_SUMMARY_PATH.exists()
    ):
        initialize_feedback_training_snapshot(
            output_root=FEEDBACK_TRAINING_DIR,
            manifest_path=FEEDBACK_TRAINING_MANIFEST_PATH,
            summary_path=FEEDBACK_TRAINING_SUMMARY_PATH,
        )

    if new_feedback < min_new_feedback:
        result = {
            'updated': False,
            'feedback_count': feedback_count,
            'last_feedback_snapshot_count': last_snapshot_count,
            'new_feedback_count': new_feedback,
            'threshold': min_new_feedback,
        }
        LOGGER.info('Feedback training snapshot unchanged: %s', result)
        return result

    summary = materialize_feedback_training_dataset(
        output_root=FEEDBACK_TRAINING_DIR,
        manifest_path=FEEDBACK_TRAINING_MANIFEST_PATH,
        summary_path=FEEDBACK_TRAINING_SUMMARY_PATH,
    )
    refreshed_state = read_json(CONTROL_STATE_PATH, {}) or {}
    refreshed_state['last_feedback_snapshot_count'] = feedback_count
    refreshed_state['last_feedback_snapshot_at'] = _utc_now_iso()
    write_json(CONTROL_STATE_PATH, refreshed_state)
    result = {
        'updated': True,
        'feedback_count': feedback_count,
        'last_feedback_snapshot_count': feedback_count,
        'new_feedback_count': new_feedback,
        'threshold': min_new_feedback,
        **summary,
    }
    LOGGER.info('Feedback training snapshot refreshed: %s', result)
    return result


def inspect_runtime_state() -> dict:
    state = read_json(CONTROL_STATE_PATH, {}) or {}
    state.pop('last_feedback_count', None)
    test_metrics = read_json(TEST_METRICS_PATH, {}) or {}
    live_metrics = read_json(LIVE_METRICS_PATH, {}) or {}
    feedback_count = _feedback_count()
    previous_feedback_count = _feedback_snapshot_count(state)
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
        'checked_at': _utc_now_iso(),
        'raw_exists': raw_exists,
        'model_exists': model_exists,
        'feedback_count': feedback_count,
        'previous_feedback_count': previous_feedback_count,
        'new_feedback_count': new_feedback,
        'last_feedback_snapshot_count': previous_feedback_count,
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
    sync_feedback_training_snapshot()
    _run_command([TRAINING_PYTHON, '-m', 'dvc', 'repro', 'report'], use_training_venv=True)


def register_best_model() -> None:
    _run_command([TRAINING_PYTHON, '-m', 'src.registry.register_best_model'], use_training_venv=True)


def refresh_report_after_registry() -> None:
    _run_command([TRAINING_PYTHON, '-m', 'src.reporting.generate_report'], use_training_venv=True)


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
    state.pop('last_feedback_count', None)
    state['last_report_sent_at'] = _utc_now_iso()
    write_json(CONTROL_STATE_PATH, state)


with DAG(
    dag_id='galaxy_morphology_control_plane',
    description='Airflow control plane for the DVC-driven galaxy pipeline',
    start_date=datetime(2026, 1, 1),
    schedule=get_config_value(CONFIG, 'continuous_improvement.monitor_schedule', '30 12 * * *'),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=['mlops', 'galaxy', 'control-plane'],
) as dag:
    decide = BranchPythonOperator(task_id='inspect_and_branch', python_callable=branch_runtime_decision)
    run_pipeline = PythonOperator(task_id='run_dvc_pipeline', python_callable=run_dvc_pipeline)
    register_model = PythonOperator(task_id='register_best_model', python_callable=register_best_model)
    refresh_report = PythonOperator(task_id='refresh_report_after_registry', python_callable=refresh_report_after_registry)
    skip_retraining = EmptyOperator(task_id='skip_retraining')
    reload_service = PythonOperator(task_id='reload_model_service', python_callable=reload_model_service)
    send_email = PythonOperator(task_id='send_report_email', python_callable=send_latest_report, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    finish = EmptyOperator(task_id='finish', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    decide >> run_pipeline >> register_model >> refresh_report >> reload_service >> send_email >> finish
    decide >> skip_retraining >> send_email >> finish
