from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.providers.smtp.hooks.smtp import SmtpHook
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.task.trigger_rule import TriggerRule

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.artifact_store import load_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging
from src.common.postgres import (
    get_control_plane_state,
    get_db_connection,
    initialize_database,
    update_control_plane_state,
)

CONFIG = load_config(PROJECT_ROOT / "config.yaml")
LOGGER = configure_logging("airflow_control_plane")
TRAINING_VENV = Path("/opt/venvs/training")
TRAINING_BIN = str(TRAINING_VENV / "bin")
TRAINING_PYTHON = str(TRAINING_VENV / "bin" / "python")
CONTROL_STATE_PATH = resolve_path(CONFIG, "paths.control_plane_state_path")
RAW_DATASET_DIR = resolve_path(CONFIG, "paths.raw_dataset_dir")
MODELS_DIR = resolve_path(CONFIG, "paths.models_dir")
FEEDBACK_TRAINING_DIR = resolve_path(CONFIG, "paths.feedback_training_dir")
FEEDBACK_TRAINING_MANIFEST_PATH = resolve_path(
    CONFIG, "paths.feedback_training_manifest_path"
)
RUNTIME_REPORT_MD_PATH = resolve_path(CONFIG, "paths.latest_runtime_report_md_path")
RUNTIME_REPORT_HTML_PATH = resolve_path(CONFIG, "paths.latest_runtime_report_html_path")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"
RUNTIME_RUNS_DIR = resolve_path(CONFIG, "paths.runtime_dir") / "runs"


def _required_config(key: str):
    value = get_config_value(CONFIG, key)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value


DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": int(_required_config("orchestration.airflow_retries")),
    "retry_delay": timedelta(
        minutes=int(_required_config("orchestration.airflow_retry_delay_minutes"))
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_env(use_training_venv: bool = False) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["APP_CONFIG_PATH"] = str(CONFIG_PATH)
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if use_training_venv:
        env["PATH"] = TRAINING_BIN + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(TRAINING_VENV)
        env["TRAINING_PYTHON"] = TRAINING_PYTHON
    return env


def _run_command(command: list[str], use_training_venv: bool = False) -> None:
    if (
        command[:2] == ["git", "add"]
        or command[:2] == ["git", "commit"]
        or command[:2] == ["git", "push"]
    ):
        raise RuntimeError(
            "Airflow DAG tasks must not run git add, git commit, or git push."
        )
    LOGGER.info("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=_build_env(use_training_venv=use_training_venv),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        LOGGER.info("STDOUT\n%s", result.stdout)
    if result.stderr:
        if result.returncode == 0:
            LOGGER.info("STDERR (non-fatal)\n%s", result.stderr)
        else:
            LOGGER.warning("STDERR\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _safe_run_id(run_id: str | None) -> str:
    value = run_id or _utc_now_iso()
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("._")
    return safe or "manual_run"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_metadata() -> dict[str, str | None]:
    return {
        "git_commit_sha": os.getenv("DEPLOYMENT_GIT_COMMIT_SHA")
        or os.getenv("GIT_COMMIT_SHA"),
        "app_version": os.getenv("APP_VERSION") or os.getenv("DEPLOYMENT_VERSION"),
        "container_image": os.getenv("CONTAINER_IMAGE") or os.getenv("IMAGE_TAG"),
        "ci_run_id": os.getenv("CI_RUN_ID") or os.getenv("GITHUB_RUN_ID"),
    }


def _latest_mlflow_run_id() -> str:
    train_metrics = (
        load_pipeline_artifact("train_metrics", config=CONFIG, default={}) or {}
    )
    run_id = train_metrics.get("run_id")
    if not run_id:
        raise RuntimeError(
            "Could not determine MLflow run_id from train_metrics for provenance logging."
        )
    return str(run_id)


def _airflow_context_run_id(context: dict) -> str:
    if context.get("run_id"):
        return str(context["run_id"])
    dag_run = context.get("dag_run")
    if dag_run and getattr(dag_run, "run_id", None):
        return str(dag_run.run_id)
    return _utc_now_iso()


def _candidate_metrics() -> dict:
    test_metrics = (
        load_pipeline_artifact("test_metrics", config=CONFIG, default={}) or {}
    )
    validation_metrics = (
        load_pipeline_artifact("validation_metrics", config=CONFIG, default={}) or {}
    )
    return {
        "accuracy": test_metrics.get("accuracy", validation_metrics.get("accuracy")),
        "macro_f1": test_metrics.get("macro_f1", validation_metrics.get("macro_f1")),
    }


def validate_candidate_thresholds() -> dict:
    metrics = _candidate_metrics()
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    accuracy_threshold = float(
        _required_config("continuous_improvement.accuracy_threshold")
    )
    macro_f1_threshold = float(
        _required_config("continuous_improvement.macro_f1_threshold")
    )
    passed = (
        isinstance(accuracy, (int, float))
        and isinstance(macro_f1, (int, float))
        and float(accuracy) >= accuracy_threshold
        and float(macro_f1) >= macro_f1_threshold
    )
    result = {
        "passed": passed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "accuracy_threshold": accuracy_threshold,
        "macro_f1_threshold": macro_f1_threshold,
    }
    LOGGER.info("Candidate validation gate: %s", result)
    return result


def _feedback_count() -> int:
    try:
        initialize_database()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS feedback_count FROM feedback_corrections"
                )
                row = cur.fetchone()
                return int(row["feedback_count"] if isinstance(row, dict) else row[0])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to read feedback count from Postgres: %s", exc)
        return 0


def _environment() -> str:
    return os.getenv("ENVIORMENT", "dev").strip().lower()


def _use_local_control_state() -> bool:
    return _environment() != "prod"


def _read_local_control_state() -> dict:
    if not _use_local_control_state():
        return {}
    state = read_json(CONTROL_STATE_PATH, {}) or {}
    state.pop("last_feedback_count", None)
    return state


def _write_local_control_state(updates: dict) -> dict:
    if not _use_local_control_state():
        return {}
    state = _read_local_control_state()
    state.update(updates)
    write_json(CONTROL_STATE_PATH, state)
    return state


def _control_plane_state() -> dict:
    legacy_state = _read_local_control_state()
    try:
        initialize_database()
        return get_control_plane_state(default_state=legacy_state)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Unable to read control-plane state from Postgres, falling back to local JSON: %s",
            exc,
        )
        return legacy_state


def _feedback_snapshot_count(state: dict | None = None) -> int:
    current_state = state if state is not None else _control_plane_state()
    return int(current_state.get("last_feedback_snapshot_count", 0) or 0)


def _pipeline_config_fingerprint() -> str:
    pipeline_config = {
        "project": {
            "random_seed": get_config_value(CONFIG, "project.random_seed"),
        },
        "paths": get_config_value(CONFIG, "paths", {}),
        "data": get_config_value(CONFIG, "data", {}),
        "training": get_config_value(CONFIG, "training", {}),
        "reporting": get_config_value(CONFIG, "reporting", {}),
    }
    serialized = json.dumps(pipeline_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sync_feedback_training_snapshot() -> dict:
    from src.data.materialize_feedback_training import (
        initialize_feedback_training_snapshot,
        materialize_feedback_training_dataset,
    )

    state = _control_plane_state()
    feedback_count = _feedback_count()
    last_snapshot_count = _feedback_snapshot_count(state)
    min_new_feedback = int(
        _required_config("continuous_improvement.min_new_feedback_samples")
    )
    new_feedback = max(feedback_count - last_snapshot_count, 0)

    if not (
        FEEDBACK_TRAINING_DIR.exists() and FEEDBACK_TRAINING_MANIFEST_PATH.exists()
    ):
        initialize_feedback_training_snapshot(
            output_root=FEEDBACK_TRAINING_DIR,
            manifest_path=FEEDBACK_TRAINING_MANIFEST_PATH,
        )

    if new_feedback < min_new_feedback:
        result = {
            "updated": False,
            "feedback_count": feedback_count,
            "last_feedback_snapshot_count": last_snapshot_count,
            "new_feedback_count": new_feedback,
            "threshold": min_new_feedback,
        }
        LOGGER.info("Feedback training snapshot unchanged: %s", result)
        return result

    summary = materialize_feedback_training_dataset(
        output_root=FEEDBACK_TRAINING_DIR,
        manifest_path=FEEDBACK_TRAINING_MANIFEST_PATH,
    )
    refreshed_state = update_control_plane_state(
        last_feedback_snapshot_count=feedback_count,
        last_feedback_snapshot_at=_utc_now_iso(),
    )
    _write_local_control_state(
        {
            "last_feedback_snapshot_count": refreshed_state[
                "last_feedback_snapshot_count"
            ],
            "last_feedback_snapshot_at": refreshed_state["last_feedback_snapshot_at"],
        }
    )
    result = {
        "updated": True,
        "feedback_count": feedback_count,
        "last_feedback_snapshot_count": refreshed_state["last_feedback_snapshot_count"],
        "new_feedback_count": new_feedback,
        "threshold": min_new_feedback,
        **summary,
    }
    LOGGER.info("Feedback training snapshot refreshed: %s", result)
    return result


def inspect_runtime_state() -> dict:
    state = _read_local_control_state()
    control_state = _control_plane_state()
    test_metrics = (
        load_pipeline_artifact("test_metrics", config=CONFIG, default={}) or {}
    )
    live_metrics = (
        load_pipeline_artifact("live_metrics", config=CONFIG, default={}) or {}
    )
    feedback_count = _feedback_count()
    previous_feedback_count = _feedback_snapshot_count(control_state)
    new_feedback = max(feedback_count - previous_feedback_count, 0)
    raw_exists = RAW_DATASET_DIR.exists() and any(RAW_DATASET_DIR.rglob("*.jpg"))
    model_exists = MODELS_DIR.exists() and (MODELS_DIR / "class_names.json").exists()

    accuracy_threshold = float(
        _required_config("continuous_improvement.accuracy_threshold")
    )
    macro_f1_threshold = float(
        _required_config("continuous_improvement.macro_f1_threshold")
    )
    min_new_feedback = int(
        _required_config("continuous_improvement.min_new_feedback_samples")
    )
    min_live_predictions = int(
        _required_config("continuous_improvement.min_live_prediction_samples")
    )
    live_prediction_count = int(live_metrics.get("prediction_count") or 0)

    offline_accuracy = test_metrics.get("accuracy")
    offline_macro_f1 = test_metrics.get("macro_f1")
    live_accuracy = live_metrics.get("accuracy")
    live_macro_f1 = live_metrics.get("macro_f1")
    current_pipeline_config_fingerprint = _pipeline_config_fingerprint()
    previous_pipeline_config_fingerprint = control_state.get(
        "last_pipeline_config_fingerprint"
    )
    pipeline_config_changed = (
        previous_pipeline_config_fingerprint != current_pipeline_config_fingerprint
    )

    model_degraded = False
    if (
        isinstance(live_accuracy, (int, float))
        and live_prediction_count >= min_live_predictions
        and live_accuracy < accuracy_threshold
    ):
        model_degraded = True
    elif (
        isinstance(live_macro_f1, (int, float))
        and live_prediction_count >= min_live_predictions
        and live_macro_f1 < macro_f1_threshold
    ):
        model_degraded = True
    elif (
        isinstance(offline_accuracy, (int, float))
        and offline_accuracy < accuracy_threshold
    ):
        model_degraded = True
    elif (
        isinstance(offline_macro_f1, (int, float))
        and offline_macro_f1 < macro_f1_threshold
    ):
        model_degraded = True

    pipeline_run_reasons: list[str] = []
    if not raw_exists:
        pipeline_run_reasons.append("raw_dataset_missing")
    if not model_exists:
        pipeline_run_reasons.append("model_missing")
    if model_degraded:
        pipeline_run_reasons.append("metrics_degraded")
    if new_feedback >= min_new_feedback:
        pipeline_run_reasons.append("new_feedback_threshold_reached")
    if pipeline_config_changed:
        pipeline_run_reasons.append("pipeline_config_changed")

    runtime_state = {
        "checked_at": _utc_now_iso(),
        "raw_exists": raw_exists,
        "model_exists": model_exists,
        "feedback_count": feedback_count,
        "live_prediction_count": live_prediction_count,
        "previous_feedback_count": previous_feedback_count,
        "new_feedback_count": new_feedback,
        "last_feedback_snapshot_count": previous_feedback_count,
        "last_feedback_snapshot_at": control_state.get("last_feedback_snapshot_at"),
        "last_report_sent_at": control_state.get("last_report_sent_at"),
        "last_pipeline_config_fingerprint": previous_pipeline_config_fingerprint,
        "last_pipeline_config_updated_at": control_state.get(
            "last_pipeline_config_updated_at"
        ),
        "current_pipeline_config_fingerprint": current_pipeline_config_fingerprint,
        "pipeline_config_changed": pipeline_config_changed,
        "offline_accuracy": offline_accuracy,
        "offline_macro_f1": offline_macro_f1,
        "live_accuracy": live_accuracy,
        "live_macro_f1": live_macro_f1,
        "model_degraded": model_degraded,
        "pipeline_run_reasons": pipeline_run_reasons,
        "should_run_pipeline": bool(pipeline_run_reasons),
    }
    _write_local_control_state({**state, **runtime_state})
    LOGGER.info("Runtime inspection: %s", runtime_state)
    return runtime_state


def branch_runtime_decision() -> str:
    state = inspect_runtime_state()
    LOGGER.info(
        "Control-plane branch decision should_run_pipeline=%s reasons=%s",
        state["should_run_pipeline"],
        state.get("pipeline_run_reasons", []),
    )
    return "run_dvc_pipeline" if state["should_run_pipeline"] else "skip_retraining"


def run_dvc_pipeline() -> None:
    sync_feedback_training_snapshot()
    _run_command(
        [TRAINING_PYTHON, "-m", "dvc", "repro", "report"], use_training_venv=True
    )
    current_pipeline_config_fingerprint = _pipeline_config_fingerprint()
    state = update_control_plane_state(
        last_pipeline_config_fingerprint=current_pipeline_config_fingerprint,
        last_pipeline_config_updated_at=_utc_now_iso(),
    )
    _write_local_control_state(
        {
            "last_pipeline_config_fingerprint": state[
                "last_pipeline_config_fingerprint"
            ],
            "last_pipeline_config_updated_at": state["last_pipeline_config_updated_at"],
            "current_pipeline_config_fingerprint": current_pipeline_config_fingerprint,
            "pipeline_config_changed": False,
        }
    )


def push_dvc_artifacts() -> None:
    if not bool(get_config_value(CONFIG, "dvc.push_on_success", True)):
        LOGGER.info("DVC push disabled by dvc.push_on_success=false; skipping.")
        return
    _run_command([TRAINING_PYTHON, "-m", "dvc", "push"], use_training_venv=True)


def save_dvc_provenance(**context) -> dict:
    run_id = _airflow_context_run_id(context)
    safe_run_id = _safe_run_id(run_id)
    run_dir = RUNTIME_RUNS_DIR / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_lock_path = run_dir / "dvc.lock"
    provenance_path = run_dir / "provenance.json"

    if not DVC_LOCK_PATH.exists():
        raise FileNotFoundError(
            f"Cannot save provenance because {DVC_LOCK_PATH} does not exist."
        )

    shutil.copy2(DVC_LOCK_PATH, copied_lock_path)
    feedback_summary = (
        load_pipeline_artifact("feedback_training_summary", config=CONFIG, default={})
        or {}
    )
    control_state = _control_plane_state()
    mlflow_run_id = _latest_mlflow_run_id()
    provenance = {
        "dvc_lock_sha256": _sha256_file(copied_lock_path),
        "airflow_run_id": run_id,
        "timestamp": _utc_now_iso(),
        "mlflow_run_id": mlflow_run_id,
        "deployment": _deployment_metadata(),
        "feedback_snapshot_id": control_state.get("last_feedback_snapshot_at")
        or control_state.get("last_feedback_snapshot_count"),
        "feedback_snapshot_path": feedback_summary.get("manifest_path")
        or str(FEEDBACK_TRAINING_MANIFEST_PATH),
        "candidate_model_version": None,
        "promotion_status": "pending_registration",
    }
    write_json(provenance_path, provenance)
    LOGGER.info("Saved DVC provenance to %s", provenance_path)
    return provenance


def log_dvc_provenance_to_mlflow(**context) -> None:
    import mlflow

    run_id = _airflow_context_run_id(context)
    safe_run_id = _safe_run_id(run_id)
    run_dir = RUNTIME_RUNS_DIR / safe_run_id
    copied_lock_path = run_dir / "dvc.lock"
    provenance_path = run_dir / "provenance.json"
    if not copied_lock_path.exists() or not provenance_path.exists():
        raise FileNotFoundError(f"Missing provenance artifacts in {run_dir}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or get_config_value(
        CONFIG, "services.mlflow_url", "http://mlflow:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    provenance = read_json(provenance_path)
    mlflow_run_id = str(provenance["mlflow_run_id"])
    with mlflow.start_run(run_id=mlflow_run_id):
        deployment = provenance.get("deployment") or {}
        for key, value in deployment.items():
            if value:
                mlflow.set_tag(f"deployment.{key}", str(value))
        mlflow.log_artifact(
            str(copied_lock_path), artifact_path=f"provenance/{safe_run_id}"
        )
        mlflow.log_artifact(
            str(provenance_path), artifact_path=f"provenance/{safe_run_id}"
        )
    LOGGER.info("Logged DVC provenance artifacts to MLflow run %s", mlflow_run_id)


def register_candidate_model() -> None:
    _run_command(
        [
            TRAINING_PYTHON,
            "-m",
            "src.registry.register_best_model",
            "register-candidate",
        ],
        use_training_venv=True,
    )


def branch_candidate_validation() -> str:
    result = validate_candidate_thresholds()
    return "promote_candidate_model" if result["passed"] else "reject_candidate_model"


def promote_candidate_model() -> None:
    _run_command(
        [
            TRAINING_PYTHON,
            "-m",
            "src.registry.register_best_model",
            "promote-candidate",
        ],
        use_training_venv=True,
    )


def reject_candidate_model() -> None:
    _run_command(
        [
            TRAINING_PYTHON,
            "-m",
            "src.registry.register_best_model",
            "promote-candidate",
        ],
        use_training_venv=True,
    )


def generate_runtime_report() -> None:
    _run_command(
        [TRAINING_PYTHON, "-m", "src.reporting.generate_runtime_report"],
        use_training_venv=True,
    )


def reload_model_service() -> None:
    response = requests.post(
        get_config_value(
            CONFIG, "services.model_service_url", "http://model-service:8001"
        )
        + "/reload",
        timeout=60,
    )
    response.raise_for_status()
    LOGGER.info("Model service reloaded successfully: %s", response.text)


def _build_report_email_html(report_text: str, report_html: str | None) -> str:
    if report_html:
        return report_html
    return f"<html><body><pre>{html.escape(report_text)}</pre></body></html>"


def send_latest_report() -> None:
    if not get_config_value(CONFIG, "email.enabled", False):
        LOGGER.info("Email delivery disabled in config; skipping report email.")
        return

    recipients = get_config_value(CONFIG, "email.recipients", [])
    if not recipients:
        raise ValueError(
            "Email config is incomplete. Set at least one recipient in config.yaml."
        )

    report_text = (
        RUNTIME_REPORT_MD_PATH.read_text(encoding="utf-8")
        if RUNTIME_REPORT_MD_PATH.exists()
        else "No runtime report generated yet."
    )
    report_html = (
        RUNTIME_REPORT_HTML_PATH.read_text(encoding="utf-8")
        if RUNTIME_REPORT_HTML_PATH.exists()
        else None
    )
    subject = f"{get_config_value(CONFIG, 'email.subject_prefix', '[Galaxy MLOps]')} Latest runtime report"
    sender = get_config_value(CONFIG, "email.sender") or None
    smtp_conn_id = get_config_value(CONFIG, "email.connection_id", "smtp_default")
    attachments = [
        str(path)
        for path in (RUNTIME_REPORT_MD_PATH, RUNTIME_REPORT_HTML_PATH)
        if path.exists()
    ]

    LOGGER.info(
        "Sending report email to %s via Airflow SMTP connection %s",
        recipients,
        smtp_conn_id,
    )
    with SmtpHook(smtp_conn_id=smtp_conn_id) as smtp_hook:
        smtp_hook.send_email_smtp(
            to=recipients,
            subject=subject,
            html_content=_build_report_email_html(report_text, report_html),
            from_email=sender,
            files=attachments,
        )

    state = update_control_plane_state(last_report_sent_at=_utc_now_iso())
    _write_local_control_state({"last_report_sent_at": state["last_report_sent_at"]})


with DAG(
    dag_id="galaxy_morphology_control_plane",
    description="Airflow control plane for the DVC-driven galaxy pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=get_config_value(
        CONFIG, "continuous_improvement.monitor_schedule", "30 12 * * *"
    ),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "galaxy", "control-plane"],
) as dag:
    decide = BranchPythonOperator(
        task_id="inspect_and_branch", python_callable=branch_runtime_decision
    )
    run_pipeline = PythonOperator(
        task_id="run_dvc_pipeline", python_callable=run_dvc_pipeline
    )
    dvc_push = PythonOperator(
        task_id="push_dvc_artifacts", python_callable=push_dvc_artifacts
    )
    save_provenance = PythonOperator(
        task_id="save_dvc_provenance", python_callable=save_dvc_provenance
    )
    log_provenance = PythonOperator(
        task_id="log_dvc_provenance_to_mlflow",
        python_callable=log_dvc_provenance_to_mlflow,
    )
    register_model = PythonOperator(
        task_id="register_candidate_model", python_callable=register_candidate_model
    )
    validate_candidate = BranchPythonOperator(
        task_id="validate_candidate_model", python_callable=branch_candidate_validation
    )
    promote_model = PythonOperator(
        task_id="promote_candidate_model", python_callable=promote_candidate_model
    )
    reject_model = PythonOperator(
        task_id="reject_candidate_model", python_callable=reject_candidate_model
    )
    runtime_report = PythonOperator(
        task_id="generate_runtime_report",
        python_callable=generate_runtime_report,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    skip_retraining = EmptyOperator(task_id="skip_retraining")
    reload_service = PythonOperator(
        task_id="reload_model_service", python_callable=reload_model_service
    )
    send_email = PythonOperator(
        task_id="send_report_email",
        python_callable=send_latest_report,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    finish = EmptyOperator(
        task_id="finish", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    (
        decide
        >> run_pipeline
        >> dvc_push
        >> save_provenance
        >> log_provenance
        >> register_model
        >> validate_candidate
    )
    (
        validate_candidate
        >> promote_model
        >> reload_service
        >> runtime_report
        >> send_email
        >> finish
    )
    validate_candidate >> reject_model >> runtime_report >> send_email >> finish
    decide >> skip_retraining >> runtime_report >> send_email >> finish
