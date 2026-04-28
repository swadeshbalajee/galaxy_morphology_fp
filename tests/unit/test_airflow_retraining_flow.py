from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class _FakeTask:
    def __init__(self, *args, **kwargs):
        self.task_id = kwargs.get('task_id')

    def __rshift__(self, other):
        return other


class _FakeDAG:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def galaxy_pipeline(monkeypatch):
    airflow_mod = types.ModuleType('airflow')
    airflow_mod.DAG = _FakeDAG
    smtp_mod = types.ModuleType('airflow.providers.smtp.hooks.smtp')
    smtp_mod.SmtpHook = _FakeTask
    empty_mod = types.ModuleType('airflow.providers.standard.operators.empty')
    empty_mod.EmptyOperator = _FakeTask
    python_mod = types.ModuleType('airflow.providers.standard.operators.python')
    python_mod.BranchPythonOperator = _FakeTask
    python_mod.PythonOperator = _FakeTask
    trigger_mod = types.ModuleType('airflow.task.trigger_rule')
    trigger_mod.TriggerRule = types.SimpleNamespace(NONE_FAILED_MIN_ONE_SUCCESS='none_failed_min_one_success')

    for name, module in {
        'airflow': airflow_mod,
        'airflow.providers.smtp.hooks.smtp': smtp_mod,
        'airflow.providers.standard.operators.empty': empty_mod,
        'airflow.providers.standard.operators.python': python_mod,
        'airflow.task.trigger_rule': trigger_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[2] / 'airflow' / 'dags' / 'galaxy_pipeline.py'
    spec = importlib.util.spec_from_file_location('test_galaxy_pipeline', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_dvc_lock_sha256_and_provenance_keys(galaxy_pipeline, tmp_path, monkeypatch):
    dvc_lock = tmp_path / 'dvc.lock'
    dvc_lock.write_text('outs:\n- md5: abc\n', encoding='utf-8')
    runs_dir = tmp_path / 'artifacts' / 'runtime' / 'runs'

    monkeypatch.setattr(galaxy_pipeline, 'DVC_LOCK_PATH', dvc_lock)
    monkeypatch.setattr(galaxy_pipeline, 'RUNTIME_RUNS_DIR', runs_dir)
    monkeypatch.setattr(galaxy_pipeline, 'FEEDBACK_TRAINING_MANIFEST_PATH', tmp_path / 'feedback.csv')
    monkeypatch.setattr(galaxy_pipeline, '_latest_mlflow_run_id', lambda: 'mlflow-run-1')
    monkeypatch.setattr(galaxy_pipeline, '_control_plane_state', lambda: {'last_feedback_snapshot_at': 'snapshot-1'})
    monkeypatch.setattr(
        galaxy_pipeline,
        'load_pipeline_artifact',
        lambda *args, **kwargs: {'manifest_path': 'artifacts/feedback_training_manifest.csv'},
    )

    provenance = galaxy_pipeline.save_dvc_provenance(run_id='scheduled__2026-04-28T00:00:00+00:00')
    saved_path = runs_dir / 'scheduled__2026-04-28T00_00_00_00_00' / 'provenance.json'
    saved = json.loads(saved_path.read_text(encoding='utf-8'))

    expected_keys = {
        'dvc_lock_sha256',
        'airflow_run_id',
        'timestamp',
        'mlflow_run_id',
        'deployment',
        'feedback_snapshot_id',
        'feedback_snapshot_path',
        'candidate_model_version',
        'promotion_status',
    }
    assert set(saved) == expected_keys
    assert provenance['dvc_lock_sha256'] == hashlib.sha256(dvc_lock.read_bytes()).hexdigest()
    assert saved['dvc_lock_sha256'] == provenance['dvc_lock_sha256']


def test_dvc_push_obeys_true_false_config(galaxy_pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr(galaxy_pipeline, '_run_command', lambda command, use_training_venv=False: calls.append(command))

    galaxy_pipeline.CONFIG['dvc'] = {'push_on_success': False}
    galaxy_pipeline.push_dvc_artifacts()
    assert calls == []

    galaxy_pipeline.CONFIG['dvc'] = {'push_on_success': True}
    galaxy_pipeline.push_dvc_artifacts()
    assert calls == [[galaxy_pipeline.TRAINING_PYTHON, '-m', 'dvc', 'push']]


def test_mlflow_logging_failure_raises(galaxy_pipeline, tmp_path, monkeypatch):
    run_dir = tmp_path / 'runs' / 'manual'
    run_dir.mkdir(parents=True)
    (run_dir / 'dvc.lock').write_text('lock', encoding='utf-8')
    (run_dir / 'provenance.json').write_text(json.dumps({'mlflow_run_id': 'run-1'}), encoding='utf-8')
    monkeypatch.setattr(galaxy_pipeline, 'RUNTIME_RUNS_DIR', tmp_path / 'runs')
    monkeypatch.setattr(galaxy_pipeline, '_safe_run_id', lambda run_id: 'manual')

    class BrokenRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    broken_mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        start_run=lambda run_id: BrokenRun(),
        log_artifact=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('mlflow artifact failed')),
    )
    monkeypatch.setitem(sys.modules, 'mlflow', broken_mlflow)

    with pytest.raises(RuntimeError, match='mlflow artifact failed'):
        galaxy_pipeline.log_dvc_provenance_to_mlflow(run_id='manual')


def test_validation_accepts_and_rejects_thresholds(galaxy_pipeline, monkeypatch):
    monkeypatch.setattr(galaxy_pipeline, '_required_config', lambda key: {'continuous_improvement.accuracy_threshold': 0.85, 'continuous_improvement.macro_f1_threshold': 0.8}[key])

    monkeypatch.setattr(galaxy_pipeline, '_candidate_metrics', lambda: {'accuracy': 0.9, 'macro_f1': 0.81})
    assert galaxy_pipeline.validate_candidate_thresholds()['passed'] is True

    monkeypatch.setattr(galaxy_pipeline, '_candidate_metrics', lambda: {'accuracy': 0.9, 'macro_f1': 0.79})
    assert galaxy_pipeline.validate_candidate_thresholds()['passed'] is False


def test_airflow_dag_contains_no_git_commit_or_push_commands():
    source = (Path(__file__).resolve().parents[2] / 'airflow' / 'dags' / 'galaxy_pipeline.py').read_text(encoding='utf-8')

    assert "_run_command(['git'" not in source
    assert '_run_command(["git"' not in source
