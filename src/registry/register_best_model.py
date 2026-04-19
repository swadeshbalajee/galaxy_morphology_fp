from __future__ import annotations

import argparse
import os
import time
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('model_registry')


def _tracking_uri(config: dict) -> str:
    return os.getenv('MLFLOW_TRACKING_URI') or get_config_value(config, 'services.mlflow_url', 'http://mlflow:5000')


def _candidate_metric(config: dict, metric_name: str) -> tuple[float | None, str | None]:
    source_names = get_config_value(config, 'registry.comparison_metric_sources', ['test_metrics', 'validation_metrics'])
    source_map = {
        'test_metrics': resolve_path(config, 'paths.test_metrics_path'),
        'validation_metrics': resolve_path(config, 'paths.validation_metrics_path'),
        'train_metrics': resolve_path(config, 'paths.train_metrics_path'),
    }
    for source_name in source_names:
        path = source_map.get(source_name)
        if not path or not path.exists():
            continue
        payload = read_json(path, {}) or {}
        if metric_name in payload and isinstance(payload[metric_name], (int, float)):
            return float(payload[metric_name]), source_name
        fallback_name = f'final_validation_{metric_name}'
        if fallback_name in payload and isinstance(payload[fallback_name], (int, float)):
            return float(payload[fallback_name]), source_name
    return None, None


def _resolve_run_id(config: dict, explicit_run_id: str | None) -> str:
    if explicit_run_id:
        return explicit_run_id
    train_metrics = read_json(resolve_path(config, 'paths.train_metrics_path'), {}) or {}
    run_id = train_metrics.get('run_id')
    if not run_id:
        raise RuntimeError('Could not determine MLflow run_id from artifacts/train_metrics.json')
    return str(run_id)


def _wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        mv = client.get_model_version(name=model_name, version=version)
        status = str(getattr(mv, 'status', 'READY'))
        if status.upper() == 'READY':
            return
        time.sleep(2)
    LOGGER.warning('Timed out waiting for model version %s/%s to become READY', model_name, version)


def _current_champion(client: MlflowClient, model_name: str, alias: str, metric_name: str) -> dict[str, Any] | None:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001
        return None
    current_metric = mv.tags.get('comparison_metric_value')
    return {
        'version': str(mv.version),
        'run_id': mv.tags.get('run_id'),
        'metric_name': mv.tags.get('comparison_metric_name', metric_name),
        'metric_value': float(current_metric) if current_metric not in (None, '') else None,
    }


def register_best_model(run_id: str | None = None) -> dict[str, Any]:
    config = load_config()
    tracking_uri = _tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    model_name = get_config_value(config, 'registry.model_name', 'galaxy_morphology_classifier')
    champion_alias = get_config_value(config, 'registry.champion_alias', 'champion')
    artifact_subpath = get_config_value(config, 'registry.artifact_subpath', 'local_export')
    metric_name = get_config_value(config, 'registry.comparison_metric', 'macro_f1')
    promote_only_if_better = bool(get_config_value(config, 'registry.promote_only_if_better_than_current', True))
    resolved_run_id = _resolve_run_id(config, run_id)
    candidate_metric, metric_source = _candidate_metric(config, metric_name)
    model_uri = f'runs:/{resolved_run_id}/{artifact_subpath}'

    LOGGER.info('Registering candidate model from %s into %s', model_uri, model_name)
    registration = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registration.version)
    _wait_until_ready(client, model_name, version)

    tag_updates = {
        'run_id': resolved_run_id,
        'source_model_uri': model_uri,
        'comparison_metric_name': metric_name,
        'comparison_metric_value': '' if candidate_metric is None else str(candidate_metric),
        'comparison_metric_source': metric_source or '',
    }
    for key, value in tag_updates.items():
        client.set_model_version_tag(name=model_name, version=version, key=key, value=value)

    current = _current_champion(client, model_name, champion_alias, metric_name)
    promote = True
    reason = 'No existing champion alias found.'
    if promote_only_if_better and current and current.get('metric_value') is not None and candidate_metric is not None:
        promote = candidate_metric > float(current['metric_value'])
        reason = (
            f'Candidate {metric_name}={candidate_metric} is better than champion {metric_name}={current["metric_value"]}.'
            if promote
            else f'Candidate {metric_name}={candidate_metric} did not beat champion {metric_name}={current["metric_value"]}.'
        )
    elif current and current.get('metric_value') is None and candidate_metric is not None:
        reason = f'Promoting because current champion has no recorded {metric_name}.'
    elif current and candidate_metric is None:
        promote = not promote_only_if_better
        reason = 'Candidate metric unavailable; promotion allowed only when strict comparison is disabled.'

    if promote:
        client.set_registered_model_alias(name=model_name, alias=champion_alias, version=version)

    status = {
        'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tracking_uri': tracking_uri,
        'model_name': model_name,
        'champion_alias': champion_alias,
        'candidate': {
            'version': version,
            'run_id': resolved_run_id,
            'model_uri': model_uri,
            'metric_name': metric_name,
            'metric_value': candidate_metric,
            'metric_source': metric_source,
        },
        'previous_champion': current,
        'champion_updated': promote,
        'current_champion_version': version if promote else (current or {}).get('version'),
        'serving_model_uri': f'models:/{model_name}@{champion_alias}',
        'decision_reason': reason,
    }
    write_json(resolve_path(config, 'paths.registry_status_path'), status)
    LOGGER.info('Registry status: %s', status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Register the latest model and optionally promote it to champion.')
    parser.add_argument('--run-id', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = register_best_model(run_id=args.run_id)
    print(result)
