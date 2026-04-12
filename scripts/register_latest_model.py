from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient


def register_latest(run_id: str, model_name: str = 'galaxy_morphology_classifier') -> str:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage='Production',
        archive_existing_versions=True,
    )
    return f'{model_name}:{result.version}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-name', default='galaxy_morphology_classifier')
    args = parser.parse_args()
    print(register_latest(args.run_id, args.model_name))
