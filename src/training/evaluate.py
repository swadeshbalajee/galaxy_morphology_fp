from __future__ import annotations

from pathlib import Path

import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets

from src.common.artifact_store import store_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database
from src.training.train import make_transforms

LOGGER = configure_logging('evaluation')


def _row_value(row, key: str, index: int):
    if isinstance(row, dict):
        return row[key]
    return row[index]


def compute_live_feedback_metrics(rows: list[dict] | list[tuple]) -> dict:
    metrics = {
        'feedback_count': 0,
        'prediction_count': 0,
        'assumed_correct_count': 0,
        'latest_model_version': None,
        'accuracy': None,
        'macro_f1': None,
    }
    if not rows:
        return metrics

    y_pred = [_row_value(row, 'predicted_label', 1) for row in rows]
    y_true = [
        _row_value(row, 'corrected_label', 2) or _row_value(row, 'predicted_label', 1)
        for row in rows
    ]
    feedback_count = sum(1 for row in rows if _row_value(row, 'corrected_label', 2))
    return {
        'feedback_count': feedback_count,
        'prediction_count': len(rows),
        'assumed_correct_count': len(rows) - feedback_count,
        'latest_model_version': _row_value(rows[-1], 'model_version', 0),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 6),
        'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 6),
    }


def evaluate_offline() -> dict:
    config = load_config()
    model_dir = resolve_path(config, 'paths.models_dir')
    test_dir = resolve_path(config, 'paths.processed_final_dir') / 'test'
    if not model_dir.exists() or not test_dir.exists():
        metrics = {
            'status': 'offline_artifacts_missing',
            'model_dir': str(model_dir),
            'test_dir': str(test_dir),
        }
        store_pipeline_artifact('test_metrics', metrics, config=config)
        LOGGER.warning('Offline evaluation skipped: %s', metrics)
        return metrics

    transform = make_transforms(config)
    batch_size = int(get_config_value(config, 'training.batch_size', 32))
    num_workers = int(get_config_value(config, 'training.num_workers', 0))
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = mlflow.pytorch.load_model(str(model_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    targets: list[int] = []
    preds: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    precision, recall, macro_f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    metrics = {
        'loss': total_loss / max(len(test_loader.dataset), 1),
        'accuracy': round(float(accuracy_score(targets, preds)), 6),
        'precision_macro': round(float(precision), 6),
        'recall_macro': round(float(recall), 6),
        'macro_f1': round(float(macro_f1), 6),
        'class_names': test_data.classes,
        'test_samples': len(test_data),
        'model_dir': str(model_dir),
    }
    store_pipeline_artifact('test_metrics', metrics, config=config)
    LOGGER.info('Offline evaluation snapshot: %s', metrics)
    return metrics


def evaluate_live_feedback(predictions_db_path: str | Path | None = None) -> dict:  # noqa: ARG001
    config = load_config()
    initialize_database()
    metrics = {
        'feedback_count': 0,
        'prediction_count': 0,
        'assumed_correct_count': 0,
        'latest_model_version': None,
        'accuracy': None,
        'macro_f1': None,
    }

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH latest_model AS (
                        SELECT model_version
                        FROM predictions
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                    SELECT
                        p.model_version,
                        p.predicted_label,
                        c.corrected_label
                    FROM predictions p
                    JOIN latest_model lm ON lm.model_version = p.model_version
                    LEFT JOIN feedback_corrections c ON c.prediction_id = p.prediction_id
                    ORDER BY p.created_at ASC
                    """,
                )
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Live feedback database not ready: %s', exc)
        store_pipeline_artifact('live_metrics', metrics, config=config)
        return metrics

    if not rows:
        store_pipeline_artifact('live_metrics', metrics, config=config)
        LOGGER.info('No live predictions available.')
        return metrics

    metrics = compute_live_feedback_metrics(rows)
    store_pipeline_artifact('live_metrics', metrics, config=config)
    LOGGER.info('Live feedback evaluation complete: %s', metrics)
    return metrics


def main() -> None:
    offline_metrics = evaluate_offline()
    live_metrics = evaluate_live_feedback()
    LOGGER.info('Evaluation complete. offline=%s live=%s', offline_metrics, live_metrics)


if __name__ == '__main__':
    main()
