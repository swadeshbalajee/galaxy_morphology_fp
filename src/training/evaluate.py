
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import mlflow.pytorch
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import write_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("evaluation")


def load_model(model_dir: str | Path):
    model = mlflow.pytorch.load_model(str(model_dir))
    model.eval()
    return model


def build_loader(test_dir: str | Path, config: dict):
    image_size = int(get_config_value(config, 'data.image_size', 224))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            get_config_value(config, 'data.normalize_mean', [0.485, 0.456, 0.406]),
            get_config_value(config, 'data.normalize_std', [0.229, 0.224, 0.225]),
        ),
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=int(get_config_value(config, 'training.batch_size', 32)), shuffle=False, num_workers=int(get_config_value(config, 'training.num_workers', 0)))
    return dataset, loader


def evaluate_test_set(test_dir: str | Path, model_dir: str | Path) -> dict:
    config = load_config()
    dataset, loader = build_loader(test_dir, config)
    model = load_model(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    targets: list[int] = []
    preds: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    precision, recall, macro_f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    metrics = {
        'accuracy': accuracy_score(targets, preds) if targets else 0.0,
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'macro_f1': float(macro_f1),
        'num_test_samples': len(targets),
        'class_names': dataset.classes,
        'model_dir': str(model_dir),
    }
    write_json(resolve_path(config, 'paths.test_metrics_path'), metrics)
    LOGGER.info('Offline test-set evaluation complete: %s', metrics)
    return metrics


def evaluate_live_feedback(predictions_db_path: str | Path | None = None) -> dict:
    config = load_config()
    db_path = Path(predictions_db_path or resolve_path(config, 'paths.predictions_db_path'))
    metrics = {
        'feedback_count': 0,
        'accuracy': None,
        'macro_f1': None,
        'ground_truth_distribution': {},
    }
    if not db_path.exists():
        write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
        return metrics

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.predicted_label, f.ground_truth_label
            FROM feedback f
            JOIN predictions p ON p.prediction_id = f.prediction_id
            ORDER BY f.created_at DESC
            """
        ).fetchall()
    if not rows:
        write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
        return metrics

    y_pred = [row[0] for row in rows]
    y_true = [row[1] for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    distribution = {label: y_true.count(label) for label in labels}
    metrics = {
        'feedback_count': len(rows),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'macro_f1': float(macro_f1),
        'labels': labels,
        'ground_truth_distribution': distribution,
    }
    write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
    LOGGER.info('Live feedback evaluation complete: %s', metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the exported model.')
    parser.add_argument('--test-dir', default=None)
    parser.add_argument('--model-dir', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config()
    processed_root = resolve_path(cfg, 'paths.processed_final_dir')
    model_dir = args.model_dir or resolve_path(cfg, 'paths.models_dir')
    evaluate_test_set(args.test_dir or processed_root / 'test', model_dir)
    evaluate_live_feedback()
