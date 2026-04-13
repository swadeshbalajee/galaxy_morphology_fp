
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import write_json
from src.common.logging_utils import configure_logging
from src.training.model_def import build_model

LOGGER = configure_logging("training")


def make_transforms(config: dict):
    image_size = int(get_config_value(config, "data.image_size", 224))
    mean = get_config_value(config, "data.normalize_mean", [0.485, 0.456, 0.406])
    std = get_config_value(config, "data.normalize_std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_dataloaders(train_dir: str, val_dir: str, test_dir: str, config: dict):
    transform = make_transforms(config)
    batch_size = int(get_config_value(config, "training.batch_size", 32))
    num_workers = int(get_config_value(config, "training.num_workers", 0))
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    return (
        train_data,
        val_data,
        test_data,
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def evaluate_loader(model, loader, device):
    model.eval()
    targets: list[int] = []
    preds: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    average_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = sum(int(a == b) for a, b in zip(preds, targets)) / max(len(targets), 1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
        "targets": targets,
        "preds": preds,
    }


def export_local_model(model, export_dir: str | Path, class_names: list[str]):
    export_dir = Path(export_dir)
    if export_dir.exists():
        import shutil
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    mlflow.pytorch.save_model(model, str(export_dir))
    (export_dir / 'class_names.json').write_text(json.dumps(class_names, indent=2), encoding='utf-8')


def train(train_dir: str, val_dir: str, test_dir: str, export_dir: str):
    config = load_config()
    training_params = get_config_value(config, 'training', {})
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI') or get_config_value(config, 'services.mlflow_url')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(get_config_value(config, 'project.name', 'galaxy-mlops'))
    mlflow.autolog(log_models=False)

    train_data, val_data, test_data, train_loader, val_loader, test_loader = make_dataloaders(train_dir, val_dir, test_dir, config)
    class_names = train_data.classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(class_names), get_config_value(config, 'training.pretrained_backbone', 'resnet18'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training_params['learning_rate']), weight_decay=float(training_params['weight_decay']))
    best_val_f1 = -1.0
    best_state = None
    patience = int(training_params['early_stopping_patience'])
    patience_counter = 0
    epochs = int(training_params['epochs'])
    start_time = time.time()

    with mlflow.start_run(run_name='galaxy_training_run') as run:
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': training_params['batch_size'],
            'learning_rate': training_params['learning_rate'],
            'weight_decay': training_params['weight_decay'],
            'backbone': training_params['pretrained_backbone'],
            'num_classes': len(class_names),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
        })
        mlflow.log_text('\n'.join(class_names), 'class_names.txt')

        epoch_history: list[dict] = []
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            epoch_start = time.time()
            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                seen += images.size(0)
                if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                    LOGGER.info('epoch=%s batch=%s/%s running_train_loss=%.4f samples_seen=%s', epoch, batch_idx, len(train_loader), running_loss / max(seen, 1), seen)

            train_loss = running_loss / max(len(train_loader.dataset), 1)
            val_metrics = evaluate_loader(model, val_loader, device)
            epoch_duration = time.time() - epoch_start
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'epoch_duration_seconds': round(epoch_duration, 3),
                'samples_per_second': round(len(train_loader.dataset) / max(epoch_duration, 1e-6), 3),
            }
            epoch_history.append(epoch_result)
            LOGGER.info('Epoch summary: %s', epoch_result)
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'samples_per_second': epoch_result['samples_per_second'],
            }, step=epoch)

            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    LOGGER.info('Early stopping triggered at epoch=%s', epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        export_local_model(model, export_dir, class_names)
        validation_metrics = evaluate_loader(model, val_loader, device)
        confusion_df = pd.DataFrame(
            confusion_matrix(validation_metrics['targets'], validation_metrics['preds'], labels=list(range(len(class_names)))),
            index=class_names,
            columns=class_names,
        )
        confusion_path = resolve_path(config, 'paths.confusion_matrix_path')
        confusion_df.to_csv(confusion_path)
        classification = classification_report(validation_metrics['targets'], validation_metrics['preds'], target_names=class_names, output_dict=True, zero_division=0)
        write_json(resolve_path(config, 'paths.classification_report_path'), classification)

        total_duration = time.time() - start_time
        train_metrics = {
            'run_id': run.info.run_id,
            'best_val_macro_f1': best_val_f1,
            'epochs_completed': len(epoch_history),
            'total_duration_seconds': round(total_duration, 3),
            'samples_per_second': round(len(train_loader.dataset) * max(len(epoch_history), 1) / max(total_duration, 1e-6), 3),
            'history': epoch_history,
            'class_names': class_names,
        }
        validation_summary = {
            'loss': validation_metrics['loss'],
            'accuracy': validation_metrics['accuracy'],
            'precision_macro': validation_metrics['precision_macro'],
            'recall_macro': validation_metrics['recall_macro'],
            'macro_f1': validation_metrics['macro_f1'],
            'class_names': class_names,
            'model_dir': str(export_dir),
        }
        write_json(resolve_path(config, 'paths.train_metrics_path'), train_metrics)
        write_json(resolve_path(config, 'paths.validation_metrics_path'), validation_summary)
        write_json(resolve_path(config, 'paths.pipeline_runtime_summary_path'), {
            'train_duration_seconds': round(total_duration, 3),
            'epochs_completed': len(epoch_history),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
        })

        mlflow.log_metrics({f'final_validation_{k}': v for k, v in validation_summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(resolve_path(config, 'paths.classification_report_path')))
        mlflow.log_artifact(str(resolve_path(config, 'paths.train_metrics_path')))
        mlflow.log_artifact(str(resolve_path(config, 'paths.validation_metrics_path')))
        mlflow.pytorch.log_model(model, artifact_path='model')
        mlflow.log_artifacts(str(export_dir), artifact_path='local_export')

        LOGGER.info('Training finished. train_metrics=%s validation_metrics=%s', train_metrics, validation_summary)
        return train_metrics, validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a galaxy morphology classifier.')
    parser.add_argument('--train-dir', default=None)
    parser.add_argument('--val-dir', default=None)
    parser.add_argument('--test-dir', default=None)
    parser.add_argument('--export-dir', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config()
    processed_root = resolve_path(cfg, 'paths.processed_final_dir')
    train(args.train_dir or processed_root / 'train', args.val_dir or processed_root / 'val', args.test_dir or processed_root / 'test', args.export_dir or resolve_path(cfg, 'paths.models_dir'))
