
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('model_loader')


class GalaxyPredictor:
    def __init__(self, model_uri: str | None = None):
        self.config = load_config()
        self.model_uri = model_uri or os.getenv('MODEL_URI') or str(resolve_path(self.config, 'paths.models_dir'))
        self.model = None
        self.class_names: list[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_size = int(get_config_value(self.config, 'data.image_size', 224))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                get_config_value(self.config, 'data.normalize_mean', [0.485, 0.456, 0.406]),
                get_config_value(self.config, 'data.normalize_std', [0.229, 0.224, 0.225]),
            ),
        ])
        self.baseline = read_json(resolve_path(self.config, 'paths.drift_baseline_path'), {'brightness_mean': 0.0, 'brightness_std': 1.0})
        self.reload()

    def reload(self):
        model_path = Path(self.model_uri)
        LOGGER.info('Reloading model from %s', model_path)
        if not model_path.exists():
            self.model = None
            self.class_names = []
            return
        self.model = mlflow.pytorch.load_model(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        class_names_path = model_path / 'class_names.json'
        self.class_names = json.loads(class_names_path.read_text(encoding='utf-8')) if class_names_path.exists() else []
        LOGGER.info('Model reload complete ready=%s classes=%s', self.ready, self.class_names)

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def predict(self, image: Image.Image, top_k: int | None = None) -> dict:
        if not self.ready:
            raise RuntimeError('Model is not loaded.')
        top_k = top_k or int(get_config_value(self.config, 'inference.top_k', 3))
        brightness = sum(sum(pixel) / 3 for pixel in image.convert('RGB').getdata()) / (image.width * image.height)
        baseline_std = self.baseline.get('brightness_std', 1.0) or 1.0
        brightness_zscore = (brightness - self.baseline.get('brightness_mean', 0.0)) / baseline_std
        tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        latency_ms = (time.perf_counter() - start) * 1000
        top_predictions = [
            {'label': self.class_names[idx], 'probability': float(score)}
            for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]
        result = {
            'predicted_label': top_predictions[0]['label'],
            'top_k': top_predictions,
            'model_version': Path(self.model_uri).name,
            'latency_ms': latency_ms,
            'brightness_zscore': float(brightness_zscore),
        }
        LOGGER.info('Inference result label=%s latency_ms=%.2f z=%.3f', result['predicted_label'], latency_ms, result['brightness_zscore'])
        return result
