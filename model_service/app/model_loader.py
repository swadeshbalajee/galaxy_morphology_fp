from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from PIL import Image
from torchvision import transforms

from src.common.artifact_store import load_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("model_loader")


class GalaxyPredictor:
    def __init__(self, model_uri: str | None = None):
        self.config = load_config()
        configured_registry_uri = get_config_value(
            self.config,
            "registry.serving_model_uri",
            "models:/galaxy_morphology_classifier@champion",
        )
        self.model_uri = model_uri or os.getenv("MODEL_URI") or configured_registry_uri
        self.local_fallback_uri = str(resolve_path(self.config, "paths.models_dir"))

        self.model = None
        self.class_names: list[str] = []
        self._download_dir: Path | None = None

        self.active_model_source: str | None = None
        self.active_model_name: str | None = None
        self.active_model_alias: str | None = None
        self.active_model_version: str | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_size = int(get_config_value(self.config, "data.image_size", 224))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    get_config_value(
                        self.config, "data.normalize_mean", [0.485, 0.456, 0.406]
                    ),
                    get_config_value(
                        self.config, "data.normalize_std", [0.229, 0.224, 0.225]
                    ),
                ),
            ]
        )
        self.baseline = load_pipeline_artifact(
            "drift_baseline",
            config=self.config,
            default=read_json(
                resolve_path(self.config, "paths.drift_baseline_path"),
                {"brightness_mean": 0.0, "brightness_std": 1.0},
            ),
        )
        self.reload()

    def _is_remote_uri(self, uri: str) -> bool:
        return uri.startswith("models:/") or uri.startswith("runs:/") or "://" in uri

    def _cleanup_download_dir(self) -> None:
        if self._download_dir and self._download_dir.exists():
            shutil.rmtree(self._download_dir, ignore_errors=True)
        self._download_dir = None

    def _configure_tracking_uri(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or get_config_value(
            self.config,
            "services.mlflow_url",
        )
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def _parse_registry_uri(
        self, uri: str
    ) -> tuple[str | None, str | None, str | None]:
        """
        Returns: (model_name, alias, version)

        Examples:
          models:/galaxy_morphology_classifier@champion -> ("galaxy_morphology_classifier", "champion", None)
          models:/galaxy_morphology_classifier/17       -> ("galaxy_morphology_classifier", None, "17")
        """
        if not uri.startswith("models:/"):
            return None, None, None

        tail = uri[len("models:/") :]

        if "@" in tail:
            model_name, alias = tail.split("@", 1)
            return model_name, alias, None

        if "/" in tail:
            model_name, version = tail.rsplit("/", 1)
            return model_name, None, version

        return tail, None, None

    def _resolve_serving_metadata(
        self, candidate_uri: str
    ) -> tuple[str | None, str | None, str | None]:
        model_name, model_alias, model_version = self._parse_registry_uri(candidate_uri)

        if model_name and model_alias:
            self._configure_tracking_uri()
            mv = MlflowClient().get_model_version_by_alias(model_name, model_alias)
            model_version = str(mv.version)

        if candidate_uri == self.local_fallback_uri:
            return "local", None, "local"

        return model_name, model_alias, model_version

    def _resolve_model_path(self, uri: str) -> Path:
        if not self._is_remote_uri(uri):
            return Path(uri)

        self._configure_tracking_uri()
        self._cleanup_download_dir()
        self._download_dir = Path(tempfile.mkdtemp(prefix="galaxy-model-"))
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=str(self._download_dir),
        )
        return Path(downloaded)

    def reload(self) -> None:
        load_errors: list[str] = []

        for candidate_uri in [self.model_uri, self.local_fallback_uri]:
            try:
                model_path = self._resolve_model_path(candidate_uri)
                LOGGER.info("Reloading model from %s", candidate_uri)

                if not model_path.exists():
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")

                self.model = mlflow.pytorch.load_model(str(model_path))
                self.model.to(self.device)
                self.model.eval()

                class_names_path = model_path / "class_names.json"
                self.class_names = (
                    json.loads(class_names_path.read_text(encoding="utf-8"))
                    if class_names_path.exists()
                    else []
                )
                if not self.class_names:
                    raise RuntimeError(f"class_names.json missing under {model_path}")

                model_name, model_alias, model_version = self._resolve_serving_metadata(
                    candidate_uri
                )
                self.active_model_source = candidate_uri
                self.active_model_name = model_name
                self.active_model_alias = model_alias
                self.active_model_version = model_version or "unknown"

                LOGGER.info(
                    "Model reload complete ready=%s classes=%s source=%s alias=%s version=%s",
                    self.ready,
                    self.class_names,
                    candidate_uri,
                    self.active_model_alias,
                    self.active_model_version,
                )
                return

            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{candidate_uri}: {exc}")
                LOGGER.warning("Failed to load model from %s: %s", candidate_uri, exc)

        self.model = None
        self.class_names = []
        self.active_model_source = None
        self.active_model_name = None
        self.active_model_alias = None
        self.active_model_version = None
        LOGGER.error("Model reload failed. Tried: %s", " | ".join(load_errors))

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def predict(self, image: Image.Image, top_k: int | None = None) -> dict:
        if not self.ready:
            raise RuntimeError("Model is not loaded.")

        top_k = top_k or int(get_config_value(self.config, "inference.top_k", 3))

        brightness = sum(sum(pixel) / 3 for pixel in image.convert("RGB").getdata()) / (
            image.width * image.height
        )
        baseline_std = self.baseline.get("brightness_std", 1.0) or 1.0
        brightness_zscore = (
            brightness - self.baseline.get("brightness_mean", 0.0)
        ) / baseline_std

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(
                probabilities, k=min(top_k, len(self.class_names))
            )
        latency_ms = (time.perf_counter() - start) * 1000

        top_predictions = [
            {"label": self.class_names[idx], "probability": float(score)}
            for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]

        result = {
            "predicted_label": top_predictions[0]["label"],
            "top_k": top_predictions,
            "model_version": self.active_model_version
            or "unknown",  # <-- numeric version now
            "model_alias": self.active_model_alias,  # optional, useful for UI/debug
            "model_source": self.active_model_source,  # optional, useful for UI/debug
            "latency_ms": latency_ms,
            "brightness_zscore": float(brightness_zscore),
        }

        LOGGER.info(
            "Inference result label=%s latency_ms=%.2f z=%.3f alias=%s version=%s",
            result["predicted_label"],
            latency_ms,
            result["brightness_zscore"],
            result["model_alias"],
            result["model_version"],
        )
        return result
