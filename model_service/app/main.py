
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from PIL import Image

from model_service.app.model_loader import GalaxyPredictor
from model_service.app.schemas import HealthResponse, ModelPredictionResponse
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('model_service')
app = FastAPI(title='Galaxy Model Service', version='2.0.0')
app.mount('/metrics', make_asgi_app())

MODEL_REQUESTS = Counter('galaxy_model_requests_total', 'Total model inference calls.', ['status'])
MODEL_LATENCY = Histogram('galaxy_model_latency_seconds', 'Model inference latency in seconds.')
MODEL_READY = Gauge('galaxy_model_ready', 'Model readiness state.')
MODEL_INPUT_BRIGHTNESS_ZSCORE = Gauge('galaxy_input_brightness_zscore', 'Brightness drift z-score for latest request.')
MODEL_LAST_LATENCY_MS = Gauge('galaxy_model_last_latency_ms', 'Last model latency in milliseconds.')

predictor = GalaxyPredictor()


@app.on_event('startup')
def startup_event():
    MODEL_READY.set(1 if predictor.ready else 0)
    LOGGER.info('Model service startup complete. ready=%s', predictor.ready)


@app.get('/health', response_model=HealthResponse)
def health():
    return HealthResponse(status='ok', service='model-service')


@app.get('/ready', response_model=HealthResponse)
def ready():
    if predictor.ready:
        MODEL_READY.set(1)
        return HealthResponse(status='ready', service='model-service')
    MODEL_READY.set(0)
    raise HTTPException(status_code=503, detail='Model artifact not available. Train/export the model first.')


@app.post('/reload', response_model=HealthResponse)
def reload_model():
    predictor.reload()
    MODEL_READY.set(1 if predictor.ready else 0)
    if not predictor.ready:
        raise HTTPException(status_code=503, detail='Model not available after reload.')
    LOGGER.info('Model reload endpoint completed successfully.')
    return HealthResponse(status='ready', service='model-service')


@app.post('/predict', response_model=ModelPredictionResponse)
def predict(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        MODEL_REQUESTS.labels('bad_request').inc()
        LOGGER.exception('Invalid image upload: %s', exc)
        raise HTTPException(status_code=400, detail='Invalid image file.') from exc

    try:
        with MODEL_LATENCY.time():
            result = predictor.predict(image)
        MODEL_INPUT_BRIGHTNESS_ZSCORE.set(result['brightness_zscore'])
        MODEL_LAST_LATENCY_MS.set(result['latency_ms'])
        MODEL_REQUESTS.labels('success').inc()
        return ModelPredictionResponse(**result)
    except Exception as exc:  # noqa: BLE001
        MODEL_REQUESTS.labels('error').inc()
        LOGGER.exception('Inference failed: %s', exc)
        raise HTTPException(status_code=500, detail='Inference failed.') from exc
