
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.app.feedback_store import FeedbackStore
from api.app.metrics import API_LATENCY, API_READY, API_REQUESTS, FEEDBACK_COUNT, RECENT_PREDICTIONS_SIZE
from api.app.model_client import ModelServiceClient
from api.app.schemas import FeedbackRequest, FeedbackResponse, HealthResponse, PredictionResponse
from src.common.config import get_config_value, load_config
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('api_gateway')
CONFIG = load_config()
app = FastAPI(title='Galaxy API Gateway', version='2.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/metrics', make_asgi_app())

store = FeedbackStore()
model_client = ModelServiceClient()
MAX_UPLOAD_MB = int(get_config_value(CONFIG, 'runtime.max_upload_mb', 10))


@app.on_event('startup')
async def startup_event():
    API_READY.set(1)
    LOGGER.info('API gateway started. max_upload_mb=%s', MAX_UPLOAD_MB)


@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(status='ok', service='api-gateway')


@app.get('/ready', response_model=HealthResponse)
async def ready():
    try:
        await model_client.ready()
        return HealthResponse(status='ready', service='api-gateway')
    except Exception as exc:  # noqa: BLE001
        API_READY.set(0)
        LOGGER.exception('Readiness check failed: %s', exc)
        raise HTTPException(status_code=503, detail='Model service not ready') from exc


@app.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail='Filename is required.')
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f'File exceeds {MAX_UPLOAD_MB} MB limit.')

    try:
        with API_LATENCY.labels('/predict').time():
            result = await model_client.predict(file.filename, content, file.content_type or 'application/octet-stream')
        prediction_id = store.create_prediction(result['predicted_label'], result['model_version'], result['latency_ms'])
        duration_ms = (time.perf_counter() - start) * 1000
        API_REQUESTS.labels('/predict', 'POST', '200').inc()
        LOGGER.info('Prediction completed file=%s api_latency_ms=%.2f model_latency_ms=%.2f', file.filename, duration_ms, result['latency_ms'])
        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_label=result['predicted_label'],
            top_k=result['top_k'],
            model_version=result['model_version'],
            latency_ms=result['latency_ms'],
        )
    except Exception as exc:  # noqa: BLE001
        API_REQUESTS.labels('/predict', 'POST', '500').inc()
        LOGGER.exception('Prediction failed: %s', exc)
        raise HTTPException(status_code=500, detail='Prediction failed') from exc


@app.post('/feedback', response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    try:
        store.add_feedback(payload.prediction_id, payload.ground_truth_label, payload.notes)
        FEEDBACK_COUNT.inc()
        API_REQUESTS.labels('/feedback', 'POST', '200').inc()
        LOGGER.info('Feedback accepted for prediction_id=%s label=%s', payload.prediction_id, payload.ground_truth_label)
        return FeedbackResponse(status='recorded', prediction_id=payload.prediction_id)
    except Exception as exc:  # noqa: BLE001
        API_REQUESTS.labels('/feedback', 'POST', '500').inc()
        LOGGER.exception('Feedback write failed: %s', exc)
        raise HTTPException(status_code=500, detail='Unable to save feedback') from exc


@app.get('/recent-predictions')
async def recent_predictions(limit: int = 10):
    try:
        items = store.recent_predictions(limit=limit)
        RECENT_PREDICTIONS_SIZE.set(len(items))
        LOGGER.info('Recent predictions requested limit=%s returned=%s', limit, len(items))
        return {'items': items, 'summary': store.feedback_summary()}
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception('Unable to fetch recent predictions: %s', exc)
        raise HTTPException(status_code=500, detail='Unable to fetch recent predictions') from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    LOGGER.exception('Unhandled exception: %s', exc)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})
