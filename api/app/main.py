from __future__ import annotations

import io
import time
import zipfile
from datetime import date

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import make_asgi_app

from api.app.feedback_store import FeedbackStore
from api.app.metrics import (
    API_LATENCY,
    API_READY,
    API_REQUESTS,
    BATCH_UPLOADS_TOTAL,
    CORRECTION_ROWS_STORED,
    CSV_UPLOADS_TOTAL,
    CSV_VALIDATION_FAILURES_TOTAL,
    DB_READY,
    FEEDBACK_COUNT,
    PREDICTION_ROWS_STORED,
    RECENT_PREDICTIONS_SIZE,
)
from api.app.model_client import ModelServiceClient
from api.app.schemas import (
    BatchPredictionResponse,
    FeedbackCsvUploadResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictionResponse,
)
from src.common.config import get_config_value, load_config
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection

LOGGER = configure_logging("api_gateway")
CONFIG = load_config()
app = FastAPI(title="Galaxy API Gateway", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app())

store: FeedbackStore | None = None
model_client = ModelServiceClient()


def _required_config(key: str):
    value = get_config_value(CONFIG, key)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value


MAX_UPLOAD_MB = int(_required_config("runtime.max_upload_mb"))
MAX_ZIP_UPLOAD_MB = int(_required_config("runtime.max_zip_upload_mb"))
ALLOWED_SUFFIXES = {
    suffix.lower()
    for suffix in get_config_value(
        CONFIG,
        "data.allowed_suffixes",
        [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    )
}


def _get_store() -> FeedbackStore:
    global store
    if store is None:
        store = FeedbackStore()
    return store


@app.on_event("startup")
async def startup_event():
    API_READY.set(1)
    try:
        _get_store()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        DB_READY.set(1)
    except Exception as exc:  # noqa: BLE001
        DB_READY.set(0)
        LOGGER.exception("Database startup check failed: %s", exc)
    LOGGER.info(
        "API gateway started. max_upload_mb=%s max_zip_upload_mb=%s",
        MAX_UPLOAD_MB,
        MAX_ZIP_UPLOAD_MB,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="api-gateway")


@app.get("/ready", response_model=HealthResponse)
async def ready():
    try:
        await model_client.ready()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        API_READY.set(1)
        DB_READY.set(1)
        return HealthResponse(status="ready", service="api-gateway")
    except Exception as exc:  # noqa: BLE001
        API_READY.set(0)
        DB_READY.set(0)
        LOGGER.exception("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Model service or database not ready") from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    try:
        with API_LATENCY.labels("/predict").time():
            result = await model_client.predict(
                file.filename,
                content,
                file.content_type or "application/octet-stream",
            )

        batch_id = _get_store().create_batch(
            source_filename=file.filename,
            source_type="single_image",
            total_files=1,
        )
        prediction_id = _get_store().create_prediction(
            batch_id=batch_id,
            original_filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            image_bytes=content,
            predicted_label=result["predicted_label"],
            top_k=result["top_k"],
            model_version=result["model_version"],  # numeric version now
            latency_ms=result["latency_ms"],
            brightness_zscore=result.get("brightness_zscore"),
        )
        PREDICTION_ROWS_STORED.inc()

        duration_ms = (time.perf_counter() - start) * 1000
        API_REQUESTS.labels("/predict", "POST", "200").inc()
        LOGGER.info(
            "Prediction completed file=%s api_latency_ms=%.2f model_latency_ms=%.2f model_version=%s",
            file.filename,
            duration_ms,
            result["latency_ms"],
            result["model_version"],
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            batch_id=batch_id,
            original_filename=file.filename,
            predicted_label=result["predicted_label"],
            top_k=result["top_k"],
            model_version=result["model_version"],
            model_alias=result.get("model_alias"),
            model_source=result.get("model_source"),
            latency_ms=result["latency_ms"],
            brightness_zscore=result.get("brightness_zscore"),
        )
    except Exception as exc:  # noqa: BLE001
        API_REQUESTS.labels("/predict", "POST", "500").inc()
        LOGGER.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(zip_file: UploadFile = File(...)):
    if not zip_file.filename or not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="A ZIP file is required.")

    archive_bytes = await zip_file.read()
    if len(archive_bytes) > MAX_ZIP_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"ZIP file exceeds {MAX_ZIP_UPLOAD_MB} MB limit.")

    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.") from exc

    image_members = [
        member
        for member in archive.infolist()
        if not member.is_dir() and any(member.filename.lower().endswith(suffix) for suffix in ALLOWED_SUFFIXES)
    ]
    if not image_members:
        raise HTTPException(status_code=400, detail="ZIP archive does not contain supported image files.")

    batch_id = _get_store().create_batch(
        source_filename=zip_file.filename,
        source_type="zip_archive",
        total_files=len(image_members),
    )

    items: list[PredictionResponse] = []
    for member in image_members:
        content = archive.read(member)
        content_type = "application/octet-stream"
        result = await model_client.predict(member.filename, content, content_type)

        prediction_id = _get_store().create_prediction(
            batch_id=batch_id,
            original_filename=member.filename,
            content_type=content_type,
            image_bytes=content,
            predicted_label=result["predicted_label"],
            top_k=result["top_k"],
            model_version=result["model_version"],  # numeric version now
            latency_ms=result["latency_ms"],
            brightness_zscore=result.get("brightness_zscore"),
        )
        PREDICTION_ROWS_STORED.inc()

        items.append(
            PredictionResponse(
                prediction_id=prediction_id,
                batch_id=batch_id,
                original_filename=member.filename,
                predicted_label=result["predicted_label"],
                top_k=result["top_k"],
                model_version=result["model_version"],
                model_alias=result.get("model_alias"),
                model_source=result.get("model_source"),
                latency_ms=result["latency_ms"],
                brightness_zscore=result.get("brightness_zscore"),
            )
        )

    BATCH_UPLOADS_TOTAL.inc()
    API_REQUESTS.labels("/predict-batch", "POST", "200").inc()
    LOGGER.info(
        "Batch prediction completed archive=%s rows=%s batch_id=%s",
        zip_file.filename,
        len(items),
        batch_id,
    )
    return BatchPredictionResponse(batch_id=batch_id, total_images=len(items), items=items)


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    try:
        _get_store().add_feedback(payload.prediction_id, payload.ground_truth_label, payload.notes)
        FEEDBACK_COUNT.inc()
        CORRECTION_ROWS_STORED.inc()
        API_REQUESTS.labels("/feedback", "POST", "200").inc()
        LOGGER.info(
            "Feedback accepted for prediction_id=%s label=%s",
            payload.prediction_id,
            payload.ground_truth_label,
        )
        return FeedbackResponse(status="recorded", prediction_id=payload.prediction_id)
    except ValueError as exc:
        API_REQUESTS.labels("/feedback", "POST", "400").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        API_REQUESTS.labels("/feedback", "POST", "500").inc()
        LOGGER.exception("Feedback write failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to save feedback") from exc


@app.post("/feedback/upload-csv", response_model=FeedbackCsvUploadResponse)
async def upload_feedback_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="A CSV file is required.")

    content = await file.read()
    result = _get_store().upload_feedback_csv(source_filename=file.filename, csv_bytes=content)
    if result["status"] == "validation_failed":
        CSV_VALIDATION_FAILURES_TOTAL.inc(len(result["errors"]))
        API_REQUESTS.labels("/feedback/upload-csv", "POST", "400").inc()
        return JSONResponse(status_code=400, content=result)

    CSV_UPLOADS_TOTAL.inc()
    CORRECTION_ROWS_STORED.inc(result["row_count"])
    API_REQUESTS.labels("/feedback/upload-csv", "POST", "200").inc()
    return FeedbackCsvUploadResponse(**result)


@app.get("/recent-predictions")
async def recent_predictions(
    limit: int = Query(default=50, ge=1, le=5000),
    start_date: date | None = None,
    end_date: date | None = None,
):
    try:
        items = _get_store().recent_predictions(limit=limit, start_date=start_date, end_date=end_date)
        RECENT_PREDICTIONS_SIZE.set(len(items))
        LOGGER.info(
            "Recent predictions requested limit=%s returned=%s start_date=%s end_date=%s",
            limit,
            len(items),
            start_date,
            end_date,
        )
        return {"items": items, "summary": _get_store().feedback_summary()}
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unable to fetch recent predictions: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to fetch recent predictions") from exc


@app.get("/recent-predictions/export")
async def export_recent_predictions(
    limit: int = Query(default=5000, ge=1, le=20000),
    start_date: date | None = None,
    end_date: date | None = None,
):
    try:
        content = _get_store().export_recent_predictions_csv(
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        filename = f'recent_predictions_{start_date or "all"}_{end_date or "all"}.csv'
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unable to export recent predictions: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to export recent predictions") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    LOGGER.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
