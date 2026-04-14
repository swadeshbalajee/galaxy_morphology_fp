from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionResponse(BaseModel):
    prediction_id: str
    batch_id: str | None = None
    original_filename: str | None = None
    predicted_label: str
    top_k: list[dict]
    model_version: str
    latency_ms: float = Field(ge=0)
    brightness_zscore: float | None = None


class BatchPredictionResponse(BaseModel):
    batch_id: str
    total_images: int
    items: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    prediction_id: str
    ground_truth_label: str
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    prediction_id: str


class FeedbackCsvUploadResponse(BaseModel):
    status: str
    row_count: int
    upload_id: str | None = None
    errors: list[dict] = Field(default_factory=list)
