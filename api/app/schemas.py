from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_label: str
    top_k: list[dict]
    model_version: str
    latency_ms: float = Field(ge=0)


class FeedbackRequest(BaseModel):
    prediction_id: str
    ground_truth_label: str
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    prediction_id: str
