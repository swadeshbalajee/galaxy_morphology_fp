from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelPredictionResponse(BaseModel):
    predicted_label: str
    top_k: list[dict]
    model_version: str
    latency_ms: float = Field(ge=0)
    brightness_zscore: float
