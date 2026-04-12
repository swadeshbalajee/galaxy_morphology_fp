
from __future__ import annotations

import os

import httpx

from src.common.config import get_config_value, load_config


class ModelServiceClient:
    def __init__(self, base_url: str | None = None):
        config = load_config()
        self.base_url = base_url or os.getenv('MODEL_SERVICE_URL') or get_config_value(config, 'services.model_service_url', 'http://model-service:8001')

    async def ready(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f'{self.base_url}/ready')
            response.raise_for_status()
            return response.json()

    async def predict(self, filename: str, content: bytes, content_type: str) -> dict:
        files = {'file': (filename, content, content_type)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f'{self.base_url}/predict', files=files)
            response.raise_for_status()
            return response.json()
