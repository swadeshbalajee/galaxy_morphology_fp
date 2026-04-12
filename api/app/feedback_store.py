
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("feedback_store")


class FeedbackStore:
    def __init__(self, db_path: str | None = None):
        config = load_config()
        self.db_path = Path(db_path) if db_path else resolve_path(config, 'paths.predictions_db_path')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    predicted_label TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    ground_truth_label TEXT NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
                )
                """
            )
            conn.commit()
        LOGGER.info('Feedback store initialized at %s', self.db_path)

    def create_prediction(self, predicted_label: str, model_version: str, latency_ms: float) -> str:
        prediction_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO predictions(prediction_id, predicted_label, model_version, latency_ms) VALUES (?, ?, ?, ?)',
                (prediction_id, predicted_label, model_version, latency_ms),
            )
            conn.commit()
        return prediction_id

    def add_feedback(self, prediction_id: str, ground_truth_label: str, notes: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO feedback(id, prediction_id, ground_truth_label, notes) VALUES (?, ?, ?, ?)',
                (str(uuid.uuid4()), prediction_id, ground_truth_label, notes),
            )
            conn.commit()
        LOGGER.info('Feedback recorded for prediction_id=%s', prediction_id)

    def recent_predictions(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT prediction_id, predicted_label, model_version, latency_ms, created_at FROM predictions ORDER BY created_at DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [
            {
                'prediction_id': row[0],
                'predicted_label': row[1],
                'model_version': row[2],
                'latency_ms': row[3],
                'created_at': row[4],
            }
            for row in rows
        ]

    def feedback_summary(self) -> dict:
        with self._connect() as conn:
            prediction_count = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
            feedback_count = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
        return {
            'prediction_count': prediction_count,
            'feedback_count': feedback_count,
        }
