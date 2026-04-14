from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date, datetime, time, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.common.logging_utils import configure_logging
from src.common.postgres import cluster_table, get_db_connection, initialize_database

LOGGER = configure_logging('feedback_store')

RECENT_PREDICTION_COLUMNS = [
    'prediction_id',
    'batch_id',
    'original_filename',
    'predicted_label',
    'model_version',
    'latency_ms',
    'created_at',
]
FEEDBACK_UPLOAD_COLUMNS = RECENT_PREDICTION_COLUMNS + ['corrected_label']


class FeedbackStore:
    def __init__(self):
        initialize_database()

    @staticmethod
    def _normalize_date_bounds(start_date: date | None, end_date: date | None) -> tuple[datetime | None, datetime | None]:
        start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc) if start_date else None
        end_dt = datetime.combine(end_date, time.max, tzinfo=timezone.utc) if end_date else None
        return start_dt, end_dt

    def create_batch(self, source_filename: str, source_type: str, total_files: int) -> str:
        batch_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_batches(batch_id, source_filename, source_type, total_files)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (batch_id, source_filename, source_type, total_files),
                )
        return batch_id

    def create_prediction(
        self,
        *,
        batch_id: str | None,
        original_filename: str,
        content_type: str,
        image_bytes: bytes,
        predicted_label: str,
        top_k: list[dict[str, Any]],
        model_version: str,
        latency_ms: float,
        brightness_zscore: float | None,
    ) -> str:
        prediction_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions(
                        prediction_id, batch_id, original_filename, content_type, image_bytes,
                        predicted_label, top_k, model_version, latency_ms, brightness_zscore
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        prediction_id,
                        batch_id,
                        original_filename,
                        content_type,
                        image_bytes,
                        predicted_label,
                        Jsonb(top_k),
                        model_version,
                        latency_ms,
                        brightness_zscore,
                    ),
                )
        return prediction_id

    def add_feedback(self, prediction_id: str, ground_truth_label: str, notes: str | None = None) -> None:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT prediction_id, original_filename, predicted_label, model_version, latency_ms, created_at
                    FROM predictions
                    WHERE prediction_id = %s
                    """,
                    (prediction_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError(f'Prediction {prediction_id} does not exist.')
                if row['predicted_label'] == ground_truth_label:
                    raise ValueError('The corrected label must be different from the predicted label.')
                cur.execute(
                    """
                    INSERT INTO feedback_corrections(
                        correction_id, upload_id, prediction_id, original_filename, predicted_label,
                        corrected_label, model_version, latency_ms, prediction_created_at, notes
                    )
                    VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id) DO UPDATE
                    SET corrected_label = EXCLUDED.corrected_label,
                        notes = EXCLUDED.notes,
                        created_at = NOW()
                    """,
                    (
                        str(uuid.uuid4()),
                        row['prediction_id'],
                        row['original_filename'],
                        row['predicted_label'],
                        ground_truth_label,
                        row['model_version'],
                        row['latency_ms'],
                        row['created_at'],
                        notes,
                    ),
                )
        cluster_table('feedback_corrections', 'idx_feedback_corrections_created_at')
        LOGGER.info('Feedback recorded for prediction_id=%s', prediction_id)

    def recent_predictions(
        self,
        *,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        start_dt, end_dt = self._normalize_date_bounds(start_date, end_date)
        if start_dt:
            where_clauses.append('p.created_at >= %s')
            params.append(start_dt)
        if end_dt:
            where_clauses.append('p.created_at <= %s')
            params.append(end_dt)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ''
        params.append(limit)
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        p.prediction_id,
                        p.batch_id,
                        p.original_filename,
                        p.predicted_label,
                        p.model_version,
                        ROUND(p.latency_ms::numeric, 2) AS latency_ms,
                        p.created_at
                    FROM predictions p
                    {where_sql}
                    ORDER BY p.created_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [
            {
                **row,
                'created_at': row['created_at'].isoformat(),
                'batch_id': str(row['batch_id']) if row['batch_id'] else None,
                'prediction_id': str(row['prediction_id']),
            }
            for row in rows
        ]

    def export_recent_predictions_csv(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 5000,
    ) -> bytes:
        rows = self.recent_predictions(limit=limit, start_date=start_date, end_date=end_date)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=FEEDBACK_UPLOAD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'corrected_label': ''})
        return output.getvalue().encode('utf-8')

    def feedback_summary(self) -> dict[str, int]:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM predictions')
                prediction_count = int(cur.fetchone()[0])
                cur.execute('SELECT COUNT(*) FROM feedback_corrections')
                feedback_count = int(cur.fetchone()[0])
        return {
            'prediction_count': prediction_count,
            'feedback_count': feedback_count,
        }

    def validate_feedback_csv(self, csv_bytes: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        text = csv_bytes.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        errors: list[dict[str, Any]] = []
        missing_columns = [column for column in FEEDBACK_UPLOAD_COLUMNS if column not in fieldnames]
        extra_columns = [column for column in fieldnames if column not in FEEDBACK_UPLOAD_COLUMNS]
        if missing_columns:
            errors.append({'row_number': 'header', 'error': f"Missing required columns: {', '.join(missing_columns)}"})
        if extra_columns:
            errors.append({'row_number': 'header', 'error': f"Unexpected columns present: {', '.join(extra_columns)}"})
        if errors:
            return [], errors, []

        valid_rows: list[dict[str, Any]] = []
        uploaded_rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: (value or '').strip() for key, value in row.items()}
            uploaded_rows.append(normalized)
            for column in FEEDBACK_UPLOAD_COLUMNS:
                if not normalized.get(column):
                    errors.append({'row_number': row_number, 'error': f'Column `{column}` is empty.'})

            predicted_label = normalized.get('predicted_label', '')
            corrected_label = normalized.get('corrected_label', '')
            if predicted_label and corrected_label and predicted_label == corrected_label:
                errors.append({'row_number': row_number, 'error': '`predicted_label` must be different from `corrected_label`.'})

            prediction_id = normalized.get('prediction_id')
            if prediction_id:
                with get_db_connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            SELECT prediction_id, batch_id, original_filename, predicted_label, model_version,
                                   ROUND(latency_ms::numeric, 2) AS latency_ms, created_at
                            FROM predictions
                            WHERE prediction_id = %s
                            """,
                            (prediction_id,),
                        )
                        db_row = cur.fetchone()
                if not db_row:
                    errors.append({'row_number': row_number, 'error': f'Prediction id `{prediction_id}` was not found in the database.'})
                else:
                    expected = {
                        'prediction_id': str(db_row['prediction_id']),
                        'batch_id': str(db_row['batch_id']) if db_row['batch_id'] else '',
                        'original_filename': db_row['original_filename'],
                        'predicted_label': db_row['predicted_label'],
                        'model_version': db_row['model_version'],
                        'latency_ms': f"{float(db_row['latency_ms']):.2f}",
                        'created_at': db_row['created_at'].isoformat(),
                    }
                    for column, expected_value in expected.items():
                        given_value = normalized.get(column, '')
                        comparable_given = given_value
                        if column == 'latency_ms':
                            try:
                                comparable_given = f"{float(given_value):.2f}"
                            except ValueError:
                                errors.append({'row_number': row_number, 'error': '`latency_ms` is not numeric.'})
                                continue
                        if comparable_given != expected_value:
                            errors.append({
                                'row_number': row_number,
                                'error': f'Column `{column}` does not match the stored prediction record.',
                            })

            if not any(error['row_number'] == row_number for error in errors):
                valid_rows.append(normalized)

        return valid_rows, errors, uploaded_rows

    def upload_feedback_csv(self, source_filename: str, csv_bytes: bytes) -> dict[str, Any]:
        valid_rows, errors, uploaded_rows = self.validate_feedback_csv(csv_bytes)
        if errors:
            return {'status': 'validation_failed', 'errors': errors, 'row_count': len(uploaded_rows)}

        upload_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback_uploads(upload_id, source_filename, raw_csv, row_count)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (upload_id, source_filename, csv_bytes, len(valid_rows)),
                )
                for row in valid_rows:
                    cur.execute(
                        """
                        INSERT INTO feedback_corrections(
                            correction_id, upload_id, prediction_id, original_filename, predicted_label,
                            corrected_label, model_version, latency_ms, prediction_created_at, notes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (prediction_id) DO UPDATE
                        SET upload_id = EXCLUDED.upload_id,
                            corrected_label = EXCLUDED.corrected_label,
                            model_version = EXCLUDED.model_version,
                            latency_ms = EXCLUDED.latency_ms,
                            prediction_created_at = EXCLUDED.prediction_created_at,
                            created_at = NOW()
                        """,
                        (
                            str(uuid.uuid4()),
                            upload_id,
                            row['prediction_id'],
                            row['original_filename'],
                            row['predicted_label'],
                            row['corrected_label'],
                            row['model_version'],
                            float(row['latency_ms']),
                            row['created_at'],
                        ),
                    )
        cluster_table('feedback_uploads', 'idx_feedback_uploads_created_at')
        cluster_table('feedback_corrections', 'idx_feedback_corrections_created_at')
        LOGGER.info('Feedback CSV uploaded source=%s rows=%s', source_filename, len(valid_rows))
        return {
            'status': 'uploaded',
            'upload_id': upload_id,
            'row_count': len(valid_rows),
            'errors': [],
        }
