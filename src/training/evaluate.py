from __future__ import annotations

from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.common.config import load_config, resolve_path
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging
from src.common.postgres import get_db_connection, initialize_database

LOGGER = configure_logging('evaluation')


def evaluate_offline() -> dict:
    config = load_config()
    metrics_path = resolve_path(config, 'paths.test_metrics_path')
    metrics = read_json(metrics_path, {}) or {}

    # Always ensure the file exists, even if empty/default
    if not metrics:
        metrics = {
            'status': 'no_offline_metrics',
        }
        write_json(metrics_path, metrics)

    LOGGER.info('Offline evaluation snapshot: %s', metrics)
    return metrics


def evaluate_live_feedback(predictions_db_path: str | Path | None = None) -> dict:  # noqa: ARG001
    config = load_config()
    initialize_database()
    metrics = {
        'feedback_count': 0,
        'accuracy': None,
        'macro_f1': None,
    }

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.predicted_label, c.corrected_label
                    FROM feedback_corrections c
                    JOIN predictions p ON p.prediction_id = c.prediction_id
                    ORDER BY c.created_at ASC
                    """,
                )
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Live feedback database not ready: %s', exc)
        write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
        return metrics

    if not rows:
        write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
        LOGGER.info('No live feedback rows available.')
        return metrics

    y_pred = [row[0] for row in rows]
    y_true = [row[1] for row in rows]
    metrics = {
        'feedback_count': len(rows),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 6),
        'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 6),
    }
    write_json(resolve_path(config, 'paths.live_metrics_path'), metrics)
    LOGGER.info('Live feedback evaluation complete: %s', metrics)
    return metrics


def main() -> None:
    offline_metrics = evaluate_offline()
    live_metrics = evaluate_live_feedback()
    LOGGER.info('Evaluation complete. offline=%s live=%s', offline_metrics, live_metrics)


if __name__ == '__main__':
    main()