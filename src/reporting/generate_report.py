from __future__ import annotations

import argparse
from datetime import UTC, datetime

from src.common.artifact_store import load_pipeline_artifact, merge_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.logging_utils import configure_logging
from src.training.evaluate import evaluate_live_feedback

LOGGER = configure_logging("reporting")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace('+00:00', 'Z')


def generate_report() -> dict:
    config = load_config()
    raw_summary = load_pipeline_artifact('raw_summary', config=config, default={})
    processed_v1_summary = load_pipeline_artifact('processed_v1_summary', config=config, default={})
    processed_final_summary = load_pipeline_artifact('processed_final_summary', config=config, default={})
    train_metrics = load_pipeline_artifact('train_metrics', config=config, default={})
    validation_metrics = load_pipeline_artifact('validation_metrics', config=config, default={})
    test_metrics = load_pipeline_artifact('test_metrics', config=config, default={})
    live_metrics = evaluate_live_feedback()
    runtime_summary = load_pipeline_artifact('pipeline_runtime_summary', config=config, default={})
    registry_status = load_pipeline_artifact('registry_status', config=config, default={})
    registry_data = registry_status if isinstance(registry_status, dict) else {}

    created_at = _utc_now_iso()
    report = {
        'generated_at': created_at,
        'title': get_config_value(config, 'reporting.report_title', 'Galaxy report'),
        'raw_summary': raw_summary,
        'processed_v1_summary': processed_v1_summary,
        'processed_final_summary': processed_final_summary,
        'train_metrics': train_metrics,
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'live_metrics': live_metrics,
        'runtime_summary': runtime_summary,
        'registry_status': registry_status,
    }

    candidate = registry_data.get('candidate') or {}
    previous = registry_data.get('previous_champion') or {}

    md_lines = [
        f"# {report['title']}",
        '',
        f"Generated at: {created_at}",
        '',
        '## Pipeline flow',
        'DVC owns the data -> preprocess v1 -> preprocess final -> train -> evaluate -> report artifact flow.',
        'Airflow acts as the control plane for monitoring, retraining decisions, registry promotion, report delivery, and service reloads.',
        '',
        '## Data summary',
        f"- Raw counts: {raw_summary.get('actual_counts', {})}",
        f"- Processed v1 total images: {processed_v1_summary.get('total_images')}",
        f"- Final split summary: {processed_final_summary.get('splits', {})}",
        '',
        '## Validation metrics from training stage',
        f"- Validation accuracy: {validation_metrics.get('accuracy')}",
        f"- Validation macro F1: {validation_metrics.get('macro_f1')}",
        '',
        '## Offline model metrics',
        f"- Accuracy: {test_metrics.get('accuracy')}",
        f"- Macro F1: {test_metrics.get('macro_f1')}",
        f"- Precision macro: {test_metrics.get('precision_macro')}",
        f"- Recall macro: {test_metrics.get('recall_macro')}",
        '',
        '## Model registry decision',
        f"- Candidate version: {candidate.get('version')}",
        f"- Candidate run id: {candidate.get('run_id')}",
        f"- Candidate metric ({candidate.get('metric_name')}): {candidate.get('metric_value')}",
        f"- Previous champion version: {previous.get('version')}",
        f"- Previous champion metric ({previous.get('metric_name')}): {previous.get('metric_value')}",
        f"- Champion updated: {registry_data.get('champion_updated')}",
        f"- Current champion version: {registry_data.get('current_champion_version')}",
        f"- Serving model URI: {registry_data.get('serving_model_uri')}",
        f"- Decision reason: {registry_data.get('decision_reason')}",
        '',
        '## Continuous improvement metrics',
        f"- Feedback count: {live_metrics.get('feedback_count')}",
        f"- Live accuracy: {live_metrics.get('accuracy')}",
        f"- Live macro F1: {live_metrics.get('macro_f1')}",
        '',
        '## Training runtime',
        f"- Training duration seconds: {runtime_summary.get('train_duration_seconds')}",
        f"- Epochs completed: {runtime_summary.get('epochs_completed')}",
    ]

    html = f"""
    <html>
      <body>
        <h1>{report['title']}</h1>
        <p><strong>Generated at:</strong> {created_at}</p>
        <h2>Validation metrics</h2>
        <ul>
          <li>Validation accuracy: {validation_metrics.get('accuracy')}</li>
          <li>Validation macro F1: {validation_metrics.get('macro_f1')}</li>
        </ul>
        <h2>Offline metrics</h2>
        <ul>
          <li>Accuracy: {test_metrics.get('accuracy')}</li>
          <li>Macro F1: {test_metrics.get('macro_f1')}</li>
          <li>Precision macro: {test_metrics.get('precision_macro')}</li>
          <li>Recall macro: {test_metrics.get('recall_macro')}</li>
        </ul>
        <h2>Registry decision</h2>
        <ul>
          <li>Candidate version: {candidate.get('version')}</li>
          <li>Candidate metric ({candidate.get('metric_name')}): {candidate.get('metric_value')}</li>
          <li>Previous champion version: {previous.get('version')}</li>
          <li>Previous champion metric ({previous.get('metric_name')}): {previous.get('metric_value')}</li>
          <li>Champion updated: {registry_data.get('champion_updated')}</li>
          <li>Current champion version: {registry_data.get('current_champion_version')}</li>
          <li>Serving model URI: {registry_data.get('serving_model_uri')}</li>
          <li>Decision reason: {registry_data.get('decision_reason')}</li>
        </ul>
        <h2>Continuous improvement</h2>
        <ul>
          <li>Feedback count: {live_metrics.get('feedback_count')}</li>
          <li>Live accuracy: {live_metrics.get('accuracy')}</li>
          <li>Live macro F1: {live_metrics.get('macro_f1')}</li>
        </ul>
        <h2>Raw data counts</h2>
        <pre>{raw_summary.get('actual_counts', {})}</pre>
      </body>
    </html>
    """

    md_path = resolve_path(config, 'paths.latest_report_md_path')
    html_path = resolve_path(config, 'paths.latest_report_html_path')
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    html_path.write_text(html, encoding='utf-8')
    merge_pipeline_artifact(
        'pipeline_runtime_summary',
        {'last_report_generated_at': created_at},
        config=config,
        stage_name='report',
        recorded_at=created_at,
    )
    LOGGER.info('Report generated at %s', md_path)
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description='Generate the latest markdown and HTML report.').parse_args()


if __name__ == '__main__':
    parse_args()
    generate_report()
