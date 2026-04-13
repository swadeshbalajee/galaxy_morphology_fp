
from __future__ import annotations

import argparse
from datetime import datetime

from src.common.config import get_config_value, load_config, resolve_path
from src.common.io_utils import read_json, write_json
from src.common.logging_utils import configure_logging
from src.training.evaluate import evaluate_live_feedback

LOGGER = configure_logging("reporting")


def generate_report() -> dict:
    config = load_config()
    raw_summary = read_json(resolve_path(config, 'paths.raw_summary_path'), {})
    processed_v1_summary = read_json(resolve_path(config, 'paths.processed_v1_summary_path'), {})
    processed_final_summary = read_json(resolve_path(config, 'paths.processed_final_summary_path'), {})
    train_metrics = read_json(resolve_path(config, 'paths.train_metrics_path'), {})
    validation_metrics = read_json(resolve_path(config, 'paths.validation_metrics_path'), {})
    test_metrics = read_json(resolve_path(config, 'paths.test_metrics_path'), {})
    live_metrics = evaluate_live_feedback()
    runtime_summary = read_json(resolve_path(config, 'paths.pipeline_runtime_summary_path'), {})

    created_at = datetime.utcnow().isoformat() + 'Z'
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
    }

    md_lines = [
        f"# {report['title']}",
        '',
        f"Generated at: {created_at}",
        '',
        '## Pipeline flow',
        'DVC owns the data -> preprocess v1 -> preprocess final -> train -> evaluate -> report artifact flow.',
        'Airflow acts as the control plane for monitoring, retraining decisions, report delivery, and service reloads.',
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
    write_json(resolve_path(config, 'paths.pipeline_runtime_summary_path'), {**runtime_summary, 'last_report_generated_at': created_at})
    LOGGER.info('Report generated at %s', md_path)
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description='Generate the latest markdown and HTML report.').parse_args()


if __name__ == '__main__':
    parse_args()
    generate_report()
