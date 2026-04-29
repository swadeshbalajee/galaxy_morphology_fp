from __future__ import annotations

import argparse
import html
from datetime import UTC, datetime
from typing import Any

from src.common.artifact_store import load_pipeline_artifact
from src.common.config import get_config_value, load_config, resolve_path
from src.common.logging_utils import configure_logging
from src.training.evaluate import evaluate_live_feedback

LOGGER = configure_logging("runtime_reporting")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    return "n/a" if value is None else value


def _has_value(value: Any) -> bool:
    return value is not None and value != "n/a"


def _metric_line(label: str, payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not _has_value(value):
        return None
    return f"- {label}: {value}"


def _html_metric_item(label: str, payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not _has_value(value):
        return None
    return f"<li>{html.escape(label)}: {html.escape(str(value))}</li>"


def _append_section(lines: list[str], title: str, entries: list[str | None]) -> None:
    present_entries = [entry for entry in entries if entry]
    if not present_entries:
        return
    lines.extend(["", f"## {title}", *present_entries])


def _html_section(title: str, items: list[str | None]) -> str:
    present_items = [item for item in items if item]
    if not present_items:
        return ""
    return f'<h2>{html.escape(title)}</h2><ul>{"".join(present_items)}</ul>'


def generate_runtime_report() -> dict[str, Any]:
    config = load_config()
    created_at = _utc_now_iso()

    raw_summary = load_pipeline_artifact("raw_summary", config=config, default={}) or {}
    test_metrics = (
        load_pipeline_artifact("test_metrics", config=config, default={}) or {}
    )
    validation_metrics = (
        load_pipeline_artifact("validation_metrics", config=config, default={}) or {}
    )
    runtime_summary = (
        load_pipeline_artifact("pipeline_runtime_summary", config=config, default={})
        or {}
    )
    registry_status = (
        load_pipeline_artifact("registry_status", config=config, default={}) or {}
    )
    live_metrics = evaluate_live_feedback()

    registry_data = registry_status if isinstance(registry_status, dict) else {}
    candidate = registry_data.get("candidate") or {}
    previous = registry_data.get("previous_champion") or {}
    title = get_config_value(
        config,
        "reporting.runtime_report_title",
        get_config_value(
            config,
            "reporting.report_title",
            "Galaxy Morphology Training and Monitoring Report",
        ),
    )

    report = {
        "generated_at": created_at,
        "title": title,
        "raw_summary": raw_summary,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "live_metrics": live_metrics,
        "runtime_summary": runtime_summary,
        "registry_status": registry_status,
    }

    md_lines = [
        f"# {title}",
        "",
        f"Generated at: {created_at}",
    ]
    _append_section(
        md_lines,
        "Validation metrics",
        [
            _metric_line("Validation accuracy", validation_metrics, "accuracy"),
            _metric_line("Validation macro F1", validation_metrics, "macro_f1"),
            _metric_line(
                "Validation precision macro", validation_metrics, "precision_macro"
            ),
            _metric_line("Validation recall macro", validation_metrics, "recall_macro"),
        ],
    )
    _append_section(
        md_lines,
        "Offline metrics",
        [
            _metric_line("Accuracy", test_metrics, "accuracy"),
            _metric_line("Macro F1", test_metrics, "macro_f1"),
            _metric_line("Precision macro", test_metrics, "precision_macro"),
            _metric_line("Recall macro", test_metrics, "recall_macro"),
        ],
    )
    _append_section(
        md_lines,
        "Registry decision",
        [
            _metric_line("Candidate version", candidate, "version"),
            _metric_line("Candidate run id", candidate, "run_id"),
            _metric_line(
                f"Candidate metric ({_value(candidate, 'metric_name')})",
                candidate,
                "metric_value",
            ),
            _metric_line(
                "Previous champion version",
                previous if isinstance(previous, dict) else {},
                "version",
            ),
            _metric_line(
                f"Previous champion metric ({_value(previous, 'metric_name') if isinstance(previous, dict) else 'n/a'})",
                previous if isinstance(previous, dict) else {},
                "metric_value",
            ),
            _metric_line("Champion updated", registry_data, "champion_updated"),
            _metric_line(
                "Current champion version", registry_data, "current_champion_version"
            ),
            _metric_line("Serving model URI", registry_data, "serving_model_uri"),
            _metric_line("Decision reason", registry_data, "decision_reason"),
        ],
    )
    _append_section(
        md_lines,
        "Continuous improvement",
        [
            _metric_line("Latest model version", live_metrics, "latest_model_version"),
            _metric_line(
                "Latest-model prediction count", live_metrics, "prediction_count"
            ),
            _metric_line("Feedback count", live_metrics, "feedback_count"),
            _metric_line(
                "Assumed correct without correction feedback",
                live_metrics,
                "assumed_correct_count",
            ),
            _metric_line("Live accuracy", live_metrics, "accuracy"),
            _metric_line("Live macro F1", live_metrics, "macro_f1"),
        ],
    )
    raw_counts = raw_summary.get("actual_counts")
    if raw_counts:
        md_lines.extend(["", "## Raw data counts", str(raw_counts)])
    _append_section(
        md_lines,
        "Training runtime",
        [
            _metric_line(
                "Training duration seconds", runtime_summary, "train_duration_seconds"
            ),
            _metric_line("Epochs completed", runtime_summary, "epochs_completed"),
        ],
    )

    html_sections = [
        _html_section(
            "Validation metrics",
            [
                _html_metric_item(
                    "Validation accuracy", validation_metrics, "accuracy"
                ),
                _html_metric_item(
                    "Validation macro F1", validation_metrics, "macro_f1"
                ),
                _html_metric_item(
                    "Validation precision macro", validation_metrics, "precision_macro"
                ),
                _html_metric_item(
                    "Validation recall macro", validation_metrics, "recall_macro"
                ),
            ],
        ),
        _html_section(
            "Offline metrics",
            [
                _html_metric_item("Accuracy", test_metrics, "accuracy"),
                _html_metric_item("Macro F1", test_metrics, "macro_f1"),
                _html_metric_item("Precision macro", test_metrics, "precision_macro"),
                _html_metric_item("Recall macro", test_metrics, "recall_macro"),
            ],
        ),
        _html_section(
            "Registry decision",
            [
                _html_metric_item("Candidate version", candidate, "version"),
                _html_metric_item("Candidate run id", candidate, "run_id"),
                _html_metric_item(
                    f"Candidate metric ({_value(candidate, 'metric_name')})",
                    candidate,
                    "metric_value",
                ),
                _html_metric_item(
                    "Previous champion version",
                    previous if isinstance(previous, dict) else {},
                    "version",
                ),
                _html_metric_item(
                    f"Previous champion metric ({_value(previous, 'metric_name') if isinstance(previous, dict) else 'n/a'})",
                    previous if isinstance(previous, dict) else {},
                    "metric_value",
                ),
                _html_metric_item(
                    "Champion updated", registry_data, "champion_updated"
                ),
                _html_metric_item(
                    "Current champion version",
                    registry_data,
                    "current_champion_version",
                ),
                _html_metric_item(
                    "Serving model URI", registry_data, "serving_model_uri"
                ),
                _html_metric_item("Decision reason", registry_data, "decision_reason"),
            ],
        ),
        _html_section(
            "Continuous improvement",
            [
                _html_metric_item(
                    "Latest model version", live_metrics, "latest_model_version"
                ),
                _html_metric_item(
                    "Latest-model prediction count", live_metrics, "prediction_count"
                ),
                _html_metric_item("Feedback count", live_metrics, "feedback_count"),
                _html_metric_item(
                    "Assumed correct without correction feedback",
                    live_metrics,
                    "assumed_correct_count",
                ),
                _html_metric_item("Live accuracy", live_metrics, "accuracy"),
                _html_metric_item("Live macro F1", live_metrics, "macro_f1"),
            ],
        ),
    ]
    if raw_counts:
        html_sections.append(
            f"<h2>Raw data counts</h2><pre>{html.escape(str(raw_counts))}</pre>"
        )
    html_sections.append(
        _html_section(
            "Training runtime",
            [
                _html_metric_item(
                    "Training duration seconds",
                    runtime_summary,
                    "train_duration_seconds",
                ),
                _html_metric_item(
                    "Epochs completed", runtime_summary, "epochs_completed"
                ),
            ],
        )
    )

    html_report = f"""
    <html>
      <body>
        <h1>{html.escape(title)}</h1>
        <p><strong>Generated at:</strong> {html.escape(created_at)}</p>
        {''.join(html_sections)}
      </body>
    </html>
    """

    md_path = resolve_path(config, "paths.latest_runtime_report_md_path")
    html_path = resolve_path(config, "paths.latest_runtime_report_html_path")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    LOGGER.info("Runtime report generated at %s", md_path)
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Generate the latest Airflow runtime report."
    ).parse_args()


if __name__ == "__main__":
    parse_args()
    generate_runtime_report()
