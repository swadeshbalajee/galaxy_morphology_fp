from src.common.config import get_config_value, load_config


def test_runtime_upload_limits_are_configured():
    config = load_config()

    assert isinstance(get_config_value(config, "runtime.max_upload_mb"), int)
    assert isinstance(get_config_value(config, "runtime.max_zip_upload_mb"), int)
    assert get_config_value(config, "runtime.max_upload_mb") > 0
    assert get_config_value(config, "runtime.max_zip_upload_mb") >= get_config_value(
        config, "runtime.max_upload_mb"
    )


def test_report_paths_are_split_by_owner():
    config = load_config()

    assert get_config_value(config, "paths.latest_report_md_path").startswith(
        "artifacts/reports/"
    )
    assert get_config_value(config, "paths.latest_report_html_path").startswith(
        "artifacts/reports/"
    )
    assert get_config_value(config, "paths.latest_runtime_report_md_path").startswith(
        "artifacts/runtime/"
    )
    assert get_config_value(config, "paths.latest_runtime_report_html_path").startswith(
        "artifacts/runtime/"
    )


def test_airflow_control_thresholds_are_configured():
    config = load_config()

    assert (
        0 < get_config_value(config, "continuous_improvement.accuracy_threshold") <= 1
    )
    assert (
        0 < get_config_value(config, "continuous_improvement.macro_f1_threshold") <= 1
    )
    assert (
        get_config_value(config, "continuous_improvement.min_new_feedback_samples") >= 0
    )
    assert (
        get_config_value(config, "continuous_improvement.min_live_prediction_samples")
        > 0
    )
    assert get_config_value(config, "orchestration.airflow_retries") >= 0
    assert get_config_value(config, "orchestration.airflow_retry_delay_minutes") > 0


def test_dvc_push_policy_is_configured():
    config = load_config()

    assert isinstance(get_config_value(config, "dvc.push_on_success"), bool)
