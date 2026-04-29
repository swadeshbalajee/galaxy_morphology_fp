from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage

import pytest

from src.common import artifact_store, email_utils, postgres
from src.common.config import get_config_value, project_root, resolve_path
from src.common.io_utils import (
    class_dirs,
    ensure_dir,
    list_image_files,
    read_json,
    reset_dir,
    write_json,
)


def test_get_config_value_reads_nested_values_and_defaults():
    config = {"runtime": {"max_upload_mb": 20}, "empty": None}

    assert get_config_value(config, "runtime.max_upload_mb") == 20
    assert get_config_value(config, "runtime.missing", default="fallback") == "fallback"
    assert get_config_value(config, "empty.value", default="fallback") == "fallback"


def test_project_root_and_resolve_path_use_config_metadata(tmp_path):
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "paths": {
            "relative": "artifacts/report.json",
            "absolute": str(tmp_path / "external" / "file.json"),
        },
    }

    assert project_root(config) == tmp_path
    assert resolve_path(config, "paths.relative") == tmp_path / "artifacts/report.json"
    assert resolve_path(config, "paths.absolute") == tmp_path / "external" / "file.json"


def test_resolve_path_raises_for_missing_config_path(tmp_path):
    config = {"_meta": {"project_root": str(tmp_path)}, "paths": {}}

    with pytest.raises(KeyError, match="Missing config path"):
        resolve_path(config, "paths.unknown")


def test_directory_helpers_create_reset_and_list_class_dirs(tmp_path):
    work_dir = ensure_dir(tmp_path / "work" / "nested")
    old_file = work_dir / "old.txt"
    old_file.write_text("stale", encoding="utf-8")

    reset_dir(tmp_path / "work")

    assert (tmp_path / "work").is_dir()
    assert not old_file.exists()

    (tmp_path / "dataset" / "spiral").mkdir(parents=True)
    (tmp_path / "dataset" / "elliptical").mkdir()
    (tmp_path / "dataset" / "README.txt").write_text("not a class", encoding="utf-8")

    assert [p.name for p in class_dirs(tmp_path / "dataset")] == [
        "elliptical",
        "spiral",
    ]
    assert class_dirs(tmp_path / "missing") == []


def test_list_image_files_filters_case_insensitively_and_sorts(tmp_path):
    (tmp_path / "b.PNG").write_bytes(b"image")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.jpg").write_bytes(b"image")
    (tmp_path / "nested" / "c.txt").write_text("skip", encoding="utf-8")

    image_names = [path.name for path in list_image_files(tmp_path)]

    assert image_names == ["b.PNG", "a.jpg"]
    assert list_image_files(tmp_path, allowed_suffixes={".txt"}) == [
        tmp_path / "nested" / "c.txt"
    ]


def test_json_helpers_round_trip_and_default_missing_file(tmp_path):
    payload = {"class": "spiral", "confidence": 0.92}

    written_path = write_json(tmp_path / "nested" / "payload.json", payload)

    assert written_path.exists()
    assert read_json(written_path) == payload
    assert read_json(tmp_path / "missing.json", default={"missing": True}) == {
        "missing": True
    }


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2025-01-02T03:04:05Z", datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)),
        (
            datetime(2025, 1, 2, 3, 4, 5),
            datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        ),
    ],
)
def test_normalize_recorded_at_returns_utc_datetimes(raw_value, expected):
    assert artifact_store._normalize_recorded_at(raw_value) == expected


def test_artifact_spec_rejects_unknown_keys():
    with pytest.raises(KeyError, match="Unknown pipeline artifact key"):
        artifact_store._artifact_spec("not_a_real_artifact")


def test_normalize_control_plane_state_defaults_empty_rows():
    assert postgres._normalize_control_plane_state(None) == {
        "last_feedback_snapshot_count": 0,
        "last_feedback_snapshot_at": None,
        "last_report_sent_at": None,
        "last_pipeline_config_fingerprint": None,
        "last_pipeline_config_updated_at": None,
        "updated_at": None,
    }


def test_normalize_control_plane_state_serializes_datetimes():
    row = {
        "last_feedback_snapshot_count": "7",
        "last_feedback_snapshot_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "last_report_sent_at": "already-serialized",
        "last_pipeline_config_fingerprint": "abc123",
        "last_pipeline_config_updated_at": None,
        "updated_at": datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    }

    assert postgres._normalize_control_plane_state(row) == {
        "last_feedback_snapshot_count": 7,
        "last_feedback_snapshot_at": "2025-01-01T00:00:00Z",
        "last_report_sent_at": "already-serialized",
        "last_pipeline_config_fingerprint": "abc123",
        "last_pipeline_config_updated_at": None,
        "updated_at": "2025-01-02T03:04:05Z",
    }


def test_send_email_report_returns_early_when_disabled(monkeypatch):
    def fail_if_used(*args, **kwargs):
        raise AssertionError("SMTP should not be created when email is disabled")

    monkeypatch.setattr(email_utils.smtplib, "SMTP", fail_if_used)

    email_utils.send_email_report(
        {"email": {"enabled": False}},
        subject="Daily report",
        body_text="Plain text",
    )


def test_send_email_report_builds_message_with_html_and_attachment(
    tmp_path, monkeypatch
):
    sent_messages: list[EmailMessage] = []
    attachment = tmp_path / "metrics.txt"
    attachment.write_text("accuracy=0.91", encoding="utf-8")

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.started_tls = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def starttls(self):
            self.started_tls = True

        def login(self, username, password):
            assert username == "user@example.com"
            assert password == "secret"

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setattr(email_utils.smtplib, "SMTP", FakeSMTP)

    email_utils.send_email_report(
        {
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "smtp_port": 2525,
                "sender": "reports@example.com",
                "recipients": ["ops@example.com", "ml@example.com"],
                "username_env": "SMTP_USER",
                "password_env": "SMTP_PASSWORD",
                "use_tls": True,
            }
        },
        subject="Daily report",
        body_text="Plain text",
        body_html="<p>HTML</p>",
        attachments=[attachment, tmp_path / "missing.txt"],
    )

    assert len(sent_messages) == 1
    message = sent_messages[0]
    assert message["Subject"] == "Daily report"
    assert message["From"] == "reports@example.com"
    assert message["To"] == "ops@example.com, ml@example.com"

    attachments = list(message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "metrics.txt"
    assert attachments[0].get_content() == "accuracy=0.91"


def test_send_email_report_requires_minimum_config():
    with pytest.raises(ValueError, match="Email config is incomplete"):
        email_utils.send_email_report(
            {"email": {"enabled": True}},
            subject="Daily report",
            body_text="Plain text",
        )
