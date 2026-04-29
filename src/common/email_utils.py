from __future__ import annotations

import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from src.common.config import get_config_value
from src.common.logging_utils import configure_logging

LOGGER = configure_logging("email")


def send_email_report(
    config: dict,
    subject: str,
    body_text: str,
    body_html: str | None = None,
    attachments: list[str | Path] | None = None,
) -> None:
    if not get_config_value(config, "email.enabled", False):
        LOGGER.info("Email delivery disabled in config; skipping report email.")
        return

    host = get_config_value(config, "email.smtp_host")
    port = int(get_config_value(config, "email.smtp_port", 587))
    sender = get_config_value(config, "email.sender")
    recipients = get_config_value(config, "email.recipients", [])
    username_env = get_config_value(config, "email.username_env")
    password_env = get_config_value(config, "email.password_env")
    username = os.getenv(username_env, "") if username_env else ""
    password = os.getenv(password_env, "") if password_env else ""

    if not host or not sender or not recipients:
        raise ValueError(
            "Email config is incomplete. Set smtp host, sender, and recipients."
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body_text)
    if body_html:
        message.add_alternative(body_html, subtype="html")

    for attachment in attachments or []:
        attachment_path = Path(attachment)
        if not attachment_path.exists():
            LOGGER.warning("Attachment missing, skipping: %s", attachment_path)
            continue
        ctype, _ = mimetypes.guess_type(str(attachment_path))
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        message.add_attachment(
            attachment_path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=attachment_path.name,
        )

    use_tls = bool(get_config_value(config, "email.use_tls", True))
    LOGGER.info("Sending email report to %s via %s:%s", recipients, host, port)
    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        if username:
            server.login(username, password)
        server.send_message(message)
    LOGGER.info("Email report sent successfully.")
