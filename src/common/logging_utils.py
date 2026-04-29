from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import sys
from typing import Any

from src.common.config import get_config_value, load_config

try:
    import psycopg
except Exception:  # noqa: BLE001
    psycopg = None


LOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS service_logs (
    log_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    service TEXT NOT NULL,
    component TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    exception TEXT,
    pathname TEXT,
    lineno INTEGER,
    func_name TEXT,
    process_id INTEGER,
    thread_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_service_logs_created_at
    ON service_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_logs_service_created_at
    ON service_logs (service, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_logs_level_created_at
    ON service_logs (level, created_at DESC);
"""


def _runtime_service_name(default_name: str) -> str:
    return os.getenv("LOG_SERVICE_NAME") or default_name


def _database_url(config: dict[str, Any]) -> str | None:
    env_name = get_config_value(config, "database.app_url_env", "DATABASE_URL")
    return os.getenv(str(env_name)) or get_config_value(config, "database.app_url")


class PostgresLogHandler(logging.Handler):
    def __init__(self, service_name: str, database_url: str | None):
        super().__init__()
        self.service_name = service_name
        self.database_url = database_url
        self.conn = None
        self.schema_ready = False

    def _connect(self):
        if psycopg is None or not self.database_url:
            return None
        if self.conn is None or self.conn.closed:
            self.conn = psycopg.connect(self.database_url, autocommit=True)
        return self.conn

    def _ensure_schema(self, conn) -> None:
        if self.schema_ready:
            return
        with conn.cursor() as cur:
            cur.execute(LOG_SCHEMA_SQL)
        self.schema_ready = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            conn = self._connect()
            if conn is None:
                return
            self._ensure_schema(conn)
            exception = (
                self.formatter.formatException(record.exc_info)
                if record.exc_info and self.formatter
                else None
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO service_logs (
                        created_at,
                        service,
                        component,
                        level,
                        message,
                        exception,
                        pathname,
                        lineno,
                        func_name,
                        process_id,
                        thread_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.fromtimestamp(record.created, tz=timezone.utc),
                        self.service_name,
                        record.name,
                        record.levelname,
                        record.getMessage(),
                        exception,
                        record.pathname,
                        record.lineno,
                        record.funcName,
                        record.process,
                        record.thread,
                    ),
                )
        except Exception:  # noqa: BLE001
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self.conn = None
            self.schema_ready = False

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
        self.conn = None
        super().close()


def configure_logging(service_name: str) -> logging.Logger:
    config = load_config()
    logger = logging.getLogger(service_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    db_handler = PostgresLogHandler(
        service_name=_runtime_service_name(service_name),
        database_url=_database_url(config),
    )
    db_handler.setFormatter(formatter)
    logger.addHandler(db_handler)

    logger.propagate = False
    return logger
