
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.common.config import load_config, resolve_path


def configure_logging(service_name: str) -> logging.Logger:
    config = load_config()
    log_dir = resolve_path(config, "paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(service_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Emit application logs to stdout so Airflow doesn't render normal logs as task errors.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=Path(log_dir) / f"{service_name}.log",
        maxBytes=5_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
