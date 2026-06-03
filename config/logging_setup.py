"""Logging configuration: console + rotating file handlers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LoggingConfig

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure the root logger with console and rotating file handlers.

    Idempotent: repeated calls won't stack duplicate handlers.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir / config.log_file

    level = getattr(logging, config.level, logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers so re-running stays clean.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("Logging initialized (level=%s, file=%s)", config.level, log_path)
    return root


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor for a module-scoped logger."""
    return logging.getLogger(name)
