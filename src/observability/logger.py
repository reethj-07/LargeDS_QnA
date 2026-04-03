"""Structured logging for queries, retrieval, and agent decisions."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logger(name: str = "bigdata_qna") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def get_logger(name: str = "bigdata_qna") -> logging.Logger:
    return _setup_logger(name)


def log_event(
    event: str,
    payload: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit one JSON line for observability."""
    log = logger or get_logger()
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **(payload or {}),
    }
    log.info(json.dumps(line, default=str, ensure_ascii=False))
