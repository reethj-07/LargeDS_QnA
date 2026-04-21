"""Structured logging with trace-id correlation for queries, retrieval, and agent decisions."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
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


def query_preview_log_payload(query: str, *, enabled: bool, max_len: int = 400) -> dict[str, Any]:
    """Fields for logging a user query (honour ``LOG_QUERY_PREVIEWS``)."""
    if enabled:
        return {"query_preview": query[:max_len], "char_len": len(query)}
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return {"query_sha256_12": digest, "char_len": len(query)}


def log_event(
    event: str,
    payload: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit one JSON line for observability, enriched with trace_id."""
    from src.observability.tracing import get_trace_id

    log = logger or get_logger()
    line: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": get_trace_id(),
        "event": event,
        **(payload or {}),
    }
    log.info(json.dumps(line, default=str, ensure_ascii=False))
