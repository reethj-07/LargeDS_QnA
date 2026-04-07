"""Clean and normalize review records for storage and embedding."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    s = _WS.sub(" ", s).strip()
    return s


def parse_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def parse_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def preprocess_records(
    raw_rows: list[dict[str, Any]],
    start_id: int = 0,
) -> list[dict[str, Any]]:
    """
    Normalize fields into a flat schema for DuckDB + embedding.

    Output keys: id, asin, category, rating, title, text, doc_text,
    timestamp, helpful_vote, verified_purchase
    """
    out: list[dict[str, Any]] = []
    for i, r in enumerate(raw_rows):
        title = clean_text(r.get("title"))
        body = clean_text(r.get("text"))
        content = f"{title} {body}".strip()
        if not content:
            continue
        cat = str(r.get("_category") or "unknown")
        cat_label = cat.replace("_", " ")
        doc_text = f"[{cat_label}] {content}"
        vp = parse_bool(r.get("verified_purchase"))
        out.append(
            {
                "id": start_id + i,
                "asin": str(r.get("asin") or r.get("parent_asin") or "")[:32],
                "category": cat,
                "rating": parse_float(r.get("rating")),
                "title": title,
                "text": body,
                "doc_text": doc_text,
                "timestamp": parse_int(r.get("timestamp")),
                "helpful_vote": parse_int(r.get("helpful_vote")) or 0,
                "verified_purchase": False if vp is None else vp,
            }
        )
    return out
