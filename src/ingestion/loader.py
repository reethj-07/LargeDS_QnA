"""Download and stream-parse McAuley-Lab Amazon-Reviews-2023 JSONL from HuggingFace Hub."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from src.config import HF_DATASET_REPO, HF_JSONL_PREFIX


def download_category_file(filename: str) -> Path:
    """Download a single category JSONL (cached under HuggingFace hub cache)."""
    rel = f"{HF_JSONL_PREFIX}/{filename}"
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=rel,
        repo_type="dataset",
    )
    return Path(path)


def iter_jsonl_lines(path: Path, max_lines: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream-parse JSONL; stop after max_lines if set."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_category_jsonl(
    category_slug: str,
    filename: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Download file if needed and return up to max_rows records with category_slug attached."""
    path = download_category_file(filename)
    rows: list[dict[str, Any]] = []
    for obj in iter_jsonl_lines(path, max_lines=max_rows):
        obj["_category"] = category_slug
        rows.append(obj)
    return rows
