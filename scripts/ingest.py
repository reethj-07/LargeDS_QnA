#!/usr/bin/env python3
"""CLI: run full ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest Amazon reviews JSONL → DuckDB + FAISS + BM25")
    p.add_argument("--rows-per-category", type=int, default=None, help="Override rows per category (default 30000)")
    p.add_argument("--skip-embed", action="store_true", help="Load data and BM25 only; skip FAISS embedding")
    args = p.parse_args()
    out = run_ingestion(rows_per_category=args.rows_per_category, skip_embed=args.skip_embed)
    print("Done:", out)


if __name__ == "__main__":
    main()
