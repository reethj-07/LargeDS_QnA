"""Orchestrate load → preprocess → DuckDB → embeddings → FAISS + BM25."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CATEGORIES,
    DUCKDB_PATH,
    PROCESSED_DIR,
    ROWS_PER_CATEGORY,
)
from src.embeddings.embedder import Embedder
from src.ingestion.loader import load_category_jsonl
from src.ingestion.preprocessor import preprocess_records
from src.observability.logger import get_logger, log_event
from src.storage.bm25_store import BM25Store
from src.storage.sql_store import SqlStore
from src.storage.vector_store import VectorStore

logger = get_logger(__name__)


def run_ingestion(
    categories: list[tuple[str, str]] | None = None,
    rows_per_category: int | None = None,
    skip_embed: bool = False,
) -> dict[str, Any]:
    categories = categories or DEFAULT_CATEGORIES
    n = rows_per_category or ROWS_PER_CATEGORY

    log_event("ingestion_start", {"categories": [c[0] for c in categories], "rows_per_category": n})

    all_raw: list[dict[str, Any]] = []
    for slug, fname in categories:
        logger.info("Loading %s from %s ...", slug, fname)
        part = load_category_jsonl(slug, fname, max_rows=n)
        all_raw.extend(part)
        log_event("category_loaded", {"category": slug, "rows": len(part)})

    # Stable global ids 0..len-1
    rows = preprocess_records(all_raw, start_id=0)
    if not rows:
        raise RuntimeError("No rows after preprocessing.")

    df = pd.DataFrame(rows)
    parquet_path = PROCESSED_DIR / "reviews.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    log_event("parquet_saved", {"path": str(parquet_path), "rows": len(df)})

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    sql = SqlStore(DUCKDB_PATH)
    sql.init_schema()
    sql.insert_rows(rows)
    sql.close()
    log_event("duckdb_built", {"path": str(DUCKDB_PATH), "rows": len(rows)})

    ids = [r["id"] for r in rows]
    texts = [r["doc_text"] for r in rows]

    bm25 = BM25Store()
    bm25.build(texts, ids)
    bm25.save()
    log_event("bm25_saved", {"path": str(bm25.path)})

    if skip_embed:
        log_event("ingestion_done_skip_embed", {"rows": len(rows)})
        return {"rows": len(rows), "parquet": str(parquet_path), "skipped_embed": True}

    embedder = Embedder()
    logger.info("Embedding %d documents (CPU)...", len(texts))
    emb = embedder.encode_batches(texts, chunk_size=5000, show_progress=True)

    vs = VectorStore()
    vs.build_from_embeddings(emb, ids)
    vs.save()
    log_event("faiss_saved", {"path": str(vs.index_path), "ntotal": vs.ntotal})

    log_event("ingestion_complete", {"rows": len(rows)})
    return {
        "rows": len(rows),
        "parquet": str(parquet_path),
        "duckdb": str(DUCKDB_PATH),
        "faiss": str(vs.index_path),
        "bm25": str(bm25.path),
    }
