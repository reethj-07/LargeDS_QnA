"""Vector + BM25 + optional SQL; merge with RRF; fetch doc text from DuckDB."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import (
    BM25_TOP_K,
    DUCKDB_PATH,
    DUCKDB_READ_ONLY,
    FINAL_TOP_K,
    RERANK_POOL,
    REVIEW_CATEGORY_NAMES,
    RRF_K,
    USE_CROSS_ENCODER,
    VECTOR_TOP_K,
)
from src.embeddings.embedder import Embedder
from src.retrieval.reranker import reciprocal_rank_fusion
from src.storage.bm25_store import BM25Store
from src.storage.sql_store import SqlStore
from src.storage.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        sql_store: SqlStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.sql_store = sql_store or SqlStore(DUCKDB_PATH, read_only=DUCKDB_READ_ONLY)

    def load_indices(self) -> None:
        self.vector_store.load()
        self.bm25_store.load()
        self.sql_store.connect()

    def vector_search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        top_k = top_k or VECTOR_TOP_K
        q_emb = self.embedder.encode([query], show_progress=False)[0]
        return self.vector_store.search(np.asarray(q_emb, dtype=np.float32), top_k)

    def keyword_search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        return self.bm25_store.search(query, top_k or BM25_TOP_K)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        *,
        use_cross_encoder: bool | None = None,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Hybrid search with optional post-filter by allowed_ids (metadata filtering).

        When *allowed_ids* is given, only documents whose id is in that set are
        retained.  Upstream candidates are over-fetched (3× top_k) so that the
        final list still reaches *top_k* after filtering.
        """
        top_k = top_k or FINAL_TOP_K
        fetch_k = VECTOR_TOP_K if allowed_ids is None else max(VECTOR_TOP_K, top_k * 3)
        bm25_k = BM25_TOP_K if allowed_ids is None else max(BM25_TOP_K, top_k * 3)

        lists: list[list[tuple[int, float]]] = []
        if use_vector:
            lists.append(self.vector_search(query, fetch_k))
        if use_bm25:
            lists.append(self.keyword_search(query, bm25_k))
        if not lists:
            return []
        merged = reciprocal_rank_fusion(lists, k=RRF_K)

        if allowed_ids is not None:
            merged = [(doc_id, s) for doc_id, s in merged if doc_id in allowed_ids]

        ce = USE_CROSS_ENCODER if use_cross_encoder is None else use_cross_encoder
        if not ce:
            return merged[:top_k]
        pool = merged[: min(RERANK_POOL, len(merged))]
        ids = [i for i, _ in pool]
        if not ids:
            return []
        docs = self.fetch_documents(ids)
        self._last_fetched = {int(d["id"]): d for d in docs}
        aligned_ids: list[int] = []
        texts: list[str] = []
        for i in ids:
            if i not in self._last_fetched:
                continue
            row = self._last_fetched[i]
            texts.append(str(row.get("doc_text") or row.get("text") or ""))
            aligned_ids.append(i)
        if not aligned_ids:
            return merged[:top_k]
        from src.retrieval.reranker import cross_encoder_rerank

        return cross_encoder_rerank(query, aligned_ids, texts, top_k=top_k)

    def fetch_documents(self, doc_ids: list[int]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        con = self.sql_store.connect()
        safe_ids = ",".join(str(int(i)) for i in doc_ids)
        sql = f"SELECT * FROM reviews WHERE id IN ({safe_ids})"
        rows = con.execute(sql).fetchdf().to_dict(orient="records")
        by_id = {int(r["id"]): r for r in rows}
        return [by_id[i] for i in doc_ids if i in by_id]

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        *,
        allowed_ids: set[int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
        """Return documents (with text) and id-score pairs."""
        ranked = self.hybrid_search(
            query, top_k=top_k, use_vector=use_vector, use_bm25=use_bm25,
            allowed_ids=allowed_ids,
        )
        ids = [i for i, _ in ranked]
        cache = getattr(self, "_last_fetched", {})
        if cache and all(i in cache for i in ids):
            docs = [cache[i] for i in ids]
        else:
            docs = self.fetch_documents(ids)
        return docs, ranked

    def sql_query(self, sql: str) -> list[dict[str, Any]]:
        return self.sql_store.query_safe(sql)

    def get_ids_for_category(self, category: str) -> set[int]:
        """Return all review ids that belong to *category* (for metadata filtering)."""
        if category not in REVIEW_CATEGORY_NAMES:
            return set()
        con = self.sql_store.connect()
        rows = con.execute(
            "SELECT id FROM reviews WHERE category = ?",
            [category],
        ).fetchall()
        return {int(r[0]) for r in rows}
