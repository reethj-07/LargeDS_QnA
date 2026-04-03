"""Vector + BM25 + optional SQL; merge with RRF; fetch doc text from DuckDB."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import BM25_TOP_K, DUCKDB_PATH, FINAL_TOP_K, RRF_K, VECTOR_TOP_K
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
        self.sql_store = sql_store or SqlStore(DUCKDB_PATH)

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
    ) -> list[tuple[int, float]]:
        top_k = top_k or FINAL_TOP_K
        lists: list[list[tuple[int, float]]] = []
        if use_vector:
            lists.append(self.vector_search(query, VECTOR_TOP_K))
        if use_bm25:
            lists.append(self.keyword_search(query, BM25_TOP_K))
        if not lists:
            return []
        merged = reciprocal_rank_fusion(lists, k=RRF_K)
        return merged[:top_k]

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
    ) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
        """Return documents (with text) and id-score pairs."""
        ranked = self.hybrid_search(query, top_k=top_k, use_vector=use_vector, use_bm25=use_bm25)
        ids = [i for i, _ in ranked]
        docs = self.fetch_documents(ids)
        return docs, ranked

    def sql_query(self, sql: str) -> list[dict[str, Any]]:
        return self.sql_store.query_safe(sql)
