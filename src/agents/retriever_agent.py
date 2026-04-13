"""Retriever: hybrid search + optional SQL from planner."""

from __future__ import annotations

from typing import Any

from src.agents.tools import retrieve_bundle
from src.config import FINAL_TOP_K
from src.observability.logger import log_event
from src.retrieval.hybrid_retriever import HybridRetriever


def run_retrieval(
    hybrid: HybridRetriever,
    query: str,
    search_query: str,
    sql_suggestion: str,
    trace: list[str],
    *,
    category_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, float]], str]:
    sql_results: list[dict[str, Any]] = []
    if sql_suggestion and sql_suggestion.strip():
        try:
            sql_results = hybrid.sql_query(sql_suggestion.strip())
            trace.append(f"SQL OK: returned {len(sql_results)} rows")
            log_event("retrieval_sql", {"rows": len(sql_results)})
        except Exception as e:
            trace.append(f"SQL skipped: {e}")
            log_event("retrieval_sql_error", {"error": str(e)})

    sq = search_query.strip() or query
    if category_filter:
        trace.append(f"Category filter (retrieval): {category_filter}")
    docs, ranked, bundle = retrieve_bundle(
        hybrid, sq, top_k=FINAL_TOP_K, category_filter=category_filter
    )
    trace.append(f"Hybrid retrieval: {len(docs)} docs")
    doc_ids = [d.get("id") for d in docs[:30]]
    scores = [float(s) for _, s in ranked[:30]]
    log_event(
        "retrieval_hybrid",
        {
            "n_docs": len(docs),
            "search_query_len": len(sq),
            "retrieved_doc_ids": doc_ids,
            "retrieval_scores_sample": scores[:10],
        },
    )
    return docs, sql_results, ranked, bundle
