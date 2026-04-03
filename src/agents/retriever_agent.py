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
    docs, ranked, bundle = retrieve_bundle(hybrid, sq, top_k=FINAL_TOP_K)
    trace.append(f"Hybrid retrieval: {len(docs)} docs")
    log_event("retrieval_hybrid", {"n_docs": len(docs)})
    return docs, sql_results, ranked, bundle
