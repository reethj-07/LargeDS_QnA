"""Shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    query_type: str
    plan: str
    route: str  # "direct" | "decompose" | "sql_first"
    sql_suggestion: str
    search_query: str
    category_filter: str  # "" or one of REVIEW_CATEGORY_NAMES — metadata filter for retrieval

    # Sub-question decomposition (multi-hop / comparison)
    sub_questions: list[str]
    sub_answers: list[dict[str, Any]]

    retrieved_docs: list[dict[str, Any]]
    sql_results: list[dict[str, Any]]
    retrieval_scores: list[tuple[int, float]]
    answer: str
    critique: str
    confidence: float
    retry_count: int
    agent_trace: Annotated[list[str], operator.add]
    needs_retry: bool
    _context_bundle: str
    _evidence_summary: str
