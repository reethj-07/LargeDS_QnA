"""Shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    query_type: str
    plan: str
    sql_suggestion: str
    search_query: str
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
