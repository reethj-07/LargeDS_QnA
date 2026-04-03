"""LangGraph: Planner → Retriever → Analyst → Critic (+ optional retry)."""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.analyst import run_analyst
from src.agents.critic import run_critic
from src.agents.planner import run_planner
from src.agents.retriever_agent import run_retrieval
from src.agents.state import AgentState
from src.config import MAX_RETRIEVAL_RETRIES
from src.observability.logger import log_event
from src.retrieval.hybrid_retriever import HybridRetriever


def _sql_block(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    try:
        return json.dumps(rows[:200], indent=2, default=str)[:20000]
    except Exception:
        return str(rows)[:20000]


def build_graph(hybrid: HybridRetriever):
    def planner(state: AgentState) -> dict[str, Any]:
        q = state["query"]
        p = run_planner(q)
        trace = [f"Planner: type={p.get('query_type')}"]
        log_event("agent_planner", {"query_type": p.get("query_type")})
        return {
            "query_type": str(p.get("query_type", "multi_hop")),
            "plan": str(p.get("plan", "")),
            "sql_suggestion": str(p.get("sql_suggestion", "")),
            "search_query": str(p.get("search_query", q)),
            "agent_trace": trace,
        }

    def retrieve(state: AgentState) -> dict[str, Any]:
        prev_trace = list(state.get("agent_trace", []))
        trace = list(prev_trace)
        base_sq = state.get("search_query", state["query"])
        rc = int(state.get("retry_count", 0))
        if rc > 0:
            crit = state.get("critique", "") or ""
            search_q = f"{base_sq} {crit}"[:2000]
        else:
            search_q = base_sq

        docs, sql_results, ranked, bundle = run_retrieval(
            hybrid,
            state["query"],
            search_q,
            state.get("sql_suggestion", ""),
            trace,
        )
        # Only append new lines — agent_trace uses operator.add; full list would duplicate prior lines.
        new_trace = trace[len(prev_trace) :]
        ev_parts = [bundle[:8000]]
        if sql_results:
            ev_parts.append(_sql_block(sql_results)[:4000])
        evidence = "\n".join(ev_parts)
        return {
            "retrieved_docs": docs,
            "sql_results": sql_results,
            "retrieval_scores": ranked,
            "_context_bundle": bundle,
            "_evidence_summary": evidence,
            "agent_trace": new_trace,
        }

    def analyst(state: AgentState) -> dict[str, Any]:
        sql_b = _sql_block(state.get("sql_results", []))
        ctx = state.get("_context_bundle", "")
        ans = run_analyst(state["query"], state.get("plan", ""), sql_b, ctx)
        log_event("agent_analyst", {"answer_len": len(ans)})
        return {"answer": ans, "agent_trace": [f"Analyst: answer_len={len(ans)}"]}

    def critic(state: AgentState) -> dict[str, Any]:
        ev = state.get("_evidence_summary", "")
        score, critique, needs = run_critic(state["query"], state.get("answer", ""), ev)
        trace = [f"Critic: score={score:.2f}; model_needs_retry={needs}"]
        log_event("agent_critic", {"score": score, "needs_retry": needs})
        return {
            "critique": critique,
            "confidence": score,
            "needs_retry": bool(needs),
            "agent_trace": trace,
        }

    def route_after_critic(state: AgentState) -> str:
        rc = int(state.get("retry_count", 0))
        if state.get("needs_retry") and rc < MAX_RETRIEVAL_RETRIES:
            return "retry_retrieve"
        return "end"

    def bump_retry(state: AgentState) -> dict[str, Any]:
        rc = int(state.get("retry_count", 0)) + 1
        return {"retry_count": rc, "agent_trace": [f"Retry retrieval pass {rc}"]}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("retrieve", retrieve)
    graph.add_node("bump_retry", bump_retry)
    graph.add_node("analyst", analyst)
    graph.add_node("critic", critic)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry_retrieve": "bump_retry", "end": END},
    )
    graph.add_edge("bump_retry", "retrieve")

    return graph.compile()


def run_agent_pipeline(
    user_query: str,
    hybrid: HybridRetriever | None = None,
) -> dict[str, Any]:
    h = hybrid or HybridRetriever()
    h.load_indices()
    app = build_graph(h)
    init: AgentState = {
        "query": user_query,
        "retry_count": 0,
        "agent_trace": [],
    }
    return app.invoke(init)
