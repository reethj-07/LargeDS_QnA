"""LangGraph: Planner -> Router -> [Decompose | SQL-first | Direct] -> Analyst -> Critic."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.analyst import run_analyst
from src.agents.critic import run_critic
from src.agents.decomposer import run_decomposer
from src.agents.planner import run_planner
from src.agents.retriever_agent import run_retrieval
from src.agents.state import AgentState
from src.agents.synthesizer import merge_sub_results
from src.config import (
    FINAL_TOP_K,
    LOG_QUERY_PREVIEWS,
    MAX_RETRIEVAL_RETRIES,
    PIPELINE_TIMEOUT_S,
    USE_SHARED_HYBRID,
    indices_artifacts_ready,
)
from src.observability.logger import log_event, query_preview_log_payload
from src.observability.tracing import get_trace_id, new_trace_id, set_trace_id, span
from src.retrieval.hybrid_retriever import HybridRetriever

_shared_hybrid: HybridRetriever | None = None
_shared_lock = threading.Lock()
# Serialize graph.invoke so timeout workers and the main path never hit FAISS concurrently.
_pipeline_invoke_lock = threading.RLock()

# Bound analyst prompt size (Groq 8b free tier: ~6k input tokens/request on many accounts).
_MAX_ANALYST_CONTEXT_CHARS = 5_500
_MAX_EVIDENCE_FOR_CRITIC = 6_000


def _shrink_cell(val: Any, max_str: int = 240) -> Any:
    if isinstance(val, str) and len(val) > max_str:
        return val[:max_str] + "…"
    return val


def _sql_block(rows: list[dict[str, Any]]) -> str:
    """Compact SQL rows for LLM prompts (avoid 413 / token overflow on wide SELECT *)."""
    if not rows:
        return "(none)"
    slim: list[dict[str, Any]] = []
    for r in rows[:24]:
        if not isinstance(r, dict):
            slim.append({"_value": _shrink_cell(r)})
            continue
        slim.append({k: _shrink_cell(v) for k, v in r.items()})
    try:
        s = json.dumps(slim, indent=2, default=str)
        if len(s) > 4500:
            s = s[:4500] + "\n… (truncated)"
        return s
    except Exception:
        return str(slim)[:4500]


def build_graph(hybrid: HybridRetriever):
    # ── Planner ──────────────────────────────────────────────────
    def planner(state: AgentState) -> dict[str, Any]:
        with span("planner", {"query": state["query"][:200]}):
            q = state["query"]
            p = run_planner(q)
            route = str(p.get("route", "direct"))
            trace = [f"Planner: type={p.get('query_type')}, route={route}"]
            log_event("agent_planner", {
                "query_type": p.get("query_type"),
                "route": route,
                "category_filter": (p.get("category_filter") or "") or None,
            })
            return {
                "query_type": str(p.get("query_type", "multi_hop")),
                "route": route,
                "plan": str(p.get("plan", "")),
                "sql_suggestion": str(p.get("sql_suggestion", "")),
                "search_query": str(p.get("search_query", q)),
                "category_filter": str(p.get("category_filter", "")),
                "agent_trace": trace,
            }

    # ── Router (conditional edge function) ───────────────────────
    def route_after_planner(state: AgentState) -> str:
        r = state.get("route", "direct")
        if r == "decompose":
            return "decompose"
        if r == "sql_first":
            return "sql_first"
        return "direct"

    # ── SQL-first node (aggregation path) ────────────────────────
    def sql_first_retrieve(state: AgentState) -> dict[str, Any]:
        trace: list[str] = []
        sql_results: list[dict[str, Any]] = []
        sql = state.get("sql_suggestion", "")
        if sql and sql.strip():
            try:
                sql_results = hybrid.sql_query(sql.strip())
                trace.append(f"SQL-first OK: {len(sql_results)} rows")
                log_event("retrieval_sql_first", {"rows": len(sql_results)})
            except Exception as e:
                trace.append(f"SQL-first error: {e}")
                log_event("retrieval_sql_first_error", {"error": str(e)})

        sq = state.get("search_query", state["query"])
        cf = (state.get("category_filter") or "").strip() or None
        from src.agents.tools import retrieve_bundle
        docs, ranked, bundle = retrieve_bundle(
            hybrid, sq, top_k=FINAL_TOP_K, category_filter=cf
        )
        trace.append(f"Supplementary retrieval: {len(docs)} docs")

        ev_parts = [bundle[:5000]]
        if sql_results:
            ev_parts.append(_sql_block(sql_results)[:3500])

        return {
            "retrieved_docs": docs,
            "sql_results": sql_results,
            "retrieval_scores": ranked,
            "_context_bundle": bundle[:_MAX_ANALYST_CONTEXT_CHARS],
            "_evidence_summary": "\n".join(ev_parts)[:_MAX_EVIDENCE_FOR_CRITIC],
            "agent_trace": trace,
        }

    # ── Decompose node (multi-hop / comparison) ─────────────────
    def decompose(state: AgentState) -> dict[str, Any]:
        with span("decomposer", {"query": state["query"][:200]}):
            subs = run_decomposer(state["query"], state.get("plan", ""))
            questions = [s["question"] for s in subs]
            log_event("agent_decomposer", {
                "n_sub_questions": len(questions),
                "sub_questions": questions,
            })
            return {
                "sub_questions": questions,
                "agent_trace": [
                    f"Decomposer: {len(questions)} sub-questions",
                    *[f"  SQ{i+1}: {q}" for i, q in enumerate(questions)],
                ],
            }

    # ── Sub-retrieval: retrieve for each sub-question, then merge
    def sub_retrieve(state: AgentState) -> dict[str, Any]:
        subs = state.get("sub_questions") or [state["query"]]
        sql = state.get("sql_suggestion", "")
        cf = (state.get("category_filter") or "").strip() or None
        sub_answers: list[dict[str, Any]] = []
        trace: list[str] = []

        for i, sq in enumerate(subs):
            t: list[str] = []
            docs, sql_results, ranked, bundle = run_retrieval(
                hybrid,
                sq,
                sq,
                sql if i == 0 else "",  # only run SQL on first sub-question
                t,
                category_filter=cf,
            )
            sub_answers.append({
                "question": sq,
                "docs": docs,
                "sql_results": sql_results,
                "ranked": ranked,
                "bundle": bundle,
            })
            trace.extend([f"SQ{i+1}: " + line for line in t])

        merged_docs, merged_sql, merged_ranked, unified = merge_sub_results(sub_answers)
        ucap = unified[:_MAX_ANALYST_CONTEXT_CHARS]
        ev_parts = [ucap]
        if merged_sql:
            ev_parts.append(_sql_block(merged_sql)[:3500])

        return {
            "sub_answers": sub_answers,
            "retrieved_docs": merged_docs,
            "sql_results": merged_sql,
            "retrieval_scores": merged_ranked,
            "_context_bundle": ucap,
            "_evidence_summary": "\n".join(ev_parts)[:_MAX_EVIDENCE_FOR_CRITIC],
            "agent_trace": trace,
        }

    # ── Direct retrieval (simple factual path) ──────────────────
    def retrieve(state: AgentState) -> dict[str, Any]:
        trace: list[str] = []
        base_sq = state.get("search_query", state["query"])
        rc = int(state.get("retry_count", 0))
        if rc > 0:
            crit = state.get("critique", "") or ""
            search_q = f"{base_sq} {crit}"[:2000]
        else:
            search_q = base_sq

        cf = (state.get("category_filter") or "").strip() or None
        docs, sql_results, ranked, bundle = run_retrieval(
            hybrid,
            state["query"],
            search_q,
            state.get("sql_suggestion", ""),
            trace,
            category_filter=cf,
        )
        bc = bundle[:_MAX_ANALYST_CONTEXT_CHARS]
        ev_parts = [bc[:5000]]
        if sql_results:
            ev_parts.append(_sql_block(sql_results)[:3500])

        return {
            "retrieved_docs": docs,
            "sql_results": sql_results,
            "retrieval_scores": ranked,
            "_context_bundle": bc,
            "_evidence_summary": "\n".join(ev_parts)[:_MAX_EVIDENCE_FOR_CRITIC],
            "agent_trace": trace,
        }

    # ── Analyst ──────────────────────────────────────────────────
    def analyst(state: AgentState) -> dict[str, Any]:
        with span("analyst", {"query": state["query"][:200]}):
            sql_b = _sql_block(state.get("sql_results", []))
            ctx = (state.get("_context_bundle") or "")[:_MAX_ANALYST_CONTEXT_CHARS]
            plan_s = (state.get("plan") or "")[:2200]
            ans = run_analyst(state["query"], plan_s, sql_b, ctx)
            log_event("agent_analyst", {"answer_len": len(ans)})
            return {"answer": ans, "agent_trace": [f"Analyst: answer_len={len(ans)}"]}

    # ── Critic ───────────────────────────────────────────────────
    def critic(state: AgentState) -> dict[str, Any]:
        with span("critic", {"query": state["query"][:200]}):
            ev = (state.get("_evidence_summary") or "")[:_MAX_EVIDENCE_FOR_CRITIC]
            score, critique, needs = run_critic(state["query"], state.get("answer", ""), ev)
            trace = [f"Critic: score={score:.2f}; needs_retry={needs}"]
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
            return "retry"
        return "end"

    def bump_retry(state: AgentState) -> dict[str, Any]:
        rc = int(state.get("retry_count", 0)) + 1
        return {"retry_count": rc, "agent_trace": [f"Retry retrieval pass {rc}"]}

    # ── Build graph ──────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("decompose", decompose)
    graph.add_node("sub_retrieve", sub_retrieve)
    graph.add_node("sql_first_retrieve", sql_first_retrieve)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyst", analyst)
    graph.add_node("critic", critic)
    graph.add_node("bump_retry", bump_retry)

    graph.set_entry_point("planner")

    # Router: planner -> one of three paths
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "decompose": "decompose",
            "sql_first": "sql_first_retrieve",
            "direct": "retrieve",
        },
    )

    # Decompose path: decompose -> sub_retrieve -> analyst
    graph.add_edge("decompose", "sub_retrieve")
    graph.add_edge("sub_retrieve", "analyst")

    # SQL-first path -> analyst
    graph.add_edge("sql_first_retrieve", "analyst")

    # Direct path -> analyst
    graph.add_edge("retrieve", "analyst")

    # All paths converge at analyst -> critic
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "bump_retry", "end": END},
    )
    graph.add_edge("bump_retry", "retrieve")

    return graph.compile()


def get_shared_hybrid_retriever() -> HybridRetriever:
    """Lazily load indices once per process (used by UI/API when ``USE_SHARED_HYBRID``)."""
    global _shared_hybrid
    with _shared_lock:
        if _shared_hybrid is None:
            _shared_hybrid = HybridRetriever()
            _shared_hybrid.load_indices()
        return _shared_hybrid


def _invoke_graph_app(app: Any, init: AgentState) -> dict[str, Any]:
    with _pipeline_invoke_lock:
        return app.invoke(init)


def _timeout_response(user_query: str, timeout_s: float) -> dict[str, Any]:
    msg = (
        f"The query exceeded the server time limit ({timeout_s:g}s) and was stopped. "
        "Try a shorter question, disable cross-encoder, or raise PIPELINE_TIMEOUT_S."
    )
    return {
        "query": user_query,
        "answer": msg,
        "error": "pipeline_timeout",
        "query_type": "unknown",
        "route": "direct",
        "plan": "",
        "sql_suggestion": "",
        "search_query": user_query,
        "category_filter": "",
        "sub_questions": [],
        "retrieved_docs": [],
        "sql_results": [],
        "retrieval_scores": [],
        "sub_answers": [],
        "critique": "",
        "confidence": 0.0,
        "needs_retry": False,
        "retry_count": 0,
        "agent_trace": [f"Error: pipeline_timeout (limit {timeout_s:g}s)"],
        "trace_id": get_trace_id(),
    }


def reset_shared_hybrid_retriever() -> None:
    """Release the shared retriever (tests or after rebuilding indices in-process)."""
    global _shared_hybrid
    with _shared_lock:
        if _shared_hybrid is not None:
            try:
                _shared_hybrid.sql_store.close()
            except Exception:
                pass
            _shared_hybrid = None


def run_agent_pipeline(
    user_query: str,
    hybrid: HybridRetriever | None = None,
    *,
    trace_id: str | None = None,
    pipeline_timeout_s: float | None = None,
) -> dict[str, Any]:
    if trace_id:
        set_trace_id(trace_id)
    else:
        new_trace_id()
    log_event(
        "pipeline_query",
        {
            **query_preview_log_payload(user_query, enabled=LOG_QUERY_PREVIEWS),
        },
    )
    if not indices_artifacts_ready():
        msg = (
            "Search indices are not available on this server. "
            "Run `python scripts/ingest.py` locally (or on the host) to build "
            "`data/indices/` (FAISS, BM25, DuckDB), then restart the app."
        )
        return {
            "query": user_query,
            "answer": msg,
            "error": "indices_missing",
            "query_type": "unknown",
            "route": "direct",
            "plan": "",
            "sql_suggestion": "",
            "search_query": user_query,
            "category_filter": "",
            "sub_questions": [],
            "retrieved_docs": [],
            "sql_results": [],
            "retrieval_scores": [],
            "sub_answers": [],
            "critique": "",
            "confidence": 0.0,
            "needs_retry": False,
            "retry_count": 0,
            "agent_trace": ["Error: indices_missing (run scripts/ingest.py)"],
            "trace_id": get_trace_id(),
        }

    if hybrid is not None:
        h = hybrid
    elif USE_SHARED_HYBRID:
        h = get_shared_hybrid_retriever()
    else:
        h = HybridRetriever()
        h.load_indices()
    app = build_graph(h)
    init: AgentState = {
        "query": user_query,
        "retry_count": 0,
        "agent_trace": [],
    }
    timeout_s = float(PIPELINE_TIMEOUT_S if pipeline_timeout_s is None else pipeline_timeout_s)
    pool: ThreadPoolExecutor | None = None
    try:
        if timeout_s > 0:
            pool = ThreadPoolExecutor(max_workers=1)
            fut = pool.submit(_invoke_graph_app, app, init)
            try:
                raw = fut.result(timeout=timeout_s)
            except FuturesTimeoutError:
                log_event(
                    "pipeline_invoke_timeout",
                    {"timeout_s": timeout_s, "note": "worker may still finish in background"},
                )
                return _timeout_response(user_query, timeout_s)
        else:
            raw = _invoke_graph_app(app, init)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)

    out: dict[str, Any] = dict(raw)
    out["trace_id"] = get_trace_id()
    return out
