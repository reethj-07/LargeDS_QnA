"""Gradio UI: question -> answer + trace + retrieved docs table."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr

from src.agents.graph import run_agent_pipeline
from src.observability.logger import log_event

EXAMPLES = [
    "What is the average rating by category?",
    "Compare average ratings across categories and name one theme from reviews in the lowest-rated category.",
    "Which ASIN has the most reviews in the dataset?",
    "Find highly rated reviews that still mention product defects.",
    "Are verified_purchase reviews higher rated on average than non-verified?",
    "What are common themes in 1-star Appliances reviews?",
    "Contrast what 5-star vs 1-star reviewers say about durability.",
]

CSS = """
.answer-box textarea { font-size: 15px !important; }
.trace-box textarea { font-size: 12px !important; font-family: monospace !important; }
.badge { font-weight: bold; }
footer { display: none !important; }
"""


def run_once(message: str) -> tuple[str, str, str, str, list[list]]:
    """Returns (answer, trace_text, query_type_badge, sub_questions_text, doc_table)."""
    if not (message or "").strip():
        return "", "", "", "", []

    log_event("ui_query", {"q_len": len(message)})
    t0 = time.time()
    out: dict[str, Any] = run_agent_pipeline(message.strip())
    elapsed = time.time() - t0

    # -- Query type badge --
    qt = out.get("query_type", "unknown")
    route = out.get("route", "direct")
    badge = f"Type: {qt}  |  Route: {route}  |  {elapsed:.1f}s"

    # -- Sub-questions --
    subs = out.get("sub_questions") or []
    sub_text = ""
    if subs:
        sub_text = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(subs))

    # -- Trace --
    trace_lines = list(out.get("agent_trace", []))
    trace_lines.append(f"confidence={out.get('confidence')}")
    critique = (out.get("critique") or "")[:600]
    trace_lines.append(f"critique={critique}")
    meta = {
        "query_type": qt,
        "route": route,
        "sub_questions": len(subs),
        "sql_rows": len(out.get("sql_results") or []),
        "retrieved_docs": len(out.get("retrieved_docs") or []),
        "latency_s": round(elapsed, 2),
    }
    trace_text = "\n".join(trace_lines) + "\n\n" + json.dumps(meta, indent=2, default=str)

    # -- Doc table --
    docs = out.get("retrieved_docs") or []
    scores_map = {i: s for i, s in (out.get("retrieval_scores") or [])}
    table_rows = []
    for d in docs[:15]:
        did = d.get("id")
        table_rows.append([
            did,
            d.get("category", ""),
            d.get("rating", ""),
            (d.get("title") or "")[:80],
            (d.get("text") or d.get("doc_text") or "")[:120],
            f"{scores_map.get(did, 0):.4f}",
        ])

    return out.get("answer", ""), trace_text, badge, sub_text, table_rows


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Big Data Q&A -- E-commerce Analytics",
        css=CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# E-commerce Analytics Agent\n"
            "**150K Amazon reviews** | **FAISS + BM25 + Cross-encoder** | "
            "**Multi-agent: Planner -> Router -> [Decompose|SQL-first|Direct] -> Analyst -> Critic** "
            "(LangGraph + Groq)\n\n"
            "Type a question or pick an example below."
        )

        with gr.Row():
            q = gr.Textbox(
                label="Your question",
                lines=2,
                placeholder="e.g. What is the average rating by category?",
                scale=4,
            )
            btn = gr.Button("Ask", variant="primary", scale=1)

        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=[q],
            label="Example questions",
        )

        badge = gr.Textbox(label="Pipeline info", lines=1, interactive=False, elem_classes=["badge"])

        with gr.Tabs():
            with gr.Tab("Answer"):
                ans = gr.Textbox(label="Answer", lines=14, elem_classes=["answer-box"])
            with gr.Tab("Agent Trace / Observability"):
                trace = gr.Textbox(label="Agent trace", lines=18, elem_classes=["trace-box"])
                sub_qs = gr.Textbox(label="Sub-questions (if decomposed)", lines=4, interactive=False)
            with gr.Tab("Retrieved Documents"):
                doc_table = gr.Dataframe(
                    headers=["ID", "Category", "Rating", "Title", "Text (preview)", "Score"],
                    label="Retrieved documents",
                    interactive=False,
                    wrap=True,
                )

        btn.click(run_once, q, [ans, trace, badge, sub_qs, doc_table])
        q.submit(run_once, q, [ans, trace, badge, sub_qs, doc_table])
    return demo


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    build_demo().launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
