"""Gradio UI: question → answer + trace."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from src.agents.graph import run_agent_pipeline
from src.observability.logger import log_event


def run_once(message: str) -> tuple[str, str]:
    if not (message or "").strip():
        return "", ""
    log_event("ui_query", {"q_len": len(message)})
    out: dict[str, Any] = run_agent_pipeline(message.strip())
    trace_lines = list(out.get("agent_trace", []))
    trace_lines.append(f"confidence={out.get('confidence')}")
    trace_lines.append(f"critique={out.get('critique', '')[:600]}")
    meta = {
        "query_type": out.get("query_type"),
        "sql_rows": len(out.get("sql_results") or []),
        "retrieved_docs": len(out.get("retrieved_docs") or []),
        "retrieved_ids": [d.get("id") for d in (out.get("retrieved_docs") or [])[:20]],
    }
    trace_text = "\n".join(trace_lines) + "\n\n" + json.dumps(meta, indent=2, default=str)
    return out.get("answer", ""), trace_text


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Big Data Q&A — E-commerce Analytics") as demo:
        gr.Markdown(
            "## E-commerce Analytics Agent\n"
            "**Flow:** Planner → Retriever (FAISS + BM25 + DuckDB) → Analyst → Critic.\n\n"
            "Requires indices from `python scripts/ingest.py` and `GROQ_API_KEY` in `.env`."
        )
        with gr.Row():
            q = gr.Textbox(label="Your question", lines=3, placeholder="e.g. What is the average rating by category?")
        with gr.Row():
            btn = gr.Button("Run", variant="primary")
        with gr.Row():
            ans = gr.Textbox(label="Answer", lines=16)
            trace = gr.Textbox(label="Agent trace + metadata", lines=16)
        btn.click(run_once, q, [ans, trace])
        q.submit(run_once, q, [ans, trace])
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
