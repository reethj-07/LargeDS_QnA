"""Generate EVALUATION_REPORT.md from eval_results.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def generate_report(results_path: Path, out_path: Path) -> None:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Evaluation Report\n")
    lines.append("## Methodology\n")
    lines.append(
        "- **Ground truth:** For each test question, `ground_truth_ids` lists `reviews.id` values from "
        "DuckDB queries that approximate *relevant* documents (e.g. keyword filter, category filter, "
        "same ASIN). This is a **strict** SQL view of relevance.\n"
    )
    lines.append(
        "- **Retrieval:** Hybrid search (FAISS + BM25 + RRF) uses `retrieval_query` per question — a "
        "short keyword-style string (see `evaluation/test_questions.json`). Full natural-language "
        "questions were found to dilute BM25; short queries align better with SQL-derived labels.\n"
    )
    lines.append(
        "- **Metrics:** **Recall@K** = |retrieved ∩ relevant| / |relevant|; **Precision@K** = "
        "|retrieved ∩ relevant| / K; **Hit@K** = 1 if at least one relevant id appears in the top-K "
        "results, else 0.\n"
    )
    lines.append(
        "- **Interpretation:** Absolute scores are often modest because SQL relevance sets are large "
        "and diverse while the retriever returns only 10 dense+sparse-fused candidates. Use metrics "
        "for **before/after** comparisons (e.g. ablations), not as an absolute ceiling.\n"
    )

    lines.append("\n## Retrieval (hybrid FAISS + BM25 + RRF)\n")
    lines.append("| id | tier | Recall@10 | Precision@10 | Hit@10 |")
    lines.append("|---:|---|---:|---:|:---:|")
    for r in rows:
        rk = r.get("recall_at_k")
        pk = r.get("precision_at_k")
        hk = r.get("hit_at_k")
        lines.append(
            f"| {r.get('id')} | {r.get('tier')} | "
            f"{rk if rk is not None else 'N/A'} | {pk if pk is not None else 'N/A'} | "
            f"{hk if hk is not None else 'N/A'} |"
        )

    vals_r = [r["recall_at_k"] for r in rows if r.get("recall_at_k") is not None]
    vals_p = [r["precision_at_k"] for r in rows if r.get("precision_at_k") is not None]
    vals_h = [r["hit_at_k"] for r in rows if r.get("hit_at_k") is not None]
    if vals_r:
        lines.append("\n### Aggregate (n=20)\n")
        lines.append(f"- **Mean Recall@10:** {sum(vals_r) / len(vals_r):.4f}")
        lines.append(f"- **Mean Precision@10:** {sum(vals_p) / len(vals_p):.4f}")
        lines.append(f"- **Hit@10 rate:** {sum(vals_h)}/{len(vals_h)} questions with ≥1 overlap in top-10")

    by_tier: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("recall_at_k") is not None:
            by_tier[str(r.get("tier"))].append(float(r["recall_at_k"]))
    if by_tier:
        lines.append("\n### Mean Recall@10 by tier\n")
        for tier in sorted(by_tier.keys()):
            v = by_tier[tier]
            lines.append(f"- **{tier}:** {sum(v) / len(v):.4f} (n={len(v)})")

    lines.append("\n## Answer quality (qualitative)\n")
    lines.append(
        "- The **Analyst** (Groq `llama-3.3-70b-versatile`) synthesizes answers with SQL + retrieved "
        "context; the **Critic** scores grounding 1–5. Spot-check complex queries via "
        "`python scripts/query.py \"...\"`.\n"
    )
    lines.append(
        "- **Example (manual):** Aggregation query *average rating by category* — SQL returned "
        "per-category averages; a follow-up tie case (All_Beauty vs Amazon_Fashion) initially "
        "mis-stated “the” lowest category until prompts required explicit **ties**; after prompt "
        "adjustment, critic score reached 5/5.\n"
    )

    lines.append("\n## Failure cases and error analysis\n")
    lines.append(
        "1. **Planner-generated SQL:** Occasionally invalid or overly narrow SQL is skipped; retrieval "
        "falls back to hybrid search only (`retrieval_sql_error` in logs).\n"
    )
    lines.append(
        "2. **Numeric ties:** Category averages can tie at displayed precision; the analyst must "
        "list all tied groups (mitigated in `src/agents/analyst.py`).\n"
    )
    lines.append(
        "3. **Retrieval vs SQL labels:** Low Recall@10 does not always mean poor UX — users may still "
        "get correct **aggregates** from SQL while retrieved snippets illustrate themes.\n"
    )
    lines.append(
        "4. **Groq rate limits:** Heavy evaluation or rapid UI clicks can hit limits; add backoff or "
        "cache for production.\n"
    )

    lines.append("\n## Trade-offs\n")
    lines.append(
        "- **CPU embeddings** (~1h+ for 150k rows) vs cloud GPU — chosen for zero cost and "
        "reproducibility.\n"
    )
    lines.append(
        "- **MiniLM (384-d)** vs larger encoders — smaller index and RAM footprint on a 12GB laptop; "
        "trade some semantic nuance.\n"
    )
    lines.append(
        "- **RRF** instead of a cross-encoder reranker — faster and no extra model load; trade some "
        "precision for latency.\n"
    )
    lines.append(
        "- **Groq free tier** — fast and free; vendor lock-in and rate limits vs self-hosted LLM.\n"
    )

    lines.append("\n## Regenerating this report\n")
    lines.append("```powershell\npython scripts/build_ground_truth.py   # after ingest; optional\npython scripts/evaluate.py\n```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "evaluation" / "cache" / "eval_results.json"
    out = root / "EVALUATION_REPORT.md"
    if not results.exists():
        out.write_text(
            "# Evaluation Report\n\nRun `python scripts/evaluate.py` after ingestion to populate metrics.\n",
            encoding="utf-8",
        )
        return
    generate_report(results, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
