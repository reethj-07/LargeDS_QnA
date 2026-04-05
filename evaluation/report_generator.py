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
        "- **Retrieval:** Hybrid search (FAISS + BM25 + RRF, then optional **cross-encoder** rerank on a "
        "pool of candidates) uses `retrieval_query` per question — a short keyword-style string (see "
        "`evaluation/test_questions.json`). Full natural-language questions were found to dilute BM25; "
        "short queries align better with SQL-derived labels.\n"
    )
    lines.append(
        "- **Metrics:** **Recall@K** = |retrieved ∩ relevant| / |relevant|; **Precision@K** = "
        "|retrieved ∩ relevant| / K; **Hit@K** = 1 if at least one relevant id appears in the top-K "
        "results; **MRR@K** = reciprocal rank of the first relevant hit; **nDCG@K** = normalized "
        "discounted cumulative gain (binary relevance).\n"
    )
    lines.append(
        "- **Interpretation:** Absolute scores are often modest because SQL relevance sets are large "
        "and diverse while the retriever returns only 10 dense+sparse-fused candidates. Use metrics "
        "for **before/after** comparisons (e.g. ablations), not as an absolute ceiling.\n"
    )

    lines.append("\n## Retrieval (hybrid FAISS + BM25 + RRF + optional cross-encoder)\n")
    lines.append("| id | tier | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Hit@10 |")
    lines.append("|---:|---|---:|---:|---:|---:|:---:|")
    for r in rows:
        rk = r.get("recall_at_k")
        pk = r.get("precision_at_k")
        mk = r.get("mrr_at_k")
        nk = r.get("ndcg_at_k")
        hk = r.get("hit_at_k")
        lines.append(
            f"| {r.get('id')} | {r.get('tier')} | "
            f"{rk if rk is not None else 'N/A'} | {pk if pk is not None else 'N/A'} | "
            f"{mk if mk is not None else 'N/A'} | {nk if nk is not None else 'N/A'} | "
            f"{hk if hk is not None else 'N/A'} |"
        )

    vals_r = [r["recall_at_k"] for r in rows if r.get("recall_at_k") is not None]
    vals_p = [r["precision_at_k"] for r in rows if r.get("precision_at_k") is not None]
    vals_m = [r["mrr_at_k"] for r in rows if r.get("mrr_at_k") is not None]
    vals_n = [r["ndcg_at_k"] for r in rows if r.get("ndcg_at_k") is not None]
    vals_h = [r["hit_at_k"] for r in rows if r.get("hit_at_k") is not None]
    if vals_r:
        lines.append("\n### Aggregate (n=20)\n")
        lines.append(f"- **Mean Recall@10:** {sum(vals_r) / len(vals_r):.4f}")
        lines.append(f"- **Mean Precision@10:** {sum(vals_p) / len(vals_p):.4f}")
        if vals_m:
            lines.append(f"- **Mean MRR@10:** {sum(vals_m) / len(vals_m):.4f}")
        if vals_n:
            lines.append(f"- **Mean nDCG@10:** {sum(vals_n) / len(vals_n):.4f}")
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
        "- **RRF + cross-encoder** — RRF fuses dense+sparse lists; a small MS MARCO cross-encoder "
        "(`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks a pool on CPU. Set `USE_CROSS_ENCODER=0` to "
        "skip rerank for speed or CI.\n"
    )
    lines.append(
        "- **Groq free tier** — fast and free; vendor lock-in and rate limits vs self-hosted LLM.\n"
    )

    # --- Ablation section (if cache exists) ---
    ablation_path = out_path.parent / "evaluation" / "cache" / "ablation.json"
    if not ablation_path.exists():
        ablation_path = out_path.parent.parent / "evaluation" / "cache" / "ablation.json"
    if ablation_path.exists():
        ab = json.loads(ablation_path.read_text(encoding="utf-8"))
        lines.append("\n## Ablation study\n")
        lines.append("| Mode | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Hit rate |")
        lines.append("|------|---:|---:|---:|---:|---:|")
        for r in ab:
            lines.append(
                f"| {r['mode']} | {r.get('recall',0):.4f} | {r.get('precision',0):.4f} | "
                f"{r.get('mrr',0):.4f} | {r.get('ndcg',0):.4f} | {r.get('hit_rate',0):.4f} |"
            )

    # --- Answer quality section (if cache exists) ---
    aq_path = out_path.parent / "evaluation" / "cache" / "answer_quality.json"
    if not aq_path.exists():
        aq_path = out_path.parent.parent / "evaluation" / "cache" / "answer_quality.json"
    if aq_path.exists():
        aq = json.loads(aq_path.read_text(encoding="utf-8"))
        scores = [r["critic_score"] for r in aq if r.get("critic_score") is not None]
        lines.append("\n## Answer quality (quantitative, batch)\n")
        if scores:
            lines.append(f"- **Questions scored:** {len(scores)}/{len(aq)}")
            lines.append(f"- **Mean critic score:** {sum(scores)/len(scores):.2f} / 5")
            lines.append(f"- **Min:** {min(scores):.1f} | **Max:** {max(scores):.1f}")
            lines.append(f"- **Score >= 4 (good/great):** {sum(1 for s in scores if s >= 4)}/{len(scores)}")
        lats = [r["latency_s"] for r in aq if r.get("latency_s") is not None]
        if lats:
            lines.append(f"- **Mean latency:** {sum(lats)/len(lats):.1f}s per question")
        errs = [r for r in aq if "error" in r]
        if errs:
            lines.append(f"- **Pipeline errors:** {len(errs)}/{len(aq)}")

    lines.append("\n## Regenerating this report\n")
    lines.append(
        "```powershell\n"
        "python scripts/build_ground_truth.py   # after ingest; optional\n"
        "python scripts/evaluate.py\n"
        "python scripts/ablation.py             # optional ablation\n"
        "python scripts/answer_quality.py       # optional batch answer quality\n"
        "```\n"
    )

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
