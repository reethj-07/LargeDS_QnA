"""Generate EVALUATION_REPORT.md from eval_results.json and supporting cache files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _find_cache(out_path: Path, filename: str) -> Path | None:
    for base in [out_path.parent, out_path.parent.parent]:
        p = base / "evaluation" / "cache" / filename
        if p.exists():
            return p
    return None


def _detect_k_values(rows: list[dict]) -> list[int]:
    ks: set[int] = set()
    for r in rows:
        for key in r:
            if key.startswith("recall@"):
                try:
                    ks.add(int(key.split("@")[1]))
                except ValueError:
                    pass
    return sorted(ks) or [10]


def generate_report(results_path: Path, out_path: Path) -> None:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    k_values = _detect_k_values(rows)
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("# Evaluation Report\n")

    # ── Methodology ───────────────────────────────────────────────────────
    lines.append("## Methodology\n")
    lines.append(
        "- **Ground truth:** For each test question, `ground_truth_ids` is a **large relevance pool** "
        "(up to 2 000 IDs) from deterministic, content-aligned DuckDB queries over the `text` column "
        "(raw review body, not category-prefixed `doc_text`). Large pools ensure retrieval metrics "
        "are meaningful — with 200–2 000 relevant docs, even top-10 retrieval has a realistic "
        "chance of overlap.\n"
    )
    lines.append(
        "- **Eval modes:** Each question has an `eval_mode`:\n"
        "  - `retrieval` — retrieval metrics are primary.\n"
        "  - `sql` — SQL aggregation answers the question; retrieval overlap is informational.\n"
        "  - `both` — SQL + retrieval contribute.\n"
    )
    lines.append(
        "- **Category filtering:** Single-category questions use metadata-filtered retrieval "
        "(DuckDB pre-filter). Multi-category / cross-cutting questions search the full corpus.\n"
    )
    lines.append(
        "- **Multi-K metrics:** Reported at K = " +
        ", ".join(str(k) for k in k_values) +
        ". Hit@K and Precision@K are the most interpretable when ground-truth pools are "
        "much larger than K; Recall@K is bounded by K / pool_size.\n"
    )

    # ── Per-question results ──────────────────────────────────────────────
    k0 = k_values[0]
    lines.append(f"\n## Per-question results (K={k0})\n")
    lines.append(
        f"| id | tier | mode | pool | Hit@{k0} | Prec@{k0} | MRR@{k0} | nDCG@{k0} | "
        f"Recall@{k0} | cat_filter |"
    )
    lines.append("|---:|---|---|---:|:---:|---:|---:|---:|---:|---|")
    for r in rows:
        def _f(key: str) -> str:
            v = r.get(key)
            return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"

        pool = r.get("ground_truth_count", 0)
        h = r.get(f"hit@{k0}")
        cf = r.get("category_filter") or ""
        lines.append(
            f"| {r['id']} | {r.get('tier')} | {r.get('eval_mode','?')} | {pool} | "
            f"{h if h is not None else 'N/A'} | "
            f"{_f(f'precision@{k0}')} | {_f(f'mrr@{k0}')} | {_f(f'ndcg@{k0}')} | "
            f"{_f(f'recall@{k0}')} | {cf} |"
        )

    # ── Aggregate by eval_mode, for each K ────────────────────────────────
    with_ground = [r for r in rows if r.get(f"recall@{k0}") is not None]
    retrieval_qs = [r for r in with_ground if r.get("eval_mode", "retrieval") != "sql"]
    sql_qs = [r for r in with_ground if r.get("eval_mode") == "sql"]

    def _agg_table(label: str, subset: list[dict]) -> None:
        if not subset:
            return
        n = len(subset)
        lines.append(f"\n### {label} (n={n})\n")
        lines.append(f"| K | Hit rate | Precision | MRR | nDCG | Recall |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for k in k_values:
            hits = sum(r.get(f"hit@{k}", 0) for r in subset)
            avg_p = _mean([r[f"precision@{k}"] for r in subset if r.get(f"precision@{k}") is not None])
            avg_m = _mean([r[f"mrr@{k}"] for r in subset if r.get(f"mrr@{k}") is not None])
            avg_n = _mean([r[f"ndcg@{k}"] for r in subset if r.get(f"ndcg@{k}") is not None])
            avg_r = _mean([r[f"recall@{k}"] for r in subset if r.get(f"recall@{k}") is not None])
            lines.append(
                f"| {k} | {hits}/{n} ({100*hits/n:.0f}%) | {avg_p:.4f} | "
                f"{avg_m:.4f} | {avg_n:.4f} | {avg_r:.4f} |"
            )

    lines.append("\n## Aggregate metrics\n")
    _agg_table("All questions", with_ground)
    _agg_table("Retrieval-primary (eval_mode != sql)", retrieval_qs)
    if sql_qs:
        _agg_table("SQL-primary (retrieval informational)", sql_qs)

    # ── By tier ───────────────────────────────────────────────────────────
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in retrieval_qs:
        by_tier[str(r.get("tier"))].append(r)
    if by_tier:
        lines.append(f"\n### By tier (retrieval-primary, K={k0})\n")
        lines.append(f"| Tier | n | Hit rate | Precision | MRR | nDCG |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for tier in sorted(by_tier.keys()):
            v = by_tier[tier]
            n = len(v)
            hits = sum(r.get(f"hit@{k0}", 0) for r in v)
            avg_p = _mean([r[f"precision@{k0}"] for r in v])
            avg_m = _mean([r[f"mrr@{k0}"] for r in v])
            avg_n = _mean([r[f"ndcg@{k0}"] for r in v])
            lines.append(
                f"| {tier} | {n} | {hits}/{n} | {avg_p:.4f} | {avg_m:.4f} | {avg_n:.4f} |"
            )

    # ── Failure / Error analysis ──────────────────────────────────────────
    lines.append("\n## Failure cases and error analysis\n")
    misses = [r for r in retrieval_qs if r.get(f"hit@{k0}") == 0]
    if misses:
        lines.append(f"**{len(misses)}/{len(retrieval_qs)}** retrieval-primary questions missed at K={k0}:\n")
        for r in misses:
            lines.append(f"- Q{r['id']} ({r.get('tier')}): \"{r['question'][:60]}\" — "
                         f"pool={r.get('ground_truth_count',0)}, "
                         f"cat_filter={r.get('category_filter') or 'none'}")
        lines.append("")
    lines.append(
        "Common miss patterns:\n"
        "- **Generic keywords** (e.g. \"quality\", \"good\") match many documents; ground truth "
        "and retrieval may find different subsets of the same semantic space.\n"
        "- **Cross-category queries** without metadata filter search 150K docs; top-10 is a "
        "very small window.\n"
        "- **Structural queries** (helpful_vote ordering, ASIN lookup) require SQL, not semantic search.\n"
    )

    # ── Trade-offs ────────────────────────────────────────────────────────
    lines.append("\n## Trade-offs\n")
    lines.append(
        "- **Large ground-truth pools** (200–2 000 docs) make Hit@K and Precision@K meaningful, "
        "but Recall@K is naturally low (bounded by K / pool_size). "
        "A pool of 1 000 means Recall@10 ≤ 0.01 even with perfect retrieval.\n"
    )
    lines.append("- **CPU embeddings** (~2h for 150K rows) vs GPU — zero cost, reproducible.\n")
    lines.append(
        "- **MiniLM (384-d)** — small, fast, fits in RAM; trades semantic depth for speed.\n"
    )
    lines.append(
        "- **Metadata filtering** narrows category queries from 150K to ~30K candidates.\n"
    )
    lines.append(
        "- **RRF + cross-encoder** — RRF fuses dense + sparse; cross-encoder rescores top pool.\n"
    )

    # ── Ablation (if cache exists) ────────────────────────────────────────
    ablation_path = _find_cache(out_path, "ablation.json")
    if ablation_path:
        ab = json.loads(ablation_path.read_text(encoding="utf-8"))
        lines.append("\n## Ablation study\n")
        ab_ks = set()
        for r in ab:
            for key in r:
                if key.startswith("recall@"):
                    try:
                        ab_ks.add(int(key.split("@")[1]))
                    except ValueError:
                        pass
        ab_ks_sorted = sorted(ab_ks) or [10]
        for k in ab_ks_sorted:
            lines.append(f"\n### K = {k}\n")
            lines.append(f"| Mode | Hit rate | Precision | MRR | nDCG | Recall |")
            lines.append("|------|---:|---:|---:|---:|---:|")
            for r in ab:
                lines.append(
                    f"| {r['mode']} | {r.get(f'hit_rate@{k}', 0):.4f} | "
                    f"{r.get(f'precision@{k}', 0):.4f} | {r.get(f'mrr@{k}', 0):.4f} | "
                    f"{r.get(f'ndcg@{k}', 0):.4f} | {r.get(f'recall@{k}', 0):.4f} |"
                )

    # ── Answer quality (batch, if cache exists) ───────────────────────────
    aq_path = _find_cache(out_path, "answer_quality.json")
    if aq_path:
        aq = json.loads(aq_path.read_text(encoding="utf-8"))
        scores = [r["critic_score"] for r in aq if r.get("critic_score") is not None]
        lines.append("\n## Answer quality (quantitative, batch)\n")
        if scores:
            lines.append(f"- **Questions scored:** {len(scores)}/{len(aq)}")
            lines.append(f"- **Mean critic score:** {sum(scores)/len(scores):.2f} / 5")
            lines.append(f"- **Min:** {min(scores):.1f} | **Max:** {max(scores):.1f}")
            lines.append(f"- **Score >= 4:** {sum(1 for s in scores if s >= 4)}/{len(scores)}")
        lats = [r["latency_s"] for r in aq if r.get("latency_s") is not None]
        if lats:
            lines.append(f"- **Mean latency:** {sum(lats)/len(lats):.1f}s per question")
        errs = [r for r in aq if "error" in r]
        if errs:
            lines.append(f"- **Pipeline errors:** {len(errs)}/{len(aq)}")

    # ── Regeneration ──────────────────────────────────────────────────────
    lines.append("\n## Regenerating this report\n")
    lines.append(
        "```powershell\n"
        "python scripts/build_ground_truth.py   # rebuild large relevance pools\n"
        "python scripts/evaluate.py             # multi-K retrieval metrics\n"
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
            "# Evaluation Report\n\n"
            "Run `python scripts/evaluate.py` after ingestion to populate metrics.\n",
            encoding="utf-8",
        )
        return
    generate_report(results, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
