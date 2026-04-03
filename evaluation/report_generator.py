"""Generate EVALUATION_REPORT.md from eval_results.json."""

from __future__ import annotations

import json
from pathlib import Path


def generate_report(results_path: Path, out_path: Path) -> None:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Evaluation Report\n")
    lines.append("## Retrieval (hybrid FAISS + BM25 + RRF)\n")
    lines.append("| id | tier | Recall@K | Precision@K | has_ground_truth |")
    lines.append("|---:|---|---:|---:|:---:|")
    for r in rows:
        rk = r.get("recall_at_k")
        pk = r.get("precision_at_k")
        gt = rk is not None
        lines.append(
            f"| {r.get('id')} | {r.get('tier')} | {rk if rk is not None else 'N/A'} | "
            f"{pk if pk is not None else 'N/A'} | {gt} |"
        )
    lines.append("\n## Notes\n")
    lines.append(
        "- Populate `ground_truth_ids` in `evaluation/test_questions.json` with relevant `reviews.id` "
        "values (from DuckDB) to enable Recall@K / Precision@K.\n"
        "- Answer quality: use Groq LLM-as-judge in the demo or add a separate script; not included in "
        "default batch to save rate limits.\n"
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
