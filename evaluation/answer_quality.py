"""Batch answer quality: run all test questions through the full agent pipeline,
collect critic scores, and append results to eval cache."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.graph import run_agent_pipeline  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402


def run_answer_quality(
    questions_path: Path,
    out_path: Path,
) -> list[dict]:
    qs = json.loads(questions_path.read_text(encoding="utf-8"))
    hybrid = HybridRetriever(embedder=Embedder())
    hybrid.load_indices()

    results: list[dict] = []
    for item in qs:
        qid = item["id"]
        q = item["question"]
        print(f"  Q{qid}: {q[:80]}...", flush=True)
        t0 = time.time()
        try:
            out = run_agent_pipeline(q, hybrid=hybrid)
            elapsed = time.time() - t0
            results.append({
                "id": qid,
                "tier": item.get("tier"),
                "question": q,
                "answer_len": len(out.get("answer", "")),
                "critic_score": out.get("confidence"),
                "critique": (out.get("critique") or "")[:300],
                "retries": out.get("retry_count", 0),
                "latency_s": round(elapsed, 2),
                "query_type": out.get("query_type"),
                "sql_rows": len(out.get("sql_results") or []),
                "retrieved_docs": len(out.get("retrieved_docs") or []),
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "id": qid,
                "tier": item.get("tier"),
                "question": q,
                "error": str(e)[:300],
                "latency_s": round(elapsed, 2),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    scores = [r["critic_score"] for r in results if r.get("critic_score") is not None]
    if scores:
        print(f"Questions scored: {len(scores)}/{len(results)}")
        print(f"Mean critic score: {sum(scores)/len(scores):.2f}")
        print(f"Min: {min(scores):.1f}  Max: {max(scores):.1f}")
        print(f"Score >= 4: {sum(1 for s in scores if s >= 4)}/{len(scores)}")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"Errors: {len(errors)}")
    return results


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Batch answer quality via full agent pipeline + critic")
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "answer_quality.json")
    args = p.parse_args()
    run_answer_quality(args.questions, args.out)


if __name__ == "__main__":
    main()
