"""Run retrieval evaluation (and optional LLM judge) over test_questions.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "eval_results.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    qs = load_questions(args.questions)
    hybrid = HybridRetriever(embedder=Embedder())
    hybrid.load_indices()

    rows_out: list[dict] = []
    for item in qs:
        q = item["question"]
        rel = set(int(x) for x in item.get("ground_truth_ids") or [])
        q_ret = (item.get("retrieval_query") or q).strip()
        ranked = hybrid.hybrid_search(q_ret, top_k=max(args.k, 10))
        retrieved_ids = [i for i, _ in ranked]
        if rel:
            r_at_k = recall_at_k(retrieved_ids, rel, args.k)
            p_at_k = precision_at_k(retrieved_ids, rel, args.k)
            m_at_k = mrr_at_k(retrieved_ids, rel, args.k)
            n_at_k = ndcg_at_k(retrieved_ids, rel, args.k)
        else:
            r_at_k = None
            p_at_k = None
            m_at_k = None
            n_at_k = None
        hit = None
        if rel:
            topk = set(retrieved_ids[: args.k])
            hit = 1 if (topk & rel) else 0
        rows_out.append(
            {
                "id": item["id"],
                "tier": item.get("tier"),
                "question": q,
                "retrieval_query_used": q_ret,
                "recall_at_k": r_at_k,
                "precision_at_k": p_at_k,
                "mrr_at_k": m_at_k,
                "ndcg_at_k": n_at_k,
                "hit_at_k": hit,
                "retrieved_top_k": retrieved_ids[: args.k],
            }
        )

    args.out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    with_ground = sum(1 for r in rows_out if r["recall_at_k"] is not None)
    print(f"Questions with ground truth: {with_ground}/{len(rows_out)}")
    if with_ground:
        avg_r = sum(r["recall_at_k"] for r in rows_out if r["recall_at_k"] is not None) / with_ground
        avg_p = sum(r["precision_at_k"] for r in rows_out if r["precision_at_k"] is not None) / with_ground
        avg_m = sum(r["mrr_at_k"] for r in rows_out if r["mrr_at_k"] is not None) / with_ground
        avg_n = sum(r["ndcg_at_k"] for r in rows_out if r["ndcg_at_k"] is not None) / with_ground
        hits = sum(r["hit_at_k"] for r in rows_out if r.get("hit_at_k") is not None)
        print(f"Mean Recall@{args.k}: {avg_r:.4f}")
        print(f"Mean Precision@{args.k}: {avg_p:.4f}")
        print(f"Mean MRR@{args.k}: {avg_m:.4f}")
        print(f"Mean nDCG@{args.k}: {avg_n:.4f}")
        print(f"Hit@{args.k} (>=1 relevant doc in top-{args.k}): {hits}/{with_ground}")


if __name__ == "__main__":
    main()
