"""Ablation study: compare retrieval modes (vector-only, BM25-only, hybrid, hybrid+CE).
Outputs a table of MRR@K, nDCG@K, Hit@K for each mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (  # noqa: E402
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.embeddings.embedder import Embedder  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def _eval_mode(
    hybrid: HybridRetriever,
    questions: list[dict],
    k: int,
    *,
    use_vector: bool,
    use_bm25: bool,
    use_cross_encoder: bool,
) -> dict[str, float]:
    recalls, precisions, mrrs, ndcgs, hits = [], [], [], [], []
    for item in questions:
        rel = set(int(x) for x in item.get("ground_truth_ids") or [])
        if not rel:
            continue
        q_ret = (item.get("retrieval_query") or item["question"]).strip()
        ranked = hybrid.hybrid_search(
            q_ret,
            top_k=max(k, 10),
            use_vector=use_vector,
            use_bm25=use_bm25,
            use_cross_encoder=use_cross_encoder,
        )
        ids = [i for i, _ in ranked]
        recalls.append(recall_at_k(ids, rel, k))
        precisions.append(precision_at_k(ids, rel, k))
        mrrs.append(mrr_at_k(ids, rel, k))
        ndcgs.append(ndcg_at_k(ids, rel, k))
        hits.append(hit_at_k(ids, rel, k))

    n = len(recalls) or 1
    return {
        "n": len(recalls),
        "recall": round(sum(recalls) / n, 4),
        "precision": round(sum(precisions) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "ndcg": round(sum(ndcgs) / n, 4),
        "hit_rate": round(sum(hits) / n, 4),
    }


MODES = [
    ("Vector only",       {"use_vector": True,  "use_bm25": False, "use_cross_encoder": False}),
    ("BM25 only",         {"use_vector": False, "use_bm25": True,  "use_cross_encoder": False}),
    ("Hybrid (RRF)",      {"use_vector": True,  "use_bm25": True,  "use_cross_encoder": False}),
    ("Hybrid + CE",       {"use_vector": True,  "use_bm25": True,  "use_cross_encoder": True}),
]


def run_ablation(questions_path: Path, k: int, out_path: Path) -> list[dict]:
    qs = json.loads(questions_path.read_text(encoding="utf-8"))
    hybrid = HybridRetriever(embedder=Embedder())
    hybrid.load_indices()

    rows: list[dict] = []
    for label, kwargs in MODES:
        print(f"  Evaluating: {label} ...", flush=True)
        metrics = _eval_mode(hybrid, qs, k, **kwargs)
        rows.append({"mode": label, **metrics})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    header = f"{'Mode':<20} {'Recall':>8} {'Prec':>8} {'MRR':>8} {'nDCG':>8} {'Hit%':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mode']:<20} {r['recall']:>8.4f} {r['precision']:>8.4f} "
            f"{r['mrr']:>8.4f} {r['ndcg']:>8.4f} {r['hit_rate']:>8.4f}"
        )
    return rows


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Retrieval ablation study")
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "ablation.json")
    args = p.parse_args()
    run_ablation(args.questions, args.k, args.out)


if __name__ == "__main__":
    main()
