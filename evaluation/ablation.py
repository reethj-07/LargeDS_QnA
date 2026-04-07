"""Ablation study: compare retrieval modes at multiple K values.

Only retrieval-primary questions (eval_mode != 'sql') contribute.
Does NOT use category filtering (raw retrieval modes are compared head-to-head).
"""

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
    k_values: list[int],
    *,
    use_vector: bool,
    use_bm25: bool,
    use_cross_encoder: bool,
) -> dict[str, float | int]:
    max_k = max(k_values)
    per_k: dict[int, dict[str, list]] = {
        k: {"recalls": [], "precisions": [], "mrrs": [], "ndcgs": [], "hits": []}
        for k in k_values
    }

    for item in questions:
        rel = set(int(x) for x in item.get("ground_truth_ids") or [])
        if not rel:
            continue
        q_ret = (item.get("retrieval_query") or item["question"]).strip()
        ranked = hybrid.hybrid_search(
            q_ret,
            top_k=max_k,
            use_vector=use_vector,
            use_bm25=use_bm25,
            use_cross_encoder=use_cross_encoder,
        )
        ids = [i for i, _ in ranked]
        for k in k_values:
            per_k[k]["recalls"].append(recall_at_k(ids, rel, k))
            per_k[k]["precisions"].append(precision_at_k(ids, rel, k))
            per_k[k]["mrrs"].append(mrr_at_k(ids, rel, k))
            per_k[k]["ndcgs"].append(ndcg_at_k(ids, rel, k))
            per_k[k]["hits"].append(hit_at_k(ids, rel, k))

    out: dict[str, float | int] = {}
    for k in k_values:
        d = per_k[k]
        n = len(d["recalls"]) or 1
        out[f"n"] = len(d["recalls"])
        out[f"recall@{k}"] = round(sum(d["recalls"]) / n, 4)
        out[f"precision@{k}"] = round(sum(d["precisions"]) / n, 4)
        out[f"mrr@{k}"] = round(sum(d["mrrs"]) / n, 4)
        out[f"ndcg@{k}"] = round(sum(d["ndcgs"]) / n, 4)
        out[f"hit_rate@{k}"] = round(sum(d["hits"]) / n, 4)
    return out


MODES = [
    ("Vector only",  {"use_vector": True,  "use_bm25": False, "use_cross_encoder": False}),
    ("BM25 only",    {"use_vector": False, "use_bm25": True,  "use_cross_encoder": False}),
    ("Hybrid (RRF)", {"use_vector": True,  "use_bm25": True,  "use_cross_encoder": False}),
    ("Hybrid + CE",  {"use_vector": True,  "use_bm25": True,  "use_cross_encoder": True}),
]


def run_ablation(questions_path: Path, k_values: list[int], out_path: Path) -> list[dict]:
    qs = json.loads(questions_path.read_text(encoding="utf-8"))
    retrieval_qs = [q for q in qs if q.get("eval_mode", "retrieval") != "sql"]
    print(f"  Using {len(retrieval_qs)} retrieval-primary questions "
          f"(skipping {len(qs) - len(retrieval_qs)} sql-only)")

    hybrid = HybridRetriever(embedder=Embedder())
    hybrid.load_indices()

    rows: list[dict] = []
    for label, kwargs in MODES:
        print(f"  Evaluating: {label} ...", flush=True)
        metrics = _eval_mode(hybrid, retrieval_qs, k_values, **kwargs)
        rows.append({"mode": label, **metrics})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    for k in k_values:
        header = f"\n  @K={k}:  {'Mode':<20} {'Recall':>8} {'Prec':>8} {'MRR':>8} {'nDCG':>8} {'Hit%':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            print(
                f"         {r['mode']:<20} "
                f"{r.get(f'recall@{k}', 0):>8.4f} {r.get(f'precision@{k}', 0):>8.4f} "
                f"{r.get(f'mrr@{k}', 0):>8.4f} {r.get(f'ndcg@{k}', 0):>8.4f} "
                f"{r.get(f'hit_rate@{k}', 0):>8.4f}"
            )
    return rows


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Retrieval ablation study")
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--k", type=int, nargs="+", default=[10, 50])
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "ablation.json")
    args = p.parse_args()
    run_ablation(args.questions, sorted(set(args.k)), args.out)


if __name__ == "__main__":
    main()
