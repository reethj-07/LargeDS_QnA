"""Run retrieval evaluation over test_questions.json with eval_mode support.

Computes metrics at multiple K values (10, 50) and separates retrieval-primary
from SQL-primary questions.  Category-specific questions use metadata-filtered
retrieval for a fairer comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from src.config import DUCKDB_PATH  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.storage.sql_store import SqlStore  # noqa: E402

_CATEGORY_KEYWORDS = {
    "All_Beauty": "All_Beauty",
    "Appliances": "Appliances",
    "Amazon_Fashion": "Amazon_Fashion",
    "Arts_Crafts_and_Sewing": "Arts_Crafts_and_Sewing",
    "Baby_Products": "Baby_Products",
}


def _detect_category(question: str, retrieval_query: str) -> str | None:
    combined = f"{question} {retrieval_query}"
    found = [cat for kw, cat in _CATEGORY_KEYWORDS.items() if kw in combined]
    if len(found) == 1:
        return found[0]
    return None


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--k", type=int, nargs="+", default=[10, 50])
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "eval_results.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    k_values = sorted(set(args.k))
    max_k = max(k_values)

    qs = load_questions(args.questions)
    hybrid = HybridRetriever(
        embedder=Embedder(),
        sql_store=SqlStore(DUCKDB_PATH, read_only=True),
    )
    hybrid.load_indices()

    category_cache: dict[str, set[int]] = {}
    rows_out: list[dict] = []

    for item in qs:
        q = item["question"]
        rel = set(int(x) for x in item.get("ground_truth_ids") or [])
        q_ret = (item.get("retrieval_query") or q).strip()
        eval_mode = item.get("eval_mode", "retrieval")

        cat = _detect_category(q, q_ret)
        allowed_ids: set[int] | None = None
        if cat and eval_mode in ("retrieval", "both"):
            if cat not in category_cache:
                category_cache[cat] = hybrid.get_ids_for_category(cat)
            allowed_ids = category_cache[cat]

        ranked = hybrid.hybrid_search(q_ret, top_k=max_k, allowed_ids=allowed_ids)
        retrieved_ids = [i for i, _ in ranked]

        row: dict = {
            "id": item["id"],
            "tier": item.get("tier"),
            "eval_mode": eval_mode,
            "question": q,
            "retrieval_query_used": q_ret,
            "category_filter": cat,
            "ground_truth_count": len(rel),
            "retrieved_top_50": retrieved_ids[:50],
        }

        for k in k_values:
            if rel:
                row[f"recall@{k}"] = recall_at_k(retrieved_ids, rel, k)
                row[f"precision@{k}"] = precision_at_k(retrieved_ids, rel, k)
                row[f"mrr@{k}"] = mrr_at_k(retrieved_ids, rel, k)
                row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, rel, k)
                topk = set(retrieved_ids[:k])
                row[f"hit@{k}"] = 1 if (topk & rel) else 0
            else:
                for m in ("recall", "precision", "mrr", "ndcg", "hit"):
                    row[f"{m}@{k}"] = None

        rows_out.append(row)

    args.out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    with_ground = [r for r in rows_out if r.get("recall@10") is not None]
    retrieval_qs = [r for r in with_ground if r.get("eval_mode", "retrieval") != "sql"]
    sql_qs = [r for r in with_ground if r.get("eval_mode") == "sql"]

    print(f"\nQuestions with ground truth: {len(with_ground)}/{len(rows_out)}")

    def _print_summary(label: str, subset: list[dict]) -> None:
        if not subset:
            return
        n = len(subset)
        print(f"\n--- {label} (n={n}) ---")
        for k in k_values:
            avg_r = sum(r[f"recall@{k}"] for r in subset) / n
            avg_p = sum(r[f"precision@{k}"] for r in subset) / n
            avg_m = sum(r[f"mrr@{k}"] for r in subset) / n
            avg_n = sum(r[f"ndcg@{k}"] for r in subset) / n
            hits = sum(r[f"hit@{k}"] for r in subset)
            print(f"  @{k:>2}: Recall={avg_r:.4f}  Prec={avg_p:.4f}  "
                  f"MRR={avg_m:.4f}  nDCG={avg_n:.4f}  Hit={hits}/{n}")

    _print_summary("All questions", with_ground)
    _print_summary("Retrieval-primary (eval_mode=retrieval|both)", retrieval_qs)
    if sql_qs:
        _print_summary("SQL-primary (informational)", sql_qs)

    # Per-question diagnostic
    print("\nPer-question Hit@10:")
    for r in rows_out:
        h = r.get("hit@10")
        pool = r.get("ground_truth_count", 0)
        emoji = "HIT" if h == 1 else "MISS" if h == 0 else "N/A"
        cat = r.get("category_filter") or ""
        print(f"  Q{r['id']:>2} [{r.get('eval_mode','?'):>9}] "
              f"pool={pool:>5}  {emoji:>4}  {cat}")


if __name__ == "__main__":
    main()
