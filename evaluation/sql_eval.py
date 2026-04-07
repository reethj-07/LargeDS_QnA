"""SQL accuracy evaluation for aggregation-type questions.

For questions with eval_mode='sql', the system's answer quality depends on SQL
correctness, not document retrieval overlap.  This module runs known SQL queries
and compares expected vs actual answers to provide a complementary evaluation
axis alongside retrieval metrics.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sql_test_cases(con: Any) -> list[dict]:
    """Return SQL test cases with expected answers derived from the database."""

    def _query(sql: str) -> list[dict]:
        return con.execute(sql).fetchdf().to_dict(orient="records")

    def _scalar(sql: str) -> Any:
        return con.execute(sql).fetchone()[0]

    cases = []

    avg_beauty = _scalar(
        "SELECT ROUND(AVG(rating), 2) FROM reviews WHERE category = 'All_Beauty'"
    )
    cases.append({
        "id": 1,
        "question": "What is the average rating in the All_Beauty category?",
        "expected_sql": "SELECT AVG(rating) FROM reviews WHERE category = 'All_Beauty'",
        "expected_value": float(avg_beauty),
        "tolerance": 0.05,
        "check_type": "numeric",
    })

    pct = _scalar(
        "SELECT ROUND(100.0 * SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) / COUNT(*), 2) "
        "FROM reviews WHERE verified_purchase = true"
    )
    cases.append({
        "id": 5,
        "question": "What percentage of verified_purchase reviews are 5-star?",
        "expected_sql": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) / COUNT(*), 2) "
            "FROM reviews WHERE verified_purchase = true"
        ),
        "expected_value": float(pct),
        "tolerance": 1.0,
        "check_type": "numeric",
    })

    best_cat = _scalar(
        "SELECT category FROM reviews GROUP BY category ORDER BY AVG(rating) DESC LIMIT 1"
    )
    cases.append({
        "id": 6,
        "question": "Which category has the highest average rating?",
        "expected_sql": (
            "SELECT category, AVG(rating) as avg_rating FROM reviews "
            "GROUP BY category ORDER BY avg_rating DESC LIMIT 1"
        ),
        "expected_value": best_cat,
        "check_type": "string_contains",
    })

    top5 = _query(
        "SELECT id, helpful_vote FROM reviews ORDER BY helpful_vote DESC NULLS LAST LIMIT 5"
    )
    cases.append({
        "id": 7,
        "question": "What are the top 5 most helpful reviews by helpful_vote?",
        "expected_sql": (
            "SELECT id, helpful_vote FROM reviews "
            "ORDER BY helpful_vote DESC NULLS LAST LIMIT 5"
        ),
        "expected_value": [r["id"] for r in top5],
        "check_type": "id_list",
    })

    dist = _query(
        "SELECT CAST(rating AS INT) as rating_bucket, COUNT(*) as cnt "
        "FROM reviews GROUP BY rating_bucket ORDER BY rating_bucket"
    )
    cases.append({
        "id": 8,
        "question": "How many reviews fall into each rating bucket (1-5)?",
        "expected_sql": (
            "SELECT CAST(rating AS INT) as rating, COUNT(*) "
            "FROM reviews GROUP BY CAST(rating AS INT) ORDER BY rating"
        ),
        "expected_value": {str(r["rating_bucket"]): int(r["cnt"]) for r in dist},
        "check_type": "distribution",
    })

    return cases


def run_sql_eval(out_path: Path) -> list[dict]:
    """Generate SQL evaluation reference data."""
    from src.config import DUCKDB_PATH

    if not DUCKDB_PATH.exists():
        print("No DuckDB found. Run ingest first.")
        sys.exit(1)

    import duckdb
    con = duckdb.connect(str(DUCKDB_PATH))

    cases = _sql_test_cases(con)
    con.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(cases)} SQL test cases to {out_path}")

    for c in cases:
        print(f"  Q{c['id']}: {c['check_type']} — expected={str(c['expected_value'])[:80]}")

    return cases


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate SQL accuracy evaluation data")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "sql_eval.json")
    args = p.parse_args()
    run_sql_eval(args.out)


if __name__ == "__main__":
    main()
