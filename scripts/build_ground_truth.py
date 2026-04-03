#!/usr/bin/env python3
"""Populate ground_truth_ids + retrieval_query in evaluation/test_questions.json from DuckDB.

`retrieval_query` is a short keyword-style string used for hybrid search in `evaluate.py`
(full natural questions dilute BM25; short queries align better with SQL-derived relevance).

Run from repo root after ingestion:
  python scripts/build_ground_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DUCKDB_PATH  # noqa: E402


def main() -> None:
    if not DUCKDB_PATH.exists():
        print("No DuckDB at", DUCKDB_PATH, "- run ingest first.")
        sys.exit(1)

    import duckdb

    con = duckdb.connect(str(DUCKDB_PATH))

    def ids(sql: str) -> list[int]:
        rows = con.execute(sql).fetchall()
        return [int(r[0]) for r in rows]

    top_asin = con.execute(
        "SELECT asin FROM reviews GROUP BY asin ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()[0]
    top_asin_esc = top_asin.replace("'", "''")

    row = con.execute(
        """
        SELECT asin FROM reviews
        GROUP BY asin
        HAVING max(rating) >= 5 AND min(rating) <= 1
        LIMIT 1
        """
    ).fetchone()
    split_asin = row[0].replace("'", "''") if row else top_asin_esc

    # Ground truth IDs + short retrieval queries aligned with each test case
    cases: list[tuple[int, list[int], str]] = [
        (1, ids("SELECT id FROM reviews WHERE category = 'All_Beauty' ORDER BY random() LIMIT 12"), "All_Beauty skincare hair beauty"),
        (2, ids("SELECT id FROM reviews WHERE lower(doc_text) LIKE '%moisturizer%' LIMIT 15"), "moisturizer cream lotion"),
        (3, ids(f"SELECT id FROM reviews WHERE asin = '{top_asin_esc}' LIMIT 20"), f"{top_asin} product reviews"),
        (
            4,
            ids(
                "SELECT id FROM reviews WHERE rating >= 5 AND lower(doc_text) LIKE '%fragrance%' LIMIT 12"
            ),
            "fragrance scent perfume smell",
        ),
        (5, ids("SELECT id FROM reviews WHERE verified_purchase = true AND rating = 5 ORDER BY random() LIMIT 12"), "verified purchase five stars"),
        (6, ids("SELECT id FROM reviews WHERE category = (SELECT category FROM reviews GROUP BY category ORDER BY AVG(rating) DESC LIMIT 1) LIMIT 12"), "best rated category quality"),
        (7, ids("SELECT id FROM reviews ORDER BY helpful_vote DESC NULLS LAST LIMIT 8"), "helpful helpful_vote useful"),
        (8, ids("SELECT id FROM reviews ORDER BY random() LIMIT 12"), "rating stars one two three four five"),
        (9, ids("(SELECT id FROM reviews WHERE category = 'All_Beauty' LIMIT 6) UNION ALL (SELECT id FROM reviews WHERE category = 'Appliances' LIMIT 6)"), "All_Beauty Appliances kitchen"),
        (10, ids("(SELECT id FROM reviews WHERE category = 'Amazon_Fashion' LIMIT 8) UNION ALL (SELECT id FROM reviews WHERE category = 'Baby_Products' LIMIT 8)"), "Amazon_Fashion Baby_Products clothing baby"),
        (11, ids("SELECT id FROM reviews ORDER BY random() LIMIT 10"), "verified purchase rating average"),
        (12, ids("SELECT id FROM reviews WHERE rating = 1 ORDER BY random() LIMIT 15"), "one star negative worst"),
        (13, ids(f"SELECT id FROM reviews WHERE asin = '{split_asin}' LIMIT 20"), f"{split_asin} reviews ratings"),
        (14, ids("SELECT id FROM reviews WHERE lower(doc_text) LIKE '%complaint%' OR lower(doc_text) LIKE '%defect%' LIMIT 12"), "complaint defect quality issue"),
        (
            15,
            ids(
                """
                SELECT id FROM reviews WHERE category = 'Arts_Crafts_and_Sewing'
                AND (lower(doc_text) LIKE '%shipping%' OR lower(doc_text) LIKE '%damage%')
                LIMIT 12
                """
            ),
            "Arts Crafts shipping damage broken",
        ),
        (16, ids("SELECT id FROM reviews WHERE rating >= 4 AND lower(doc_text) LIKE '%defect%' LIMIT 10"), "defect defective problem quality"),
        (17, ids("SELECT id FROM reviews WHERE rating <= 2 ORDER BY random() LIMIT 12"), "negative complaint poor quality disappointed"),
        (18, ids("SELECT id FROM reviews WHERE category = 'Baby_Products' ORDER BY random() LIMIT 12"), "Baby_Products infant toddler safety"),
        (19, ids("SELECT id FROM reviews WHERE lower(doc_text) LIKE '%quality%' ORDER BY random() LIMIT 12"), "quality manufacturing defect issue"),
        (20, ids("SELECT id FROM reviews WHERE lower(doc_text) LIKE '%durab%' ORDER BY random() LIMIT 12"), "durability durable lasting quality"),
    ]

    rq_map = {cid: rq for cid, _, rq in cases}
    gt_map = {cid: g for cid, g, _ in cases}

    path = ROOT / "evaluation" / "test_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        qid = item["id"]
        if qid in gt_map and gt_map[qid]:
            item["ground_truth_ids"] = gt_map[qid]
        if qid in rq_map:
            item["retrieval_query"] = rq_map[qid]

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Updated", path)
    con.close()


if __name__ == "__main__":
    main()
