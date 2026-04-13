#!/usr/bin/env python3
"""Populate ground_truth_ids + retrieval_query + eval_mode in evaluation/test_questions.json.

Ground truth uses **large relevance pools** from deterministic, content-aligned SQL.
Queries use the `text` column (raw review body, no category prefix) so keyword filters
genuinely reflect review content.  Pools are capped at 2000 for JSON size.

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

POOL_CAP = 2000


def main() -> None:
    if not DUCKDB_PATH.exists():
        print("No DuckDB at", DUCKDB_PATH, "- run ingest first.")
        sys.exit(1)

    import duckdb

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    def ids(sql: str, cap: int = POOL_CAP) -> list[int]:
        rows = con.execute(sql).fetchall()
        return [int(r[0]) for r in rows[:cap]]

    top_asin = con.execute(
        "SELECT asin FROM reviews GROUP BY asin ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()[0]
    top_asin_esc = top_asin.replace("'", "''")

    row = con.execute(
        """
        SELECT asin FROM reviews
        GROUP BY asin
        HAVING max(rating) >= 5 AND min(rating) <= 1
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()
    split_asin = row[0].replace("'", "''") if row else top_asin_esc

    # ── Ground truth: large pools, content-aligned, deterministic ─────────
    # Uses `text` column (raw review body) not `doc_text` (has category prefix)
    # eval_mode: "retrieval" | "sql" | "both"
    cases: list[tuple[int, list[int], str, str]] = [
        # ── SIMPLE ────────────────────────────────────────────────────────
        # Q1: avg rating in All_Beauty → SQL primary; retrieval secondary
        #   Pool = All_Beauty reviews whose TEXT mentions beauty-related words
        (1,
         ids(
             "SELECT id FROM reviews WHERE category = 'All_Beauty' "
             "AND (lower(text) LIKE '%beauty%' OR lower(text) LIKE '%skin%' "
             "     OR lower(text) LIKE '%hair%' OR lower(text) LIKE '%cream%') "
             "ORDER BY id"
         ),
         "All_Beauty beauty skin hair cream product rating",
         "both"),

        # Q2: reviews mentioning moisturizer → full content match
        (2,
         ids(
             "SELECT id FROM reviews "
             "WHERE lower(text) LIKE '%moisturizer%' OR lower(title) LIKE '%moisturizer%' "
             "ORDER BY id"
         ),
         "moisturizer cream lotion skincare hydrating",
         "retrieval"),

        # Q3: ASIN with most reviews → all reviews for that ASIN
        (3,
         ids(f"SELECT id FROM reviews WHERE asin = '{top_asin_esc}' ORDER BY id"),
         f"{top_asin} product reviews",
         "both"),

        # Q4: 5-star fragrance reviews → content match
        (4,
         ids(
             "SELECT id FROM reviews WHERE rating >= 5 "
             "AND (lower(text) LIKE '%fragrance%' OR lower(text) LIKE '%scent%' "
             "     OR lower(text) LIKE '%perfume%' OR lower(text) LIKE '%smell%') "
             "ORDER BY id"
         ),
         "fragrance scent perfume smell five star amazing",
         "retrieval"),

        # ── AGGREGATION ──────────────────────────────────────────────────
        # Q5: verified 5-star % → SQL
        (5,
         ids(
             "SELECT id FROM reviews WHERE verified_purchase = true AND rating = 5 "
             "AND (lower(text) LIKE '%great%' OR lower(text) LIKE '%excellent%' "
             "     OR lower(text) LIKE '%amazing%' OR lower(text) LIKE '%love%') "
             "ORDER BY id"
         ),
         "verified purchase five star great excellent love",
         "sql"),

        # Q6: highest rated category → SQL
        (6,
         ids(
             "SELECT id FROM reviews WHERE category = ("
             "  SELECT category FROM reviews GROUP BY category "
             "  ORDER BY AVG(rating) DESC LIMIT 1"
             ") AND (lower(text) LIKE '%good%' OR lower(text) LIKE '%great%' "
             "        OR lower(text) LIKE '%quality%') "
             "ORDER BY id"
         ),
         "highest rated category best quality average rating",
         "sql"),

        # Q7: most helpful reviews → top by helpful_vote
        (7,
         ids(
             "SELECT id FROM reviews WHERE helpful_vote >= 5 "
             "ORDER BY helpful_vote DESC"
         ),
         "helpful useful informative review recommendation",
         "both"),

        # Q8: rating distribution → SQL
        (8,
         ids(
             "SELECT id FROM ("
             "  SELECT id, ROW_NUMBER() OVER (PARTITION BY CAST(rating AS INT) ORDER BY id) AS rn "
             "  FROM reviews"
             ") sub WHERE rn <= 400 ORDER BY id"
         ),
         "rating distribution one two three four five star count",
         "sql"),

        # ── COMPARISON ───────────────────────────────────────────────────
        # Q9: All_Beauty vs Appliances
        (9,
         ids(
             "(SELECT id FROM reviews WHERE category = 'All_Beauty' "
             " AND (lower(text) LIKE '%beauty%' OR lower(text) LIKE '%skin%' "
             "      OR lower(text) LIKE '%product%') ORDER BY id LIMIT 1000) "
             "UNION ALL "
             "(SELECT id FROM reviews WHERE category = 'Appliances' "
             " AND (lower(text) LIKE '%appliance%' OR lower(text) LIKE '%kitchen%' "
             "      OR lower(text) LIKE '%machine%') ORDER BY id LIMIT 1000)"
         ),
         "All_Beauty Appliances compare rating average quality",
         "both"),

        # Q10: Fashion vs Baby_Products
        (10,
         ids(
             "(SELECT id FROM reviews WHERE category = 'Amazon_Fashion' "
             " AND (lower(text) LIKE '%fashion%' OR lower(text) LIKE '%clothing%' "
             "      OR lower(text) LIKE '%wear%' OR lower(text) LIKE '%fit%') "
             " ORDER BY id LIMIT 1000) "
             "UNION ALL "
             "(SELECT id FROM reviews WHERE category = 'Baby_Products' "
             " AND (lower(text) LIKE '%baby%' OR lower(text) LIKE '%infant%' "
             "      OR lower(text) LIKE '%child%') "
             " ORDER BY id LIMIT 1000)"
         ),
         "Amazon_Fashion Baby_Products clothing baby sentiment compare",
         "both"),

        # Q11: verified vs non-verified rating
        (11,
         ids(
             "(SELECT id FROM reviews WHERE verified_purchase = true "
             " AND (lower(text) LIKE '%quality%' OR lower(text) LIKE '%recommend%') "
             " ORDER BY id LIMIT 1000) "
             "UNION ALL "
             "(SELECT id FROM reviews WHERE verified_purchase = false "
             " AND (lower(text) LIKE '%quality%' OR lower(text) LIKE '%recommend%') "
             " ORDER BY id LIMIT 1000)"
         ),
         "verified purchase non-verified rating average quality recommend",
         "both"),

        # Q12: 1-star reviews with negative sentiment
        (12,
         ids(
             "SELECT id FROM reviews WHERE rating = 1 "
             "AND (lower(text) LIKE '%terrible%' OR lower(text) LIKE '%worst%' "
             "     OR lower(text) LIKE '%horrible%' OR lower(text) LIKE '%bad%' "
             "     OR lower(text) LIKE '%waste%' OR lower(text) LIKE '%awful%') "
             "ORDER BY id"
         ),
         "one star negative terrible worst horrible bad waste review",
         "both"),

        # ── MULTI-HOP ───────────────────────────────────────────────────
        # Q13: ASIN with both 5-star and 1-star
        (13,
         ids(f"SELECT id FROM reviews WHERE asin = '{split_asin}' ORDER BY id"),
         f"{split_asin} reviews mixed ratings five star one star",
         "both"),

        # Q14: complaints / defects across reviews
        (14,
         ids(
             "SELECT id FROM reviews "
             "WHERE lower(text) LIKE '%defect%' OR lower(text) LIKE '%broke%' "
             "   OR lower(text) LIKE '%broken%' OR lower(text) LIKE '%complaint%' "
             "   OR lower(text) LIKE '%malfunction%' "
             "ORDER BY id"
         ),
         "complaint defect broke broken malfunction quality issue recurring",
         "retrieval"),

        # Q15: Arts_Crafts + shipping damage
        (15,
         ids(
             "SELECT id FROM reviews WHERE category = 'Arts_Crafts_and_Sewing' "
             "AND (lower(text) LIKE '%shipping%' OR lower(text) LIKE '%damage%' "
             "     OR lower(text) LIKE '%broken%' OR lower(text) LIKE '%arrived%') "
             "ORDER BY id"
         ),
         "Arts Crafts Sewing shipping damage broken arrived package",
         "retrieval"),

        # Q16: high-rated reviews mentioning defects
        (16,
         ids(
             "SELECT id FROM reviews WHERE rating >= 4 "
             "AND (lower(text) LIKE '%defect%' OR lower(text) LIKE '%problem%' "
             "     OR lower(text) LIKE '%issue%' OR lower(text) LIKE '%flaw%' "
             "     OR lower(text) LIKE '%broken%') "
             "ORDER BY id"
         ),
         "defect defective problem issue flaw broken high rated despite good",
         "retrieval"),

        # ── TREND ────────────────────────────────────────────────────────
        # Q17: 1-star Appliances themes
        (17,
         ids(
             "SELECT id FROM reviews WHERE category = 'Appliances' AND rating <= 2 "
             "AND (lower(text) LIKE '%broke%' OR lower(text) LIKE '%terrible%' "
             "     OR lower(text) LIKE '%disappointed%' OR lower(text) LIKE '%waste%' "
             "     OR lower(text) LIKE '%return%' OR lower(text) LIKE '%defect%') "
             "ORDER BY id"
         ),
         "Appliances one star broke terrible disappointed waste return defect",
         "retrieval"),

        # Q18: Baby_Products strengths/weaknesses
        (18,
         ids(
             "SELECT id FROM reviews WHERE category = 'Baby_Products' "
             "AND (lower(text) LIKE '%safe%' OR lower(text) LIKE '%sturdy%' "
             "     OR lower(text) LIKE '%quality%' OR lower(text) LIKE '%recommend%' "
             "     OR lower(text) LIKE '%flimsy%' OR lower(text) LIKE '%broke%') "
             "ORDER BY id"
         ),
         "Baby_Products baby infant toddler safe quality sturdy recommend",
         "retrieval"),

        # Q19: quality issues across products
        (19,
         ids(
             "SELECT id FROM reviews "
             "WHERE lower(text) LIKE '%quality%' "
             "  AND (lower(text) LIKE '%issue%' OR lower(text) LIKE '%problem%' "
             "       OR lower(text) LIKE '%defect%' OR lower(text) LIKE '%poor%') "
             "ORDER BY id"
         ),
         "quality issue problem defect poor manufacturing",
         "retrieval"),

        # Q20: durability mentions
        (20,
         ids(
             "SELECT id FROM reviews "
             "WHERE lower(text) LIKE '%durabl%' OR lower(text) LIKE '%sturdy%' "
             "   OR lower(text) LIKE '%long lasting%' OR lower(text) LIKE '%well built%' "
             "   OR lower(text) LIKE '%well made%' "
             "ORDER BY id"
         ),
         "durability durable sturdy long lasting well built quality",
         "retrieval"),
    ]

    rq_map = {cid: rq for cid, _, rq, _ in cases}
    gt_map = {cid: g for cid, g, _, _ in cases}
    em_map = {cid: em for cid, _, _, em in cases}

    path = ROOT / "evaluation" / "test_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        qid = item["id"]
        if qid in gt_map and gt_map[qid]:
            item["ground_truth_ids"] = gt_map[qid]
        if qid in rq_map:
            item["retrieval_query"] = rq_map[qid]
        if qid in em_map:
            item["eval_mode"] = em_map[qid]

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated {path}")
    total_gt = 0
    for cid, g, _, em in cases:
        total_gt += len(g)
        print(f"  Q{cid:>2}: {len(g):>5} ground truth ids  (eval_mode={em})")
    print(f"  Total ground truth docs: {total_gt}")
    con.close()


if __name__ == "__main__":
    main()
