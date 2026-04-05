"""Synthesizer: merge per-sub-question retrieval results into unified context."""

from __future__ import annotations

import json
from typing import Any


def merge_sub_results(
    sub_answers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, float]], str]:
    """Combine docs, sql_results, ranked, and context bundles from sub-question retrievals.

    Returns (merged_docs, merged_sql, merged_ranked, unified_context_bundle).
    """
    all_docs: list[dict[str, Any]] = []
    all_sql: list[dict[str, Any]] = []
    all_ranked: list[tuple[int, float]] = []
    seen_doc_ids: set[int] = set()
    ctx_parts: list[str] = []

    for idx, sa in enumerate(sub_answers):
        sq = sa.get("question", f"Sub-question {idx + 1}")
        ctx_parts.append(f"=== Sub-question {idx + 1}: {sq} ===")

        for doc in sa.get("docs", []):
            did = int(doc.get("id", -1))
            if did not in seen_doc_ids:
                seen_doc_ids.add(did)
                all_docs.append(doc)

        for rid, score in sa.get("ranked", []):
            if int(rid) not in {r for r, _ in all_ranked}:
                all_ranked.append((int(rid), float(score)))

        sql_rows = sa.get("sql_results", [])
        if sql_rows:
            all_sql.extend(sql_rows)
            ctx_parts.append(
                "SQL results:\n" + json.dumps(sql_rows[:50], indent=2, default=str)[:4000]
            )

        bundle = sa.get("bundle", "")
        if bundle:
            ctx_parts.append(bundle[:6000])

        ctx_parts.append("")

    unified = "\n\n".join(ctx_parts)
    return all_docs, all_sql, all_ranked, unified
