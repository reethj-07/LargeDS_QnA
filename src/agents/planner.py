"""Planner: classify query and produce execution plan + optional SQL hint."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.chat import get_chat_llm, require_any_llm_key
from src.observability.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You are a planner for an e-commerce analytics Q&A system over Amazon product reviews.

Table `reviews` has EXACTLY these columns (no others exist):
id, asin, category, rating, title, text, doc_text, timestamp, helpful_vote, verified_purchase

SQL rules:
- Use ONLY the columns above. Never invent buyer_id, user_id, customer_id, sku, revenue, order_id, etc.
- "Multiple buyers / reviewers for the same product" means multiple review rows per asin: use COUNT(*) or COUNT(DISTINCT id) grouped by asin, not buyer_id.
- If you cannot answer with this schema, set sql_suggestion to empty string and rely on search_query + retrieval.

Categories in this corpus include: All_Beauty, Appliances, Amazon_Fashion, Arts_Crafts_and_Sewing, Baby_Products.

Respond with ONLY a JSON object (no markdown) with keys:
- query_type: one of factual, aggregation, comparison, trend, multi_hop
- route: one of "direct", "decompose", "sql_first"
    "direct"    = simple factual or single-topic lookup, one retrieval pass suffices
    "decompose" = multi-hop, comparison, or complex question requiring sub-questions
    "sql_first" = aggregation / counting / stats that should run SQL before retrieval
- plan: 2-5 bullet steps as a single string
- sql_suggestion: either empty string OR a single safe DuckDB SELECT query using table reviews ONLY, if the user needs aggregates/filters/counts. Use only SELECT. No semicolons. If unsure, use empty string.
- search_query: a short English query optimized for semantic + keyword retrieval over review text (combine key entities).
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in planner output")
    return json.loads(m.group(0))


def run_planner(user_query: str) -> dict[str, Any]:
    require_any_llm_key()
    llm = get_chat_llm("planner")
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=user_query),
        ]
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        data = _extract_json(content)
    except Exception as e:
        logger.warning("Planner JSON parse failed: %s; raw=%s", e, content[:500])
        data = {
            "query_type": "multi_hop",
            "plan": "1) Retrieve relevant reviews. 2) Synthesize answer from evidence.",
            "sql_suggestion": "",
            "search_query": user_query,
        }
    data.setdefault("query_type", "multi_hop")
    data.setdefault("plan", "")
    data.setdefault("sql_suggestion", "")
    data.setdefault("search_query", user_query)

    # Derive route if the LLM didn't return one
    if "route" not in data or data["route"] not in ("direct", "decompose", "sql_first"):
        qt = data["query_type"]
        if qt in ("aggregation",):
            data["route"] = "sql_first"
        elif qt in ("multi_hop", "comparison"):
            data["route"] = "decompose"
        else:
            data["route"] = "direct"
    return data
