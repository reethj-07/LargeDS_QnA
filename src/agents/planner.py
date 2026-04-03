"""Planner: classify query and produce execution plan + optional SQL hint."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, MODEL_PLANNER
from src.observability.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You are a planner for an e-commerce analytics Q&A system over Amazon product reviews.
The data has columns: id, asin, category, rating, title, text, doc_text, timestamp, helpful_vote, verified_purchase.
Categories in this corpus include: All_Beauty, Appliances, Amazon_Fashion, Arts_Crafts_and_Sewing, Baby_Products.

Respond with ONLY a JSON object (no markdown) with keys:
- query_type: one of factual, aggregation, comparison, trend, multi_hop
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
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_PLANNER,
        temperature=0.1,
    )
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
    return data
