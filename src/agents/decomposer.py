"""Decomposer: break complex / multi-hop queries into 2-4 independent sub-questions."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.chat import get_chat_llm, require_any_llm_key
from src.observability.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You decompose complex questions about Amazon product reviews into 2-4 simpler sub-questions.
Each sub-question should be independently answerable with either:
  - A SQL aggregation on table `reviews` using ONLY: id, asin, category, rating, title, text,
    doc_text, timestamp, helpful_vote, verified_purchase (no buyer_id, user_id, or other columns), OR
  - A semantic / keyword search over review text.

Return ONLY a JSON object (no markdown):
{
  "sub_questions": [
    {"question": "...", "strategy": "sql" | "retrieval" | "both"},
    ...
  ]
}

Guidelines:
- Keep sub-questions atomic: one fact, one comparison leg, or one aggregation each.
- For comparisons (A vs B), create one sub-question per side.
- 2 sub-questions is usually enough; use 3-4 only for genuinely complex queries.
"""


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON in decomposer output")
    return json.loads(m.group(0))


def run_decomposer(user_query: str, plan: str) -> list[dict[str, Any]]:
    """Return list of {"question": str, "strategy": str}."""
    require_any_llm_key()
    llm = get_chat_llm("decomposer")
    human = f"Original question: {user_query}\nPlanner plan: {plan}"
    msg = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=human)])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        data = _extract_json(content)
        subs = data.get("sub_questions") or []
        if not isinstance(subs, list) or len(subs) < 1:
            raise ValueError("Empty sub_questions list")
        return [
            {
                "question": str(s.get("question", user_query)),
                "strategy": str(s.get("strategy", "both")),
            }
            for s in subs[:4]
        ]
    except Exception as e:
        logger.warning("Decomposer parse failed (%s); falling back to single question", e)
        return [{"question": user_query, "strategy": "both"}]
