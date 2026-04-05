"""Analyst: synthesize answer with citations."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.chat import get_chat_llm, require_any_llm_key


SYSTEM = """You are an analyst answering questions using ONLY the provided evidence:
(1) SQL result rows if any,
(2) Retrieved review excerpts.

Rules:
- Cite review ids as [id=123] when you use a review.
- If SQL results are present, use them for numbers (averages, counts).
- When identifying lowest/highest category (or any min/max from SQL): if multiple categories share the same
  value, state explicitly that they are **tied** and list **all** tied categories. Do not name only one
  as "the" lowest/highest unless it is unique.
- If evidence is insufficient, say so clearly.
- Do not invent product names, ratings, or statistics not supported by evidence.
- Keep answers structured and concise unless the user asks for depth.
"""


def run_analyst(
    user_query: str,
    plan: str,
    sql_block: str,
    context_block: str,
) -> str:
    require_any_llm_key()
    llm = get_chat_llm("analyst")
    human = f"""User question: {user_query}

Planner plan:
{plan}

SQL results (if any):
{sql_block}

Retrieved reviews:
{context_block}
"""
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=human),
        ]
    )
    return msg.content if isinstance(msg.content, str) else str(msg.content)
