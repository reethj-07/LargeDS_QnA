"""Critic: score answer quality 1-5 and give short feedback."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, CRITIC_PASS_THRESHOLD, MODEL_CRITIC
from src.observability.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You evaluate answers for grounding in provided evidence.
Return ONLY JSON: {"score": <1-5 float>, "critique": "<short>", "needs_retry": <true|false>}
Score rubric:
5 = fully grounded, complete
4 = mostly grounded, minor gaps
3 = partially grounded or incomplete
2 = significant hallucination risk or missing key evidence
1 = not grounded

Set needs_retry true only if score < 3 AND more retrieval could help.
"""


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON in critic output")
    return json.loads(m.group(0))


def run_critic(
    user_query: str,
    answer: str,
    evidence_summary: str,
) -> tuple[float, str, bool]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_CRITIC,
        temperature=0.0,
    )
    human = f"""Question: {user_query}

Proposed answer:
{answer}

Evidence available (excerpt):
{evidence_summary[:12000]}
"""
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=human),
        ]
    )
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        data = _extract_json(content)
    except Exception as e:
        logger.warning("Critic parse failed: %s", e)
        return 3.0, "Critic parse failed; accepting answer.", False
    score = float(data.get("score", 3))
    critique = str(data.get("critique", ""))
    needs = bool(data.get("needs_retry", False)) and score < CRITIC_PASS_THRESHOLD
    return score, critique, needs
