"""Unit tests for planner route derivation (no LLM calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


@patch("src.agents.planner.get_chat_llm")
def test_planner_returns_route_direct(mock_get_llm):
    payload = {
        "query_type": "factual",
        "plan": "1) search 2) answer",
        "sql_suggestion": "",
        "search_query": "product quality",
        "route": "direct",
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    from src.agents.planner import run_planner
    result = run_planner("What is the best product?")
    assert result["route"] == "direct"


@patch("src.agents.planner.get_chat_llm")
def test_planner_derives_route_from_query_type(mock_get_llm):
    payload = {
        "query_type": "aggregation",
        "plan": "1) SQL 2) answer",
        "sql_suggestion": "SELECT AVG(rating) FROM reviews",
        "search_query": "average rating",
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    from src.agents.planner import run_planner
    result = run_planner("What is the average rating?")
    assert result["route"] == "sql_first"


@patch("src.agents.planner.get_chat_llm")
def test_planner_derives_decompose_for_multi_hop(mock_get_llm):
    payload = {
        "query_type": "multi_hop",
        "plan": "1) break down 2) retrieve 3) merge",
        "sql_suggestion": "",
        "search_query": "themes across categories",
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    from src.agents.planner import run_planner
    result = run_planner("Compare themes across categories")
    assert result["route"] == "decompose"
