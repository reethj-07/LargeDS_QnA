"""Unit tests for agents.decomposer (parsing / fallback logic, no LLM calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agents.decomposer import _extract_json, run_decomposer


def test_extract_json_valid():
    raw = '{"sub_questions": [{"question": "Q1", "strategy": "sql"}]}'
    data = _extract_json(raw)
    assert data["sub_questions"][0]["question"] == "Q1"


def test_extract_json_with_wrapping_text():
    raw = 'Here is the plan:\n{"sub_questions": [{"question": "A", "strategy": "both"}]}\nDone.'
    data = _extract_json(raw)
    assert len(data["sub_questions"]) == 1


def test_extract_json_no_json_raises():
    import pytest
    with pytest.raises(ValueError, match="No JSON"):
        _extract_json("no json here")


@patch("src.agents.decomposer.get_chat_llm")
def test_run_decomposer_parses_llm_output(mock_get_llm):
    payload = {
        "sub_questions": [
            {"question": "What is avg rating?", "strategy": "sql"},
            {"question": "Common themes in low-rated?", "strategy": "retrieval"},
        ]
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    result = run_decomposer("Compare ratings and themes", "some plan")
    assert len(result) == 2
    assert result[0]["question"] == "What is avg rating?"
    assert result[1]["strategy"] == "retrieval"


@patch("src.agents.decomposer.get_chat_llm")
def test_run_decomposer_fallback_on_bad_json(mock_get_llm):
    mock_msg = MagicMock()
    mock_msg.content = "I cannot parse this"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    result = run_decomposer("Some query", "plan")
    assert len(result) == 1
    assert result[0]["question"] == "Some query"
    assert result[0]["strategy"] == "both"


@patch("src.agents.decomposer.get_chat_llm")
def test_run_decomposer_caps_at_four(mock_get_llm):
    payload = {
        "sub_questions": [
            {"question": f"Q{i}", "strategy": "both"} for i in range(8)
        ]
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_get_llm.return_value = mock_llm

    result = run_decomposer("Big query", "plan")
    assert len(result) == 4
