"""Tests for LLM routing (no real API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.llm.chat as chat_mod


def test_require_any_llm_key_raises_when_no_keys():
    with patch.object(chat_mod, "GEMINI_API_KEY", ""), patch.object(chat_mod, "GROQ_API_KEY", ""):
        with pytest.raises(RuntimeError, match="No LLM API key"):
            chat_mod.require_any_llm_key()


def test_require_any_llm_key_ok_with_groq_only():
    with patch.object(chat_mod, "GEMINI_API_KEY", ""), patch.object(chat_mod, "GROQ_API_KEY", "gsk_test"):
        chat_mod.require_any_llm_key()


def test_require_any_llm_key_ok_with_gemini_only():
    with patch.object(chat_mod, "GEMINI_API_KEY", "gemini_key"), patch.object(chat_mod, "GROQ_API_KEY", ""):
        chat_mod.require_any_llm_key()


@patch.object(chat_mod, "_groq_chat")
@patch.object(chat_mod, "_gemini_chat")
def test_build_llm_gemini_with_groq_fallback(mock_gemini, mock_groq):
    chat_mod.clear_llm_cache()
    with patch.object(chat_mod, "GEMINI_INVOKE_RETRY_ATTEMPTS", 1):
        with patch.object(chat_mod, "GEMINI_API_KEY", "g"), patch.object(chat_mod, "GROQ_API_KEY", "q"):
            with patch.object(chat_mod, "LLM_PRIMARY", "auto"):
                gemini_llm = MagicMock()
                gemini_llm.with_fallbacks = MagicMock(return_value="combined")
                mock_gemini.return_value = gemini_llm
                mock_groq.return_value = "groq_inst"

                out = chat_mod._build_llm("planner")
                assert out == "combined"
                gemini_llm.with_fallbacks.assert_called_once_with(["groq_inst"])


@patch.object(chat_mod, "_groq_chat")
def test_build_llm_groq_only_when_no_gemini_key(mock_groq):
    chat_mod.clear_llm_cache()
    with patch.object(chat_mod, "GEMINI_API_KEY", ""), patch.object(chat_mod, "GROQ_API_KEY", "q"):
        with patch.object(chat_mod, "LLM_PRIMARY", "auto"):
            mock_groq.return_value = "groq_inst"
            out = chat_mod._build_llm("analyst")
            assert out == "groq_inst"


@patch.object(chat_mod, "_groq_chat")
def test_build_llm_force_groq_skips_gemini(mock_groq):
    chat_mod.clear_llm_cache()
    with patch.object(chat_mod, "GEMINI_API_KEY", "g"), patch.object(chat_mod, "GROQ_API_KEY", "q"):
        with patch.object(chat_mod, "LLM_PRIMARY", "groq"):
            mock_groq.return_value = "groq_inst"
            out = chat_mod._build_llm("planner")
            assert out == "groq_inst"
