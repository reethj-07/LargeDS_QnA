"""Chat models: Gemini first when configured, Groq as fallback on errors / rate limits."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config import (
    GEMINI_API_KEY,
    GEMINI_INVOKE_RETRY_ATTEMPTS,
    GEMINI_MAX_RETRIES,
    GROQ_API_KEY,
    LLM_PRIMARY,
    LLM_REQUEST_TIMEOUT_S,
    MODEL_ANALYST,
    MODEL_CRITIC,
    MODEL_GEMINI_ANALYST,
    MODEL_GEMINI_CRITIC,
    MODEL_GEMINI_PLANNER,
    MODEL_PLANNER,
)
from src.observability.logger import get_logger, log_event

logger = get_logger(__name__)

ChatRole = Literal["planner", "decomposer", "analyst", "critic"]

_llm_cache: dict[str, Runnable] = {}


def clear_llm_cache() -> None:
    """Clear cached LLM runnables (e.g. in tests)."""
    _llm_cache.clear()


def require_any_llm_key() -> None:
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        raise RuntimeError(
            "No LLM API key set. Add GEMINI_API_KEY or GOOGLE_API_KEY (Google AI Studio) "
            "and/or GROQ_API_KEY to .env — see .env.example."
        )


def _groq_chat(model_name: str, temperature: float) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required for Groq but is empty.")
    kwargs: dict[str, Any] = {
        "groq_api_key": GROQ_API_KEY,
        "model_name": model_name,
        "temperature": temperature,
    }
    if LLM_REQUEST_TIMEOUT_S > 0:
        kwargs["timeout"] = LLM_REQUEST_TIMEOUT_S
    try:
        return ChatGroq(**kwargs)
    except TypeError:
        kwargs.pop("timeout", None)
        return ChatGroq(**kwargs)


def _gemini_chat(model_name: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": model_name,
        "google_api_key": GEMINI_API_KEY,
        "temperature": temperature,
        "max_retries": GEMINI_MAX_RETRIES,
    }
    if LLM_REQUEST_TIMEOUT_S > 0:
        kwargs["timeout"] = LLM_REQUEST_TIMEOUT_S
    try:
        return ChatGoogleGenerativeAI(**kwargs)
    except TypeError:
        kwargs.pop("timeout", None)
        return ChatGoogleGenerativeAI(**kwargs)


def _build_llm(role: ChatRole) -> Runnable:
    """Gemini primary + Groq fallback when both keys exist; otherwise whichever is available."""
    force_groq = LLM_PRIMARY == "groq"
    if force_groq and not GROQ_API_KEY:
        raise RuntimeError("LLM_PRIMARY=groq requires GROQ_API_KEY in .env.")
    use_gemini = bool(GEMINI_API_KEY) and not force_groq

    if role in ("planner", "decomposer"):
        g_model, g_temp = MODEL_GEMINI_PLANNER, 0.1
        q_model, q_temp = MODEL_PLANNER, 0.1
    elif role == "analyst":
        g_model, g_temp = MODEL_GEMINI_ANALYST, 0.2
        q_model, q_temp = MODEL_ANALYST, 0.2
    else:
        g_model, g_temp = MODEL_GEMINI_CRITIC, 0.0
        q_model, q_temp = MODEL_CRITIC, 0.0

    groq_llm = _groq_chat(q_model, q_temp)

    if not use_gemini:
        log_event("llm_routing", {"role": role, "provider": "groq", "model": q_model})
        return groq_llm

    try:
        gemini_llm = _gemini_chat(g_model, g_temp)
    except Exception as e:
        logger.warning("Gemini init failed (%s); using Groq only for role=%s", e, role)
        log_event("llm_routing", {"role": role, "provider": "groq", "reason": "gemini_init_failed"})
        return groq_llm

    if GEMINI_INVOKE_RETRY_ATTEMPTS > 1:
        try:
            from google.api_core.exceptions import (
                ResourceExhausted,
                ServiceUnavailable,
                TooManyRequests,
            )

            gemini_llm = gemini_llm.with_retry(  # type: ignore[assignment]
                retry_if_exception_type=(ResourceExhausted, ServiceUnavailable, TooManyRequests),
                stop_after_attempt=GEMINI_INVOKE_RETRY_ATTEMPTS,
                wait_exponential_jitter=True,
                exponential_jitter_params={"initial": 2.0, "max": 120.0},
            )
        except ImportError:
            logger.warning("google.api_core not available; skipping Gemini invoke-level rate-limit retries")

    if GROQ_API_KEY:
        log_event(
            "llm_routing",
            {"role": role, "provider": "gemini_with_groq_fallback", "gemini_model": g_model},
        )
        return gemini_llm.with_fallbacks([groq_llm])

    log_event("llm_routing", {"role": role, "provider": "gemini", "model": g_model})
    return gemini_llm


def get_chat_llm(role: ChatRole) -> Runnable:
    """Return a cached chat runnable (invoke/ainvoke same as BaseChatModel)."""
    if role not in _llm_cache:
        _llm_cache[role] = _build_llm(role)
    return _llm_cache[role]
