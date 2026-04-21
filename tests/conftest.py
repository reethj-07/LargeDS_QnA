"""Pytest hooks (avoid importing the full agent graph at collection time)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_shared_hybrid_at_session_end() -> None:
    """Release DuckDB connection held by the process-wide HybridRetriever singleton."""
    yield
    try:
        from src.agents.graph import reset_shared_hybrid_retriever

        reset_shared_hybrid_retriever()
    except Exception:
        pass
