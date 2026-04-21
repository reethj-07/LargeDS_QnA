"""Agent graph and nodes (lazy to keep ``import src.agents`` lightweight)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_graph",
    "get_shared_hybrid_retriever",
    "reset_shared_hybrid_retriever",
    "run_agent_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import src.agents.graph as graph
        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
