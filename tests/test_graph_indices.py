"""Graph entry behaviour when indices are missing."""

from __future__ import annotations

from unittest.mock import patch

from src.agents.graph import run_agent_pipeline


def test_pipeline_short_circuits_when_indices_missing() -> None:
    with patch("src.agents.graph.indices_artifacts_ready", return_value=False):
        out = run_agent_pipeline("What is the average rating?")
    assert out.get("error") == "indices_missing"
    assert "ingest" in (out.get("answer") or "").lower()
