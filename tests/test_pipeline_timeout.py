"""Wall-clock timeout around LangGraph invoke."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.agents.graph import run_agent_pipeline


@patch("src.agents.graph.indices_artifacts_ready", return_value=True)
@patch("src.agents.graph.build_graph")
def test_pipeline_timeout_returns_error_when_invoke_slow(
    mock_build: MagicMock,
    _mock_indices: MagicMock,
) -> None:
    app = MagicMock()

    def _slow(_state: object) -> dict:
        time.sleep(2.0)
        return {"answer": "never", "agent_trace": []}

    app.invoke.side_effect = _slow
    mock_build.return_value = app

    hybrid = MagicMock()
    out = run_agent_pipeline("hello", hybrid=hybrid, pipeline_timeout_s=0.25)

    assert out.get("error") == "pipeline_timeout"
    assert "time limit" in (out.get("answer") or "").lower()
