"""Minimal production-style JSON API over the agent pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.graph import run_agent_pipeline
from src.config import indices_artifacts_ready
from src.observability.logger import get_logger, log_event
from src.observability.tracing import new_trace_id

logger = get_logger(__name__)

_STATIC = Path(__file__).resolve().parent / "static"


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=16_000)


def create_app() -> FastAPI:
    app = FastAPI(title="Big Data Q&A API", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        ready = indices_artifacts_ready()
        return {"ready": ready, "indices": ready}

    @app.post("/v1/query")
    def query_v1(body: QueryBody) -> dict[str, Any]:
        tid = new_trace_id()
        log_event("api_query", {"trace_id": tid, "q_len": len(body.query)})
        try:
            out = run_agent_pipeline(body.query.strip(), trace_id=tid)
        except Exception as e:
            logger.exception("api_query_failed trace_id=%s", tid)
            log_event("api_query_error", {"trace_id": tid, "error": str(e)[:500]})
            raise HTTPException(
                status_code=500,
                detail={"trace_id": tid, "error": str(e), "type": type(e).__name__},
            ) from e
        # Return a JSON-serializable subset for clients
        return {
            "trace_id": tid,
            "answer": out.get("answer", ""),
            "query_type": out.get("query_type"),
            "route": out.get("route"),
            "confidence": out.get("confidence"),
            "critique": out.get("critique"),
            "error": out.get("error"),
            "latency_hint": "see server logs for pipeline_query",
            "retrieved_doc_count": len(out.get("retrieved_docs") or []),
        }

    if _STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

        @app.get("/")
        def spa_index() -> FileResponse:
            index = _STATIC / "index.html"
            if not index.is_file():
                raise HTTPException(status_code=404)
            return FileResponse(index)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
