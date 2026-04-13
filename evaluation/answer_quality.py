"""Batch answer quality: run all test questions through the full agent pipeline,
collect critic scores, and append results to eval cache."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CLI flag parsed before imports (so config sees Groq model choice).
if "--use-70b" in sys.argv:
    os.environ["ANSWER_QUALITY_USE_70B"] = "1"

# Before src imports: avoid cross-encoder load (matches CI, prevents huge prompts).
os.environ.setdefault("USE_CROSS_ENCODER", "0")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Groq free tier: llama-3.3-70b has a small tokens-per-day budget; batch runs 20 queries ×
# (planner + analyst + critic + …) and exhausts TPD if analyst+critic stay on 70b.
# Default this script to 8B for those roles on Groq unless ANSWER_QUALITY_USE_70B=1 or --use-70b.
if os.getenv("ANSWER_QUALITY_USE_70B", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["MODEL_ANALYST"] = "llama-3.1-8b-instant"
    os.environ["MODEL_CRITIC"] = "llama-3.1-8b-instant"

from src.agents.graph import run_agent_pipeline  # noqa: E402
from src.config import DUCKDB_PATH, MODEL_ANALYST, MODEL_CRITIC  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.storage.sql_store import SqlStore  # noqa: E402


def _transient_groq_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if type(exc).__name__ == "RateLimitError":
        return True
    return (
        "429" in msg
        or "413" in msg
        or "rate_limit" in msg
        or "too large" in msg
        or "tokens per minute" in msg
    )


def run_answer_quality(
    questions_path: Path,
    out_path: Path,
) -> list[dict]:
    qs = json.loads(questions_path.read_text(encoding="utf-8"))
    print(
        f"Groq models (analyst/critic): {MODEL_ANALYST}, {MODEL_CRITIC} "
        "(set ANSWER_QUALITY_USE_70B=1 or --use-70b for .env defaults)\n",
        flush=True,
    )
    # Read-only DuckDB allows running while the UI or another script holds a write connection.
    hybrid = HybridRetriever(
        embedder=Embedder(),
        sql_store=SqlStore(DUCKDB_PATH, read_only=True),
    )
    hybrid.load_indices()

    delay = float(os.getenv("ANSWER_QUALITY_DELAY_SEC", "1.25").strip() or "1.25")
    max_attempts = max(1, int(os.getenv("ANSWER_QUALITY_MAX_ATTEMPTS", "4").strip() or "4"))

    results: list[dict] = []
    for item in qs:
        qid = item["id"]
        q = item["question"]
        print(f"  Q{qid}: {q[:80]}...", flush=True)
        t0 = time.time()
        try:
            out = None
            last_err: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    out = run_agent_pipeline(q, hybrid=hybrid)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts - 1 and _transient_groq_error(e):
                        wait = min(32.0, (2.0**attempt) + delay)
                        time.sleep(wait)
                        continue
                    raise
            if out is None and last_err is not None:
                raise last_err
            assert out is not None
            elapsed = time.time() - t0
            results.append({
                "id": qid,
                "tier": item.get("tier"),
                "question": q,
                "answer_len": len(out.get("answer", "")),
                "critic_score": out.get("confidence"),
                "critique": (out.get("critique") or "")[:300],
                "retries": out.get("retry_count", 0),
                "latency_s": round(elapsed, 2),
                "query_type": out.get("query_type"),
                "sql_rows": len(out.get("sql_results") or []),
                "retrieved_docs": len(out.get("retrieved_docs") or []),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Q{qid} ERROR: {type(e).__name__}: {e!s}", flush=True)
            if not _transient_groq_error(e):
                traceback.print_exc()
            results.append({
                "id": qid,
                "tier": item.get("tier"),
                "question": q,
                "error": f"{type(e).__name__}: {str(e)[:400]}",
                "latency_s": round(elapsed, 2),
            })

        time.sleep(delay)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    scores = [r["critic_score"] for r in results if r.get("critic_score") is not None]
    if scores:
        print(f"Questions scored: {len(scores)}/{len(results)}")
        print(f"Mean critic score: {sum(scores)/len(scores):.2f}")
        print(f"Min: {min(scores):.1f}  Max: {max(scores):.1f}")
        print(f"Score >= 4: {sum(1 for s in scores if s >= 4)}/{len(scores)}")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"Errors: {len(errors)}")
    return results


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Batch answer quality via full agent pipeline + critic")
    p.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "test_questions.json")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cache" / "answer_quality.json")
    p.add_argument(
        "--use-70b",
        action="store_true",
        help="Use MODEL_ANALYST / MODEL_CRITIC from .env (often 70b). Must appear on argv before imports; same as ANSWER_QUALITY_USE_70B=1.",
    )
    args = p.parse_args()
    run_answer_quality(args.questions, args.out)


if __name__ == "__main__":
    main()
