#!/usr/bin/env python3
"""CLI: run a single query through the agent pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.graph import run_agent_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default=None, help="Question text")
    p.add_argument("-q", "--question", default=None, help="Question text (alternative)")
    args = p.parse_args()
    q = args.question or args.query
    if not q:
        print("Usage: python scripts/query.py \"Your question\"")
        sys.exit(1)
    out = run_agent_pipeline(q)
    print("=== Answer ===\n")
    print(out.get("answer", ""))
    print("\n=== Trace ===\n")
    for line in out.get("agent_trace", []):
        print(line)
    print("\n=== Confidence ===", out.get("confidence"))
    print("=== Critique ===", out.get("critique", "")[:500])


if __name__ == "__main__":
    main()
