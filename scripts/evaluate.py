#!/usr/bin/env python3
"""CLI: run evaluation/evaluate.py and regenerate EVALUATION_REPORT.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.evaluate import main as eval_main
    from evaluation.report_generator import main as report_main

    eval_main()
    report_main()


if __name__ == "__main__":
    main()
