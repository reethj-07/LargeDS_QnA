#!/usr/bin/env python3
"""Regenerate requirements.lock from requirements.in (requires pip-tools)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        str(ROOT / "requirements.in"),
        "-o",
        str(ROOT / "requirements.lock"),
        "--resolver=backtracking",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
