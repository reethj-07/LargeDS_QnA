#!/usr/bin/env python3
"""Generate SQL accuracy evaluation data for aggregation questions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.sql_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
