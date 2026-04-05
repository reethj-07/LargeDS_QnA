"""List Gemini model names your API key can use (helps fix 404 NotFound on wrong model id).

Run from repo root (with .env loaded):
  python scripts/list_gemini_models.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
if not KEY:
    print("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

URL = "https://generativelanguage.googleapis.com/v1beta/models"
r = requests.get(URL, params={"key": KEY}, timeout=60)
if not r.ok:
    print(f"HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
    sys.exit(1)

data = r.json()
models = data.get("models") or []
usable = []
for m in models:
    methods = m.get("supportedGenerationMethods") or []
    if "generateContent" in methods:
        name = m.get("name", "")
        # API returns "models/gemini-..."; LangChain wants the short id usually
        short = name.split("/")[-1] if "/" in name else name
        usable.append((short, m.get("displayName", "")))

print(f"Found {len(usable)} models supporting generateContent:\n")
for short, display in sorted(usable, key=lambda x: x[0]):
    print(f"  {short}")
    if display:
        print(f"      ({display})")

print(
    "\nUse these in .env, e.g. MODEL_GEMINI_PLANNER=gemini-2.5-flash "
    "(must match a line above exactly)."
)
