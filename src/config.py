"""Central configuration loaded from environment and defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDICES_DIR = DATA_DIR / "indices"

for d in (RAW_DIR, PROCESSED_DIR, INDICES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset (JSONL files under raw/review_categories/)
HF_DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_JSONL_PREFIX = "raw/review_categories"

# Five categories × 30K = 150K rows (plan)
DEFAULT_CATEGORIES: list[tuple[str, str]] = [
    ("All_Beauty", "All_Beauty.jsonl"),
    ("Appliances", "Appliances.jsonl"),
    ("Amazon_Fashion", "Amazon_Fashion.jsonl"),
    ("Arts_Crafts_and_Sewing", "Arts_Crafts_and_Sewing.jsonl"),
    ("Baby_Products", "Baby_Products.jsonl"),
]
ROWS_PER_CATEGORY = 30_000

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))

# FAISS
FAISS_INDEX_PATH = INDICES_DIR / "faiss.index"
FAISS_META_PATH = INDICES_DIR / "faiss_meta.json"
BM25_PATH = INDICES_DIR / "bm25.pkl"
DUCKDB_PATH = INDICES_DIR / "reviews.duckdb"

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_PLANNER = os.getenv("MODEL_PLANNER", "llama-3.1-8b-instant")
MODEL_ANALYST = os.getenv("MODEL_ANALYST", "llama-3.3-70b-versatile")
MODEL_CRITIC = os.getenv("MODEL_CRITIC", "llama-3.3-70b-versatile")

# Google Gemini (Google AI Studio). Accept GEMINI_API_KEY or GOOGLE_API_KEY.
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
# auto = use Gemini when GEMINI_API_KEY/GOOGLE_API_KEY is set, else Groq; groq = always Groq
LLM_PRIMARY = os.getenv("LLM_PRIMARY", "auto").strip().lower()
# Use current API model IDs (gemini-1.5-* often 404 on v1beta; see ai.google.dev/gemini-api/docs/models)
MODEL_GEMINI_PLANNER = os.getenv("MODEL_GEMINI_PLANNER", "gemini-2.5-flash")
MODEL_GEMINI_ANALYST = os.getenv("MODEL_GEMINI_ANALYST", "gemini-2.5-pro")
MODEL_GEMINI_CRITIC = os.getenv("MODEL_GEMINI_CRITIC", "gemini-2.5-flash")
# LangChain Gemini client retries ResourceExhausted / ServiceUnavailable / GoogleAPIError up to N times with exponential backoff.
# Default 6 matches langchain-google-genai; use 1–2 only if you want Groq fallback sooner (weaker on 429 / quota bursts).
GEMINI_MAX_RETRIES = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "6")))
# Extra full invoke-level retries (LangChain RunnableRetry) after inner retries exhaust; helps batch jobs between quota windows.
GEMINI_INVOKE_RETRY_ATTEMPTS = max(1, int(os.getenv("GEMINI_INVOKE_RETRY_ATTEMPTS", "3")))

# Retrieval
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))
# RRF pool before optional cross-encoder rerank (CPU-friendly small model)
RERANK_POOL = int(os.getenv("RERANK_POOL", "40"))
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "1").strip().lower() in ("1", "true", "yes")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Agent
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
CRITIC_PASS_THRESHOLD = float(os.getenv("CRITIC_PASS_THRESHOLD", "3.0"))
