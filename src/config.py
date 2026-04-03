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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_PLANNER = os.getenv("MODEL_PLANNER", "llama-3.1-8b-instant")
MODEL_ANALYST = os.getenv("MODEL_ANALYST", "llama-3.3-70b-versatile")
MODEL_CRITIC = os.getenv("MODEL_CRITIC", "llama-3.3-70b-versatile")

# Retrieval
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Agent
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
CRITIC_PASS_THRESHOLD = float(os.getenv("CRITIC_PASS_THRESHOLD", "3.0"))
