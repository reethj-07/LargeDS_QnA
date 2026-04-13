"""Retrieval tool wrappers for the retriever step (programmatic, not LLM tool-calling)."""

from __future__ import annotations

from typing import Any

from src.config import REVIEW_CATEGORY_NAMES
from src.retrieval.hybrid_retriever import HybridRetriever


def format_doc_for_context(doc: dict[str, Any], score: float | None = None) -> str:
    parts = [
        f"[id={doc.get('id')}]",
        f"category={doc.get('category')}",
        f"rating={doc.get('rating')}",
        f"asin={doc.get('asin')}",
    ]
    if score is not None:
        parts.append(f"score={score:.4f}")
    head = " | ".join(parts)
    title = doc.get("title") or ""
    text = doc.get("text") or doc.get("doc_text") or ""
    # Keep excerpts short so Groq 8b free-tier requests stay under ~6k-token input limits.
    return f"{head}\nTitle: {title[:200]}\nReview: {text[:650]}"


def retrieve_bundle(
    hybrid: HybridRetriever,
    query: str,
    top_k: int = 10,
    *,
    category_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, float]], str]:
    allowed_ids: set[int] | None = None
    if category_filter and category_filter.strip() in REVIEW_CATEGORY_NAMES:
        allowed_ids = hybrid.get_ids_for_category(category_filter.strip())
    docs, ranked = hybrid.retrieve_context(query, top_k=top_k, allowed_ids=allowed_ids)
    score_by_id = {i: s for i, s in ranked}
    ctx_parts = []
    for d in docs:
        did = int(d["id"])
        ctx_parts.append(format_doc_for_context(d, score_by_id.get(did)))
    return docs, ranked, "\n\n---\n\n".join(ctx_parts)
