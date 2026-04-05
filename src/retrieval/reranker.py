"""Reciprocal Rank Fusion (RRF) for merging ranked lists + optional cross-encoder rerank."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.config import CROSS_ENCODER_MODEL

_ce_model: Any = None


def _get_cross_encoder(model_name: str | None = None):
    global _ce_model
    name = model_name or CROSS_ENCODER_MODEL
    if _ce_model is None or getattr(_ce_model, "_model_name", None) != name:
        from sentence_transformers import CrossEncoder

        _ce_model = CrossEncoder(name)
        _ce_model._model_name = name  # type: ignore[attr-defined]
    return _ce_model


def cross_encoder_rerank(
    query: str,
    doc_ids: list[int],
    texts: list[str],
    top_k: int,
    model_name: str | None = None,
) -> list[tuple[int, float]]:
    """
    Score (query, doc) pairs with a cross-encoder; return top_k (doc_id, score).
    `doc_ids` and `texts` must align 1:1.
    """
    if not doc_ids or not texts or len(doc_ids) != len(texts):
        return []
    pairs = [(query, t[:5120]) for t in texts]
    ce = _get_cross_encoder(model_name)
    scores = ce.predict(pairs, show_progress_bar=False)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    out: list[tuple[int, float]] = []
    for j in order[:top_k]:
        idx = int(j)
        out.append((int(doc_ids[idx]), float(scores[idx])))
    return out


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge multiple lists of (doc_id, score_or_dummy) by RRF score.
    Order within each list is used; original scores are ignored for ranking.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [(doc_id, s) for doc_id, s in merged]
