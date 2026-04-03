"""Reciprocal Rank Fusion (RRF) for merging ranked lists."""

from __future__ import annotations

from collections import defaultdict


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
