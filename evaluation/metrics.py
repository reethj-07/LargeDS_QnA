"""Retrieval metrics without paid APIs: Recall@K, Precision@K, MRR@K, nDCG@K."""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = set(retrieved[:k])
    if not top:
        return 0.0
    return len(top & relevant) / min(k, len(top))


def average_precision(retrieved: list[int], relevant: set[int]) -> float:
    """AP over full ranked list (for diagnostics)."""
    if not relevant:
        return 0.0
    hits = 0
    precisions: list[float] = []
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant)


def mrr_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Mean reciprocal rank: 1/rank of first relevant doc in top-k, else 0."""
    if not relevant or k <= 0:
        return 0.0
    for i, rid in enumerate(retrieved[:k]):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Binary nDCG@k: relevance 1 if id in relevant, else 0."""
    if k <= 0:
        return 0.0
    rel_flags = [1.0 if rid in relevant else 0.0 for rid in retrieved[:k]]
    if not rel_flags or sum(rel_flags) == 0:
        return 0.0
    dcg = sum(rel_flags[i] / math.log2(i + 2) for i in range(len(rel_flags)))
    ideal = sorted(rel_flags, reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> int:
    """1 if at least one relevant doc appears in top-k, else 0."""
    if not relevant or k <= 0:
        return 0
    return 1 if (set(retrieved[:k]) & relevant) else 0
