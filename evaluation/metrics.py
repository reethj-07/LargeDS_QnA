"""Retrieval metrics without paid APIs: Recall@K, Precision@K."""

from __future__ import annotations


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
