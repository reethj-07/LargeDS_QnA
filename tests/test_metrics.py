"""Unit tests for retrieval metrics."""

from __future__ import annotations

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k():
    rel = {1, 2, 3}
    assert recall_at_k([10, 1, 2], rel, k=3) == 2 / 3
    assert recall_at_k([], rel, k=5) == 0.0


def test_mrr_at_k_first_hit():
    rel = {7, 8}
    assert mrr_at_k([7, 1, 2], rel, k=5) == 1.0
    assert mrr_at_k([1, 7, 2], rel, k=5) == 0.5


def test_mrr_at_k_no_hit():
    assert mrr_at_k([1, 2, 3], {9}, k=5) == 0.0


def test_ndcg_at_k_perfect_ordering():
    rel = {1, 2}
    retrieved = [1, 2, 3, 4]
    assert ndcg_at_k(retrieved, rel, k=4) == 1.0


def test_ndcg_at_k_zero_when_no_relevant_in_top():
    assert ndcg_at_k([5, 6, 7], {1, 2}, k=3) == 0.0
