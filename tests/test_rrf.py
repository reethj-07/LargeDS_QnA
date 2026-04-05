"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from src.retrieval.reranker import reciprocal_rank_fusion


def test_rrf_two_lists_agreement_boosts_rank():
    a = [(1, 0.0), (2, 0.0)]
    b = [(2, 0.0), (3, 0.0)]
    merged = reciprocal_rank_fusion([a, b], k=60)
    ids = [i for i, _ in merged]
    assert ids[0] == 2


def test_rrf_single_list():
    ranked = [(10, 1.0), (20, 0.5)]
    out = reciprocal_rank_fusion([ranked], k=60)
    assert [i for i, _ in out] == [10, 20]
