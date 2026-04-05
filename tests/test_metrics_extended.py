"""Extended metric tests: average_precision, hit_at_k, edge cases."""

from __future__ import annotations

from evaluation.metrics import average_precision, hit_at_k, precision_at_k


def test_average_precision_perfect():
    assert average_precision([1, 2, 3], {1, 2, 3}) == 1.0


def test_average_precision_partial():
    ap = average_precision([1, 99, 2, 99, 3], {1, 2, 3})
    assert 0 < ap < 1


def test_average_precision_no_hits():
    assert average_precision([10, 20], {1, 2}) == 0.0


def test_hit_at_k_positive():
    assert hit_at_k([5, 6, 7], {7}, k=3) == 1


def test_hit_at_k_negative():
    assert hit_at_k([5, 6, 7], {8}, k=3) == 0


def test_hit_at_k_empty_relevant():
    assert hit_at_k([1, 2], set(), k=5) == 0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {1, 2}, k=5) == 0.0


def test_precision_at_k_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0
