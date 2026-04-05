"""Unit tests for agents.synthesizer (merge logic, no LLM calls)."""

from __future__ import annotations

from src.agents.synthesizer import merge_sub_results


def test_merge_empty():
    docs, sql, ranked, ctx = merge_sub_results([])
    assert docs == []
    assert sql == []
    assert ranked == []
    assert ctx == ""


def test_merge_single_sub():
    sa = {
        "question": "What is avg rating?",
        "docs": [{"id": 1, "text": "Good"}, {"id": 2, "text": "Bad"}],
        "sql_results": [{"avg_rating": 4.2}],
        "ranked": [(1, 0.9), (2, 0.7)],
        "bundle": "doc1 doc2",
    }
    docs, sql, ranked, ctx = merge_sub_results([sa])
    assert len(docs) == 2
    assert len(sql) == 1
    assert len(ranked) == 2
    assert "Sub-question 1" in ctx
    assert "What is avg rating?" in ctx


def test_merge_deduplicates_docs():
    sa1 = {
        "question": "Q1",
        "docs": [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}],
        "sql_results": [],
        "ranked": [(1, 0.9), (2, 0.8)],
        "bundle": "bundle1",
    }
    sa2 = {
        "question": "Q2",
        "docs": [{"id": 2, "text": "B"}, {"id": 3, "text": "C"}],
        "sql_results": [{"count": 5}],
        "ranked": [(2, 0.85), (3, 0.6)],
        "bundle": "bundle2",
    }
    docs, sql, ranked, ctx = merge_sub_results([sa1, sa2])
    doc_ids = [d["id"] for d in docs]
    assert doc_ids == [1, 2, 3]
    assert len(sql) == 1
    ranked_ids = [r for r, _ in ranked]
    assert 1 in ranked_ids
    assert 2 in ranked_ids
    assert 3 in ranked_ids
    assert "Sub-question 1" in ctx
    assert "Sub-question 2" in ctx


def test_merge_preserves_order():
    sa1 = {"question": "Q1", "docs": [{"id": 10}], "sql_results": [], "ranked": [(10, 1.0)], "bundle": "b1"}
    sa2 = {"question": "Q2", "docs": [{"id": 20}], "sql_results": [], "ranked": [(20, 0.5)], "bundle": "b2"}
    docs, _, ranked, _ = merge_sub_results([sa1, sa2])
    assert [d["id"] for d in docs] == [10, 20]
    assert [r for r, _ in ranked] == [10, 20]
