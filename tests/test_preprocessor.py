"""Unit tests for ingestion.preprocessor."""

from __future__ import annotations

from src.ingestion.preprocessor import clean_text, parse_bool, parse_float, parse_int, preprocess_records


def test_clean_text_strips_html():
    assert clean_text("Good <b>product</b>!") == "Good product !"


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_none():
    assert clean_text(None) == ""


def test_parse_bool_true():
    assert parse_bool(True) is True
    assert parse_bool("true") is True
    assert parse_bool("1") is True


def test_parse_bool_false():
    assert parse_bool(False) is False
    assert parse_bool("false") is False
    assert parse_bool("0") is False


def test_parse_bool_none():
    assert parse_bool(None) is None
    assert parse_bool("maybe") is None


def test_parse_float():
    assert parse_float("3.5") == 3.5
    assert parse_float(None) is None
    assert parse_float("abc") is None


def test_parse_int():
    assert parse_int("42") == 42
    assert parse_int(3.9) == 3
    assert parse_int(None) is None


def test_preprocess_records_filters_empty():
    rows = [
        {"title": "", "text": "", "_category": "test"},
        {"title": "Good", "text": "Nice product", "_category": "test", "rating": 5.0},
    ]
    out = preprocess_records(rows, start_id=0)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["category"] == "test"
    assert out[0]["doc_text"] == "Good Nice product"
