"""Test that SqlStore.query_safe blocks dangerous SQL."""

from __future__ import annotations

import pytest

from src.storage.sql_store import SqlStore


class FakeSqlStore(SqlStore):
    """Bypass actual DuckDB connection for safety-check tests."""

    def __init__(self):
        self.db_path = None
        self._conn = None

    def connect(self):
        raise RuntimeError("Should not reach connect in safety tests")


def _store():
    return FakeSqlStore()


def test_rejects_insert():
    with pytest.raises(ValueError, match="Only SELECT"):
        _store().query_safe("INSERT INTO reviews VALUES (1)")


def test_rejects_drop_via_semicolon():
    with pytest.raises(ValueError, match="Multiple statements"):
        _store().query_safe("SELECT 1; DROP TABLE reviews")


def test_rejects_forbidden_keyword():
    with pytest.raises(ValueError, match="forbidden"):
        _store().query_safe("SELECT * FROM reviews WHERE 1=1 UNION ALL DELETE FROM reviews")


def test_rejects_non_select():
    with pytest.raises(ValueError, match="Only SELECT"):
        _store().query_safe("UPDATE reviews SET rating=5")


def test_rejects_multiple_statements():
    with pytest.raises(ValueError, match="Multiple statements"):
        _store().query_safe("SELECT 1; SELECT 2")


def test_allows_select_with_where():
    """Safety check passes for legitimate SELECT (connection will fail, but that's fine)."""
    s = _store()
    with pytest.raises(RuntimeError, match="Should not reach connect"):
        s.query_safe("SELECT * FROM reviews WHERE rating > 4")
