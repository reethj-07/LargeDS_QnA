"""Test that SqlStore.query_safe blocks dangerous SQL."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from src.storage.sql_store import SqlStore


class MemSqlStore(SqlStore):
    """In-memory DuckDB with a minimal reviews row for happy-path SQL tests."""

    def __init__(self) -> None:
        self.db_path = Path(":memory:")
        self._read_only = False
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(database=":memory:")
            self.init_schema()
            self.insert_rows([
                {
                    "id": 1,
                    "asin": "B00TEST",
                    "category": "All_Beauty",
                    "rating": 5.0,
                    "title": "Nice",
                    "text": "Great product",
                    "doc_text": "All_Beauty Nice Great product",
                    "timestamp": 0,
                    "helpful_vote": 0,
                    "verified_purchase": True,
                },
            ])
        return self._conn


def test_rejects_insert():
    with pytest.raises(ValueError, match="Only SELECT"):
        MemSqlStore().query_safe("INSERT INTO reviews VALUES (1)")


def test_rejects_drop_via_semicolon():
    with pytest.raises(ValueError, match="Multiple statements"):
        MemSqlStore().query_safe("SELECT 1; DROP TABLE reviews")


def test_rejects_forbidden_keyword():
    with pytest.raises(ValueError, match="forbidden"):
        MemSqlStore().query_safe(
            "SELECT * FROM reviews WHERE 1=1 UNION ALL DELETE FROM reviews"
        )


def test_rejects_non_select():
    with pytest.raises(ValueError, match="Only SELECT"):
        MemSqlStore().query_safe("UPDATE reviews SET rating=5")


def test_rejects_multiple_statements():
    with pytest.raises(ValueError, match="Multiple statements"):
        MemSqlStore().query_safe("SELECT 1; SELECT 2")


def test_rejects_union():
    with pytest.raises(ValueError, match="UNION"):
        MemSqlStore().query_safe("SELECT id FROM reviews UNION SELECT id FROM reviews")


def test_rejects_join():
    with pytest.raises(ValueError, match="JOIN"):
        MemSqlStore().query_safe(
            "SELECT a.id FROM reviews a JOIN reviews b ON a.id = b.id"
        )


def test_rejects_without_reviews_table():
    with pytest.raises(ValueError, match="reviews"):
        MemSqlStore().query_safe("SELECT 1 AS x")


def test_allows_select_with_where():
    rows = MemSqlStore().query_safe("SELECT id, rating FROM reviews WHERE rating > 4")
    assert len(rows) == 1
    assert rows[0]["id"] == 1
