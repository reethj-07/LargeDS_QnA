"""DuckDB storage for structured review fields and safe read-only SQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

from src.observability.logger import get_logger

logger = get_logger(__name__)

# Only SELECT allowed; block multiple statements and dangerous keywords
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA)\b",
    re.IGNORECASE,
)


class SqlStore:
    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        self.db_path = db_path
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path), read_only=self._read_only)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        con = self.connect()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY,
                asin VARCHAR,
                category VARCHAR,
                rating DOUBLE,
                title VARCHAR,
                text VARCHAR,
                doc_text VARCHAR,
                timestamp BIGINT,
                helpful_vote INTEGER,
                verified_purchase BOOLEAN
            );
            """
        )

    def insert_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        con = self.connect()
        con.executemany(
            """
            INSERT INTO reviews
            (id, asin, category, rating, title, text, doc_text, timestamp, helpful_vote, verified_purchase)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["id"],
                    r["asin"],
                    r["category"],
                    r["rating"],
                    r["title"],
                    r["text"],
                    r["doc_text"],
                    r["timestamp"],
                    r["helpful_vote"],
                    r["verified_purchase"],
                )
                for r in rows
            ],
        )

    def query_safe(self, sql: str) -> list[dict[str, Any]]:
        """Execute a single SELECT if it passes safety checks."""
        s = sql.strip().rstrip(";")
        if not s.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
        if ";" in s:
            raise ValueError("Multiple statements are not allowed.")
        if _FORBIDDEN.search(s):
            raise ValueError("Query contains forbidden keywords.")
        con = self.connect()
        try:
            df = con.execute(s).fetchdf()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.warning("SQL error: %s", e)
            raise

    def table_info(self) -> str:
        return (
            "Table reviews columns: id, asin, category, rating, title, text, doc_text, "
            "timestamp (unix ms), helpful_vote, verified_purchase (boolean)."
        )
