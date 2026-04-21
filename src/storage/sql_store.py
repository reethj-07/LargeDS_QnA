"""DuckDB storage for structured review fields and safe read-only SQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

from src.config import SQL_MAX_ROWS
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Only SELECT allowed; block dangerous keywords (DuckDB surface)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|"
    r"CALL|LOAD|INSTALL|CHECKPOINT|VACUUM)\b",
    re.IGNORECASE,
)

# Chained statement after semicolon (ignore semicolons inside string literals for v1)
_CHAINED_STMT = re.compile(
    r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|PRAGMA|ATTACH)\b",
    re.IGNORECASE,
)

_FROM_REVIEWS = re.compile(r"\bfrom\s+reviews\b", re.IGNORECASE)


def _strip_sql_comments(sql: str) -> str:
    """Remove line comments so keywords in comments do not trip guards."""
    lines = []
    for line in sql.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _validate_and_cap_llm_sql(sql: str, *, max_rows: int) -> str:
    """Extra checks for LLM-generated SELECT (single-table reviews corpus)."""
    s = sql.strip().rstrip(";")
    s = _strip_sql_comments(s)
    if not s.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")
    if _CHAINED_STMT.search(s):
        raise ValueError("Multiple statements are not allowed.")
    if _FORBIDDEN.search(s):
        raise ValueError("Query contains forbidden keywords.")
    if re.search(r"\bUNION\b", s, re.IGNORECASE):
        raise ValueError("UNION is not allowed in agent SQL.")
    if re.search(r"\bJOIN\b", s, re.IGNORECASE):
        raise ValueError("JOIN is not allowed in agent SQL.")
    if not _FROM_REVIEWS.search(s):
        raise ValueError("Query must read from the reviews table only.")

    # Hard row cap: wrap so an inner LIMIT cannot exceed *max_rows*.
    return f"SELECT * FROM ({s}) AS _agent_limited LIMIT {int(max_rows)}"


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

    def query_safe(self, sql: str, *, max_rows: int | None = None) -> list[dict[str, Any]]:
        """Execute a single SELECT on ``reviews`` if it passes safety checks."""
        cap = max_rows if max_rows is not None else SQL_MAX_ROWS
        s = _validate_and_cap_llm_sql(sql, max_rows=cap)
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
