"""The only module that knows the database drivers: SQLite (default) and PostgreSQL.

Differences isolated here: connection set-up, placeholders, how a write transaction takes the
single writer lock (SQLite `BEGIN IMMEDIATE`, PostgreSQL a transaction-level advisory lock),
how a consistent read starts, and which driver exceptions mean "store unavailable".
The SQL itself (including `INSERT ... ON CONFLICT ... DO UPDATE`) is shared.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

ADVISORY_LOCK_KEY = 7_340_032  # arbitrary constant: the platform's single writer lock


class Dialect:
    name = "base"
    unavailable_errors: tuple[type[BaseException], ...] = ()
    shared_connection = False

    def connect(self) -> Any:
        raise NotImplementedError

    def sql(self, text: str) -> str:
        return text

    def begin_write(self, conn: Any) -> None:
        raise NotImplementedError

    def begin_read(self, conn: Any) -> None:
        raise NotImplementedError

    def commit(self, conn: Any) -> None:
        conn.execute("COMMIT")

    def rollback(self, conn: Any) -> None:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001 - nothing to roll back
            pass

    def schema_types(self) -> dict[str, str]:
        return {"BIGINT": "BIGINT", "REAL": "DOUBLE PRECISION"}

    def describe(self) -> str:
        return self.name


class SQLiteDialect(Dialect):
    """`path` ':memory:' gives one shared in-memory connection (virtual driver, harness)."""

    name = "sqlite"
    # only operational errors (locked, I/O, unreachable) mean "unavailable"; integrity errors are bugs
    unavailable_errors = (sqlite3.OperationalError,)

    def __init__(self, path: str = ":memory:", busy_timeout_ms: int = 10_000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self.shared_connection = path == ":memory:"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False, timeout=self.busy_timeout_ms / 1000.0
        )
        conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def begin_write(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    def begin_read(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN DEFERRED")

    def describe(self) -> str:
        return f"sqlite {sqlite3.sqlite_version}"


class PostgresDialect(Dialect):
    name = "postgres"

    def __init__(self, dsn: str, schema: str | None = None) -> None:
        import psycopg  # imported here so that the default path never needs it

        self._psycopg = psycopg
        self.dsn = dsn
        self.schema = schema
        self.unavailable_errors = (psycopg.OperationalError, psycopg.InterfaceError)
        self._lock = threading.Lock()

    def connect(self) -> Any:
        conn = self._psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)
        if self.schema:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            conn.execute(f'SET search_path TO "{self.schema}"')
        return _PgConn(conn)

    def sql(self, text: str) -> str:
        return text.replace("?", "%s")

    def begin_write(self, conn: Any) -> None:
        conn.execute("BEGIN")
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))

    def begin_read(self, conn: Any) -> None:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")

    def describe(self) -> str:
        return "postgres"


class _PgConn:
    """Gives a psycopg connection the `execute(...) -> cursor` shape of sqlite3."""

    def __init__(self, conn: Any) -> None:
        self.raw = conn

    def execute(self, sql: str, params: tuple | list = ()) -> Any:
        cur = self.raw.cursor()
        cur.execute(sql, params)
        return cur

    def close(self) -> None:
        self.raw.close()
