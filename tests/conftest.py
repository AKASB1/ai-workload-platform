"""Shared fixtures. PostgreSQL tests run when AWP_PG_DSN points at a reachable server."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable, Iterator

import pytest

from ai_workload_platform.models import Namespace, NodeInfo
from ai_workload_platform.store.dialect import PostgresDialect, SQLiteDialect
from ai_workload_platform.store.sql import Store

PG_DSN = os.environ.get("AWP_PG_DSN")


def _pg_reachable() -> str | None:
    if not PG_DSN:
        return "AWP_PG_DSN is not set (start PostgreSQL, e.g. docker compose, and export the DSN)"
    try:
        import psycopg

        with psycopg.connect(PG_DSN, connect_timeout=3):
            return None
    except Exception as e:  # noqa: BLE001
        return f"PostgreSQL at AWP_PG_DSN is not reachable: {e}"


_PG_SKIP = _pg_reachable()


def make_store_factory(kind: str, tmp: str) -> tuple[Callable[..., Store], Callable[[], None]]:
    made: list[Store] = []
    schemas: list[str] = []

    def factory(seed: int = 0) -> Store:
        if kind == "sqlite-memory":
            s = Store(SQLiteDialect(":memory:"), seed=seed)
        elif kind == "sqlite-file":
            s = Store(SQLiteDialect(os.path.join(tmp, f"awp-{uuid.uuid4().hex[:8]}.db")), seed=seed)
        else:
            schema = f"t_{uuid.uuid4().hex[:12]}"
            schemas.append(schema)
            s = Store(PostgresDialect(PG_DSN or "", schema=schema), seed=seed)
        made.append(s)
        return s

    def cleanup() -> None:
        for s in made:
            s.close()
        if schemas:
            import psycopg

            with psycopg.connect(PG_DSN or "", autocommit=True) as c:
                for sc in schemas:
                    c.execute(f'DROP SCHEMA IF EXISTS "{sc}" CASCADE')

    return factory, cleanup


STORE_KINDS = [
    "sqlite-memory",
    "sqlite-file",
    pytest.param(
        "postgres",
        marks=[pytest.mark.postgres, pytest.mark.skipif(_PG_SKIP is not None, reason=_PG_SKIP or "")],
    ),
]
THREADED_KINDS = STORE_KINDS[1:]


@pytest.fixture(params=STORE_KINDS)
def store_factory(request: pytest.FixtureRequest) -> Iterator[Callable[..., Store]]:
    with tempfile.TemporaryDirectory() as tmp:
        factory, cleanup = make_store_factory(request.param, tmp)
        try:
            yield factory
        finally:
            cleanup()


@pytest.fixture(params=THREADED_KINDS)
def threaded_store_factory(request: pytest.FixtureRequest) -> Iterator[Callable[..., Store]]:
    with tempfile.TemporaryDirectory() as tmp:
        factory, cleanup = make_store_factory(request.param, tmp)
        try:
            yield factory
        finally:
            cleanup()


@pytest.fixture
def mem_store() -> Iterator[Store]:
    s = Store(seed=0)
    yield s
    s.close()


REF_NODES = [
    NodeInfo("r0-n00", "r0", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("r0-n01", "r0", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("r1-n00", "r1", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("r1-n01", "r1", "a100", 1.0, 8, 128, 1_024_000),
]


def setup_reference(store: Store, *, cap: int = 32, max_queued: int = 200) -> None:
    for name, q in (("team-a", 12), ("team-b", 9), ("team-c", 9)):
        store.put_namespace(Namespace(name, q, cap, 9, max_queued), 0)
    store.save_inventory(list(REF_NODES), 0, None)
