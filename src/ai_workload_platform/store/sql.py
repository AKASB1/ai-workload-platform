"""The store: the only source of truth (SQLite by default, PostgreSQL through the same code).

Every mutation is one write transaction (writers are serialized by the dialect). Rows carry a
`version`; `Tx.cas_update` is the compare-and-set every mutation goes through. Each mutation
appends exactly one event with the next gap-free `seq` (store/ops.py).
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, NamedTuple

from ai_workload_platform.models import (
    Attempt,
    AttemptState,
    Event,
    LeaseLost,
    Namespace,
    NodeInfo,
    StoreUnavailable,
    VersionConflict,
    Workload,
    WorkloadState,
)
from ai_workload_platform.store.dialect import Dialect, SQLiteDialect

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS counters (id INTEGER PRIMARY KEY, next_seq BIGINT NOT NULL, "
    "last_at_ms BIGINT NOT NULL, next_workload BIGINT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS namespaces (name TEXT PRIMARY KEY, quota_gpus INTEGER NOT NULL, "
    "cap_gpus INTEGER NOT NULL, max_priority INTEGER NOT NULL, max_queued INTEGER NOT NULL, "
    "updated_ms BIGINT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS workloads (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, spec TEXT NOT NULL, "
    "spec_hash TEXT NOT NULL, priority INTEGER NOT NULL, gpus INTEGER NOT NULL, workers INTEGER NOT NULL, "
    "cpus INTEGER NOT NULL, mem_mb BIGINT NOT NULL, submit_ms BIGINT NOT NULL, submit_seq BIGINT NOT NULL, "
    "state TEXT NOT NULL, version INTEGER NOT NULL, cancel_requested INTEGER NOT NULL, counted INTEGER NOT NULL, "
    "attempts INTEGER NOT NULL, retry_at_ms BIGINT, state_since_ms BIGINT NOT NULL, first_started_ms BIGINT, "
    "terminal_ms BIGINT, retained_ms BIGINT NOT NULL DEFAULT 0, preemptions INTEGER NOT NULL DEFAULT 0)",
    "CREATE INDEX IF NOT EXISTS workloads_state ON workloads (state)",
    "CREATE INDEX IF NOT EXISTS workloads_ns_seq ON workloads (namespace, submit_seq)",
    "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, workload_id TEXT NOT NULL, namespace TEXT NOT NULL, "
    "n INTEGER NOT NULL, state TEXT NOT NULL, placement TEXT NOT NULL, gpus INTEGER NOT NULL, "
    "workers INTEGER NOT NULL, cpus INTEGER NOT NULL, mem_mb BIGINT NOT NULL, version INTEGER NOT NULL, "
    "started_ms BIGINT NOT NULL, state_since_ms BIGINT NOT NULL, stop_reason TEXT, stop_requested_ms BIGINT, "
    "running_ms BIGINT, observed_started_ms BIGINT, observed_ended_ms BIGINT, ended_ms BIGINT, end_reason TEXT, "
    "exit_code INTEGER, counted INTEGER, observed_nodes TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS attempts_state ON attempts (state)",
    "CREATE INDEX IF NOT EXISTS attempts_workload ON attempts (workload_id)",
    "CREATE TABLE IF NOT EXISTS events (seq BIGINT PRIMARY KEY, at_ms BIGINT NOT NULL, type TEXT NOT NULL, "
    "namespace TEXT NOT NULL, workload_id TEXT NOT NULL, attempt_id TEXT, data TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS idempotency (namespace TEXT NOT NULL, key TEXT NOT NULL, spec_hash TEXT NOT NULL, "
    "workload_id TEXT NOT NULL, PRIMARY KEY (namespace, key))",
    "CREATE TABLE IF NOT EXISTS lease (id INTEGER PRIMARY KEY, holder TEXT, epoch BIGINT NOT NULL, "
    "expires_ms BIGINT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS inventory (name TEXT PRIMARY KEY, rack TEXT NOT NULL, class TEXT NOT NULL, "
    "speed DOUBLE PRECISION NOT NULL, gpus INTEGER NOT NULL, cpus INTEGER NOT NULL, mem_mb BIGINT NOT NULL, "
    "ready INTEGER NOT NULL, not_ready_since_ms BIGINT, updated_ms BIGINT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS books (namespace TEXT NOT NULL, node TEXT NOT NULL, gpus INTEGER NOT NULL, "
    "cpus INTEGER NOT NULL, mem_mb BIGINT NOT NULL, PRIMARY KEY (namespace, node))",
]

W_COLS = (
    "id, namespace, spec, spec_hash, priority, gpus, workers, cpus, mem_mb, submit_ms, submit_seq, state, version, "
    "cancel_requested, counted, attempts, retry_at_ms, state_since_ms, first_started_ms, terminal_ms, retained_ms, "
    "preemptions"
)
A_COLS = (
    "id, workload_id, namespace, n, state, placement, gpus, workers, cpus, mem_mb, version, started_ms, "
    "state_since_ms, stop_reason, stop_requested_ms, running_ms, observed_started_ms, observed_ended_ms, ended_ms, "
    "end_reason, exit_code, counted, observed_nodes"
)
TERMINAL_SQL = "('SUCCEEDED','FAILED','DEAD_LETTER','CANCELLED')"


class Fence(NamedTuple):
    holder: str
    epoch: int


class Inventory(NamedTuple):
    nodes: list[NodeInfo]
    not_ready_since: dict[str, int]
    updated_ms: int


class Tx:
    """One transaction. Read helpers plus the low-level writes the mutations use."""

    def __init__(self, store: Store, conn: Any, write: bool) -> None:
        self.store = store
        self.conn = conn
        self.write = write
        self._counters: list[int] | None = None
        self._counters_dirty = False
        self._new_specs: dict[str, dict] = {}

    # --- plumbing ----------------------------------------------------------------------------
    def q(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        return list(self.conn.execute(self.store.dialect.sql(sql), tuple(params)).fetchall())

    def one(self, sql: str, params: tuple | list = ()) -> tuple | None:
        return self.conn.execute(self.store.dialect.sql(sql), tuple(params)).fetchone()

    def x(self, sql: str, params: tuple | list = ()) -> int:
        cur = self.conn.execute(self.store.dialect.sql(sql), tuple(params))
        return cur.rowcount

    # --- counters ----------------------------------------------------------------------------
    def _load_counters(self) -> list[int]:
        if self._counters is None:
            row = self.one("SELECT next_seq, last_at_ms, next_workload FROM counters WHERE id = 1")
            assert row is not None
            self._counters = [int(row[0]), int(row[1]), int(row[2])]
        return self._counters

    def event_time(self, now_ms: int) -> int:
        """Event time: the caller's now, never before the last event (monotone log)."""
        return max(int(now_ms), self._load_counters()[1])

    def next_workload_number(self) -> int:
        c = self._load_counters()
        n = c[2]
        c[2] = n + 1
        self._counters_dirty = True
        return n

    def flush(self) -> None:
        if self._counters_dirty and self._counters is not None:
            self.x(
                "UPDATE counters SET next_seq = ?, last_at_ms = ?, next_workload = ? WHERE id = 1",
                self._counters,
            )
            self._counters_dirty = False

    # --- fence -------------------------------------------------------------------------------
    def check_fence(self, fence: Fence) -> None:
        row = self.one("SELECT holder, epoch FROM lease WHERE id = 1")
        if row is None or row[0] != fence.holder or int(row[1]) != fence.epoch:
            raise LeaseLost(
                f"lease lost by {fence.holder} (epoch {fence.epoch})",
                {"holder": row[0] if row else None, "epoch": int(row[1]) if row else None},
            )

    # --- reads -------------------------------------------------------------------------------
    def workload(self, wid: str) -> Workload | None:
        row = self.one(f"SELECT {W_COLS} FROM workloads WHERE id = ?", (wid,))
        return self.store._w(row) if row else None

    def attempt(self, aid: str) -> Attempt | None:
        row = self.one(f"SELECT {A_COLS} FROM attempts WHERE id = ?", (aid,))
        return self.store._a(row) if row else None

    def namespace(self, name: str) -> Namespace | None:
        row = self.one(
            "SELECT name, quota_gpus, cap_gpus, max_priority, max_queued FROM namespaces WHERE name = ?",
            (name,),
        )
        return Namespace(*row) if row else None

    def ns_allocated_gpus(self, name: str) -> int:
        row = self.one("SELECT COALESCE(SUM(gpus), 0) FROM books WHERE namespace = ?", (name,))
        return int(row[0]) if row else 0

    def count_waiting(self, name: str) -> int:
        row = self.one(
            "SELECT COUNT(*) FROM workloads WHERE namespace = ? AND state IN ('QUEUED','RETRY_WAIT')", (name,)
        )
        return int(row[0]) if row else 0

    def idempotency(self, ns: str, key: str) -> tuple[str, str] | None:
        row = self.one(
            "SELECT spec_hash, workload_id FROM idempotency WHERE namespace = ? AND key = ?", (ns, key)
        )
        return (row[0], row[1]) if row else None

    def exists(self, wid: str) -> bool:
        return self.one("SELECT 1 FROM workloads WHERE id = ?", (wid,)) is not None

    def inventory(self) -> Inventory:
        return self.store._inventory(self)

    # --- writes ------------------------------------------------------------------------------
    def cas_update(self, table: str, row_id: str, expected_version: int, values: dict[str, Any]) -> None:
        """Compare and set: write `values` and version+1 only if the row still has `expected_version`."""
        sets = ", ".join(f"{k} = ?" for k in values)
        n = self.x(
            f"UPDATE {table} SET {sets}, version = ? WHERE id = ? AND version = ?",
            [*values.values(), expected_version + 1, row_id, expected_version],
        )
        if n != 1:
            raise VersionConflict(
                f"{table} {row_id}: version {expected_version} is stale",
                {"table": table, "id": row_id, "expected_version": expected_version},
            )

    def insert_workload(self, w: Workload) -> None:
        self.x(f"INSERT INTO workloads ({W_COLS}) VALUES ({', '.join('?' * 22)})", self.store._w_values(w))
        self._new_specs[w.id] = w.spec

    def update_workload(self, new: Workload, expected_version: int) -> None:
        vals = self.store._w_values(new)
        cols = [c.strip() for c in W_COLS.split(",")]
        mutable = {
            c: v
            for c, v in zip(cols, vals, strict=True)
            if c
            not in (
                "id",
                "namespace",
                "spec",
                "spec_hash",
                "priority",
                "gpus",
                "workers",
                "cpus",
                "mem_mb",
                "submit_ms",
                "submit_seq",
                "version",
            )
        }
        self.cas_update("workloads", new.id, expected_version, mutable)

    def insert_attempt(self, a: Attempt) -> None:
        self.x(f"INSERT INTO attempts ({A_COLS}) VALUES ({', '.join('?' * 23)})", self.store._a_values(a))

    def update_attempt(self, new: Attempt, expected_version: int) -> None:
        vals = self.store._a_values(new)
        cols = [c.strip() for c in A_COLS.split(",")]
        mutable = {
            c: v
            for c, v in zip(cols, vals, strict=True)
            if c
            not in (
                "id",
                "workload_id",
                "namespace",
                "n",
                "placement",
                "gpus",
                "workers",
                "cpus",
                "mem_mb",
                "version",
                "started_ms",
            )
        }
        self.cas_update("attempts", new.id, expected_version, mutable)

    def add_books(
        self, namespace: str, placement: list[dict], gpus: int, cpus: int, mem_mb: int, sign: int
    ) -> None:
        for p in placement:
            k = int(p["workers"]) * sign
            self.x(
                "INSERT INTO books (namespace, node, gpus, cpus, mem_mb) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (namespace, node) DO UPDATE SET gpus = books.gpus + excluded.gpus, "
                "cpus = books.cpus + excluded.cpus, mem_mb = books.mem_mb + excluded.mem_mb",
                (namespace, p["node"], gpus * k, cpus * k, mem_mb * k),
            )

    def append_event(
        self,
        at_ms: int,
        etype: str,
        namespace: str,
        workload_id: str,
        attempt_id: str | None,
        data: dict[str, Any],
    ) -> Event:
        c = self._load_counters()
        seq = c[0]
        at = max(int(at_ms), c[1])
        c[0] = seq + 1
        c[1] = at
        self._counters_dirty = True
        self.x(
            "INSERT INTO events (seq, at_ms, type, namespace, workload_id, attempt_id, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (seq, at, str(etype), namespace, workload_id, attempt_id, json.dumps(data, sort_keys=True)),
        )
        return Event(seq, at, str(etype), namespace, workload_id, attempt_id, data)

    def put_idempotency(self, ns: str, key: str, spec_hash: str, wid: str) -> None:
        self.x(
            "INSERT INTO idempotency (namespace, key, spec_hash, workload_id) VALUES (?, ?, ?, ?)",
            (ns, key, spec_hash, wid),
        )


class Store:
    """Connections, transactions, and read queries. Mutations live in store/ops.py."""

    def __init__(self, dialect: Dialect | None = None, *, seed: int = 0, metrics: Any = None) -> None:
        self.dialect = dialect or SQLiteDialect()
        self.metrics = metrics
        self.injector: Any = None  # fault hooks of the harness (None in production)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._conns: list[Any] = []
        self._conns_lock = threading.Lock()
        self._shared: Any = None
        self._spec_cache: dict[str, dict] = {}
        self._closed = False
        self._init(seed)

    # --- connections and transactions ----------------------------------------------------------
    def _conn(self) -> Any:
        if self.dialect.shared_connection:
            if self._shared is None:
                self._shared = self.dialect.connect()
                self._conns.append(self._shared)
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = self.dialect.connect()
            except self.dialect.unavailable_errors as e:  # type: ignore[misc]
                self._count_error()
                raise StoreUnavailable(f"store unreachable: {e}") from e
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def _count_error(self) -> None:
        if self.metrics is not None:
            self.metrics.store_errors.inc()

    @contextmanager
    def transaction(self, fence: Fence | None = None, *, write: bool = True) -> Iterator[Tx]:
        if self._closed:
            raise StoreUnavailable("store is closed")
        guard = self._lock if self.dialect.shared_connection else nullcontext()
        with guard:
            conn = self._conn()
            try:
                if self.injector is not None:
                    self.injector.before_begin(write)
                if write:
                    self.dialect.begin_write(conn)
                else:
                    self.dialect.begin_read(conn)
            except StoreUnavailable:
                self._count_error()
                raise
            except self.dialect.unavailable_errors as e:  # type: ignore[misc]
                self._count_error()
                self._drop_conn(conn)
                raise StoreUnavailable(f"store unavailable: {e}") from e
            tx = Tx(self, conn, write)
            try:
                if write and fence is not None:
                    tx.check_fence(fence)
                yield tx
                if write:
                    tx.flush()
                    if self.injector is not None:
                        self.injector.before_commit()
                self.dialect.commit(conn)
            except self.dialect.unavailable_errors as e:  # type: ignore[misc]
                self.dialect.rollback(conn)
                self._count_error()
                raise StoreUnavailable(f"store unavailable: {e}") from e
            except StoreUnavailable:
                self.dialect.rollback(conn)
                self._count_error()
                raise
            except BaseException:
                self.dialect.rollback(conn)
                raise
            self._spec_cache.update(tx._new_specs)

    def read(self) -> Any:
        return self.transaction(write=False)

    def _drop_conn(self, conn: Any) -> None:
        if self.dialect.shared_connection:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._local.conn = None
        with self._conns_lock:
            if conn in self._conns:
                self._conns.remove(conn)

    def close(self) -> None:
        self._closed = True
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conns.clear()
        self._shared = None

    def _init(self, seed: int) -> None:
        with self.transaction() as tx:
            for stmt in SCHEMA:
                tx.x(stmt)
            if tx.one("SELECT 1 FROM counters WHERE id = 1") is None:
                tx.x("INSERT INTO counters (id, next_seq, last_at_ms, next_workload) VALUES (1, 1, 0, 1)")
                tx.x("INSERT INTO lease (id, holder, epoch, expires_ms) VALUES (1, NULL, 0, 0)")
                tx.x("INSERT INTO meta (key, value) VALUES ('instance_id', ?)", (uuid.uuid4().hex[:12],))
                tx.x("INSERT INTO meta (key, value) VALUES ('seed', ?)", (str(int(seed)),))

    # --- meta --------------------------------------------------------------------------------
    def meta(self, key: str) -> str | None:
        with self.read() as tx:
            row = tx.one("SELECT value FROM meta WHERE key = ?", (key,))
        return row[0] if row else None

    def set_meta_if_absent(self, key: str, value: str) -> str:
        with self.transaction() as tx:
            row = tx.one("SELECT value FROM meta WHERE key = ?", (key,))
            if row:
                return row[0]
            tx.x("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))
            return value

    @property
    def seed(self) -> int:
        v = self.meta("seed")
        return int(v) if v is not None else 0

    @property
    def instance_id(self) -> str:
        return self.meta("instance_id") or ""

    # --- row mapping -------------------------------------------------------------------------
    def _w(self, r: tuple) -> Workload:
        spec = self._spec_cache.get(r[0])
        if spec is None:
            spec = json.loads(r[2])
            self._spec_cache[r[0]] = spec
        return Workload(
            id=r[0],
            namespace=r[1],
            spec=spec,
            spec_hash=r[3],
            priority=int(r[4]),
            gpus=int(r[5]),
            workers=int(r[6]),
            cpus=int(r[7]),
            mem_mb=int(r[8]),
            submit_ms=int(r[9]),
            submit_seq=int(r[10]),
            state=WorkloadState(r[11]),
            version=int(r[12]),
            cancel_requested=bool(r[13]),
            counted=int(r[14]),
            attempts=int(r[15]),
            retry_at_ms=None if r[16] is None else int(r[16]),
            state_since_ms=int(r[17]),
            first_started_ms=None if r[18] is None else int(r[18]),
            terminal_ms=None if r[19] is None else int(r[19]),
            retained_ms=int(r[20]),
            preemptions=int(r[21]),
        )

    @staticmethod
    def _w_values(w: Workload) -> list[Any]:
        return [
            w.id,
            w.namespace,
            json.dumps(w.spec, sort_keys=True, separators=(",", ":")),
            w.spec_hash,
            w.priority,
            w.gpus,
            w.workers,
            w.cpus,
            w.mem_mb,
            w.submit_ms,
            w.submit_seq,
            str(w.state),
            w.version,
            int(w.cancel_requested),
            w.counted,
            w.attempts,
            w.retry_at_ms,
            w.state_since_ms,
            w.first_started_ms,
            w.terminal_ms,
            w.retained_ms,
            w.preemptions,
        ]

    @staticmethod
    def _a(r: tuple) -> Attempt:
        return Attempt(
            id=r[0],
            workload_id=r[1],
            namespace=r[2],
            n=int(r[3]),
            state=AttemptState(r[4]),
            placement=json.loads(r[5]),
            gpus=int(r[6]),
            cpus=int(r[8]),
            mem_mb=int(r[9]),
            version=int(r[10]),
            started_ms=int(r[11]),
            state_since_ms=int(r[12]),
            stop_reason=r[13],
            stop_requested_ms=None if r[14] is None else int(r[14]),
            running_ms=None if r[15] is None else int(r[15]),
            observed_started_ms=None if r[16] is None else int(r[16]),
            observed_ended_ms=None if r[17] is None else int(r[17]),
            ended_ms=None if r[18] is None else int(r[18]),
            end_reason=r[19],
            exit_code=None if r[20] is None else int(r[20]),
            counted=None if r[21] is None else bool(r[21]),
            observed_nodes=json.loads(r[22]),
        )

    @staticmethod
    def _a_values(a: Attempt) -> list[Any]:
        return [
            a.id,
            a.workload_id,
            a.namespace,
            a.n,
            str(a.state),
            json.dumps(a.placement, separators=(",", ":")),
            a.gpus,
            a.workers,
            a.cpus,
            a.mem_mb,
            a.version,
            a.started_ms,
            a.state_since_ms,
            a.stop_reason,
            a.stop_requested_ms,
            a.running_ms,
            a.observed_started_ms,
            a.observed_ended_ms,
            a.ended_ms,
            a.end_reason,
            a.exit_code,
            None if a.counted is None else int(a.counted),
            json.dumps(list(a.observed_nodes), separators=(",", ":")),
        ]

    # --- reads -------------------------------------------------------------------------------
    def get_workload(self, wid: str) -> Workload | None:
        with self.read() as tx:
            return tx.workload(wid)

    def get_attempt(self, aid: str) -> Attempt | None:
        with self.read() as tx:
            return tx.attempt(aid)

    def attempts_of(self, wid: str) -> list[Attempt]:
        with self.read() as tx:
            return [
                self._a(r)
                for r in tx.q(f"SELECT {A_COLS} FROM attempts WHERE workload_id = ? ORDER BY n", (wid,))
            ]

    def list_workloads(
        self,
        namespace: str | None = None,
        state: str | None = None,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[Workload]:
        sql = f"SELECT {W_COLS} FROM workloads WHERE submit_seq > ?"
        params: list[Any] = [after_seq]
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        if state is not None:
            sql += " AND state = ?"
            params.append(state)
        sql += " ORDER BY submit_seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.read() as tx:
            return [self._w(r) for r in tx.q(sql, params)]

    def all_workloads(self, tx: Tx | None = None) -> list[Workload]:
        if tx is not None:
            return [self._w(r) for r in tx.q(f"SELECT {W_COLS} FROM workloads ORDER BY id")]
        with self.read() as t:
            return [self._w(r) for r in t.q(f"SELECT {W_COLS} FROM workloads ORDER BY id")]

    def all_attempts(self, tx: Tx | None = None) -> list[Attempt]:
        if tx is not None:
            return [self._a(r) for r in tx.q(f"SELECT {A_COLS} FROM attempts ORDER BY id")]
        with self.read() as t:
            return [self._a(r) for r in t.q(f"SELECT {A_COLS} FROM attempts ORDER BY id")]

    def live_workloads(self, tx: Tx) -> list[Workload]:
        return [
            self._w(r)
            for r in tx.q(f"SELECT {W_COLS} FROM workloads WHERE state NOT IN {TERMINAL_SQL} ORDER BY id")
        ]

    def open_attempts(self, tx: Tx) -> list[Attempt]:
        return [self._a(r) for r in tx.q(f"SELECT {A_COLS} FROM attempts WHERE state <> 'ENDED' ORDER BY id")]

    def events(self, after: int = 0, limit: int | None = None) -> list[Event]:
        sql = "SELECT seq, at_ms, type, namespace, workload_id, attempt_id, data FROM events WHERE seq > ? ORDER BY seq"
        params: list[Any] = [after]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.read() as tx:
            rows = tx.q(sql, params)
        return [Event(int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], json.loads(r[6])) for r in rows]

    def last_seq(self) -> int:
        with self.read() as tx:
            row = tx.one("SELECT next_seq FROM counters WHERE id = 1")
        return int(row[0]) - 1 if row else 0

    def books(self, tx: Tx | None = None) -> dict[tuple[str, str], tuple[int, int, int]]:
        def rows(t: Tx) -> list[tuple]:
            return t.q("SELECT namespace, node, gpus, cpus, mem_mb FROM books ORDER BY namespace, node")

        rs = rows(tx) if tx is not None else self._with_read(rows)
        return {
            (r[0], r[1]): (int(r[2]), int(r[3]), int(r[4])) for r in rs if (r[2], r[3], r[4]) != (0, 0, 0)
        }

    def _with_read(self, fn: Any) -> Any:
        with self.read() as tx:
            return fn(tx)

    # --- namespaces --------------------------------------------------------------------------
    def put_namespace(self, ns: Namespace, now_ms: int) -> None:
        with self.transaction() as tx:
            tx.x(
                "INSERT INTO namespaces (name, quota_gpus, cap_gpus, max_priority, max_queued, updated_ms) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (name) DO UPDATE SET quota_gpus = excluded.quota_gpus, "
                "cap_gpus = excluded.cap_gpus, max_priority = excluded.max_priority, "
                "max_queued = excluded.max_queued, updated_ms = excluded.updated_ms",
                (ns.name, ns.quota_gpus, ns.cap_gpus, ns.max_priority, ns.max_queued, int(now_ms)),
            )

    def get_namespace(self, name: str) -> Namespace | None:
        with self.read() as tx:
            return tx.namespace(name)

    def namespaces(self, tx: Tx | None = None) -> list[Namespace]:
        sql = "SELECT name, quota_gpus, cap_gpus, max_priority, max_queued FROM namespaces ORDER BY name"
        rs = tx.q(sql) if tx is not None else self._with_read(lambda t: t.q(sql))
        return [Namespace(*r) for r in rs]

    # --- inventory ---------------------------------------------------------------------------
    def _inventory(self, tx: Tx) -> Inventory:
        rows = tx.q(
            "SELECT name, rack, class, speed, gpus, cpus, mem_mb, ready, not_ready_since_ms, updated_ms "
            "FROM inventory ORDER BY rack, name"
        )
        nodes = [
            NodeInfo(r[0], r[1], r[2], float(r[3]), int(r[4]), int(r[5]), int(r[6]), bool(r[7])) for r in rows
        ]
        nrs = {r[0]: int(r[8]) for r in rows if r[8] is not None}
        upd = max((int(r[9]) for r in rows), default=-1)
        return Inventory(nodes, nrs, upd)

    def inventory(self) -> Inventory:
        with self.read() as tx:
            return self._inventory(tx)

    def save_inventory(self, nodes: list[NodeInfo], now_ms: int, fence: Fence | None) -> Inventory:
        """Store the controller's inventory snapshot; track since when each node is not ready."""
        with self.transaction(fence) as tx:
            old = self._inventory(tx)
            old_by = {n.name: n for n in old.nodes}
            nrs: dict[str, int] = {}
            names = set()
            for n in nodes:
                names.add(n.name)
                since = None if n.ready else old.not_ready_since.get(n.name, int(now_ms))
                if since is not None:
                    nrs[n.name] = since
                prev = old_by.get(n.name)
                if prev == n and old.not_ready_since.get(n.name) == since:
                    continue
                tx.x(
                    "INSERT INTO inventory (name, rack, class, speed, gpus, cpus, mem_mb, ready, not_ready_since_ms, "
                    "updated_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (name) DO UPDATE SET "
                    "rack = excluded.rack, class = excluded.class, speed = excluded.speed, gpus = excluded.gpus, "
                    "cpus = excluded.cpus, mem_mb = excluded.mem_mb, ready = excluded.ready, "
                    "not_ready_since_ms = excluded.not_ready_since_ms, updated_ms = excluded.updated_ms",
                    (
                        n.name,
                        n.rack,
                        n.gpu_class,
                        n.speed,
                        n.gpus,
                        n.cpus,
                        n.mem_mb,
                        int(n.ready),
                        since,
                        int(now_ms),
                    ),
                )
            for gone in sorted(set(old_by) - names):
                tx.x("DELETE FROM inventory WHERE name = ?", (gone,))
            ordered = sorted(nodes, key=lambda n: (n.rack, n.name))
            return Inventory(ordered, nrs, int(now_ms))

    # --- lease -------------------------------------------------------------------------------
    def acquire_lease(self, holder: str, epoch: int | None, now_ms: int, ttl_ms: int) -> int:
        """Renew our lease or take a free/expired one (epoch + 1). Raises LeaseLost if fenced out.

        Returns the epoch held. A controller that never held the lease (epoch None) and finds
        it held by another live holder gets LeaseLost with details {"standby": True}.
        """
        with self.transaction() as tx:
            row = tx.one("SELECT holder, epoch, expires_ms FROM lease WHERE id = 1")
            assert row is not None
            cur_holder, cur_epoch, expires = row[0], int(row[1]), int(row[2])
            if epoch is not None and cur_holder == holder and cur_epoch == epoch:
                tx.x("UPDATE lease SET expires_ms = ? WHERE id = 1", (int(now_ms) + ttl_ms,))
                return epoch
            if epoch is not None:
                raise LeaseLost(
                    f"{holder} lost the lease (epoch {epoch}); now {cur_holder} epoch {cur_epoch}",
                    {"holder": cur_holder, "epoch": cur_epoch},
                )
            if cur_holder is None or expires <= now_ms or cur_holder == holder:
                new = cur_epoch + 1
                tx.x(
                    "UPDATE lease SET holder = ?, epoch = ?, expires_ms = ? WHERE id = 1",
                    (holder, new, int(now_ms) + ttl_ms),
                )
                return new
            raise LeaseLost(
                f"lease held by {cur_holder}", {"holder": cur_holder, "epoch": cur_epoch, "standby": True}
            )

    def release_lease(self, fence: Fence) -> None:
        with self.transaction() as tx:
            tx.x(
                "UPDATE lease SET holder = NULL, expires_ms = 0 WHERE id = 1 AND holder = ? AND epoch = ?",
                (fence.holder, fence.epoch),
            )

    def lease(self) -> tuple[str | None, int, int]:
        with self.read() as tx:
            row = tx.one("SELECT holder, epoch, expires_ms FROM lease WHERE id = 1")
        assert row is not None
        return row[0], int(row[1]), int(row[2])
