"""Check 8: real-process kill. The live service runs as a child process (no shell, PID recorded) over a
SQLite file with the local backend on a scaled clock; its process tree is killed by PID three times while
workloads run, a new one starts over the same file, and the invariants are checked from outside (API and
file). No process of the test is left."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from ai_workload_platform.client import Client
from ai_workload_platform.models import TERMINAL_STATES, AttemptState
from ai_workload_platform.procutil import child_env, kill_tree, pid_alive, popen_kwargs
from ai_workload_platform.store.dialect import SQLiteDialect
from ai_workload_platform.store.replay import diff, replay
from ai_workload_platform.store.sql import Store

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {
    "start_retry_ms": 2000,
    "start_timeout_ms": 60_000,
    "node_grace_ms": 5000,
    "lost_grace_ms": 6000,
    "stop_retry_ms": 3000,
    "lease_ttl_ms": 3000,
    "observe_interval_ms": 50,
}
TERMINAL = {s.value for s in TERMINAL_STATES}


def start(tmp: Path, n: int) -> tuple[subprocess.Popen, str]:
    pf = tmp / f"port-{n}"
    log = open(tmp / f"service-{n}.log", "wb")  # noqa: SIM115 - closed by the caller via proc
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ai_workload_platform",
            "up",
            "--db",
            str(tmp / "awp.db"),
            "--port",
            "0",
            "--port-file",
            str(pf),
            "--scale",
            "50",
            "--thresholds",
            json.dumps(THRESHOLDS),
            "--log-level",
            "WARNING",
            "--cluster",
            str(ROOT / "configs" / "clusters" / "reference.json"),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=child_env(drop=("AWP_PG_DSN",)),
        **popen_kwargs(),
    )
    proc._log = log  # type: ignore[attr-defined]
    deadline = time.monotonic() + 60
    while not pf.exists():
        assert proc.poll() is None, (tmp / f"service-{n}.log").read_text(errors="replace")
        assert time.monotonic() < deadline, "service did not start"
        time.sleep(0.05)
    return proc, f"http://127.0.0.1:{pf.read_text().strip()}"


def stop(proc: subprocess.Popen, db: Path) -> None:
    kill_tree(proc.pid)
    proc.wait(timeout=30)
    proc._log.close()  # type: ignore[attr-defined]
    wait_released(db)


def wait_released(db: Path, timeout_s: float = 15.0) -> None:
    """Wait until a write lock on the SQLite file can be taken. On Windows the PID of a venv's python.exe is a
    launcher whose interpreter runs as its child: the launcher can be reaped while the killed interpreter
    still holds the database files, and opening them then fails with "disk I/O error"."""
    if not db.exists():
        return
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn = sqlite3.connect(db, timeout=0.5)
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("ROLLBACK")
                return
            finally:
                conn.close()
        except sqlite3.OperationalError:
            assert time.monotonic() < deadline, "the killed service still holds the SQLite file"
            time.sleep(0.1)


@pytest.mark.slow
def test_kill_the_service_three_times_and_converge(tmp_path: Path) -> None:
    pids: list[int] = []
    proc, url = start(tmp_path, 0)
    pids.append(proc.pid)
    try:
        with Client(url) as c:
            deadline = time.monotonic() + 30
            while True:  # the first tick stores the inventory
                try:
                    c.submit("team-a", {"id": "probe", "gpus": 1, "sim": {"runtime_s": 1}}, "probe")
                    break
                except Exception:  # noqa: BLE001
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
            for i in range(12):
                spec = {
                    "id": f"k{i:02d}",
                    "gpus": (1, 2, 4, 8)[i % 4],
                    "workers": 1 + (i % 5 == 0),
                    "sim": {"runtime_s": 60 + 20 * i},
                    "retry": {"max_attempts": 5, "backoff_base_s": 1, "jitter": "none"},
                }
                c.submit(("team-a", "team-b", "team-c")[i % 3], spec, f"key-{i}")
        for k in range(3):  # kill while workloads run, then restart over the same file
            time.sleep(0.6 + 0.4 * k)
            stop(proc, tmp_path / "awp.db")
            assert not pid_alive(proc.pid)
            proc, url = start(tmp_path, k + 1)
            pids.append(proc.pid)
            with Client(url) as c:  # a client retry after the crash: same key, one workload
                assert (
                    c.submit(
                        "team-b",
                        {
                            "id": "k01",
                            "gpus": 2,
                            "workers": 1,
                            "sim": {"runtime_s": 80},
                            "retry": {"max_attempts": 5, "backoff_base_s": 1, "jitter": "none"},
                        },
                        "key-1",
                    )[0]
                    == 200
                )
        with Client(url) as c:
            deadline = time.monotonic() + 180
            while True:
                states = {w["id"]: w["state"] for ns in ("team-a", "team-b", "team-c") for w in c.list(ns)}
                if len(states) == 13 and all(s in TERMINAL for s in states.values()):
                    break
                assert time.monotonic() < deadline, states
                time.sleep(0.1)
            api_events = c.events()
            assert [e["seq"] for e in api_events] == list(range(1, len(api_events) + 1))
            assert all(e["at_ms"] <= f["at_ms"] for e, f in zip(api_events, api_events[1:], strict=False))
    finally:
        stop(proc, tmp_path / "awp.db")
    # from the file: the replay equals the tables, nothing is held, one terminal event per workload
    s = Store(SQLiteDialect(str(tmp_path / "awp.db")))
    try:
        evs = s.events()
        assert len(evs) == len(api_events)
        assert diff(replay(evs), s.all_workloads(), s.all_attempts(), s.books()) == []
        assert all(a.state == AttemptState.ENDED for a in s.all_attempts())
        assert s.books() == {}
        term = Counter(
            e.workload_id
            for e in evs
            if e.type in ("attempt_ended", "cancel_requested") and e.data.get("state") in TERMINAL
        )
        assert len(term) == 13 and set(term.values()) == {1}
        epochs = [e.data["epoch"] for e in evs if "epoch" in e.data]
        assert epochs == sorted(epochs) and len(set(epochs)) >= 4  # one lease epoch per incarnation
        reasons = Counter(a.end_reason for a in s.all_attempts())
        assert reasons["succeeded"] >= 1
    finally:
        s.close()
    assert not any(pid_alive(p) for p in pids), "a process of the test is still running"
