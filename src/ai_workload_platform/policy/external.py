"""ExternalPolicy: a child process that speaks the external-policy protocol v1 over stdin/stdout.

Started without a shell, binary pipes, PYTHONUTF8=1 and PYTHONDONTWRITEBYTECODE=1. stdout carries
protocol lines only; stderr goes to a log file. Every reply is read through a reader thread with a
timeout (pipes cannot be polled on Windows). The child is closed and reaped on every path.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
from typing import Any

from ai_workload_platform.policy import PolicyFailure
from ai_workload_platform.policy.view import wire_view
from ai_workload_platform.procutil import child_env, kill_tree, pid_alive, popen_kwargs

log = logging.getLogger("awp.policy")

DECISION_KEYS = {"type", "seq", "actions", "wake_at_s", "reservations", "solver"}


def _shape_problem(reply: dict[str, Any]) -> str | None:
    """Wire-form checks of a decision (unknown fields and wrong types are `malformed`, not `invalid`)."""
    for i, a in enumerate(reply["actions"]):
        if not isinstance(a, dict) or a.get("op") not in ("start", "preempt"):
            return f"action {i}: not an object with op start or preempt"
        keys = {"op", "job_id", "placement"} if a["op"] == "start" else {"op", "job_id"}
        if set(a) != keys or not isinstance(a["job_id"], str):
            return f"action {i}: fields must be exactly {sorted(keys)} with a string job_id"
        if a["op"] == "start":
            pl = a["placement"]
            if not isinstance(pl, list) or not all(
                isinstance(e, dict)
                and set(e) == {"node", "workers"}
                and isinstance(e["node"], str)
                and isinstance(e["workers"], int)
                and not isinstance(e["workers"], bool)
                for e in pl
            ):
                return f"action {i}: placement entries are {{node: string, workers: integer}}"
    w = reply.get("wake_at_s")
    if w is not None and (isinstance(w, bool) or not isinstance(w, int | float)):
        return "wake_at_s must be a number or null"
    if not isinstance(reply.get("reservations", []), list):
        return "reservations must be a list"
    s = reply.get("solver")
    if s is not None and not isinstance(s, dict):
        return "solver must be an object or null"
    return None


_EOF = object()


class ExternalPolicy:
    def __init__(
        self,
        cmd: list[str],
        *,
        name: str,
        wire_name: str,
        params: dict[str, Any] | None = None,
        seed: int = 0,
        timeout_s: float = 60.0,
        stderr_path: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.cmd = list(cmd)
        self.cwd = cwd
        self.wire_name = wire_name
        self.params = params or {}
        self.seed = seed
        self.timeout_s = timeout_s
        self.stderr_path = stderr_path or os.path.join(tempfile.gettempdir(), f"awp-policy-{os.getpid()}.log")
        self.proc: subprocess.Popen | None = None
        self.pids: list[int] = []  # every child ever started (tests check none is left)
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_file: Any = None
        self._seq = 0

    # --- process handling ------------------------------------------------------------------
    def _start(self) -> None:
        self._stderr_file = open(self.stderr_path, "ab")  # noqa: SIM115 - closed in _reap
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            env=child_env(),
            shell=False,
            **popen_kwargs(),
        )
        self.pids.append(self.proc.pid)
        self._lines = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(self.proc, self._lines),
            daemon=True,
            name=f"policy-reader-{self.proc.pid}",
        )
        self._reader.start()
        self._seq = 0

    @staticmethod
    def _read_loop(proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        except Exception:  # noqa: BLE001 - the pipe broke; the reader reports EOF
            pass
        q.put(_EOF)

    def _stderr_tail(self, n: int = 2000) -> str:
        try:
            with open(self.stderr_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - n))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _fail(self, kind: str, msg: str) -> PolicyFailure:
        self._reap(force=True)
        return PolicyFailure(kind, f"policy {self.name}: {msg}", self._stderr_tail())

    def _send(self, obj: dict[str, Any]) -> None:
        """Write one line through a helper thread with the same timeout as reads (a pipe can fill up when the
        child stops reading)."""
        proc = self.proc
        assert proc is not None and proc.stdin is not None
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        errors: list[BaseException] = []

        def write() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(data)
                proc.stdin.flush()
            except (OSError, ValueError) as e:
                errors.append(e)

        t = threading.Thread(target=write, daemon=True, name=f"policy-writer-{proc.pid}")
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            raise self._fail("timeout", f"writing to the policy process blocked for {self.timeout_s} s")
        if errors:
            raise self._fail("crash", f"cannot write to the policy process: {errors[0]}")

    def _recv(self) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=self.timeout_s)
        except queue.Empty:
            raise self._fail("timeout", f"no reply within {self.timeout_s} s") from None
        if line is _EOF:
            raise self._fail("crash", "the policy process closed stdout")
        if not line.endswith(b"\n"):
            raise self._fail("crash", "the policy process closed stdout in the middle of a line")
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise self._fail("malformed", f"not a JSON line: {line[:200]!r}") from None
        if not isinstance(obj, dict):
            raise self._fail("malformed", "reply is not a JSON object")
        if obj.get("type") == "error":
            if set(obj) - {"type", "message"}:
                raise self._fail("malformed", "error reply with unknown fields")
            raise self._fail("error", str(obj.get("message")))
        return obj

    def _reap(self, force: bool = False) -> None:
        p = self.proc
        self.proc = None
        if p is not None:
            if force and p.poll() is None:
                kill_tree(p.pid)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_tree(p.pid)
                p.wait(timeout=10)
            for s in (p.stdin, p.stdout):
                try:
                    if s is not None:
                        s.close()
                except OSError:
                    pass
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    # --- protocol ------------------------------------------------------------------------
    def hello(self, msg: dict[str, Any]) -> None:
        if self.proc is not None:
            self.close()
        self._start()
        self._send(
            {
                "type": "hello",
                "protocol_version": 1,
                "policy": {"name": self.wire_name, "params": self.params},
                "seed": self.seed,
                "cluster": msg["cluster"],
            }
        )
        reply = self._recv()
        if reply.get("type") != "hello" or set(reply) - {"type", "name", "version"}:
            raise self._fail("malformed", f"bad hello reply: {reply}")

    def schedule(self, view: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None:
            raise PolicyFailure("crash", f"policy {self.name}: no session")
        self._seq += 1
        self._send({"type": "schedule", "seq": self._seq, "view": wire_view(view)})
        reply = self._recv()
        if reply.get("type") != "decision":
            raise self._fail("malformed", f"expected a decision, got type {reply.get('type')!r}")
        unknown = set(reply) - DECISION_KEYS
        if unknown:
            raise self._fail("malformed", f"unknown fields in the decision: {sorted(unknown)}")
        seq = reply.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq != self._seq:
            raise self._fail("malformed", f"decision seq {seq!r} for request {self._seq}")
        if not isinstance(reply.get("actions"), list):
            raise self._fail("malformed", "actions is not a list")
        problem = _shape_problem(reply)
        if problem:
            raise self._fail("malformed", problem)
        return reply

    def close(self) -> None:
        """Send bye, wait for the reply and the exit; kill on any problem."""
        if self.proc is None:
            return
        ok = False
        try:
            self._send({"type": "bye"})
            reply = self._lines.get(timeout=min(5.0, self.timeout_s))
            ok = reply is not _EOF
        except (PolicyFailure, queue.Empty):
            ok = False
        if self.proc is not None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5 if ok else 0.5)
            except subprocess.TimeoutExpired:
                pass
        self._reap(force=True)

    def alive_children(self) -> list[int]:
        return [p for p in self.pids if pid_alive(p)]
