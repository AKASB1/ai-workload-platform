"""Policy interface: `schedule(view) -> decision` on the wire form of the external-policy protocol v1.

A policy decides who starts where; it never executes anything. Built-ins run in process;
`ExternalPolicy` runs a child process that speaks the protocol (docs/contracts.md §6).
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

FAILURE_KINDS = ("invalid", "error", "timeout", "crash", "malformed")


class PolicyFailure(Exception):
    def __init__(self, kind: str, message: str, stderr_tail: str = "") -> None:
        assert kind in FAILURE_KINDS, kind
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.stderr_tail = stderr_tail


class Policy(Protocol):
    name: str

    def hello(self, msg: dict[str, Any]) -> None:
        """Start a session with the cluster of `msg` (the protocol's hello message)."""

    def schedule(self, view: dict[str, Any]) -> dict[str, Any]:
        """Return a decision: {"actions": [...], "wake_at_s": ..., ...}."""

    def close(self) -> None:
        """End the session (external: `bye`, then reap the child)."""


BUILTIN_NAMES = (
    "fifo+first_fit",
    "priority+best_fit",
    "quota+first_fit",
    "priority+best_fit+preempt",
    "quota+first_fit+reclaim",
)


def stub_command() -> list[str]:
    """The committed stub server (fifo+first_fit over the wire form)."""
    return [sys.executable, "-m", "ai_workload_platform.policy.stub_server"]


def make_policy(
    name: str,
    *,
    cmd: list[str] | None = None,
    cwd: str | None = None,
    params: dict | None = None,
    seed: int = 0,
    timeout_s: float = 60.0,
    stderr_path: str | None = None,
) -> Policy:
    """`fifo+first_fit`, `priority+best_fit`, `quota+first_fit`, `stub` (the stub server through the
    protocol), or `external:<name>` with `cmd` (AWP_POLICY_CMD / AWP_POLICY_CWD)."""
    from ai_workload_platform.policy.builtin import make_builtin
    from ai_workload_platform.policy.external import ExternalPolicy

    if name in BUILTIN_NAMES:
        return make_builtin(name)
    if name == "stub":
        return ExternalPolicy(
            stub_command(),
            name="stub",
            wire_name="fifo+first_fit",
            params=params or {},
            seed=seed,
            timeout_s=timeout_s,
            stderr_path=stderr_path,
            cwd=cwd,
        )
    if name.startswith("external:"):
        if not cmd:
            raise ValueError("an external policy needs a command (AWP_POLICY_CMD)")
        return ExternalPolicy(
            cmd,
            name=name,
            wire_name=name.split(":", 1)[1],
            params=params or {},
            seed=seed,
            timeout_s=timeout_s,
            stderr_path=stderr_path,
            cwd=cwd,
        )
    raise ValueError(f"unknown policy {name!r}; known: {', '.join(BUILTIN_NAMES)}, stub, external:<name>")
