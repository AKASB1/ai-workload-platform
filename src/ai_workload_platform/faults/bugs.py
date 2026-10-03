"""Known bugs, each applied as a test-only patch: a context manager that replaces one function.
Production code has no flag for them. The harness must catch each within 200 schedules."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from ai_workload_platform.controller import rules
from ai_workload_platform.models import EventType, VersionConflict
from ai_workload_platform.store import ops
from ai_workload_platform.store.sql import Tx


@contextmanager
def _patch(obj: Any, name: str, repl: Any) -> Iterator[None]:
    orig = getattr(obj, name)
    setattr(obj, name, repl)
    try:
        yield
    finally:
        setattr(obj, name, orig)


def release_at_stop() -> AbstractContextManager[None]:
    """Bug 1: resources released at the stop request instead of at the end (breaks R9)."""

    def books_delta(event_type: str, attempt_state_before: str | None) -> int:
        if event_type == EventType.STARTED:
            return 1
        if event_type == EventType.STOP_REQUESTED:
            return -1
        if event_type == EventType.ATTEMPT_ENDED:
            return 0 if attempt_state_before == "STOPPING" else -1
        return 0

    return _patch(ops, "books_delta", books_delta)


def no_epoch_check() -> AbstractContextManager[None]:
    """Bug 2: no epoch check on the lease (a paused controller keeps writing)."""
    return _patch(Tx, "check_fence", lambda self, fence: None)


def no_lost_grace() -> AbstractContextManager[None]:
    """Bug 3: an attempt is ended on the first snapshot that lacks it (R5 without its grace)."""
    return _patch(rules, "r5_due", lambda now_ms, since_ms, grace_ms: True)


def no_version_check() -> AbstractContextManager[None]:
    """Bug 4: no version check (writes overwrite whatever another writer committed)."""

    def cas_update(self: Tx, table: str, row_id: str, expected_version: int, values: dict[str, Any]) -> None:
        sets = ", ".join(f"{k} = ?" for k in values)
        n = self.x(
            f"UPDATE {table} SET {sets}, version = ? WHERE id = ?",
            [*values.values(), expected_version + 1, row_id],
        )
        if n != 1:
            raise VersionConflict(f"{table} {row_id} missing")

    return _patch(Tx, "cas_update", cas_update)


BUGS: dict[str, Callable[[], AbstractContextManager[None]]] = {
    "release_at_stop": release_at_stop,
    "no_epoch_check": no_epoch_check,
    "no_lost_grace": no_lost_grace,
    "no_version_check": no_version_check,
}


@contextmanager
def none() -> Iterator[None]:
    yield


def bug(name: str | None) -> AbstractContextManager[None]:
    if not name or name == "none":
        return none()
    if name not in BUGS:
        raise ValueError(f"unknown bug {name!r}; known: {', '.join(BUGS)}")
    return BUGS[name]()
