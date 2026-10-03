"""Named crash points. Production code runs them as no-ops; the fault harness installs a hook.

A hook may raise (a controller crash: the open transaction rolls back) or run another actor's
step at that instant (an interleaving: a client cancel, or another controller's ticks).
"""

from __future__ import annotations

from collections.abc import Callable

CRASH_POINTS = (
    "submit.after_commit",  # after the submit commit
    "start.after_commit",  # after the `started` commit and before the backend call
    "start.after_call",  # after the backend call
    "ended.before_forget",  # after a terminal observation (attempt_ended committed) and before `forget`
    "stop.before_call",  # between a stop request (committed) and the `stop` call
    "requeue.before_commit",  # inside the retry release, before `requeued` is committed
    "stopped.before_commit",  # after the backend confirms a stop and before `attempt_ended` is committed
)

# Interleaving points (outside transactions) where the harness may run a client action or
# another controller; they never sit inside an open transaction.
RACE_POINTS = (
    "cycle.after_view",  # the scheduling view is built, no action committed yet
    "rules.before_write",  # a repair was decided from the tick's read, before its transaction
    "tick.after_lease",  # the lease was acquired or renewed
)


class Crash(Exception):
    """Raised by the harness hook to simulate a controller crash at a named point."""

    def __init__(self, point: str) -> None:
        super().__init__(point)
        self.point = point


_hook: Callable[[str, str | None], None] | None = None


def crashpoint(name: str, subject: str | None = None) -> None:
    """`subject`: the workload the code is about to write, when there is one (interleaving points)."""
    if _hook is not None:
        _hook(name, subject)


def install(hook: Callable[[str, str | None], None] | None) -> Callable[[str, str | None], None] | None:
    """Install a hook (None removes it); returns the previous one."""
    global _hook
    prev = _hook
    _hook = hook
    return prev
