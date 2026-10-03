"""Backend contract. The backend executes and observes; it knows nothing about queues, quotas,
retries, or policies. The platform talks to it only through `SchedulerAdapter` (docs/contracts.md §7).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_workload_platform.models import AttemptRequest, BackendError, NodeInfo, Snapshot


@runtime_checkable
class SchedulerAdapter(Protocol):
    def inventory(self) -> list[NodeInfo]:
        """The nodes with name, rack, class, speed, total GPUs, CPUs, memory (MB), and readiness."""

    def start(self, attempt: AttemptRequest) -> None:
        """Begin the attempt. Idempotent on the attempt id."""

    def stop(self, attempt_id: str) -> None:
        """Ask the attempt to end. Idempotent; unknown or ended attempts are not an error."""

    def observe(self) -> Snapshot:
        """Every attempt the backend knows, ended ones included until forgotten. Complete or an error."""

    def forget(self, attempt_id: str) -> None:
        """The platform recorded the terminal state; release what is kept for the attempt."""

    def next_event_ms(self) -> int | None:
        """Virtual-time backends: the time of their next internal event (None: nothing scheduled)."""


__all__ = ["BackendError", "SchedulerAdapter"]
