"""Policy failure handling: counting, the fifo+first_fit fallback, and recovery (docs/contracts.md §6)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ai_workload_platform.policy import Policy, PolicyFailure
from ai_workload_platform.policy.builtin import make_builtin

log = logging.getLogger("awp.policy")


class PolicyRunner:
    """Calls the configured policy; after `max_failures` consecutive failures it degrades to the
    fallback and retries the configured policy at min(60 s, 1 s * 2^j) on the platform clock."""

    def __init__(self, policy: Policy, *, max_failures: int = 3, metrics: Any = None) -> None:
        self.policy = policy
        self.fallback = make_builtin("fifo+first_fit")
        self.max_failures = max_failures
        self.metrics = metrics
        self.consecutive = 0
        self.degraded = False
        self.retry_j = 0
        self.next_try_ms: int | None = None
        self._hello: dict[str, Any] | None = None  # hello of the current session of `policy`
        self._fb_hello: dict[str, Any] | None = None
        self.failures: dict[str, int] = {}
        self.session_id = 0  # bumped whenever the configured policy gets a new hello
        self._hist_session = -1
        self._hist_seq = 0
        self.wake_ms: int | None = None  # the policy's requested wake-up (platform ms)
        self.last_failure: str | None = None

    @property
    def is_fallback(self) -> bool:
        return self.policy.name == "fifo+first_fit"

    def next_wakeup_ms(self) -> int | None:
        """The earliest of the degraded-mode retry and the wake-up the policy asked for (`wake_at_s`)."""
        c = [t for t in (self.next_try_ms if self.degraded else None, self.wake_ms) if t is not None]
        return min(c) if c else None

    def _session(
        self, policy: Policy, hello: dict[str, Any], current: dict[str, Any] | None
    ) -> dict[str, Any]:
        if current is None or current["cluster"] != hello["cluster"]:
            if current is not None:
                policy.close()
            if policy is self.policy:
                self.session_id += 1
            policy.hello(hello)
        return hello

    def decide(
        self,
        view: dict[str, Any],
        hello: dict[str, Any],
        now_ms: int,
        validate: Callable[[dict[str, Any]], list[Any]],
        history: Callable[[int], tuple[list[dict[str, Any]], int]] | None = None,
    ) -> list[Any]:
        """`history(after_seq)` returns (completions after that log position, the new position); the runner
        sends a session's first call all of them and later calls only the new ones."""
        if not self.degraded or (self.next_try_ms is not None and now_ms >= self.next_try_ms):
            try:
                try:
                    self._hello = self._session(self.policy, hello, self._hello)
                    if self._hist_session != self.session_id:
                        self._hist_session, self._hist_seq = self.session_id, 0
                    items, upto = history(self._hist_seq) if history is not None else ([], self._hist_seq)
                    decision = self.policy.schedule({**view, "history_new": items})
                except PolicyFailure:
                    raise
                except Exception as e:  # noqa: BLE001 - e.g. the command cannot start: a crash, never an abort
                    raise PolicyFailure("crash", f"{type(e).__name__}: {e}") from e
                try:
                    actions = validate(decision)
                except PolicyFailure:
                    raise
                except Exception as e:  # noqa: BLE001 - a reply of the wrong shape
                    raise PolicyFailure("malformed", f"{type(e).__name__}: {e}") from e
            except PolicyFailure as f:
                self._hello = None
                try:
                    self.policy.close()
                except Exception:  # noqa: BLE001 - closing a broken session must not stop the cycle
                    pass
                self._record(f, now_ms)
                if not self.degraded:
                    return []
            else:
                self._hist_seq = upto
                w = decision.get("wake_at_s")
                self.wake_ms = int(round(float(w) * 1000)) if w is not None else None
                self.consecutive = 0
                if self.degraded:
                    log.info("policy recovered", extra={"fields": {"policy": self.policy.name}})
                    self._set_degraded(False)
                return actions
        # degraded: the fallback decides
        self._fb_hello = self._session(self.fallback, hello, self._fb_hello)
        try:
            return validate(self.fallback.schedule(view))
        except Exception as e:  # noqa: BLE001 - cannot happen for a correct built-in; never abort
            log.error("fallback policy failed", extra={"fields": {"reason": str(e)}})
            return []

    def _record(self, f: PolicyFailure, now_ms: int) -> None:
        self.consecutive += 1
        self.failures[f.kind] = self.failures.get(f.kind, 0) + 1
        self.last_failure = f"{f.kind}: {f.message}"
        if self.metrics is not None:
            self.metrics.policy_failures.labels(kind=f.kind).inc()
        log.warning(
            "policy failure",
            extra={
                "fields": {
                    "policy": self.policy.name,
                    "kind": f.kind,
                    "reason": f.message,
                    "stderr_tail": f.stderr_tail[-500:],
                }
            },
        )
        if self.degraded:
            self.retry_j += 1
            self.next_try_ms = now_ms + min(60_000, 1000 * 2**self.retry_j)
        elif self.consecutive >= self.max_failures:
            self._set_degraded(True)
            self.retry_j = 0
            self.next_try_ms = now_ms + 1000

    def _set_degraded(self, on: bool) -> None:
        self.degraded = on
        if not on:
            self.next_try_ms = None
            self.retry_j = 0
        if self.metrics is not None:
            self.metrics.policy_degraded.set(1 if on else 0)
        if on:
            log.warning(
                "policy degraded; fifo+first_fit decides", extra={"fields": {"policy": self.policy.name}}
            )

    def close(self) -> None:
        for p in (self.policy, self.fallback):
            try:
                p.close()
            except Exception:  # noqa: BLE001
                pass
        self._hello = self._fb_hello = None
