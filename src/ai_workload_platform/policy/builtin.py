"""The built-in baselines, pure functions of the hello cluster and the view (wire form).

`+preempt` and `+reclaim` (Tier 2) follow gpu-cluster-scheduler's policy semantics (its docs/contracts.md §7):
victims by least lost work, then lower priority, then latest start, then id, at most 8, unneeded ones dropped
last added first. On this platform a stop takes time and R9 keeps the GPUs until the attempt has ended, so a
preemption only frees capacity later: the policy emits the preemptions, counts capacity that is already being
stopped for a preemption (`awp_stopping`) as soon free, and starts the waiting workload in a later cycle.
"""

from __future__ import annotations

import math
from typing import Any

BASE_NAMES = ("fifo+first_fit", "priority+best_fit", "quota+first_fit")
PREEMPT_NAMES = ("priority+best_fit+preempt", "quota+first_fit+reclaim")
MIN_GAP = 1
MAX_VICTIMS = 8


class _Free:
    def __init__(self, cluster: dict[str, Any], view: dict[str, Any]) -> None:
        self.order = [n["name"] for n in cluster["nodes"]]  # (rack, name) order from hello
        self.cls = {n["name"]: n["class"] for n in cluster["nodes"]}
        self.free = {n["name"]: [n["free_gpus"], n["free_cpus"], n["free_mem_mb"]] for n in view["nodes"]}

    @classmethod
    def of(cls, order: list[str], classes: dict[str, str], free: dict[str, list[int]]) -> _Free:
        f = cls.__new__(cls)
        f.order, f.cls, f.free = order, classes, {n: list(v) for n, v in free.items()}
        return f

    def add(self, entries: list[dict[str, Any]], gpus: int, cpus: int, mem: int, sign: int = 1) -> None:
        for e in entries:
            v = self.free.setdefault(e["node"], [0, 0, 0])
            k = int(e["workers"]) * sign
            v[0] += gpus * k
            v[1] += cpus * k
            v[2] += mem * k

    def per_node(self, name: str, job: dict[str, Any]) -> int:
        if name not in self.free or (job["gpu_class"] and self.cls.get(name) != job["gpu_class"]):
            return 0
        g, c, m = self.free[name]
        k = g // job["gpus"]
        if job["cpus"] > 0:
            k = min(k, c // job["cpus"])
        if job["mem_mb"] > 0:
            k = min(k, m // job["mem_mb"])
        return max(0, k)

    def take(self, placement: list[tuple[str, int]], job: dict[str, Any]) -> None:
        for name, k in placement:
            f = self.free[name]
            f[0] -= job["gpus"] * k
            f[1] -= job["cpus"] * k
            f[2] -= job["mem_mb"] * k

    def first_fit(self, job: dict[str, Any]) -> list[tuple[str, int]] | None:
        left = job["workers"]
        out = []
        for name in self.order:
            if left == 0:
                break
            k = min(left, self.per_node(name, job))
            if k > 0:
                out.append((name, k))
                left -= k
        return out if left == 0 else None

    def best_fit(self, job: dict[str, Any]) -> list[tuple[str, int]] | None:
        idx = {n: i for i, n in enumerate(self.order)}
        trial = {n: list(v) for n, v in self.free.items()}
        counts: dict[str, int] = {}
        for _ in range(job["workers"]):
            best = None
            for name in self.order:
                if name not in trial or (job["gpu_class"] and self.cls.get(name) != job["gpu_class"]):
                    continue
                g, c, m = trial[name]
                if g >= job["gpus"] and c >= job["cpus"] and m >= job["mem_mb"]:
                    key = (g - job["gpus"], idx[name])
                    if best is None or key < best[0]:
                        best = (key, name)
            if best is None:
                return None
            name = best[1]
            trial[name][0] -= job["gpus"]
            trial[name][1] -= job["cpus"]
            trial[name][2] -= job["mem_mb"]
            counts[name] = counts.get(name, 0) + 1
        return [(n, counts[n]) for n in self.order if n in counts]


def _caps(view: dict[str, Any]) -> dict[str, list[int]]:
    """namespace -> [quota, cap, allocated] from the in-process extension (absent: unlimited)."""
    return {
        n["tenant"]: [n["quota_gpus"], n["cap_gpus"], n["allocated_gpus"]]
        for n in view.get("awp_namespaces", [])
    }


def _action(job: dict[str, Any], placement: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        "op": "start",
        "job_id": job["job_id"],
        "placement": [{"node": n, "workers": k} for n, k in placement],
    }


def _decision(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "decision", "actions": actions, "wake_at_s": None, "reservations": [], "solver": None}


class BuiltinPolicy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cluster: dict[str, Any] | None = None

    def hello(self, msg: dict[str, Any]) -> None:
        self.cluster = msg["cluster"]

    def close(self) -> None:
        self.cluster = None

    def schedule(self, view: dict[str, Any]) -> dict[str, Any]:
        assert self.cluster is not None, "hello first"
        free = _Free(self.cluster, view)
        caps = _caps(view)
        if self.name == "fifo+first_fit":
            return _decision(self._fifo(view, free, caps))
        if self.name.startswith("priority+best_fit"):
            acts = self._priority(view, free, caps)
        else:
            acts = self._quota(view, free, caps)
        if self.name in PREEMPT_NAMES:
            acts += self._preempt_pass(view, free, caps, {a["job_id"] for a in acts})
        return _decision(acts)

    @staticmethod
    def _lost(r: dict[str, Any]) -> float:
        """Work a running job would lose: work done minus what its checkpoints retain (07's rule)."""
        done, ck = float(r["work_done_s"]), float(r["checkpoint_interval_s"] or 0)
        kept = min(done, math.floor(done / ck * (1 + 1e-9)) * ck) if ck > 0 else 0.0
        return r["gpus"] * r["workers"] * (done - kept)

    def _preempt_pass(self, view: dict, free: _Free, caps: dict, started: set[str]) -> list[dict]:
        reclaim = self.name.endswith("+reclaim")
        soon = {n: list(v) for n, v in (view.get("awp_stopping") or {}).items()}
        future = _Free.of(free.order, free.cls, free.free)
        for n, v in soon.items():
            fv = future.free.setdefault(n, [0, 0, 0])
            for i in range(3):
                fv[i] += v[i]
        fit = future.first_fit if reclaim else future.best_fit
        chosen: set[str] = set()
        out: list[dict] = []
        for job in view["pending"]:
            if job["job_id"] in started or not self._within(caps, job, 1):
                continue
            if reclaim and not self._within(caps, job, 0):
                continue  # only a workload within its namespace's quota may reclaim
            pl = fit(job)
            if pl is not None:  # fits once the stops already under way finish: preempt nothing more
                future.take(pl, job)
                continue

            def eligible(r: dict[str, Any], job: dict[str, Any] = job) -> bool:
                if not r["preemptible"] or r["job_id"] in chosen:
                    return False
                if reclaim:
                    c = caps.get(r["tenant"])
                    return r["tenant"] != job["tenant"] and c is not None and c[2] > c[0]
                return r["priority"] <= job["priority"] - MIN_GAP

            cands = sorted(
                (r for r in view["running"] if eligible(r)),
                key=lambda r: (self._lost(r), r["priority"], -r["run_start_s"], r["job_id"]),
            )
            victims: list[dict] = []
            trial = _Free.of(future.order, future.cls, future.free)
            for r in cands:
                if len(victims) >= MAX_VICTIMS:
                    break
                trial.add(r["placement"], r["gpus"], r["cpus"], r["mem_mb"])
                victims.append(r)
                if (trial.first_fit if reclaim else trial.best_fit)(job) is not None:
                    break
            if not victims or (trial.first_fit if reclaim else trial.best_fit)(job) is None:
                continue
            for r in reversed(list(victims)):  # drop victims the workload does not need, last added first
                trial.add(r["placement"], r["gpus"], r["cpus"], r["mem_mb"], -1)
                if (trial.first_fit if reclaim else trial.best_fit)(job) is not None:
                    victims.remove(r)
                else:
                    trial.add(r["placement"], r["gpus"], r["cpus"], r["mem_mb"])
            for r in victims:
                chosen.add(r["job_id"])
                out.append({"op": "preempt", "job_id": r["job_id"]})
                future.add(r["placement"], r["gpus"], r["cpus"], r["mem_mb"])
            pl = fit(job)
            if pl is not None:
                future.take(pl, job)  # later workloads do not count on the same capacity
        return out

    @staticmethod
    def _within(caps: dict[str, list[int]], job: dict[str, Any], limit: int) -> bool:
        c = caps.get(job["tenant"])
        if c is None:
            return True
        return c[2] + job["gpus"] * job["workers"] <= c[limit]

    @staticmethod
    def _book(caps: dict[str, list[int]], job: dict[str, Any]) -> None:
        if job["tenant"] in caps:
            caps[job["tenant"]][2] += job["gpus"] * job["workers"]

    def _fifo(self, view: dict, free: _Free, caps: dict) -> list[dict]:
        out = []
        for job in sorted(view["pending"], key=lambda j: (j["submit_s"], j["job_id"])):
            if not self._within(caps, job, 1):
                break
            pl = free.first_fit(job)
            if pl is None:
                break
            free.take(pl, job)
            self._book(caps, job)
            out.append(_action(job, pl))
        return out

    def _priority(self, view: dict, free: _Free, caps: dict) -> list[dict]:
        out = []
        for job in view["pending"]:  # already (priority high first, submit time, id)
            if not self._within(caps, job, 1):
                continue
            pl = free.best_fit(job)
            if pl is None:
                continue
            free.take(pl, job)
            self._book(caps, job)
            out.append(_action(job, pl))
        return out

    def _quota(self, view: dict, free: _Free, caps: dict) -> list[dict]:
        out = []
        deferred = []
        for job in view["pending"]:
            if not self._within(caps, job, 0):
                deferred.append(job)
                continue
            pl = free.first_fit(job)
            if pl is None:
                continue
            free.take(pl, job)
            self._book(caps, job)
            out.append(_action(job, pl))
        for job in deferred:
            if not self._within(caps, job, 1):
                continue
            pl = free.first_fit(job)
            if pl is None:
                continue
            free.take(pl, job)
            self._book(caps, job)
            out.append(_action(job, pl))
        return out


def make_builtin(name: str) -> BuiltinPolicy:
    if name not in BASE_NAMES + PREEMPT_NAMES:
        raise ValueError(name)
    return BuiltinPolicy(name)
