"""Trace schema v1 of gpu-cluster-scheduler (16 columns) with its manifest: load, write, convert.

A loader rejects a missing or different header, a wrong field count, an unparsable or out-of-range
value, an empty required field, a duplicate job_id, and a row whose submit_s is smaller than the
previous row's; each error names the 1-based line number. A header with no rows is a valid empty
trace; a zero-byte file is invalid.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_workload_platform.controller.drivers import Submission

HEADER = (
    "job_id,submit_s,tenant,user,priority,gpus,workers,gpu_class,topology,cpus,mem_gb,runtime_s,estimate_s,"
    "preemptible,checkpoint_interval_s,max_wait_s"
).split(",")
DEC = re.compile(r"^\d+(\.\d{1,3})?$")


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class TraceRow:
    job_id: str
    submit_s: float
    tenant: str
    user: str
    priority: int
    gpus: int
    workers: int
    gpu_class: str
    topology: str
    cpus: int
    mem_gb: float
    runtime_s: float
    estimate_s: float
    preemptible: int
    checkpoint_interval_s: float
    max_wait_s: float | None

    def to_spec(self) -> dict[str, Any]:
        return {
            "id": self.job_id,
            "priority": self.priority,
            "gpus": self.gpus,
            "workers": self.workers,
            "gpu_class": self.gpu_class or None,
            "topology": self.topology,
            "cpus": self.cpus,
            "mem_gb": self.mem_gb,
            "estimate_s": self.estimate_s,
            "preemptible": bool(self.preemptible),
            "checkpoint_interval_s": self.checkpoint_interval_s,
            "max_wait_s": self.max_wait_s,
            "labels": {"user": self.user or self.tenant},
            "sim": {"runtime_s": self.runtime_s},
        }


def fmt_dec(x: float | None) -> str:
    """Shortest plain decimal with at most three decimals (12, 12.5, 0.001)."""
    if x is None:
        return ""
    s = f"{round(float(x), 3):.3f}".rstrip("0").rstrip(".")
    return s or "0"


def _dec(v: str, line: int, col: str, *, positive: bool = False, optional: bool = False) -> float | None:
    if v == "":
        if optional:
            return None
        raise TraceError(f"line {line}: {col} is empty")
    if not DEC.match(v):
        raise TraceError(f"line {line}: {col} {v!r} is not a decimal with at most three decimals")
    x = float(v)
    if positive and x <= 0:
        raise TraceError(f"line {line}: {col} must be > 0")
    return x


def _int(v: str, line: int, col: str, lo: int, hi: int | None = None) -> int:
    if not re.fullmatch(r"\d+", v or ""):
        raise TraceError(f"line {line}: {col} {v!r} is not an integer")
    x = int(v)
    if x < lo or (hi is not None and x > hi):
        raise TraceError(f"line {line}: {col} {x} out of range")
    return x


def parse_trace(data: bytes) -> list[TraceRow]:
    if len(data) == 0:
        raise TraceError("line 1: empty file (a trace needs at least the header)")
    text = data.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows: list[TraceRow] = []
    seen: set[str] = set()
    prev = -1.0
    for i, rec in enumerate(reader, start=1):
        if i == 1:
            if rec != HEADER:
                raise TraceError(f"line 1: header must be exactly {','.join(HEADER)}")
            continue
        if not rec:
            continue
        if len(rec) != len(HEADER):
            raise TraceError(f"line {i}: {len(rec)} fields, expected {len(HEADER)}")
        f = dict(zip(HEADER, rec, strict=True))
        jid = f["job_id"]
        if not jid:
            raise TraceError(f"line {i}: job_id is empty")
        if jid in seen:
            raise TraceError(f"line {i}: duplicate job_id {jid}")
        seen.add(jid)
        if not f["tenant"]:
            raise TraceError(f"line {i}: tenant is empty")
        submit = _dec(f["submit_s"], i, "submit_s")
        assert submit is not None
        if submit < prev:
            raise TraceError(f"line {i}: submit_s {submit} is smaller than the previous row's {prev}")
        prev = submit
        topo = f["topology"]
        if topo not in ("any", "rack", "node"):
            raise TraceError(f"line {i}: topology must be any, rack, or node")
        rows.append(
            TraceRow(
                job_id=jid,
                submit_s=submit,
                tenant=f["tenant"],
                user=f["user"],
                priority=_int(f["priority"], i, "priority", 0, 9),
                gpus=_int(f["gpus"], i, "gpus", 1),
                workers=_int(f["workers"], i, "workers", 1),
                gpu_class=f["gpu_class"],
                topology=topo,
                cpus=_int(f["cpus"], i, "cpus", 0),
                mem_gb=_dec(f["mem_gb"], i, "mem_gb") or 0.0,
                runtime_s=_dec(f["runtime_s"], i, "runtime_s", positive=True) or 0.0,
                estimate_s=_dec(f["estimate_s"], i, "estimate_s", positive=True) or 0.0,
                preemptible=_int(f["preemptible"], i, "preemptible", 0, 1),
                checkpoint_interval_s=_dec(f["checkpoint_interval_s"], i, "checkpoint_interval_s") or 0.0,
                max_wait_s=_dec(f["max_wait_s"], i, "max_wait_s", optional=True),
            )
        )
    return rows


def write_trace_bytes(rows: list[TraceRow]) -> bytes:
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(HEADER)
    for r in rows:
        w.writerow(
            [
                r.job_id,
                fmt_dec(r.submit_s),
                r.tenant,
                r.user,
                r.priority,
                r.gpus,
                r.workers,
                r.gpu_class,
                r.topology,
                r.cpus,
                fmt_dec(r.mem_gb),
                fmt_dec(r.runtime_s),
                fmt_dec(r.estimate_s),
                r.preemptible,
                fmt_dec(r.checkpoint_interval_s),
                fmt_dec(r.max_wait_s),
            ]
        )
    return out.getvalue().encode("utf-8")


def manifest_for(data: bytes, rows: list[TraceRow], generator: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": generator,
        "seed": seed,
        "jobs": len(rows),
        "duration_s": rows[-1].submit_s if rows else 0,
        "content_sha256": hashlib.sha256(data).hexdigest(),
    }


def write_trace(
    path: str | Path, rows: list[TraceRow], generator: dict[str, Any], seed: int
) -> dict[str, Any]:
    p = Path(path)
    data = write_trace_bytes(rows)
    p.write_bytes(data)
    man = manifest_for(data, rows, generator, seed)
    manifest_path(p).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man


def manifest_path(path: str | Path) -> Path:
    p = Path(path)
    return (
        p.with_name(p.name[: -len(".csv")] + ".manifest.json")
        if p.name.endswith(".csv")
        else Path(f"{p}.manifest.json")
    )


def load_trace(path: str | Path, *, check_manifest: bool = True) -> list[TraceRow]:
    p = Path(path)
    data = p.read_bytes()
    rows = parse_trace(data)
    if check_manifest:
        mp = manifest_path(p)
        if not mp.exists():
            raise TraceError(f"{mp.name}: manifest missing")
        man = json.loads(mp.read_text(encoding="utf-8"))
        if man.get("schema_version") != 1:
            raise TraceError(f"{mp.name}: schema_version must be 1")
        if man.get("content_sha256") != hashlib.sha256(data).hexdigest():
            raise TraceError(f"{mp.name}: content_sha256 does not match the CSV")
        if man.get("jobs") != len(rows):
            raise TraceError(f"{mp.name}: jobs {man.get('jobs')} but the CSV has {len(rows)} rows")
    return rows


def to_submissions(rows: list[TraceRow], *, time_scale: float = 1.0) -> list[Submission]:
    """Submissions on the platform clock: submit_s (divided by time_scale for a live replay) in ms."""
    return [
        Submission(int(round(r.submit_s * 1000 / time_scale)), r.tenant, r.to_spec(), key=f"trace-{r.job_id}")
        for r in rows
    ]
