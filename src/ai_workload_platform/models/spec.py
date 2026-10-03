"""Workload specification: validation, defaults, canonical form, and hash (docs/contracts.md §3)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ai_workload_platform.models import PlatformError

DNS1123 = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MAX_ID_LEN = 40


def is_dns_label(value: str, max_len: int = MAX_ID_LEN) -> bool:
    return isinstance(value, str) and 0 < len(value) <= max_len and DNS1123.match(value) is not None


def _three_decimals(v: float) -> float:
    if isinstance(v, bool):
        raise ValueError("must be a number")
    scaled = float(v) * 1000.0
    if abs(scaled - round(scaled)) > 1e-6:
        raise ValueError("at most three decimals")
    return float(v)


def _norm_num(v: float | int | None) -> float | int | None:
    """Canonical number: an integral value becomes an int, others keep three decimals."""
    if v is None:
        return None
    f = round(float(v), 3)
    return int(f) if f == int(f) else f


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetrySpec(_Strict):
    max_attempts: int = Field(3, ge=1, le=10)
    backoff_base_s: float = Field(5, ge=0, le=86400)
    backoff_cap_s: float = Field(300, ge=0, le=86400)
    jitter: Literal["full", "none"] = "full"
    fatal_exit_codes: list[int] = Field(default_factory=list, max_length=32)

    @field_validator("backoff_base_s", "backoff_cap_s")
    @classmethod
    def _dec(cls, v: float) -> float:
        return _three_decimals(v)

    @field_validator("fatal_exit_codes")
    @classmethod
    def _codes(cls, v: list[int]) -> list[int]:
        for c in v:
            if not 1 <= c <= 255:
                raise ValueError("exit codes are 1..255")
        return sorted(set(v))


class SimSpec(_Strict):
    runtime_s: float = Field(gt=0, le=10_000_000)
    fail_after_s: float | None = Field(None, gt=0, le=10_000_000)
    fail_attempts: int = Field(0, ge=0, le=10)
    exit_code: int = Field(1, ge=1, le=255)

    @field_validator("runtime_s", "fail_after_s")
    @classmethod
    def _dec(cls, v: float | None) -> float | None:
        return None if v is None else _three_decimals(v)

    @model_validator(mode="after")
    def _fail(self) -> SimSpec:
        if self.fail_attempts > 0 and self.fail_after_s is None:
            raise ValueError("fail_after_s is required when fail_attempts > 0")
        return self


class WorkloadSpec(_Strict):
    id: str | None = None
    priority: int = Field(4, ge=0, le=9)
    gpus: int = Field(ge=1, le=1024)
    workers: int = Field(1, ge=1, le=256)
    gpu_class: str | None = Field(None, max_length=40)
    topology: Literal["any", "rack", "node"] = "any"
    cpus: int = Field(0, ge=0, le=100_000)
    mem_gb: float = Field(0, ge=0, le=1_000_000)
    estimate_s: float = Field(3600, gt=0, le=10_000_000)
    preemptible: bool = False
    checkpoint_interval_s: float = Field(0, ge=0, le=10_000_000)
    max_wait_s: float | None = Field(None, ge=0, le=10_000_000)
    retry: RetrySpec = Field(default_factory=RetrySpec)
    labels: dict[str, str] = Field(default_factory=dict)
    image: str | None = Field(None, max_length=512)
    command: list[str] | None = Field(None, max_length=64)
    sim: SimSpec

    @field_validator("id")
    @classmethod
    def _id(cls, v: str | None) -> str | None:
        if v is not None and not is_dns_label(v):
            raise ValueError("must be a DNS-1123 label of at most 40 characters")
        return v

    @field_validator("gpu_class")
    @classmethod
    def _cls(cls, v: str | None) -> str | None:
        return v or None

    @field_validator("mem_gb", "estimate_s", "checkpoint_interval_s", "max_wait_s")
    @classmethod
    def _dec(cls, v: float | None) -> float | None:
        return None if v is None else _three_decimals(v)

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 8:
            raise ValueError("at most 8 labels")
        for k, val in v.items():
            if not is_dns_label(k, 63):
                raise ValueError(f"label key {k!r} is not a DNS-1123 label")
            if len(val) > 63:
                raise ValueError(f"label value of {k!r} is longer than 63 characters")
        return v


def _loc(err: dict[str, Any]) -> str:
    parts = [str(p) for p in err.get("loc", ()) if not (isinstance(p, str) and p.startswith("function-"))]
    return ".".join(parts) if parts else "(body)"


def validate_spec(raw: Any) -> dict[str, Any]:
    """Validate a submission body; return the canonical specification (defaults filled).

    Raises PlatformError INVALID_SPEC (422) whose details list every failing field path.
    """
    if not isinstance(raw, dict):
        raise PlatformError(
            "specification must be a JSON object", {"fields": ["(body)"]}, code="INVALID_SPEC", status=422
        )
    try:
        spec = WorkloadSpec.model_validate(raw)
    except ValidationError as e:
        fields: list[str] = []
        problems: dict[str, str] = {}
        for err in e.errors():
            path = _loc(err)
            if path not in problems:
                fields.append(path)
                problems[path] = err.get("msg", "invalid")
        raise PlatformError(
            "invalid specification", {"fields": fields, "problems": problems}, code="INVALID_SPEC", status=422
        ) from None
    return canonical(spec)


def canonical(spec: WorkloadSpec) -> dict[str, Any]:
    d = spec.model_dump()
    for k in ("mem_gb", "estimate_s", "checkpoint_interval_s", "max_wait_s"):
        d[k] = _norm_num(d[k])
    for k in ("backoff_base_s", "backoff_cap_s"):
        d["retry"][k] = _norm_num(d["retry"][k])
    for k in ("runtime_s", "fail_after_s"):
        d["sim"][k] = _norm_num(d["sim"][k])
    d["labels"] = dict(sorted(d["labels"].items()))
    return _sorted(d)


def _sorted(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _sorted(x[k]) for k in sorted(x)}
    if isinstance(x, list):
        return [_sorted(v) for v in x]
    return x


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(canon: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(canon).encode("utf-8")).hexdigest()


def mem_mb(mem_gb: float | int) -> int:
    """Memory in whole MB, as in 07: round(mem_gb * 1000)."""
    return int(round(float(mem_gb) * 1000.0))
