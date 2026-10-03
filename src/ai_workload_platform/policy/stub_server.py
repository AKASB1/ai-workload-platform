"""Stub external policy: speaks protocol v1 over stdin/stdout and implements fifo+first_fit on the
wire form. Standard library only, deliberately independent of the platform package, so that it
behaves like any external policy (gpu-cluster-scheduler's policy server is started the same way).

Run: python -m ai_workload_platform.policy.stub_server

Test faults, chosen through `hello.policy.params`: {"fault": kind, "at_seq": n} with kind one of
error, crash, malformed, invalid, hang, unknown_field, wrong_seq (from `schedule` number n on).
"""

from __future__ import annotations

import json
import sys
import time


def first_fit(job, order, cls, free):
    left = job["workers"]
    out = []
    for name in order:
        if left == 0:
            break
        if name not in free or (job["gpu_class"] and cls[name] != job["gpu_class"]):
            continue
        g, c, m = free[name]
        k = g // job["gpus"]
        if job["cpus"] > 0:
            k = min(k, c // job["cpus"])
        if job["mem_mb"] > 0:
            k = min(k, m // job["mem_mb"])
        k = min(k, left)
        if k > 0:
            out.append((name, k))
            left -= k
    return out if left == 0 else None


def decide(cluster, view):
    order = [n["name"] for n in cluster["nodes"]]
    cls = {n["name"]: n["class"] for n in cluster["nodes"]}
    free = {n["name"]: [n["free_gpus"], n["free_cpus"], n["free_mem_mb"]] for n in view["nodes"]}
    actions = []
    for job in sorted(view["pending"], key=lambda j: (j["submit_s"], j["job_id"])):
        pl = first_fit(job, order, cls, free)
        if pl is None:
            break
        for name, k in pl:
            free[name][0] -= job["gpus"] * k
            free[name][1] -= job["cpus"] * k
            free[name][2] -= job["mem_mb"] * k
        actions.append(
            {"op": "start", "job_id": job["job_id"], "placement": [{"node": n, "workers": k} for n, k in pl]}
        )
    return actions


def send(out, obj):
    out.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    out.flush()


def main() -> int:
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    cluster = None
    params = {}
    for raw in inp:
        msg = json.loads(raw.decode("utf-8"))
        t = msg.get("type")
        if t == "hello":
            cluster = msg["cluster"]
            params = (msg.get("policy") or {}).get("params") or {}
            sys.stderr.write(f"stub: hello with {len(cluster['nodes'])} nodes\n")
            sys.stderr.flush()
            send(out, {"type": "hello", "name": "fifo+first_fit", "version": "awp-stub 1"})
        elif t == "schedule":
            seq = msg["seq"]
            fault = params.get("fault")
            if fault and seq >= int(params.get("at_seq", 1)):
                sys.stderr.write(f"stub: injecting {fault} at seq {seq}\n")
                sys.stderr.flush()
                if fault == "error":
                    send(out, {"type": "error", "message": "stub error on request"})
                    continue
                if fault == "crash":
                    return 3
                if fault == "malformed":
                    out.write(b"this is not json\n")
                    out.flush()
                    continue
                if fault == "hang":
                    time.sleep(3600)
                if fault == "invalid":
                    send(
                        out,
                        {
                            "type": "decision",
                            "seq": seq,
                            "actions": [
                                {
                                    "op": "start",
                                    "job_id": "no-such-job",
                                    "placement": [{"node": "nowhere", "workers": 1}],
                                }
                            ],
                            "wake_at_s": None,
                            "reservations": [],
                            "solver": None,
                        },
                    )
                    continue
                if fault == "unknown_field":
                    send(
                        out,
                        {
                            "type": "decision",
                            "seq": seq,
                            "actions": [],
                            "wake_at_s": None,
                            "reservations": [],
                            "solver": None,
                            "surprise": 1,
                        },
                    )
                    continue
                if fault == "wrong_seq":
                    send(
                        out,
                        {
                            "type": "decision",
                            "seq": seq + 1,
                            "actions": [],
                            "wake_at_s": None,
                            "reservations": [],
                            "solver": None,
                        },
                    )
                    continue
            send(
                out,
                {
                    "type": "decision",
                    "seq": seq,
                    "actions": decide(cluster, msg["view"]),
                    "wake_at_s": None,
                    "reservations": [],
                    "solver": None,
                },
            )
        elif t == "bye":
            send(out, {"type": "bye"})
            return 0
        else:
            send(out, {"type": "error", "message": f"unknown message type {t!r}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
