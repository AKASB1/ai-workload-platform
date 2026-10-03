"""Cluster configuration v1 (the format of gpu-cluster-scheduler, its docs/contracts.md §3) and topology.

Nodes are ordered by (rack, name); that order is the node index used everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_workload_platform.models import NodeInfo
from ai_workload_platform.models.spec import is_dns_label, mem_mb

TOP_KEYS = {
    "schema_version",
    "name",
    "classes",
    "node_groups",
    "nodes",
    "cross_node_factor",
    "cross_rack_factor",
    "restart_overhead_s",
    "preempt_grace_s",
    "sources",
}
GROUP_KEYS = {"count", "prefix", "rack", "class", "gpus", "cpus", "mem_gb"}
NODE_KEYS = {"name", "rack", "class", "gpus", "cpus", "mem_gb"}


class ClusterConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    classes: dict[str, float]
    nodes: tuple[NodeInfo, ...]
    cross_node_factor: float = 1.1
    cross_rack_factor: float = 1.25
    restart_overhead_s: float = 120.0
    preempt_grace_s: float = 0.0
    sources: dict[str, Any] = field(default_factory=dict)

    @property
    def total_gpus(self) -> int:
        return sum(n.gpus for n in self.nodes)

    @property
    def work_capacity(self) -> float:
        """Reference-GPU-seconds per second: sum of GPUs x class speed."""
        return sum(n.gpus * n.speed for n in self.nodes)

    def with_factors(self, cross_node: float, cross_rack: float) -> ClusterConfig:
        return ClusterConfig(
            self.name,
            self.classes,
            self.nodes,
            cross_node,
            cross_rack,
            self.restart_overhead_s,
            self.preempt_grace_s,
            self.sources,
        )

    def hello_cluster(self, nodes: list[NodeInfo] | None = None) -> dict[str, Any]:
        """The `cluster` object of the protocol's `hello`, from the given inventory (default: config)."""
        ns = sorted(nodes if nodes is not None else self.nodes, key=lambda n: (n.rack, n.name))
        classes = sorted({n.gpu_class: n.speed for n in ns}.items()) or sorted(self.classes.items())
        return {
            "classes": [{"name": c, "speed": s} for c, s in classes],
            "racks": sorted({n.rack for n in ns}),
            "cross_node_factor": self.cross_node_factor,
            "cross_rack_factor": self.cross_rack_factor,
            "restart_overhead_s": self.restart_overhead_s,
            "preempt_grace_s": self.preempt_grace_s,
            "nodes": [
                {
                    "name": n.name,
                    "rack": n.rack,
                    "class": n.gpu_class,
                    "speed": n.speed,
                    "gpus": n.gpus,
                    "cpus": n.cpus,
                    "mem_mb": n.mem_mb,
                }
                for n in ns
            ],
        }


def _num(d: dict, k: str, where: str, *, positive: bool = True, integer: bool = False) -> float:
    v = d.get(k)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ClusterConfigError(f"{where}.{k}: must be a number")
    if integer and int(v) != v:
        raise ClusterConfigError(f"{where}.{k}: must be an integer")
    if positive and v <= 0:
        raise ClusterConfigError(f"{where}.{k}: must be > 0")
    return v


def parse_cluster(obj: Any) -> ClusterConfig:
    if not isinstance(obj, dict):
        raise ClusterConfigError("cluster configuration must be a JSON object")
    unknown = set(obj) - TOP_KEYS
    if unknown:
        raise ClusterConfigError(f"unknown fields: {sorted(unknown)}")
    if obj.get("schema_version") != 1:
        raise ClusterConfigError("schema_version must be 1")
    classes: dict[str, float] = {}
    for i, c in enumerate(obj.get("classes") or []):
        if not isinstance(c, dict) or set(c) - {"name", "speed"} or not c.get("name"):
            raise ClusterConfigError(f"classes[{i}]: needs exactly name and speed")
        if c["name"] in classes:
            raise ClusterConfigError(f"classes[{i}]: duplicate class {c['name']}")
        classes[c["name"]] = float(_num(c, "speed", f"classes[{i}]"))
    if not classes:
        raise ClusterConfigError("classes: at least one class")
    raw_nodes: list[dict[str, Any]] = []
    for i, g in enumerate(obj.get("node_groups") or []):
        if not isinstance(g, dict) or set(g) != GROUP_KEYS:
            raise ClusterConfigError(f"node_groups[{i}]: fields must be exactly {sorted(GROUP_KEYS)}")
        count = int(_num(g, "count", f"node_groups[{i}]", integer=True))
        for k in range(count):
            raw_nodes.append({"name": f"{g['prefix']}{k:02d}", **{x: g[x] for x in NODE_KEYS - {"name"}}})
    for i, n in enumerate(obj.get("nodes") or []):
        if not isinstance(n, dict) or set(n) != NODE_KEYS:
            raise ClusterConfigError(f"nodes[{i}]: fields must be exactly {sorted(NODE_KEYS)}")
        raw_nodes.append(dict(n))
    if not raw_nodes:
        raise ClusterConfigError("at least one node")
    nodes: list[NodeInfo] = []
    seen: set[str] = set()
    for n in raw_nodes:
        where = f"node {n.get('name')}"
        if not isinstance(n["name"], str) or not is_dns_label(n["name"]):
            raise ClusterConfigError(f"{where}: name must be a DNS-1123 label of at most 40 characters")
        if n["name"] in seen:
            raise ClusterConfigError(f"{where}: duplicate name")
        seen.add(n["name"])
        if not isinstance(n["rack"], str) or not n["rack"]:
            raise ClusterConfigError(f"{where}: rack must be non-empty")
        if n["class"] not in classes:
            raise ClusterConfigError(f"{where}: unknown class {n['class']}")
        nodes.append(
            NodeInfo(
                n["name"],
                n["rack"],
                n["class"],
                classes[n["class"]],
                int(_num(n, "gpus", where, integer=True)),
                int(_num(n, "cpus", where, integer=True)),
                mem_mb(_num(n, "mem_gb", where)),
            )
        )
    cnf = float(obj.get("cross_node_factor", 1.0))
    crf = float(obj.get("cross_rack_factor", 1.0))
    if not 1.0 <= cnf <= crf:
        raise ClusterConfigError("need 1 <= cross_node_factor <= cross_rack_factor")
    ro = float(obj.get("restart_overhead_s", 0))
    pg = float(obj.get("preempt_grace_s", 0))
    if ro < 0 or pg < 0:
        raise ClusterConfigError("restart_overhead_s and preempt_grace_s must be >= 0")
    return ClusterConfig(
        str(obj.get("name", "")),
        classes,
        tuple(sorted(nodes, key=lambda x: (x.rack, x.name))),
        cnf,
        crf,
        ro,
        pg,
        dict(obj.get("sources") or {}),
    )


def load_cluster(path: str | Path) -> ClusterConfig:
    with open(path, encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ClusterConfigError(f"{path}: not JSON: {e}") from e
    return parse_cluster(obj)


def span(nodes: list[NodeInfo]) -> str:
    if len({n.name for n in nodes}) <= 1:
        return "node"
    if len({n.rack for n in nodes}) <= 1:
        return "rack"
    return "cluster"


def topology_factor(nodes: list[NodeInfo], topology: str, cross_node: float, cross_rack: float) -> float:
    """1 when the span is within the workload's sensitivity; else the cross-node or cross-rack factor."""
    sp = span(nodes)
    if topology == "any" or sp == "node":
        return 1.0
    if topology == "rack":
        return cross_rack if sp == "cluster" else 1.0
    # topology == "node"
    return cross_rack if sp == "cluster" else cross_node


def rate_of(nodes: list[NodeInfo], topology: str, cross_node: float, cross_rack: float) -> float:
    return min(n.speed for n in nodes) / topology_factor(nodes, topology, cross_node, cross_rack)
