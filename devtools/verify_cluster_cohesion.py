"""Validate that proposed clusters from the projection would be cohesive.

For each declared subpackage in ``docs/plans/topology-target.yaml``, build
the import graph using stdlib ``ast`` and report:

  * cross-cluster imports through internals (bypassing ``__init__.py``)
  * cycles between clusters
  * "lonely" clusters (< min-cluster-size modules)
  * external fan-in count per cluster (rough cohesion signal)

The check uses TARGET paths, not current paths — it answers
"would the proposed split actually be cohesive?" without anyone moving a
file. This is what catches bad cuts before the refactor PR is opened.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "docs" / "plans" / "topology-target.yaml"


def parse_yaml(text: str) -> list[dict[str, Any]]:
    """Same minimal YAML parser as verify_topology.py."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "files:":
            continue
        if line.startswith("  - path: "):
            if current is not None:
                rows.append(current)
            current = {"path": line[len("  - path: ") :].strip()}
            continue
        if current is None:
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent != 4:
            continue
        if ": " in stripped:
            key, _, value = stripped.partition(": ")
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"')
            current[key] = value
    if current is not None:
        rows.append(current)
    return rows


def cluster_for(target: str) -> str | None:
    """Map a target path to its cluster identifier.

    Cluster = the first two segments under polylogue/ that produce a
    subpackage, e.g. polylogue/archive/query/foo.py → 'archive/query'.
    Files at polylogue/<dir>/<file>.py with no further nesting are in cluster
    '<dir>'.
    """
    if not target.startswith("polylogue/") or target == "TBD":
        return None
    parts = target[len("polylogue/") :].split("/")
    if len(parts) == 1:
        return "<root>"
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def module_name_to_cluster(name: str, cluster_modules: dict[str, str]) -> str | None:
    """Look up the cluster for a fully-qualified polylogue module name."""
    if not name.startswith("polylogue."):
        return None
    rel = name[len("polylogue.") :].replace(".", "/")
    candidates = [rel + ".py", rel + "/__init__.py"]
    for cand in candidates:
        target = f"polylogue/{cand}"
        if target in cluster_modules:
            return cluster_modules[target]
    # try matching prefixes
    for path, cluster in cluster_modules.items():
        if path.startswith(f"polylogue/{rel}/") or path == f"polylogue/{rel}.py":
            return cluster
    return None


def collect_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=PROJECTION)
    p.add_argument("--cluster", help="Limit report to one cluster name (e.g. 'archive/query').")
    p.add_argument("--min-cluster-size", type=int, default=3)
    args = p.parse_args(argv)

    rows = parse_yaml(args.yaml.read_text())

    # Map current path → cluster (using target).
    path_to_cluster: dict[str, str] = {}
    for row in rows:
        target = row.get("target", "")
        if target == "TBD":
            continue
        c = cluster_for(target)
        if c is not None:
            # Use current path because we're scanning current files.
            path_to_cluster[row["path"]] = c

    cluster_members: dict[str, list[str]] = defaultdict(list)
    for path, cluster in path_to_cluster.items():
        cluster_members[cluster].append(path)

    # Build import edges using current paths.
    cross_internals: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    cluster_edges: dict[str, set[str]] = defaultdict(set)
    cluster_sizes = {k: len(v) for k, v in cluster_members.items()}

    for path, cluster in path_to_cluster.items():
        full_path = ROOT / path
        if not full_path.exists():
            continue
        for imp in collect_imports(full_path):
            other = module_name_to_cluster(imp, path_to_cluster)
            if other is None or other == cluster:
                continue
            cluster_edges[cluster].add(other)
            # Detect through-internals: import names a module path that ends
            # with a non-__init__ name and is not the cluster's own __init__.
            if imp.startswith("polylogue."):
                rel = imp[len("polylogue.") :].replace(".", "/")
                # If the import names polylogue.<a>.<b>.<c>, the imported
                # module is `c.py` inside cluster <a>/<b>; cohesion failure
                # if c != __init__ and rel doesn't exactly equal a cluster
                # __init__ path.
                init_path = f"polylogue/{rel}/__init__.py"
                if init_path not in path_to_cluster:
                    cross_internals[cluster].append((path, imp, other))

    # Cycles between clusters
    cycles: list[tuple[str, str]] = []
    seen_pairs: set[frozenset[str]] = set()
    for src, tgts in cluster_edges.items():
        for tgt in tgts:
            if src in cluster_edges.get(tgt, set()):
                pair = frozenset([src, tgt])
                if pair not in seen_pairs and src != tgt:
                    cycles.append((src, tgt))
                    seen_pairs.add(pair)

    # Lonely clusters
    lonely = sorted(c for c, n in cluster_sizes.items() if n < args.min_cluster_size and c not in {"<root>"})

    print("=" * 72)
    print("CLUSTER COHESION REPORT")
    print("=" * 72)
    print()

    print(f"{'cluster':<40s} {'size':>6s} {'fan-out':>9s} {'inbound':>9s}")
    inbound: dict[str, int] = defaultdict(int)
    for tgts in cluster_edges.values():
        for tgt in tgts:
            inbound[tgt] += 1
    targets_to_show = sorted(cluster_sizes, key=lambda k: -cluster_sizes[k])
    if args.cluster:
        targets_to_show = [args.cluster] if args.cluster in cluster_sizes else []
    for c in targets_to_show:
        size = cluster_sizes[c]
        fan_out = len(cluster_edges.get(c, set()))
        ib = inbound.get(c, 0)
        print(f"{c:<40s} {size:>6d} {fan_out:>9d} {ib:>9d}")

    print()
    print(f"Cycles between proposed clusters: {len(cycles)}")
    for a, b in cycles[:20]:
        print(f"  {a} <-> {b}")
    if len(cycles) > 20:
        print(f"  ... and {len(cycles) - 20} more")

    print()
    print(f"Lonely clusters (< {args.min_cluster_size} modules): {len(lonely)}")
    for c in lonely:
        print(f"  {c} ({cluster_sizes[c]})")

    print()
    cross_total = sum(len(v) for v in cross_internals.values())
    print(f"Cross-cluster imports through internals (bypass __init__): {cross_total}")
    for c in sorted(cross_internals, key=lambda k: -len(cross_internals[k]))[:5]:
        items = cross_internals[c]
        print(f"  {c}: {len(items)} (top samples)")
        for src, imp, dst in items[:3]:
            print(f"    {src} -> {imp}  (cluster {dst})")

    print()
    blocking = bool(cycles)
    print(f"BLOCKING={blocking}  (cycles between clusters)")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
