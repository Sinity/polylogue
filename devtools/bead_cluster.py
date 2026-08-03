"""bead-cluster: execution-frontier clustering for polylogue Beads.

Implements polylogue-2yax: reads bd ready (or a supplied JSON export), extracts
file/package/resource footprints from each bead's design+notes+ac+description
text, builds
a weighted overlap graph, and emits:

  - FRONTIER-READY   -- horizon:frontier, no blocking deps -> claim now
  - BLOCKED          -- has unmet hard dependencies
  - IN-PROGRESS       -- already claimed
  - DESIGN-HORIZON    -- mid/vision horizon, not yet executable

For frontier-ready beads it computes:
  - OVERLAPPING clusters  -> one branch/PR sweep (share context, single rewrite pass)
  - DISJOINT sets         -> safe to run as parallel lanes / separate worktrees
  - CONTENTION points     -> same migration tier/slot, same generated surface, same DB

Clustering algorithm (polylogue-1ebm follow-up, fixing the degeneracy
measured on the 2026-07-30 live backlog): the original clustering used
unweighted connected-components over "any shared footprint token" --
a small number of hub files/packages (e.g. docs/internals.md,
polylogue/config.py) appear in 7-8 beads each, and any-shared-token
transitivity chains those beads into one mega-cluster (measured: 165 beads
in a single cluster, 93 true singletons). Hub tokens carry almost no
information about whether two beads should share a branch.

The fix is TF-IDF-flavored edge weighting instead of unweighted adjacency:

  1. Compute document frequency ``df[token]`` for every overlap token
     (exact file path or specific sub-package dir) across the population
     of beads being clustered.
  2. Pairwise edge weight between two beads = sum over their SHARED tokens
     of ``log2(N / df[token])`` (rare, shared tokens score high; hub tokens
     shared by many beads score near zero) plus ``0.5`` per shared
     ``area:`` label.
  3. Keep only edges with weight >= ``--min-similarity`` (default 3.0 --
     roughly "one token shared by only ~1/8 of the population").
  4. Sort edges by weight descending; greedy union-find merge, REFUSING
     any merge that would grow a cluster past ``--max-cluster-size``
     (default 6) -- this caps how far one strong edge can drag in
     unrelated beads through transitive chaining.

Measured on the 2026-07-31 backlog at merge time (PR #3462, ``--all-open
--json``), this produced 33 multi-bead clusters covering 161 beads instead
of one blob -- coherent themes (a claude-ai-export parser-fidelity cluster,
a CLI read/render-vocabulary cluster, a repair.py shipped-but-dead cluster).
An earlier draft of this docstring quoted a different, never-actually-
measured "40 clusters / 156 beads / 457 singletons" split -- that number was
aspirational and is corrected here. The backlog moves daily (beads close,
new ones land), so an exact split is not durable; re-measure with
``devtools workspace bead-cluster --all-open --json`` and count
``frontier_clusters`` entries with ``len(beads) > 1`` rather than trusting a
frozen number. The plain any-overlap graph (``_build_overlap_graph`` /
``_connected_components``) is kept as a general-purpose utility (also used
by ``devtools workspace lane-brief`` and other footprint tooling) but is no
longer used to shape the FRONTIER-READY cluster output.

Usage:
    # From repo root, using live bd output:
    devtools workspace bead-cluster

    # Read a pre-exported JSON (e.g. bd ready --json --limit 0 > ready.json):
    devtools workspace bead-cluster --input ready.json

    # Include all open beads (not just ready):
    devtools workspace bead-cluster --all-open

    # Machine-readable JSON output:
    devtools workspace bead-cluster --json

    # Filter to a priority ceiling:
    devtools workspace bead-cluster --max-priority 1

    # Tune the weighted-clustering thresholds:
    devtools workspace bead-cluster --min-similarity 2.5 --max-cluster-size 8

    # Replay-validate against the 2026-07-13 migration-slot collision:
    devtools workspace bead-cluster --validate-roster

Notes:
  Footprints are extracted heuristically -- they are ADVISORY. A missing or
  ambiguous footprint lowers readiness and is reported as NEEDS-CONFIRM.
  Never treat inferred footprints as correctness proof.

History: originally shipped 2026-07-16 as .agent/tools/bead-cluster.py
(commit 49182a7f2) but never registered as a devtools CommandSpec, so it was
invisible to `devtools --help` / `render devtools-reference` / the repo's
catalog-completeness checks. PR #3188 (2026-07-20) classified the unregistered
file as dead code ("no live references") and deleted it. polylogue-1ebm
recovers it here, registered, so the analysis is discoverable again. The
mega-cluster degeneracy above was measured and fixed the same week.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

BeadDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Footprint extraction
# ---------------------------------------------------------------------------

# Regex: file paths mentioning known project trees
_FILE_PAT = re.compile(
    r"(?:polylogue|tests|\.agent|docs|storage|pipeline|daemon|cli|mcp"
    r"|browser[_-]extension|browser_capture|coordination|archive|insights"
    r"|context|core|hooks|maintenance|artifacts)"
    r"/[\w./\-]+\.(?:py|ts|js|yaml|yml|md|json|sql|html)",
    re.IGNORECASE,
)

# Regex: migration slot references  e.g. "migration 008", "slot 008", "NNN_"
_MIGRATION_PAT = re.compile(r"\b(?:migration|slot)\s*[#]?(\d{3,4})\b|(\d{3})_\w+\.sql", re.IGNORECASE)

# Regex: generated surface families
_GENERATED_SURFACES = {
    "topology-target.yaml": "generated:topology-target",
    "cli-reference.md": "generated:cli-reference",
    "openapi/": "generated:openapi",
}

# Area label -> primary package(s)
_AREA_TO_PACKAGES: dict[str, list[str]] = {
    "area:lineage": ["storage/sqlite/", "polylogue/archive/"],
    "area:storage": ["storage/sqlite/", "polylogue/storage/"],
    "area:daemon": ["polylogue/daemon/"],
    "area:ingest": ["pipeline/", "polylogue/daemon/"],
    "area:mcp": ["polylogue/mcp/"],
    "area:query": ["polylogue/archive/query/", "polylogue/storage/search/"],
    "area:cli": ["polylogue/cli/"],
    "area:browser": ["browser-extension/", "polylogue/browser_capture/"],
    "area:sources": ["polylogue/pipeline/", "polylogue/core/"],
    "area:context": ["polylogue/context/"],
    "area:substrate": ["polylogue/storage/"],
    "area:analytics": ["polylogue/cost/", "polylogue/insights/"],
    "area:perf": ["polylogue/archive/query/", "polylogue/storage/"],
    "area:coordination": ["polylogue/coordination/"],
    "area:web": ["browser-extension/"],
    "area:sinex": ["polylogue/coordination/"],
    "area:test": ["tests/"],
    "area:audit": ["devtools/", ".agent/"],
    "area:devtools": ["devtools/"],
    "area:ops": ["polylogue/maintenance/"],
    "area:architecture": [],
    "area:beads": [".beads/", ".agent/"],
    "area:capture": ["browser-extension/", "polylogue/browser_capture/"],
}


# Top-level package dirs that are too broad to use as overlap discriminators.
# Nearly every bead touches these, so they would collapse the entire frontier
# into one mega-cluster if used as clustering keys.
_BROAD_PACKAGES: frozenset[str] = frozenset(
    {
        "tests/",
        "polylogue/storage/",
        "polylogue/archive/",
        "polylogue/",
        ".agent/",
        ".agent/scratch/",
        ".agent/handoffs/",
        "polylogue/sources/",
        "storage/sqlite/",
        "pipeline/",
        "docs/",
    }
)


@dataclass
class Footprint:
    files: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    migration_slots: list[str] = field(default_factory=list)
    generated_surfaces: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.files or self.packages or self.areas)

    def overlap_keys(self) -> set[str]:
        """Keys used to detect footprint overlap between beads.

        Uses concrete file paths and specific sub-package dirs only.
        Broad roots (tests/, polylogue/storage/, etc.) are excluded -- they
        would collapse the entire frontier into one mega-cluster.
        Area labels are metadata, not clustering keys.
        Migration slots belong in contention_keys only.
        """
        keys: set[str] = set()
        # Exact file paths (most discriminating)
        keys.update(self.files)
        # Package dirs: exclude broad shared roots
        for pkg in self.packages:
            if pkg not in _BROAD_PACKAGES:
                keys.add(pkg)
        return keys

    def contention_keys(self) -> set[str]:
        keys: set[str] = set()
        keys.update(f"migration:{s}" for s in self.migration_slots)
        keys.update(self.generated_surfaces)
        return keys


def _extract_footprint(item: BeadDict) -> Footprint:
    text = " ".join(
        filter(
            None,
            [
                item.get("design", ""),
                item.get("notes", ""),
                item.get("acceptance_criteria", ""),
                item.get("description", ""),
            ],
        )
    )
    labels: list[str] = item.get("labels", [])

    # Raw file paths (most specific footprint signal)
    raw_files = list(dict.fromkeys(_FILE_PAT.findall(text)))  # deduplicate, preserve order

    # Package families: second path segment (e.g. polylogue/mcp/, tests/unit/)
    packages: set[str] = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 3:
            # Use the two-level prefix for specificity
            packages.add(f"{parts[0]}/{parts[1]}/")
        elif len(parts) == 2:
            packages.add(f"{parts[0]}/{parts[1]}/")
        elif parts:
            packages.add(parts[0] + "/")

    # Only fall back to area->package mapping when NO files found at all.
    # When files ARE found, area->package would add broad roots that over-merge.
    if not raw_files:
        for label in labels:
            for pkg in _AREA_TO_PACKAGES.get(label, []):
                packages.add(pkg)

    # Area labels -- stored for display and metadata, NOT used as overlap keys.
    areas: list[str] = [lbl for lbl in labels if lbl.startswith("area:")]

    # Migration slots
    migration_slots: list[str] = []
    for m in _MIGRATION_PAT.finditer(text):
        slot = m.group(1) or m.group(2)
        if slot:
            migration_slots.append(slot)

    # Generated surfaces
    gen_surfaces: list[str] = []
    for key, tag in _GENERATED_SURFACES.items():
        if key in text:
            gen_surfaces.append(tag)

    return Footprint(
        files=raw_files,
        packages=sorted(packages),
        areas=areas,
        migration_slots=list(dict.fromkeys(migration_slots)),
        generated_surfaces=gen_surfaces,
    )


# ---------------------------------------------------------------------------
# Bead classification
# ---------------------------------------------------------------------------

FRONTIER_LABELS = {"horizon:frontier"}
VISION_LABELS = {"horizon:vision"}
MID_LABELS = {"horizon:mid"}


def _horizon(item: BeadDict) -> str:
    labels = set(item.get("labels", []))
    if labels & FRONTIER_LABELS:
        return "frontier"
    if labels & VISION_LABELS:
        return "vision"
    if labels & MID_LABELS:
        return "mid"
    return "unset"


def _classify(item: BeadDict) -> str:
    status = item.get("status", "open")
    if status in ("in_progress", "claimed"):
        return "IN-PROGRESS"
    dep_count = item.get("dependency_count", 0)
    if dep_count and dep_count > 0:
        return "BLOCKED"
    h = _horizon(item)
    if h == "frontier":
        return "FRONTIER-READY"
    if h in ("vision", "mid"):
        return "DESIGN-HORIZON"
    # No horizon label -- treat as frontier-ready if P0/P1, else design-horizon
    prio = item.get("priority", 4)
    return "FRONTIER-READY" if prio <= 1 else "DESIGN-HORIZON"


# ---------------------------------------------------------------------------
# Overlap graph + clustering
# ---------------------------------------------------------------------------


def _build_overlap_graph(beads: list[BeadDict], footprints: dict[str, Footprint]) -> dict[str, set[str]]:
    """Return adjacency map: id -> set of overlapping ids."""
    # Build inverted index: overlap_key -> list of bead ids
    key_to_ids: dict[str, list[str]] = defaultdict(list)
    for b in beads:
        bid = b["id"]
        for key in footprints[bid].overlap_keys():
            key_to_ids[key].append(bid)

    adj: dict[str, set[str]] = {b["id"]: set() for b in beads}
    for ids in key_to_ids.values():
        if len(ids) > 1:
            for i, a in enumerate(ids):
                for other in ids[i + 1 :]:
                    adj[a].add(other)
                    adj[other].add(a)
    return adj


def _connected_components(ids: list[str], adj: dict[str, set[str]]) -> list[set[str]]:
    visited: set[str] = set()
    components: list[set[str]] = []
    for start in ids:
        if start in visited:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor in ids and neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def _find_contention(beads: list[BeadDict], footprints: dict[str, Footprint]) -> list[dict[str, Any]]:
    """Return list of contention events (migration slot / generated surface collisions)."""
    key_to_ids: dict[str, list[str]] = defaultdict(list)
    for b in beads:
        for key in footprints[b["id"]].contention_keys():
            key_to_ids[key].append(b["id"])
    events: list[dict[str, Any]] = []
    for key, ids in key_to_ids.items():
        if len(ids) > 1:
            events.append({"key": key, "beads": ids})
    return events


# ---------------------------------------------------------------------------
# Weighted clustering (fixes the any-overlap mega-cluster degeneracy)
# ---------------------------------------------------------------------------

DEFAULT_MIN_SIMILARITY = 3.0
DEFAULT_MAX_CLUSTER_SIZE = 6


def _document_frequencies(beads: list[BeadDict], footprints: dict[str, Footprint]) -> dict[str, int]:
    """Count, over ``beads``, how many beads carry each overlap token."""
    df: dict[str, int] = defaultdict(int)
    for b in beads:
        for key in footprints[b["id"]].overlap_keys():
            df[key] += 1
    return df


def _pair_weight(
    a_fp: Footprint,
    b_fp: Footprint,
    df: dict[str, int],
    n_docs: int,
) -> tuple[float, list[str]]:
    """TF-IDF-flavored similarity between two beads' footprints.

    Shared tokens that are rare across the population (low df) score high;
    shared hub tokens (high df, e.g. a file touched by 8 beads) score near
    zero -- this is what stops hub files from collapsing the whole frontier
    into one component. Shared ``area:`` labels add a small flat bonus since
    they are coarse-grained but still a real signal of thematic overlap.
    """
    shared = sorted(a_fp.overlap_keys() & b_fp.overlap_keys())
    score = 0.0
    for key in shared:
        d = df.get(key, 1)
        if d <= 0 or d >= n_docs:
            continue
        score += math.log2(n_docs / d)
    shared_areas = set(a_fp.areas) & set(b_fp.areas)
    score += 0.5 * len(shared_areas)
    return score, shared


class _UnionFind:
    """Union-find with a live per-root size, so merges can be capped."""

    def __init__(self, ids: list[str]) -> None:
        self.parent: dict[str, str] = {i: i for i in ids}
        self.size: dict[str, int] = dict.fromkeys(ids, 1)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Attach the smaller root under the larger for shallower trees.
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass
class WeightedCluster:
    beads: set[str] = field(default_factory=set)
    score: float = 0.0
    shared_files: list[str] = field(default_factory=list)


def _weighted_clusters(
    beads: list[BeadDict],
    footprints: dict[str, Footprint],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
) -> list[WeightedCluster]:
    """Greedy weight-descending union-find clustering with a size cap.

    Builds every pairwise edge with weight >= ``min_similarity``, sorts by
    weight descending, and merges greedily -- refusing a merge that would
    push the resulting cluster past ``max_cluster_size``. This bounds how
    far a single strong edge can transitively drag in unrelated beads,
    which is the exact failure mode of unweighted connected-components.
    """
    ids = [b["id"] for b in beads]
    n_docs = len(ids)
    if n_docs == 0:
        return []

    df = _document_frequencies(beads, footprints)

    edges: list[tuple[float, str, str, list[str]]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            weight, shared = _pair_weight(footprints[a], footprints[b], df, n_docs)
            if weight >= min_similarity:
                edges.append((weight, a, b, shared))
    edges.sort(key=lambda e: e[0], reverse=True)

    uf = _UnionFind(ids)
    # Per-root accumulated score and shared-file evidence, used to render
    # the merged cluster's headline stats.
    root_score: dict[str, float] = defaultdict(float)
    root_shared_files: dict[str, set[str]] = defaultdict(set)

    for weight, a, b, shared in edges:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if uf.size[ra] + uf.size[rb] > max_cluster_size:
            continue
        uf.union(a, b)
        new_root = uf.find(a)
        old_root = rb if new_root == ra else ra
        root_score[new_root] = max(root_score[new_root], root_score.get(old_root, 0.0), weight)
        merged_files = {tok for tok in shared if tok in footprints[a].files or tok in footprints[b].files}
        root_shared_files[new_root] = (
            root_shared_files.get(old_root, set()) | root_shared_files.get(new_root, set()) | merged_files
        )

    groups: dict[str, set[str]] = defaultdict(set)
    for bead_id in ids:
        groups[uf.find(bead_id)].add(bead_id)

    clusters: list[WeightedCluster] = []
    for root, members in groups.items():
        if len(members) > 1:
            # Files appearing in more than one member of THIS cluster,
            # regardless of whether they happened to be on the winning
            # merge edge -- gives a fuller "what do these beads share" view.
            file_counts: dict[str, int] = defaultdict(int)
            for bead_id in members:
                for f in footprints[bead_id].files:
                    file_counts[f] += 1
            shared_files = sorted(f for f, count in file_counts.items() if count > 1)
        else:
            shared_files = []
        clusters.append(
            WeightedCluster(
                beads=members,
                score=round(root_score.get(root, 0.0), 3),
                shared_files=shared_files,
            )
        )
    return clusters


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _short(s: str, n: int = 70) -> str:
    return s[:n] + "..." if len(s) > n else s


def _render_human(
    all_beads: list[BeadDict],
    footprints: dict[str, Footprint],
    frontier_clusters: list[WeightedCluster],
    contention: list[dict[str, Any]],
    design_horizon: list[BeadDict],
    blocked: list[BeadDict],
    in_progress: list[BeadDict],
) -> None:
    id_to_bead = {b["id"]: b for b in all_beads}

    print("=" * 72)
    print("POLYLOGUE EXECUTION FRONTIER CLUSTERS")
    print("=" * 72)
    print()

    # In-progress first (context)
    if in_progress:
        print(f"-- IN-PROGRESS ({len(in_progress)}) -----------------------------------------------")
        for b in sorted(in_progress, key=lambda x: x.get("priority", 4)):
            print(f"  {b['id']:25s} P{b['priority']}  {_short(b['title'])}")
        print()

    multi = sum(1 for c in frontier_clusters if len(c.beads) > 1)
    solo = len(frontier_clusters) - multi
    # Frontier clusters
    print(
        f"-- FRONTIER-READY CLUSTERS ({len(frontier_clusters)} clusters: "
        f"{multi} multi-bead, {solo} singleton) ------------------------"
    )
    for i, cluster in enumerate(
        sorted(
            frontier_clusters,
            key=lambda c: min(id_to_bead[bid].get("priority", 4) for bid in c.beads),
        )
    ):
        members = sorted(cluster.beads, key=lambda bid: id_to_bead[bid].get("priority", 4))
        min_p = id_to_bead[members[0]].get("priority", 4)
        shape = "SWEEP (overlapping)" if len(cluster.beads) > 1 else "SOLO"
        print(f"\n  Cluster {i + 1}  [{shape}]  min-P{min_p}  score={cluster.score}")
        if cluster.shared_files:
            top_shared = ", ".join(cluster.shared_files[:6])
            if len(cluster.shared_files) > 6:
                top_shared += f" +{len(cluster.shared_files) - 6}"
            print(f"    shared files: {top_shared}")
        for bid in members:
            b = id_to_bead[bid]
            fp = footprints[bid]
            needsconf = "  [NEEDS-CONFIRM: no footprint]" if fp.is_empty else ""
            print(f"    {bid:30s} P{b['priority']}  {_short(b['title'], 55)}{needsconf}")
            if fp.packages:
                pkgs = ", ".join(fp.packages[:4])
                if len(fp.packages) > 4:
                    pkgs += f" +{len(fp.packages) - 4}"
                print(f"      pkgs: {pkgs}")
            if fp.migration_slots:
                print(f"      migration slots: {', '.join(fp.migration_slots)}")
            if fp.generated_surfaces:
                print(f"      generated surfaces: {', '.join(fp.generated_surfaces)}")
        # Parallel hint
        if len(cluster.beads) > 1:
            print("    -> Execute as one branch/PR sweep (shared file footprint)")
    print()

    # Contention warnings
    if contention:
        print(f"-- CONTENTION WARNINGS ({len(contention)}) -------------------------------------")
        for ev in contention:
            print(f"  {ev['key']}")
            for bid in ev["beads"]:
                b = id_to_bead.get(bid, {})
                print(f"    {bid:30s} {_short(b.get('title', '?'), 50)}")
        print()

    # Blocked
    if blocked:
        print(f"-- BLOCKED ({len(blocked)}) -----------------------------------------------------")
        for b in sorted(blocked, key=lambda x: x.get("priority", 4)):
            ndeps = b.get("dependency_count", "?")
            print(f"  {b['id']:25s} P{b['priority']}  [{ndeps} dep]  {_short(b['title'])}")
        print()

    # Design-horizon summary (just counts)
    if design_horizon:
        by_h: dict[str, int] = defaultdict(int)
        for b in design_horizon:
            by_h[_horizon(b)] += 1
        horizon_str = "  ".join(f"{h}:{c}" for h, c in sorted(by_h.items()))
        print(f"-- DESIGN-HORIZON ({len(design_horizon)})  [{horizon_str}]  (not shown -- use --all-open to list)")
        print()

    # Disjoint parallel-safe set across all clusters
    solo_clusters = [c for c in frontier_clusters if len(c.beads) == 1]
    disjoint_frontier = [next(iter(c.beads)) for c in solo_clusters]
    if len(disjoint_frontier) > 1:
        print(f"-- SAFE PARALLEL LANES ({len(disjoint_frontier)} solo beads, disjoint footprints) --")
        for bid in sorted(disjoint_frontier, key=lambda b: id_to_bead[b].get("priority", 4)):
            b = id_to_bead[bid]
            print(f"  {bid:25s} P{b['priority']}  {_short(b['title'])}")
        print()

    print("Advisory: footprints are extracted heuristically. Verify on claim.")


def _render_json(
    all_beads: list[BeadDict],
    footprints: dict[str, Footprint],
    frontier_clusters: list[WeightedCluster],
    contention: list[dict[str, Any]],
    design_horizon: list[BeadDict],
    blocked: list[BeadDict],
    in_progress: list[BeadDict],
) -> None:
    id_to_bead = {b["id"]: b for b in all_beads}

    clusters_out: list[dict[str, Any]] = []
    for cluster in sorted(frontier_clusters, key=lambda c: min(id_to_bead[bid].get("priority", 4) for bid in c.beads)):
        members = sorted(cluster.beads, key=lambda bid: id_to_bead[bid].get("priority", 4))
        shape = "sweep" if len(cluster.beads) > 1 else "solo"
        clusters_out.append(
            {
                "shape": shape,
                "min_priority": min(id_to_bead[bid].get("priority", 4) for bid in members),
                "score": cluster.score,
                "shared_files": cluster.shared_files,
                "beads": [
                    {
                        "id": bid,
                        "priority": id_to_bead[bid].get("priority"),
                        "title": id_to_bead[bid].get("title"),
                        "footprint": {
                            "packages": footprints[bid].packages,
                            "files": footprints[bid].files[:8],
                            "areas": footprints[bid].areas,
                            "migration_slots": footprints[bid].migration_slots,
                            "generated_surfaces": footprints[bid].generated_surfaces,
                            "needs_confirm": footprints[bid].is_empty,
                        },
                    }
                    for bid in members
                ],
            }
        )

    out = {
        "frontier_clusters": clusters_out,
        "contention": contention,
        "blocked": [
            {
                "id": b["id"],
                "priority": b.get("priority"),
                "title": b["title"],
                "dep_count": b.get("dependency_count", 0),
            }
            for b in blocked
        ],
        "in_progress": [{"id": b["id"], "priority": b.get("priority"), "title": b["title"]} for b in in_progress],
        "design_horizon_count": len(design_horizon),
    }
    json.dump(out, sys.stdout, indent=2)
    print()


# ---------------------------------------------------------------------------
# Roster validation (AC3: predict the 2026-07-13 migration-slot collision)
# ---------------------------------------------------------------------------

# The hand-built LANES dict from the retired fanout_gen_prompts.py encoded two
# lanes both touching migration slot 008 -- the tool should have flagged them.
ROSTER_2026_07_13_KNOWN_COLLISION = {
    "lane_a": "storage-rebuild-bytes",  # touched migrations/source/008_*.sql
    "lane_b": "schema-write-audit",  # also touched migrations/source/008_*.sql
    "slot": "008",
    "note": "Two fanout PR lanes collided on durable-tier migration slot 008; "
    "the hand roster missed it; bead-cluster must predict it.",
}


def _validate_roster(beads: list[BeadDict], footprints: dict[str, Footprint]) -> None:
    print("=== ROSTER VALIDATION: 2026-07-13 migration-slot collision ===")
    print()
    slot = ROSTER_2026_07_13_KNOWN_COLLISION["slot"]
    slot_beads = [b for b in beads if slot in footprints[b["id"]].migration_slots]
    if len(slot_beads) >= 2:
        print(f"  PREDICTED: {len(slot_beads)} beads share migration slot {slot}:")
        for b in slot_beads:
            print(f"    {b['id']}: {_short(b['title'])}")
    elif len(slot_beads) == 1:
        print(f"  PARTIAL: only 1 bead mentions slot {slot} (live roster may differ from 2026-07-13)")
        print(f"    {slot_beads[0]['id']}: {_short(slot_beads[0]['title'])}")
    else:
        print(f"  NOT-DETECTED in current roster: no beads mention slot {slot}")
        print("    (Either the collision lanes are merged/closed, or footprint extraction")
        print("     missed the migration reference. Verify with: grep -r '008_' storage/sqlite/migrations/)")
    print()
    print(f"  Note: {ROSTER_2026_07_13_KNOWN_COLLISION['note']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_json_stream(text: str) -> Any:
    """Parse a JSON value from ``text``, tolerating trailing non-JSON text.

    bd's ``--json`` output can be followed by a plain-text pagination notice
    on stdout (e.g. ``Showing 100 of 391 ready issues...``) when a command's
    default ``--limit`` truncates the result set. ``json.loads`` rejects the
    trailing text outright, so decode only the leading JSON value.
    """
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(text)
    return value


def _load_beads(args: argparse.Namespace) -> list[BeadDict]:
    if args.input:
        with open(args.input) as f:
            return list(json.load(f))

    cmd = (
        ["bd", "list", "--status=open", "--json", "--limit", "0"]
        if args.all_open
        else ["bd", "ready", "--json", "--limit", "0"]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running bd: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    try:
        return list(_parse_json_stream(result.stdout))
    except json.JSONDecodeError as exc:
        print(f"ERROR parsing bd output: {exc}", file=sys.stderr)
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", metavar="FILE", help="Read beads from JSON file instead of bd")
    parser.add_argument("--all-open", action="store_true", help="Include all open beads (not just ready)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Machine-readable JSON output")
    parser.add_argument(
        "--max-priority", type=int, default=4, metavar="N", help="Only show beads at priority <= N (default: 4)"
    )
    parser.add_argument(
        "--validate-roster", action="store_true", help="Replay-validate 2026-07-13 collision prediction"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        metavar="F",
        help=(
            "Minimum TF-IDF-flavored pairwise edge weight to merge two beads into "
            f"the same cluster (default: {DEFAULT_MIN_SIMILARITY})"
        ),
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=DEFAULT_MAX_CLUSTER_SIZE,
        metavar="N",
        help=f"Refuse a merge that would grow a cluster past this size (default: {DEFAULT_MAX_CLUSTER_SIZE})",
    )
    args = parser.parse_args(argv)

    all_beads = _load_beads(args)

    # Filter by priority ceiling
    if args.max_priority < 4:
        all_beads = [b for b in all_beads if b.get("priority", 4) <= args.max_priority]

    # Extract footprints
    footprints: dict[str, Footprint] = {b["id"]: _extract_footprint(b) for b in all_beads}

    # Classify
    frontier_beads = [b for b in all_beads if _classify(b) == "FRONTIER-READY"]
    blocked = [b for b in all_beads if _classify(b) == "BLOCKED"]
    in_progress = [b for b in all_beads if _classify(b) == "IN-PROGRESS"]
    design_horizon = [b for b in all_beads if _classify(b) == "DESIGN-HORIZON"]

    # Weighted overlap clustering over frontier-ready beads only. See the
    # module docstring for why this replaced unweighted connected-components.
    frontier_clusters = _weighted_clusters(
        frontier_beads,
        footprints,
        min_similarity=args.min_similarity,
        max_cluster_size=args.max_cluster_size,
    )

    # Contention (migration slots / generated surfaces) across ALL frontier beads
    contention = _find_contention(frontier_beads, footprints)

    # Roster validation
    if args.validate_roster:
        _validate_roster(all_beads, footprints)

    if args.json_out:
        _render_json(all_beads, footprints, frontier_clusters, contention, design_horizon, blocked, in_progress)
    else:
        _render_human(
            all_beads,
            footprints,
            frontier_clusters,
            contention,
            design_horizon,
            blocked,
            in_progress,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
