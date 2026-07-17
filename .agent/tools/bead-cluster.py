#!/usr/bin/env python3
"""
bead-cluster.py  —  Execution-frontier clustering for polylogue Beads.

Implements polylogue-2yax: reads bd ready (or a supplied JSON export), extracts
file/package/resource footprints from each bead's design+notes+ac text, builds
an overlap and dependency graph, and emits:

  - FRONTIER-READY   — horizon:frontier, no blocking deps → claim now
  - BLOCKED          — has unmet hard dependencies
  - IN-PROGRESS      — already claimed
  - DESIGN-HORIZON   — mid/vision horizon, not yet executable

For frontier-ready beads it computes:
  - OVERLAPPING clusters  → one branch/PR sweep (share context, single rewrite pass)
  - DISJOINT sets         → safe to run as parallel lanes / separate worktrees
  - CONTENTION points     → same migration tier/slot, same generated surface, same DB

Usage:
    # From repo root, using live bd output:
    python3 .agent/tools/bead-cluster.py

    # Read a pre-exported JSON (e.g. bd ready --json > ready.json):
    python3 .agent/tools/bead-cluster.py --input ready.json

    # Include all open beads (not just ready):
    python3 .agent/tools/bead-cluster.py --all-open

    # Machine-readable JSON output:
    python3 .agent/tools/bead-cluster.py --json

    # Filter to a priority ceiling:
    python3 .agent/tools/bead-cluster.py --max-priority 1

    # Replay-validate against the 2026-07-13 migration-slot collision:
    python3 .agent/tools/bead-cluster.py --validate-roster

Notes:
  Footprints are extracted heuristically — they are ADVISORY. A missing or
  ambiguous footprint lowers readiness and is reported as NEEDS-CONFIRM.
  Never treat inferred footprints as correctness proof.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

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
    "topology-status.md": "generated:topology-status",
    "topology-target.yaml": "generated:topology-target",
    "cli-reference.md": "generated:cli-reference",
    "openapi/": "generated:openapi",
}

# Area label → primary package(s)
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
        "storage/sqlite/",
        "pipeline/",
        "docs/",
    }
)


def _extract_footprint(item: dict) -> Footprint:
    text = " ".join(filter(None, [item.get("design", ""), item.get("notes", ""), item.get("acceptance_criteria", "")]))
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

    # Only fall back to area→package mapping when NO files found at all.
    # When files ARE found, area→package would add broad roots that over-merge.
    if not raw_files:
        for label in labels:
            for pkg in _AREA_TO_PACKAGES.get(label, []):
                packages.add(pkg)

    # Area labels — stored for display and metadata, NOT used as overlap keys.
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
        Broad roots (tests/, polylogue/storage/, etc.) are excluded — they would
        collapse the entire frontier into one mega-cluster.
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


# ---------------------------------------------------------------------------
# Bead classification
# ---------------------------------------------------------------------------

FRONTIER_LABELS = {"horizon:frontier"}
VISION_LABELS = {"horizon:vision"}
MID_LABELS = {"horizon:mid"}


def _horizon(item: dict) -> str:
    labels = set(item.get("labels", []))
    if labels & FRONTIER_LABELS:
        return "frontier"
    if labels & VISION_LABELS:
        return "vision"
    if labels & MID_LABELS:
        return "mid"
    return "unset"


def _classify(item: dict) -> str:
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
    # No horizon label — treat as frontier-ready if P0/P1, else design-horizon
    prio = item.get("priority", 4)
    return "FRONTIER-READY" if prio <= 1 else "DESIGN-HORIZON"


# ---------------------------------------------------------------------------
# Overlap graph + clustering
# ---------------------------------------------------------------------------


def _build_overlap_graph(beads: list[dict], footprints: dict[str, Footprint]) -> dict[str, set[str]]:
    """Return adjacency map: id → set of overlapping ids."""
    # Build inverted index: overlap_key → list of bead ids
    key_to_ids: dict[str, list[str]] = defaultdict(list)
    for b in beads:
        bid = b["id"]
        for key in footprints[bid].overlap_keys():
            key_to_ids[key].append(bid)

    adj: dict[str, set[str]] = {b["id"]: set() for b in beads}
    for ids in key_to_ids.values():
        if len(ids) > 1:
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    adj[a].add(b)
                    adj[b].add(a)
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


def _find_contention(beads: list[dict], footprints: dict[str, Footprint]) -> list[dict]:
    """Return list of contention events (migration slot / generated surface collisions)."""
    key_to_ids: dict[str, list[str]] = defaultdict(list)
    for b in beads:
        for key in footprints[b["id"]].contention_keys():
            key_to_ids[key].append(b["id"])
    events = []
    for key, ids in key_to_ids.items():
        if len(ids) > 1:
            events.append({"key": key, "beads": ids})
    return events


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _short(s: str, n: int = 70) -> str:
    return s[:n] + "…" if len(s) > n else s


def _render_human(
    all_beads: list[dict],
    footprints: dict[str, Footprint],
    frontier_clusters: list[set[str]],
    contention: list[dict],
    design_horizon: list[dict],
    blocked: list[dict],
    in_progress: list[dict],
    max_priority: int,
) -> None:
    id_to_bead = {b["id"]: b for b in all_beads}

    print("=" * 72)
    print("POLYLOGUE EXECUTION FRONTIER CLUSTERS")
    print("=" * 72)
    print()

    # In-progress first (context)
    if in_progress:
        print(f"── IN-PROGRESS ({len(in_progress)}) ─────────────────────────────────────────")
        for b in sorted(in_progress, key=lambda x: x.get("priority", 4)):
            print(f"  {b['id']:25s} P{b['priority']}  {_short(b['title'])}")
        print()

    # Frontier clusters
    print(f"── FRONTIER-READY CLUSTERS ({len(frontier_clusters)} clusters) ──────────────────────────")
    for i, cluster in enumerate(
        sorted(frontier_clusters, key=lambda c: min(id_to_bead[bid].get("priority", 4) for bid in c))
    ):
        members = sorted(cluster, key=lambda bid: id_to_bead[bid].get("priority", 4))
        min_p = id_to_bead[members[0]].get("priority", 4)
        shape = "SWEEP (overlapping)" if len(cluster) > 1 else "SOLO"
        print(f"\n  Cluster {i + 1}  [{shape}]  min-P{min_p}")
        for bid in members:
            b = id_to_bead[bid]
            fp = footprints[bid]
            needsconf = "  ⚠ NEEDS-CONFIRM (no footprint)" if fp.is_empty else ""
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
        if len(cluster) > 1:
            print("    → Execute as one branch/PR sweep (shared file footprint)")
    print()

    # Contention warnings
    if contention:
        print(f"── CONTENTION WARNINGS ({len(contention)}) ─────────────────────────────────────")
        for ev in contention:
            print(f"  {ev['key']}")
            for bid in ev["beads"]:
                b = id_to_bead.get(bid, {})
                print(f"    {bid:30s} {_short(b.get('title', '?'), 50)}")
        print()

    # Blocked
    if blocked:
        print(f"── BLOCKED ({len(blocked)}) ──────────────────────────────────────────────────")
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
        print(f"── DESIGN-HORIZON ({len(design_horizon)})  [{horizon_str}]  (not shown — use --all to list)")
        print()

    # Disjoint parallel-safe set across all clusters
    solo_clusters = [c for c in frontier_clusters if len(c) == 1]
    disjoint_frontier = [next(iter(c)) for c in solo_clusters]
    if len(disjoint_frontier) > 1:
        print(f"── SAFE PARALLEL LANES ({len(disjoint_frontier)} solo beads, disjoint footprints) ──")
        for bid in sorted(disjoint_frontier, key=lambda b: id_to_bead[b].get("priority", 4)):
            b = id_to_bead[bid]
            print(f"  {bid:25s} P{b['priority']}  {_short(b['title'])}")
        print()

    print("Advisory: footprints are extracted heuristically. Verify on claim.")


def _render_json(
    all_beads: list[dict],
    footprints: dict[str, Footprint],
    frontier_clusters: list[set[str]],
    contention: list[dict],
    design_horizon: list[dict],
    blocked: list[dict],
    in_progress: list[dict],
) -> None:
    id_to_bead = {b["id"]: b for b in all_beads}

    clusters_out = []
    for cluster in sorted(frontier_clusters, key=lambda c: min(id_to_bead[bid].get("priority", 4) for bid in c)):
        members = sorted(cluster, key=lambda bid: id_to_bead[bid].get("priority", 4))
        shape = "sweep" if len(cluster) > 1 else "solo"
        clusters_out.append(
            {
                "shape": shape,
                "min_priority": min(id_to_bead[bid].get("priority", 4) for bid in members),
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

# The hand-built LANES dict from fanout_gen_prompts.py encoded these two lanes
# both touching migration slot 008 — the tool should have flagged them.
ROSTER_2026_07_13_KNOWN_COLLISION = {
    "lane_a": "storage-rebuild-bytes",  # touched migrations/source/008_*.sql
    "lane_b": "schema-write-audit",  # also touched migrations/source/008_*.sql
    "slot": "008",
    "note": "Two fanout PR lanes collided on durable-tier migration slot 008; "
    "the hand roster missed it; bead-cluster must predict it.",
}


def _validate_roster(beads: list[dict], footprints: dict[str, Footprint]) -> None:
    print("=== ROSTER VALIDATION: 2026-07-13 migration-slot collision ===")
    print()
    slot = ROSTER_2026_07_13_KNOWN_COLLISION["slot"]
    slot_beads = [b for b in beads if slot in footprints[b["id"]].migration_slots]
    if len(slot_beads) >= 2:
        print(f"  ✓ PREDICTED: {len(slot_beads)} beads share migration slot {slot}:")
        for b in slot_beads:
            print(f"    {b['id']}: {_short(b['title'])}")
    elif len(slot_beads) == 1:
        print(f"  ~ PARTIAL: only 1 bead mentions slot {slot} (live roster may differ from 2026-07-13)")
        print(f"    {slot_beads[0]['id']}: {_short(slot_beads[0]['title'])}")
    else:
        print(f"  ✗ NOT-DETECTED in current roster: no beads mention slot {slot}")
        print("    (Either the collision lanes are merged/closed, or footprint extraction")
        print("     missed the migration reference. Verify with: grep -r '008_' storage/sqlite/migrations/)")
    print()
    print(f"  Note: {ROSTER_2026_07_13_KNOWN_COLLISION['note']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_beads(args: argparse.Namespace) -> list[dict]:
    if args.input:
        with open(args.input) as f:
            return json.load(f)

    cmd = ["bd", "list", "--status=open", "--json", "--limit=2000"] if args.all_open else ["bd", "ready", "--json"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running bd: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", metavar="FILE", help="Read beads from JSON file instead of bd")
    parser.add_argument("--all-open", action="store_true", help="Include all open beads (not just ready)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Machine-readable JSON output")
    parser.add_argument(
        "--max-priority", type=int, default=4, metavar="N", help="Only show beads at priority ≤ N (default: 4)"
    )
    parser.add_argument(
        "--validate-roster", action="store_true", help="Replay-validate 2026-07-13 collision prediction"
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

    # Overlap graph over frontier-ready beads only
    adj = _build_overlap_graph(frontier_beads, footprints)
    frontier_clusters = _connected_components([b["id"] for b in frontier_beads], adj)

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
            args.max_priority,
        )


if __name__ == "__main__":
    main()
