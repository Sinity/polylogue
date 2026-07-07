#!/usr/bin/env python3
"""delivery-gate-status: per-release-gate progress board over .beads/issues.jsonl.

The delivery overlay (2026-07-07, corpus escrowed at
.agent/scratch/corpus-gpt-pro-2026-07-07/) labels every active bead with a
delivery:<release> gate and lane:<lane>. This script computes, per gate:
open / in_progress / closed / blocked / ready counts, percent complete, and
the gate's exit criterion (embedded below — the corpus copy is ephemeral).

Gate ORDER is the implementation sequence: a gate is "up next" when every
earlier gate is complete or explicitly waived. Exit criteria are prose —
they need human/agent judgment, so the board prints them as reminders, not
as computed booleans.

Usage: python3 .agent/tools/delivery-gate-status.py [--json] [--fresh]
       [--gate delivery:A-trust-floor] [path-to-issues.jsonl]
--fresh runs `bd export -o .beads/issues.jsonl` first (bd updates do NOT
immediately re-export; a stale file yields stale counts).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# (gate-id, short name, exit criterion) in implementation order.
# Source: delivery overlay release_gates (adjudicated 2026-07-07); durable copy.
GATES: list[tuple[str, str, str]] = [
    (
        "R0-normalize",
        "Backlog normalization",
        "Every active bead has a delivery release, lane, readiness grade, proof lane, and either acceptance criteria or a deliberate horizon/spec status.",
    ),
    (
        "A-trust-floor",
        "Trust floor",
        "Full verification classified; security negative tests pass; missing bytes classified; numbers/time/prose-mined fields carry honest provenance; agent writes land as candidates.",
    ),
    (
        "B-storage-rebuild-bytes",
        "Storage/rebuild/bytes",
        "Blue-green derived rebuilds cannot show partial archives as ready; restore drill passes; blob refs resolve or carry classified missing state.",
    ),
    (
        "C-read-evidence-contract",
        "Read + evidence contract",
        "CLI, daemon, MCP, Python API, web, reports, and docs read through the same contract; content-hash citations expose drift states.",
    ),
    (
        "D-agent-context-coordination",
        "Agent context/coordination",
        "Hooks install; MCP roles/prompts discoverable; context scheduler emits ledgers; two-agent worktree proof exists.",
    ),
    (
        "E-variants-preferences",
        "Variants + preferences",
        "Variant refs/nodes/alignment/storage exist; reader/query/projection can show source, variant, and side-by-side views honestly.",
    ),
    (
        "F-lineage-compaction",
        "Lineage + compaction",
        "Shared content is stored/counted once, compaction loss is queryable, and regrounding packs pass through the context scheduler.",
    ),
    (
        "G-live-performance",
        "Live performance",
        "Named SLOs and regression gates exist; daemon push/live cache paths work; capture reliability is visible and tested.",
    ),
    (
        "H-web-cockpit",
        "Web evidence cockpit",
        "Evidence basket to citable export works; web UI shows stale/partial/degraded states instead of pretending readiness.",
    ),
    (
        "I-analytics-experiments",
        "Analytics + experiments",
        "Measures are registered, experiments are first-class, analytics render caveats and evidence tiers.",
    ),
    (
        "J-embeddings-retrieval",
        "Embeddings + retrieval",
        "FTS/vector/hybrid quality evals exist; local/cloud providers share one interface; large sessions bound vector work.",
    ),
    (
        "K-interop-origin-export",
        "Interop + origin + export",
        "Each origin has detector/parser/fixtures/fidelity docs; Polylogue export/import is content-hash idempotent; outbound citations preserve provenance.",
    ),
    (
        "L-external-legibility",
        "External legibility",
        "README first screen is clear; one-command demo passes; public claims ledger covers launch claims; cold-reader proof passes.",
    ),
    (
        "M-substrate-consolidation",
        "Substrate consolidation",
        "Storage twins collapse behind a clear boundary; public models are frozen; dead abstractions are deleted or adopted.",
    ),
    (
        "N-horizon",
        "Horizon",
        "Each item either has an implementation-grade spec or remains parked with a decision record.",
    ),
]
GATE_IDS = [g[0] for g in GATES]


def load(path: Path):
    issues: dict[str, dict] = {}
    deps: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("_type") == "issue":
            issues[d["id"]] = d
            for dep in d.get("dependencies") or []:
                deps.append((d["id"], dep.get("depends_on_id"), dep.get("type", "blocks")))
        elif d.get("_type") == "dependency":
            deps.append((d.get("issue_id"), d.get("depends_on_id"), d.get("type", "blocks")))
    return issues, deps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".beads/issues.jsonl")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="bd export first")
    ap.add_argument("--gate", help="only this gate (accepts 'A-trust-floor' or 'delivery:A-trust-floor')")
    args = ap.parse_args()

    if args.fresh:
        subprocess.run(["bd", "export", "-o", args.path], check=True, capture_output=True)

    issues, deps = load(Path(args.path))
    blockers = defaultdict(list)
    for src, dst, kind in deps:
        if kind == "blocks" and dst in issues:
            blockers[src].append(dst)

    def gate_of(d) -> str | None:
        for lab in d.get("labels") or []:
            # delivery:ac-patched is an overlay marker, not a gate assignment
            if lab.startswith("delivery:") and lab != "delivery:ac-patched":
                return lab.removeprefix("delivery:")
        return None

    by_gate: dict[str, list[dict]] = defaultdict(list)
    unlabeled_open = 0
    for d in issues.values():
        g = gate_of(d)
        if g is None:
            if d.get("status") in ("open", "in_progress"):
                unlabeled_open += 1
            continue
        by_gate[g].append(d)

    want = args.gate.removeprefix("delivery:") if args.gate else None
    rows = []
    for gid, name, exit_crit in GATES:
        if want and gid != want:
            continue
        beads = by_gate.get(gid, [])
        closed = [b for b in beads if b["status"] == "closed"]
        in_prog = [b for b in beads if b["status"] == "in_progress"]
        open_ = [b for b in beads if b["status"] == "open"]
        blocked = [b for b in open_ if any(issues[x]["status"] != "closed" for x in blockers.get(b["id"], []))]
        ready = [b for b in open_ if b not in blocked]
        rows.append(
            {
                "gate": gid,
                "name": name,
                "total": len(beads),
                "closed": len(closed),
                "in_progress": len(in_prog),
                "ready": len(ready),
                "blocked": len(blocked),
                "pct": round(100 * len(closed) / len(beads)) if beads else None,
                "exit": exit_crit,
                "ready_ids": sorted(b["id"] for b in ready)[:12],
                "in_progress_ids": sorted(b["id"] for b in in_prog),
            }
        )

    unknown_gates = sorted(set(by_gate) - set(GATE_IDS))

    if args.json:
        print(json.dumps({"gates": rows, "unlabeled_open": unlabeled_open, "unknown_gates": unknown_gates}, indent=2))
        return 0

    frontier_shown = False
    for r in rows:
        if r["total"] == 0:
            bar = "(no beads)"
        else:
            done = int(round((r["pct"] or 0) / 10))
            bar = "#" * done + "." * (10 - done) + f" {r['pct']:>3}%"
        marker = " "
        if not frontier_shown and r["total"] and r["closed"] < r["total"]:
            marker = ">"  # first incomplete gate = active frontier
            frontier_shown = True
        print(
            f"{marker} {r['gate']:<32} {bar}  closed {r['closed']:>3} | wip {r['in_progress']:>2} | ready {r['ready']:>3} | blocked {r['blocked']:>3}"
        )
        if marker == ">" or (want and rows):
            print(f"    exit: {r['exit']}")
            if r["in_progress_ids"]:
                print(f"    wip:   {', '.join(r['in_progress_ids'])}")
            if r["ready_ids"]:
                print(f"    ready: {', '.join(r['ready_ids'])}")
    if unlabeled_open:
        print(f"\n  {unlabeled_open} open/in_progress beads carry no delivery:* label")
    if unknown_gates:
        print(f"  labels outside the gate registry: {', '.join(unknown_gates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
