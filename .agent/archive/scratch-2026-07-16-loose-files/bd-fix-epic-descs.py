#!/usr/bin/env python3
"""Write descriptions for the 5 design-only epics. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:120] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])


bd(
    "update",
    "polylogue-8jg9",
    "--description",
    "WHY: an archive whose pitch is durable evidence must itself survive incidents — daemon death "
    "mid-write, bad deploys, disk loss. Durable tiers (source.db/user.db) are irreplaceable; a "
    "restore path that has never been drilled is a hope, not a capability. ENABLES: trusting the "
    "archive as system-of-record; the backup-manifest gate that durable-tier migrations (60i5) "
    "already assume. MEMBER BEADS: polylogue-4be (backup-restore + quarterly restore drill), "
    "polylogue-peo (daemon-death recovery), polylogue-s8q (deploy trust; parked P4 while prod "
    "polylogued is inactive). Epic closes when a restore drill has actually run against a copy of "
    "the live archive and daemon-death recovery is regression-tested.",
)

bd(
    "update",
    "polylogue-kwsb",
    "--description",
    "WHY: a personal archive of ALL AI work is the most sensitive database on the machine — it must "
    "be able to forget on purpose (excision that provably removes bytes, not just rows) and must "
    "never leak (localhost daemon reachable from a hostile page, secrets in captured content). "
    "Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but backlog "
    "ownership was missing. MEMBER BEADS: polylogue-kwsb.1 (Host/Origin gate + receiver token + "
    "spool governor — the live DNS-rebinding hole), polylogue-27m (excision), polylogue-jnj.5 "
    "(reset-mutation ordering bug: reset.py tombstones before the preview/--yes gate), polylogue-jsy "
    "(crawl-source permissions). Epic closes when the covenant doc's claims are each backed by a "
    "test or an explicit non-goal.",
)

bd(
    "update",
    "polylogue-f2qv",
    "--description",
    "WHY: token/cost accounting is a correctness surface with a track record of silent large errors "
    "(7.69x Codex inflation; per-model partition double-count #2472) — and cost numbers are exactly "
    "what operators quote publicly, so wrong numbers are reputational. Four invariants define "
    "honest accounting (full doctrine in design): disjoint token lanes; one pricing source "
    "(vendored LiteLLM catalog, last-path-segment match); dual view (API-list-equivalent vs "
    "subscription-credit); stale-row hygiene (376.6B-token class artifacts re-ingest away). "
    "ENABLES: credible cost analytics (9l5.4), provider comparisons, the flight-recorder byte-"
    "resolution promise applied to money. Epic members carry the per-surface work; this epic owns "
    "the invariants staying true across new providers/models.",
)

bd(
    "update",
    "polylogue-a7xr",
    "--description",
    "WHY: internal duplication is where correctness quietly dies — the sync/async storage twins "
    "must be edited in pairs or daemon and CLI diverge (standing trap, see storage twins memory), "
    "god-modules resist review, and dead/double-declared tables (fts_freshness_state, 1ty) mislead "
    "every new reader. Distinct from t46 (public surface): this is inside-the-walls consolidation "
    "with no behavior change intended. MEMBER BEADS: polylogue-1ty, polylogue-0aj, polylogue-yp0, "
    "polylogue-48h, polylogue-pf1, polylogue-1a9, polylogue-dab, polylogue-c9y (see design for the "
    "cluster map). Epic closes when the twins are generated-or-unified (single source of truth) and "
    "no module in polylogue/ exceeds the agreed size/responsibility bar.",
)

bd(
    "update",
    "polylogue-l4kf",
    "--description",
    "WHY: the cross-provider claim is only as strong as origin breadth — every AI surface the "
    "operator actually uses must land in the archive, and evidence must flow OUT as citable objects "
    "(two-way interop), or Polylogue is a roach motel. Distinct from fs1 (the Hermes bridge "
    "specifically). MEMBER BEADS in two lanes — ingest breadth: polylogue-t0p (rest-of-Claude "
    "capture), polylogue-uiw, polylogue-2qx (OriginSpec), polylogue-0cg, polylogue-7xv, "
    "polylogue-611, polylogue-ale; export/interop: polylogue-wmj, polylogue-7k7, polylogue-r47, "
    "polylogue-4g5, polylogue-l4kf.2 (federation, vision-tier). Epic closes when the operator's "
    "live surface list has zero uncaptured origins and at least one external tool consumes a "
    "Polylogue export by contract.",
)

print("--- epic descs done")
