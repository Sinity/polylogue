#!/usr/bin/env python3
"""Apply pass — Script 4: completion-state + remaining epics (operator-authorized). --live to execute."""

import re
import subprocess
import sys

LIVE = "--live" in sys.argv
ledger = []
ids = {}


def run(a):
    if not LIVE:
        ledger.append("DRY: bd " + " ".join(x if len(x) < 40 else x[:37] + "..." for x in a))
        return ""
    r = subprocess.run(["bd"] + a, capture_output=True, text=True)
    ledger.append(f"bd {a[0]} {a[1] if len(a) > 1 else ''} rc={r.returncode}")
    if r.returncode != 0:
        ledger.append("  ERR: " + (r.stdout + r.stderr)[:150])
    return r.stdout + r.stderr


def create(title, typ, prio, parent=None, design=None, acc=None, labels=None):
    a = ["create", title, "--type", typ, "--priority", str(prio)]
    if parent:
        a += ["--parent", parent]
    if design:
        a += ["--design", design]
    if acc:
        a += ["--acceptance", acc]
    if labels:
        a += ["--labels", labels]
    out = run(a)
    m = re.search(r"polylogue-[a-z0-9.]+", out)
    nid = m.group(0) if m else "DRY-" + title[:6]
    ids[title] = nid
    ledger.append(f"  -> {nid} : {title[:55]}")
    return nid


def update(bid, *f):
    run(["update", bid, *f])


def close(bid, reason):
    run(["close", bid, "--reason", reason])


def comment(bid, t):
    run(["comment", bid, t])


# ===== COMPLETION-STATE (authorized) =====
close(
    "polylogue-s7ae.1",
    "Coordination envelope + agents CLI (7 subcommands) + MCP agent_coordination tool/prompt + CLI/coordination/MCP tests shipped in 32ff31651; AC met (E3 file-level verify). Archive-evidence composition (originally over-claimed in design) split to s7ae.4.",
)
close(
    "polylogue-s7ae.2",
    ".beads-hooks/ git-hook wiring + MCP agent_coordination tool+prompt + regenerated CLI/topology docs shipped in 32ff31651. Full-suite verify classification tracked as the new verify-gap bead under s7ae.",
)
create(
    "Classify the 74%-aborted full verify from the coordination commit before deploy",
    "task",
    1,
    parent="polylogue-s7ae",
    design="Commit 32ff31651 shipped ~1376 LOC with only verify --quick + focused tests green; full devtools verify was aborted at 74% with unclassified scattered failures. Before any deploy/switch, run full devtools verify and classify each failure coordination-caused vs pre-existing/flaky.",
    acc="A full devtools verify run is recorded; every failure classified (coordination-caused fixed; pre-existing referenced); s7ae deploy-clean. Verify: devtools verify (full).",
    labels="area:coordination",
)
comment(
    "polylogue-7ry",
    "Closed with an empty reason; its AC is satisfied by 4bu (converging-state contract, 16 tests passing). Backfill reference for audit legibility.",
)

# ===== ADDITIONAL EPICS (operator: 'as you see fit') =====
capext = create(
    "Capture extension: reliability, coverage, and in-page presence",
    "epic",
    2,
    design="The browser-capture extension surface (spool health, capture-state UX, in-page overlay, concurrent-instance safety) had no owning epic. 3v1 (reliability/status), 90y (in-page overlay), 3v1.1 (concurrent instances) form a coherent MV3 capability distinct from bby (the daemon reader).",
    acc="Spool health + capture completeness are observable; per-chat capture-state indicator ships once (badge and in-page chip share one signal, not two); concurrent instances dedup by content hash. Verify: the extension smoke + dedup test.",
    labels="area:ingest,spine",
)
for a in ["polylogue-3v1", "polylogue-90y"]:
    update(a, "--parent", capext)

interop = create(
    "Ecosystem interop + origin breadth: more sources in, two-way citable export out",
    "epic",
    3,
    design="A large orphan cluster is source-ingest breadth (t0p capture-rest-of-claude, uiw, 2qx OriginSpec, 0cg, 7xv, 611, ale) + two-way export (wmj, 7k7, r47, 4g5). Coherent as 'widen what Polylogue can ingest and emit', distinct from fs1 (the Hermes source bridge specifically).",
    acc="Each new origin has detector+parser+fixture+schema+docs (devtools lab provider completeness green); export paths are citable interchange, not bespoke dumps. Verify: devtools lab provider completeness per origin.",
    labels="area:ingest,spine",
)
for a in [
    "polylogue-t0p",
    "polylogue-uiw",
    "polylogue-2qx",
    "polylogue-0cg",
    "polylogue-7xv",
    "polylogue-611",
    "polylogue-ale",
    "polylogue-wmj",
    "polylogue-7k7",
    "polylogue-r47",
    "polylogue-4g5",
]:
    update(a, "--parent", interop)

subcon = create(
    "Substrate consolidation: kill the storage twins and split the god-modules",
    "epic",
    3,
    design="Internal-debt cluster distinct from t46's surface focus: storage sync/async twins (hiu-adjacent), god-modules, dead tables, the fts_freshness_state double-declaration (1ty), and other consolidation debt (0aj, yp0, 48h, pf1, 1a9, dab, c9y). Refactor/consolidation, not new capability.",
    acc="Each twin/god-module has a consolidation bead with before/after; no duplicate schema declarations remain; devtools verify layering + schema-versioning stay green. Verify: devtools verify + the dead-table audit (9e5.5).",
    labels="area:substrate,spine",
)
for a in [
    "polylogue-0aj",
    "polylogue-yp0",
    "polylogue-48h",
    "polylogue-pf1",
    "polylogue-1ty",
    "polylogue-1a9",
    "polylogue-dab",
    "polylogue-c9y",
]:
    update(a, "--parent", subcon)

# ===== work-evidence-rail: dissolved (critic killed the epic); reparent members individually =====
update("polylogue-4c0", "--parent", "polylogue-rii")  # beads<->session cross-links = live intake evidence
update("polylogue-7fj", "--parent", "polylogue-rii")  # beads-history ingestion as evidence source
update("polylogue-kph", "--parent", "polylogue-s7ae")  # provenance-carrying PRs = coordination evidence
comment(
    "polylogue-lio",
    "Kept standalone (cross-repo devloop contract with Sinex sinex-hlv); not folded into an epic — the work-evidence-rail grouping was dropped as ceremony overlapping s7ae/rii.",
)

# ===== 9e5 read-only contract (encode the drift the audit found) =====
update(
    "polylogue-9e5",
    "--acceptance",
    "Every child produces a READ-ONLY evidence artifact and never mutates product code. Children that must ship tooling/deletions (9e5.9 heuristics lane, 9e5.15 dead-code sweep, 9e5.16 api-doc gate) are split so the audit/analysis half stays read-only and the execution half is a separate tracked bead. Verify: each closed 9e5 child cites an artifact, not a product-code diff, unless explicitly split.",
)

print("\n".join(ledger))
print(f"\n=== SUMMARY ({'LIVE' if LIVE else 'DRY'}) created={len(ids)} ===")
