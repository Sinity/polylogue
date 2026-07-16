#!/usr/bin/env python3
"""Apply pass — Script 2: creations + new-bead deps. --live to execute."""

import ast
import json
import re
import subprocess
import sys

OUT = "/tmp/claude-1000/-realm-project-polylogue/900c6128-99a3-4986-8ed4-d5d7aacb38fe/scratchpad"
LIVE = "--live" in sys.argv
ledger = []
created = {}


def load(n):
    return json.load(open(f"{OUT}/{n}"))


def as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return []


def create(title, typ, prio, parent=None, design=None, acc=None, labels=None, extref=None):
    prio = str(prio).replace("P", "")
    args = ["bd", "create", title, "--type", typ, "--priority", prio]
    if parent:
        args += ["--parent", parent]
    if design:
        args += ["--design", design]
    if acc:
        args += ["--acceptance", acc]
    if labels:
        args += ["--labels", labels]
    if extref:
        args += ["--external-ref", extref]
    if not LIVE:
        ledger.append(f"DRY CREATE {typ} P{prio} <- {title[:55]}")
        created[title] = "DRY-" + title[:8]
        return created[title]
    r = subprocess.run(args, capture_output=True, text=True)
    m = re.search(r"polylogue-[a-z0-9.]+", r.stdout + r.stderr)
    nid = m.group(0) if m else None
    created[title] = nid
    ledger.append(f"CREATE {typ} P{prio} {nid} <- {title[:50]} rc={r.returncode}")
    if r.returncode != 0:
        ledger.append("   ERR: " + (r.stdout + r.stderr)[:160])
    return nid


def dep(blocked, blocker, typ="blocks"):
    if not (blocked and blocker):
        return
    if not LIVE:
        ledger.append(f"DRY DEP {blocker} -{typ}-> {blocked}")
        return
    r = subprocess.run(["bd", "dep", "add", blocked, blocker, "-t", typ], capture_output=True, text=True)
    ledger.append(f"DEP {blocker} -{typ}-> {blocked} rc={r.returncode}")


def reparent(bid, parent):
    if not (bid and parent):
        return
    if not LIVE:
        ledger.append(f"DRY ADOPT {bid} -> {parent}")
        return
    r = subprocess.run(["bd", "update", bid, "--parent", parent], capture_output=True, text=True)
    ledger.append(f"ADOPT {bid} -> {parent} rc={r.returncode}")


# ---- 1. Usage/cost epic (S10) ----
s10 = load("draft_S10_usage.json")
ue = s10["new_epic"]
usage_id = create(
    ue["title"],
    "epic",
    ue.get("priority", 2),
    design=ue.get("design"),
    acc=ue.get("acceptance"),
    labels="area:analytics,spine",
    extref="gh-2316",
)
for a in as_list(s10.get("adopts")):
    reparent(a, usage_id)
for c in s10.get("new_children", []):
    create(
        c["title"],
        c["type"],
        c.get("priority", 2),
        parent=usage_id,
        design=c.get("design"),
        acc=c.get("acceptance"),
        labels="area:analytics",
    )

# ---- 2. Config-doctrine epic (S6) ----
s6 = load("draft_S6_reorg.json")
cfg = [e for e in s6["new_epics"] if e["title"].startswith("Configuration")][0]
config_id = create(
    cfg["title"], "epic", 2, design=cfg.get("design"), acc=cfg.get("acceptance"), labels="area:ops,spine"
)
for a in as_list(cfg.get("adopts")):
    reparent(a, config_id)

# ---- 3. 1xc children (S1) ----
s1 = load("draft_S1_1xc.json")
s1ids = []
for c in s1:
    nid = create(
        c["title"],
        c["type"],
        c.get("priority", 1),
        parent="polylogue-1xc",
        design=c.get("design"),
        acc=c.get("acceptance"),
        labels="area:storage",
    )
    s1ids.append((c["title"], nid))
lane = [nid for t, nid in s1ids if "regression lane" in t.lower()]
if lane:
    for t, nid in s1ids:
        if nid and nid != lane[0] and "regression lane" not in t.lower():
            dep(lane[0], nid)

# ---- 4. s7ae children (S7) ----
s7 = load("draft_S7_s7ae.json")
s7ids = {}
for c in s7["new_beads"]:
    nid = create(
        c["title"],
        c["type"],
        c.get("priority", 1),
        parent="polylogue-s7ae",
        design=c.get("design"),
        acc=c.get("acceptance"),
        labels="area:coordination",
    )
    s7ids[c["title"]] = nid
    for d in as_list(c.get("deps")):
        if str(d).startswith("polylogue-"):
            dep(nid, d)
items = list(s7ids.items())
A = [nid for t, nid in items if any(w in t.lower() for w in ("compose", "archive", "topology"))]
B = [nid for t, nid in items if nid not in A]
if A and B:
    dep(B[0], A[0])

# ---- 5. Judgment queue (S11) ----
s11 = load("draft_S11_judgment.json")
jid = None
for c in s11["new_beads"]:
    jid = create(
        c["title"],
        c["type"],
        c.get("priority", 2),
        parent="polylogue-37t",
        design=c.get("design"),
        acc=c.get("acceptance"),
        labels="area:context",
    )
for e in s11.get("edges", []):
    b = e.get("blocker")
    k = e.get("blocks")
    t = e.get("type", "blocks")
    if b and "judgment-NEW" in b:
        b = jid
    if k and "judgment-NEW" in k:
        k = jid
    typ = "relates-to" if t in ("relates-to", "related", "related-to") else "blocks"
    if b and k and str(b).startswith("polylogue") and str(k).startswith("polylogue"):
        dep(k, b, typ)

# ---- 6. Decision residual beads (S2) — closures/retype EXCLUDED (checkpoint) ----
s2 = load("draft_S2_decisions.json")
for c in s2.get("residual_beads", []):
    create(
        c["title"],
        c.get("type", "task"),
        c.get("priority", 3),
        parent=c.get("parent"),
        design=c.get("design"),
        acc=c.get("acceptance"),
        labels="area:coordination",
    )

# ---- 7. Authored gap beads (E2 tf2-regen, N3 x2) ----
create(
    "Regenerate agent-forensics finding on the v24 archive",
    "task",
    1,
    parent="polylogue-3tl",
    design="tf2 (forensics campaign) was closed with a v23 artifact now retired to .agent/archive/retired-demos/2026-07-04-v23-demo-packets/; RETIRED-DEMO.md states the cardinality is stale post-v24 rebuild. Unlike jxe (re-run tracked by cfk) and sru (refreshed on v24), tf2's finding has no v24 coverage. Regenerate the agent_forensics packet after session-profile convergence on the current archive; publish through the same finding lane as sru.",
    acc="A v24 agent-forensics finding artifact exists under .agent/demos/ with current archive cardinality; cited counts match `polylogue` live reads; the retired-demos path is no longer the only copy; cold-reader-legible per the 3tl gate.",
    labels="area:legibility",
)
create(
    "Multiple concurrent browser-capture extension instances: attribution, dedup, spool safety",
    "task",
    2,
    parent="polylogue-3v1",
    design="Raw-log 2026-07-04 19:00: with agent-private + live Chrome both able to run the capture extension against the single loopback receiver, >1 instance can post concurrently. No bead covers per-instance attribution, duplicate-post dedup, or spool-file write safety under concurrent posters. Define an instance id on the POST envelope, dedup by (native_id, content_hash), and make the receiver spool writer concurrency-safe.",
    acc="Two simultaneous extension instances posting the same session produce one archived session (dedup by content hash); each capture carries an attributable instance id; concurrent spool writes never corrupt or interleave a spool file (test with 2 simulated posters).",
    labels="area:ingest",
)
create(
    "README de-meta / de-persuasion pass with reproducible capability claims",
    "task",
    2,
    parent="polylogue-3tl",
    design="Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run). Distinct axis from 3tl.1's structure work.",
    acc="README first screen names the category and four verbs without persuasion register; every coined term is defined at first use; each capability claim links a runnable `polylogue`/`devtools` command; a fresh no-context reader can reproduce >=2 claims. Verify: docs-commands lint green + cold-reader pass.",
    labels="area:legibility",
)

print("\n".join(ledger))
print(f"\n=== SUMMARY ({'LIVE' if LIVE else 'DRY'}) created={sum(1 for v in created.values() if v)} ===")
with open(f"{OUT}/ledger_apply2.json", "w") as fh:
    json.dump({"ledger": ledger, "created": created}, fh, indent=1)
