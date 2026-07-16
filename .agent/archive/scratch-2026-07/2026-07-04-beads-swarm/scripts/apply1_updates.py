#!/usr/bin/env python3
"""Apply pass — Script 1: updates referencing existing bead IDs.
Field-fills (only where live field empty), reprioritizations, reparents into
EXISTING epics, and dep edges. EXCLUDES all checkpoint items. Idempotent-safe.
Usage: apply1_updates.py [--live]   (default dry-run)
"""

import json
import os
import subprocess
import sys

OUT = "/tmp/claude-1000/-realm-project-polylogue/900c6128-99a3-4986-8ed4-d5d7aacb38fe/scratchpad"
LIVE = "--live" in sys.argv
ledger = []

EXISTING_EPICS = {
    "polylogue-" + e
    for e in [
        "1xc",
        "37t",
        "s7ae",
        "rii",
        "9l5",
        "20d",
        "3tl",
        "fs1",
        "83u",
        "t46",
        "bby",
        "fnm",
        "mhx",
        "4smp",
        "4ts",
        "jnj",
        "fnm",
    ]
}
# 6 HELD new epics — their adoptees must NOT be reparented now (epic not created)
HELD_ADOPTEES = set()  # filled from S6 held epics below
CHECKPOINT_IDS = {
    "polylogue-f94",
    "polylogue-gjg",
    "polylogue-jnj.5",
    "polylogue-7aw",
    "polylogue-20d.4",
    "polylogue-jgp",
    "polylogue-6mv",
    "polylogue-lnd",
    "polylogue-s7ae.1",
    "polylogue-s7ae.2",
}


def load(name):
    p = f"{OUT}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def bd(args, capture=True):
    if not LIVE:
        ledger.append("DRY: bd " + " ".join(a if len(a) < 60 else a[:57] + "..." for a in args))
        return ("", 0)
    r = subprocess.run(["bd"] + args, capture_output=True, text=True)
    return (r.stdout + r.stderr, r.returncode)


def live_fields(bid):
    r = subprocess.run(["bd", "show", bid, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
    except Exception:
        return None
    if isinstance(d, list):
        d = d[0] if d else {}
    if not isinstance(d, dict):
        return None
    return {
        "design": d.get("design") or "",
        "acc": d.get("acceptance_criteria") or "",
        "notes": d.get("notes") or "",
        "priority": d.get("priority"),
        "parent": d.get("parent"),
        "status": d.get("status"),
    }


# ---------- gather field-fill updates ----------
def norm_update(item):
    return {
        "id": item.get("id"),
        "design": item.get("set_design") or item.get("proposed_design"),
        "acc": item.get("set_acceptance") or item.get("proposed_acceptance"),
        "title": item.get("set_title"),
        "priority": item.get("set_priority"),
    }


updates = []
for name in ["draft_S3_ac_A.json", "draft_S4_ac_B.json"]:
    d = load(name) or []
    for it in d:
        updates.append(norm_update(it))
for name in ["draft_S7_s7ae.json", "draft_S9_launch.json", "draft_S11_judgment.json"]:
    d = load(name) or {}
    for it in d.get("updates", []):
        updates.append(norm_update(it))
# S8 4p1 decision content
s8 = load("draft_S8_variants.json") or {}
if s8.get("decision_4p1"):
    dd = s8["decision_4p1"]
    updates.append(
        {
            "id": "polylogue-4p1",
            "design": dd.get("set_design"),
            "acc": dd.get("set_acceptance"),
            "title": None,
            "priority": None,
        }
    )
# S10 5hf underspecified
s10 = load("draft_S10_usage.json") or {}
for it in s10.get("underspecified", []):
    updates.append(norm_update(it))

# ---------- apply field-fills ----------
fill_ok = fill_skip = 0
seen = set()
for u in updates:
    bid = u["id"]
    if not bid or bid in CHECKPOINT_IDS:
        continue
    lf = live_fields(bid) if LIVE else {"design": "", "acc": "", "notes": "", "priority": None}
    if lf is None:
        ledger.append(f"MISS {bid} (not found)")
        continue
    args = ["update", bid]
    did = []
    if u.get("design") and not lf["design"]:
        args += ["--design", u["design"]]
        did.append("design")
    if u.get("acc") and not lf["acc"]:
        args += ["--acceptance", u["acc"]]
        did.append("acc")
    if u.get("title"):
        args += ["--title", u["title"]]
        did.append("title")
    if u.get("priority") is not None:
        args += ["-p", str(u["priority"])]
        did.append("prio")
    if len(args) > 2 and did:
        out, rc = bd(args)
        fill_ok += 1
        ledger.append(f"FILL {bid}: {'+'.join(did)} rc={rc if LIVE else 'dry'}")
    else:
        fill_skip += 1

# ---------- reprioritizations ----------
reprios = {}
for it in s8.get("reprioritizations") or []:
    reprios[it["id"]] = str(it.get("to_priority", "")).replace("P", "")
s6 = load("draft_S6_reorg.json") or {}
for it in s6.get("reprioritizations") or []:
    # skip retypes (checkpoint); only pure priority moves
    if it.get("id") in CHECKPOINT_IDS:
        continue
    tp = it.get("to_priority")
    if tp is not None:
        reprios[it["id"]] = str(tp).replace("P", "")
reprios["polylogue-1ty"] = "2"  # D16: schema-dup correctness bug P3->P2
rep_ok = 0
for bid, p in reprios.items():
    if not p or bid in CHECKPOINT_IDS:
        continue
    out, rc = bd(["update", bid, "-p", p])
    rep_ok += 1
    ledger.append(f"PRIO {bid} -> P{p} rc={rc if LIVE else 'dry'}")

# ---------- reparents into EXISTING epics only ----------
rep2_ok = rep2_skip = 0
for it in s6.get("reparents") or []:
    bid = it.get("id")
    to = it.get("to_parent") or it.get("to")
    if not bid or not to:
        continue
    if bid in CHECKPOINT_IDS:
        rep2_skip += 1
        continue  # jnj.5 etc held
    if to not in EXISTING_EPICS:
        rep2_skip += 1
        continue  # target is a HELD new epic -> skip
    out, rc = bd(["update", bid, "--parent", to])
    rep2_ok += 1
    ledger.append(f"REPARENT {bid} -> {to} rc={rc if LIVE else 'dry'}")
# 0ns -> mhx (S6 ruling)
out, rc = bd(["update", "polylogue-0ns", "--parent", "polylogue-mhx"])
rep2_ok += 1
ledger.append(f"REPARENT polylogue-0ns -> polylogue-mhx rc={rc if LIVE else 'dry'}")

# ---------- dep edges ----------
s5 = load("draft_S5_deps.json") or {}
# removals: only 2 real edges exist (alias edges emb-* never resolved -> nothing to drop).
# Must drop backwards 37t.5->mhx.1 before adding mhx.1->37t.5 (avoid 2-cycle).
REAL_REMOVALS = [
    ("polylogue-mhx.1", "polylogue-37t.5"),  # (blocked, blocker) backwards edge
    ("polylogue-rlsb", "polylogue-0v9p"),
]  # downgrade blocks->relates
rem_ok = 0
for blocked, blocker in REAL_REMOVALS:
    out, rc = bd(["dep", "remove", blocked, blocker])
    rem_ok += 1
    ledger.append(f"DEP-RM {blocker}->{blocked} rc={rc if LIVE else 'dry'}")
add_ok = 0
for e in s5.get("edges") or []:
    b = e.get("blocker")
    k = e.get("blocks")
    t = e.get("type", "blocks")
    if not b or not k:
        continue
    typ = "relates-to" if t in ("relates-to", "related") else "blocks"
    out, rc = bd(["dep", "add", k, b, "-t", typ])
    add_ok += 1
    ledger.append(f"DEP {b} -{typ}-> {k} rc={rc if LIVE else 'dry'}")

print("\n".join(ledger[-60:]) if not LIVE else "\n".join(ledger))
print(f"\n=== SUMMARY ({'LIVE' if LIVE else 'DRY'}) ===")
print(f"field-fills applied={fill_ok} skipped(already-had)={fill_skip}")
print(f"reprioritizations={rep_ok}")
print(f"reparents(existing epics)={rep2_ok} skipped(held/checkpoint)={rep2_skip}")
print(f"dep removals={rem_ok} dep adds={add_ok}")
with open(f"{OUT}/ledger_apply1.json", "w") as fh:
    json.dump(ledger, fh, indent=0)
