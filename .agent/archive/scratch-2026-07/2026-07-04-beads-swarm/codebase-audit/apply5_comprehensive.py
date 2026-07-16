#!/usr/bin/env python3
"""Comprehensive-pass apply: AC fills + prose edges + reality actions. --live to execute."""

import json
import subprocess
import sys

OUT = "/tmp/claude-1000/-realm-project-polylogue/900c6128-99a3-4986-8ed4-d5d7aacb38fe/scratchpad"
LIVE = "--live" in sys.argv
ledger = []


def bd(a):
    if not LIVE:
        ledger.append("DRY: bd " + " ".join(x if len(x) < 40 else x[:37] + "..." for x in a))
        return 0
    r = subprocess.run(["bd"] + a, capture_output=True, text=True)
    if r.returncode != 0:
        ledger.append(f"  ERR bd {a[0]} {a[1] if len(a) > 1 else ''}: " + (r.stdout + r.stderr)[:120])
    return r.returncode


def live(bid):
    r = subprocess.run(["bd", "show", bid, "--json"], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        d = d[0] if isinstance(d, list) else d
        return {"design": (d.get("design") or "").strip(), "acc": (d.get("acceptance_criteria") or "").strip()}
    except Exception:
        return None


# ---- AC fills (dedupe batch1+batch2) ----
acc = {}
for f in ["cc_ac_batch1.json", "cc_ac_batch2.json"]:
    try:
        with open(f"{OUT}/{f}") as fh:
            arr = json.load(fh)
    except Exception as e:
        ledger.append(f"load {f}: {e}")
        arr = []
    for it in arr:
        bid = it.get("id")
        if not bid:
            continue
        cur = acc.setdefault(bid, {})
        if it.get("set_design") and not cur.get("set_design"):
            cur["set_design"] = it["set_design"]
        if it.get("set_acceptance") and not cur.get("set_acceptance"):
            cur["set_acceptance"] = it["set_acceptance"]
fill = skip = 0
for bid, v in acc.items():
    lf = live(bid) if LIVE else {"design": "", "acc": ""}
    if lf is None:
        ledger.append(f"MISS {bid}")
        continue
    args = ["update", bid]
    did = []
    if v.get("set_design") and not lf["design"]:
        args += ["--design", v["set_design"]]
        did.append("design")
    if v.get("set_acceptance") and not lf["acc"]:
        args += ["--acceptance", v["set_acceptance"]]
        did.append("acc")
    if did:
        bd(args)
        fill += 1
        ledger.append(f"AC {bid}: {'+'.join(did)}")
    else:
        skip += 1

# ---- prose edges ----
with open(f"{OUT}/cc_edges.json") as fh:
    edges = json.load(fh)
eok = 0
for e in edges:
    b = e.get("blocker")
    k = e.get("blocks")
    t = e.get("type", "blocks")
    if not b or not k:
        continue
    typ = "relates-to" if t in ("relates-to", "related") else "blocks"
    bd(["dep", "add", k, b, "-t", typ])
    eok += 1
    ledger.append(f"EDGE {b} -{typ}-> {k}")

# ---- reality actions ----
bd(
    [
        "close",
        "polylogue-z7rv",
        "--reason",
        "Already shipped: durable-tier additive migration framework (migration_runner.py backup-gate + one-step advance, runner tests) landed in commit 5b28e91b9; two-regime docs reconciled this session (architecture-spine.md, internals.md); devtools lab policy schema-versioning green. Filed then found already-done (code-outruns-beads).",
    ]
)
ledger.append("CLOSE z7rv (already shipped 5b28e91b9)")
bd(
    [
        "comment",
        "polylogue-1xc.2",
        "REALITY PASS (2026-07-04): the rebuild-index-from-source.db path already SHIPPED; residual scope narrowed to (a) source.db is still in the DEFAULT `reset --database` deletion set — remove it or gate it, and (b) the unresolvable-raw-row guard is missing. Close on those two, not the rebuild path.",
    ]
)
ledger.append("NARROW 1xc.2 (residual: deletion-set + guard)")

print("\n".join(ledger))
print(f"\n=== SUMMARY ({'LIVE' if LIVE else 'DRY'}) AC-filled={fill} skipped={skip} edges={eok} ===")
