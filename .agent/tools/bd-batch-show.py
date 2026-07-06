#!/usr/bin/env python3
"""Batch-show beads: id, status, prio, title, desc head, deps, notes tail."""

import json
import subprocess
import sys


def dep_str(x: dict) -> str:
    target = x.get("depends_on_id") or x.get("to_id") or x.get("id") or "?"
    return f"{target}({x.get('type') or x.get('dep_type') or '?'})"


for bid in sys.argv[1:]:
    r = subprocess.run(["bd", "show", bid, "--json"], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)[0]
    except Exception:
        print(f"== {bid} MISSING/ERROR: {r.stderr.strip()[:120]}")
        continue
    print(f"== {d['id']} [{d['status']}] P{d['priority']} {d['issue_type']} | {d['title']}")
    desc = (d.get("description") or "").replace("\n", " ")[:280]
    print(f"   DESC: {desc}")
    deps = d.get("dependencies") or []
    if deps:
        print("   DEPS:", ", ".join(dep_str(x) if isinstance(x, dict) else str(x) for x in deps))
    notes = (d.get("notes") or "").replace("\n", " ")
    if notes:
        print(f"   NOTES-tail: ...{notes[-240:]}")
    print()
