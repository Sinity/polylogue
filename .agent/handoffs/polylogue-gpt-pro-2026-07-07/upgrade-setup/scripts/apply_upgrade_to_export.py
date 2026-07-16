#!/usr/bin/env python3
"""Apply the Polylogue delivery-upgrade patch to a Beads export JSONL.

Usage:
  python scripts/apply_upgrade_to_export.py /path/to/polylogue-beads-export.jsonl patch_manifest.json /tmp/upgraded.jsonl

Safety properties:
  - only appends labels not already present;
  - only sets acceptance_criteria when the existing field is empty;
  - only adds dependency edges not already present;
  - appends notes without replacing existing notes;
  - adds memories only when the key does not already exist.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(2)
source = Path(sys.argv[1])
patch = Path(sys.argv[2])
out = Path(sys.argv[3])
records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
manifest = json.loads(patch.read_text())
issues = {r["id"]: r for r in records if r.get("_type") == "issue"}
mem_keys = {r["key"] for r in records if r.get("_type") == "memory"}
now = manifest.get("generated_at", "delivery-upgrade")
creator = "delivery-upgrade-apply-script"

for lp in manifest.get("labels_to_add", []):
    o = issues.get(lp["id"])
    if not o:
        continue
    labels = o.setdefault("labels", [])
    if lp["label"] not in labels:
        labels.append(lp["label"])
        o["updated_at"] = now

for ap in manifest.get("acceptance_criteria_patches", []):
    o = issues.get(ap["id"])
    if not o:
        continue
    if not (o.get("acceptance_criteria") or "").strip():
        o["acceptance_criteria"] = ap["proposed_acceptance_criteria"]
        o["updated_at"] = now

for e in manifest.get("dependency_edges_to_add", []):
    o = issues.get(e["issue_id"])
    if not o or e["depends_on_id"] not in issues:
        continue
    deps = o.setdefault("dependencies", [])
    if not any(d.get("depends_on_id") == e["depends_on_id"] and d.get("type") == e.get("type", "blocks") for d in deps):
        deps.append(
            {
                "created_at": now,
                "created_by": creator,
                "depends_on_id": e["depends_on_id"],
                "issue_id": e["issue_id"],
                "metadata": json.dumps({"reason": e.get("reason", "delivery upgrade")}),
                "type": e.get("type", "blocks"),
            }
        )
        o["updated_at"] = now

for n in manifest.get("notes_to_append", []):
    o = issues.get(n["id"])
    if not o:
        continue
    old = o.get("notes") or ""
    text = n["text"]
    if text not in old:
        o["notes"] = (old.rstrip() + ("\n\n" if old.strip() else "") + text).strip()
        o["updated_at"] = now

# recompute blocks counts
block_deps = Counter()
block_dependents = Counter()
for o in issues.values():
    for d in o.get("dependencies") or []:
        if d.get("type") == "blocks":
            block_deps[o["id"]] += 1
            block_dependents[d["depends_on_id"]] += 1
for o in issues.values():
    o["dependency_count"] = block_deps[o["id"]]
    o["dependent_count"] = block_dependents[o["id"]]

for m in manifest.get("memories_to_add", []):
    if m["key"] not in mem_keys:
        records.append({"_type": "memory", "key": m["key"], "value": m["value"]})
        mem_keys.add(m["key"])

out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
print(f"wrote {out}")
