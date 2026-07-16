#!/usr/bin/env python3
"""Validate a Polylogue Beads delivery-upgrade package.

Usage:
  python scripts/validate_upgrade_manifest.py patch_manifest.json delivery_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)
patch = json.loads(Path(sys.argv[1]).read_text())
delivery = json.loads(Path(sys.argv[2]).read_text())
ids = {item["id"] for item in delivery["items"]}
all_ok = True
for e in patch.get("dependency_edges_to_add", []):
    if e["issue_id"] not in ids:
        print("missing target in delivery manifest", e)
        all_ok = False
    # blocker may be closed in theory, but in this package it should exist in delivery if active.
for item in delivery["items"]:
    for field in ["release", "lane", "upgraded_readiness_grade", "verification_lane", "proof_artifact_path"]:
        if not item.get(field):
            print("missing", field, "for", item["id"])
            all_ok = False
print("active beads:", len(ids))
print("acceptance patches:", len(patch.get("acceptance_criteria_patches", [])))
print("dependency edges:", len(patch.get("dependency_edges_to_add", [])))
print("label additions:", len(patch.get("labels_to_add", [])))
print("memories:", len(patch.get("memories_to_add", [])))
if not all_ok:
    sys.exit(1)
print("upgrade manifest validates")
