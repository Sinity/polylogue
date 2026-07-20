"""Batch-show beads: id, status, prio, title, desc head, deps, notes tail.

Usage: devtools workspace bead-batch-show <bead-id> [<bead-id> ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def _dep_str(dep: dict[str, Any]) -> str:
    target = dep.get("depends_on_id") or dep.get("to_id") or dep.get("id") or "?"
    return f"{target}({dep.get('type') or dep.get('dep_type') or '?'})"


def _show_one(bead_id: str) -> None:
    result = subprocess.run(["bd", "show", bead_id, "--json"], capture_output=True, text=True)
    try:
        record = json.loads(result.stdout)[0]
    except Exception:
        print(f"== {bead_id} MISSING/ERROR: {result.stderr.strip()[:120]}")
        return
    print(f"== {record['id']} [{record['status']}] P{record['priority']} {record['issue_type']} | {record['title']}")
    desc = (record.get("description") or "").replace("\n", " ")[:280]
    print(f"   DESC: {desc}")
    deps = record.get("dependencies") or []
    if deps:
        print("   DEPS:", ", ".join(_dep_str(d) if isinstance(d, dict) else str(d) for d in deps))
    notes = (record.get("notes") or "").replace("\n", " ")
    if notes:
        print(f"   NOTES-tail: ...{notes[-240:]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bead_ids", nargs="+", help="Bead ids to show (e.g. polylogue-kapb)")
    args = parser.parse_args(argv)
    for bead_id in args.bead_ids:
        _show_one(bead_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
