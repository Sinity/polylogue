#!/usr/bin/env python3
"""Apply the accepted subset of the GPT-Pro delivery-upgrade package to live Beads.

Accepted (2026-07-07 triage, see chatlog-2026-07-07.md):
  - all 191 dependency edges (mechanically validated: no dangling/dup/cycle/closed)
  - AC fills where the live field is still empty
  - delivery:* and lane:* labels (ready:* grades SKIPPED — decay; kept in notes)
  - per-bead delivery notes (append-only, carry proof-artifact specs)
  - 5 durable memories

Skipped: ready:* labels.
Only open/in_progress beads are touched.
"""

import json
import subprocess
import sys
import time

BASE = ".agent/scratch/new/extracted/polylogue_beads_upgrade_setup/patches/"
LOG = []


def bd(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        LOG.append(f"FAIL bd {' '.join(args[:3])}...: {out[:200]}")
    return proc.returncode, out


def main() -> None:
    live = {}
    with open(".beads/issues.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            live[d["id"]] = d

    def active(bead_id: str) -> bool:
        b = live.get(bead_id)
        return b is not None and b["status"] != "closed"

    with open(BASE + "beads_delta_ops.jsonl") as fh:
        ops = [json.loads(line) for line in fh]
    counts = {"label": 0, "ac": 0, "note": 0, "dep": 0, "mem": 0, "skip": 0}
    t0 = time.time()

    for i, op in enumerate(ops):
        kind = op["op"]
        if kind == "append_label":
            if op["label"].startswith("ready:") or not active(op["id"]):
                counts["skip"] += 1
                continue
            rc, _ = bd("label", "add", op["id"], op["label"])
            counts["label"] += rc == 0
        elif kind == "set_acceptance_if_empty":
            b = live.get(op["id"])
            if b is None or b["status"] == "closed" or (b.get("acceptance_criteria") or "").strip():
                counts["skip"] += 1
                continue
            rc, _ = bd("update", op["id"], "--acceptance", op["proposed_acceptance_criteria"])
            counts["ac"] += rc == 0
        elif kind == "append_note":
            if not active(op["id"]):
                counts["skip"] += 1
                continue
            rc, _ = bd("update", op["id"], "--append-notes", op["text"])
            counts["note"] += rc == 0
        elif kind == "add_dependency":
            if not (active(op["issue_id"]) and active(op["depends_on_id"])):
                counts["skip"] += 1
                continue
            rc, _ = bd("dep", "add", op["issue_id"], op["depends_on_id"], "--no-cycle-check")
            counts["dep"] += rc == 0
        elif kind == "add_memory":
            rc, _ = bd("remember", op["value"], "--key", op["key"])
            counts["mem"] += rc == 0
        if i % 100 == 99:
            print(f"[{i + 1}/{len(ops)}] {counts} {time.time() - t0:.0f}s", flush=True)

    print(f"DONE {counts} in {time.time() - t0:.0f}s")
    if LOG:
        print(f"{len(LOG)} failures:")
        for entry in LOG[:30]:
            print(" ", entry)
        sys.exit(1)


if __name__ == "__main__":
    main()
