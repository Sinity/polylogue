#!/usr/bin/env python3
"""Trim over-blocking edges from the GPT-Pro delivery overlay (2026-07-07 adjudication).

Principle applied: beads convention is blocks = HARD only. An edge survives only
if doing the dependent bead correctly is impossible/unsafe without the blocker.
Removed classes:
  1. b5l gates that contradict the operator-endorsed "b5l early" sequencing
     doctrine: blob-GC safety (8jg9.*) and backup restore drill (4be) are
     source-tier concerns orthogonal to an index-tier blue-green rebuild
     (rollback story = old generation, not backups). Kept: 1xc.8 (rebuild-safety
     scenario IS the safety proof for the rebuild itself) + pre-existing 20d.15.
  2. s7ae.5 <- ahqd: a passive observation bead cannot be a prerequisite of the
     live proof it observes.
  3. 2qx (OriginSpec) fan-out trimmed to genuinely NEW session-origin
     detectors/parsers (611 grok, 0cg otel-ingest, fs1.2 nemo, fs1.8 nous-chat,
     uiw origin-breadth, 7aw agent-config family, l4kf.1 CIF import). Removed
     for exports/analysis/existing-origin extensions where OriginSpec is
     refactor-churn avoidance (soft "after"), not a hard correctness gate.
Each removal appends a note-trail entry on the dependent bead? No — one summary
note on 2qx and b5l instead; per-bead noise avoided.
"""

import subprocess
import sys

REMOVALS = [
    # (dependent, blocker)
    ("polylogue-b5l", "polylogue-8jg9.4"),
    ("polylogue-b5l", "polylogue-8jg9.2"),
    ("polylogue-b5l", "polylogue-4be"),
    ("polylogue-s7ae.5", "polylogue-ahqd"),
    # 2qx fan-out: exports / analysis / existing-origin extensions
    ("polylogue-4g5", "polylogue-2qx"),  # HPI/Promnesia exposure = export integration
    ("polylogue-wmj", "polylogue-2qx"),  # OTel trace EXPORT lane
    ("polylogue-ale", "polylogue-2qx"),  # external link archival, not a session origin
    ("polylogue-r47", "polylogue-2qx"),  # Obsidian export profile
    ("polylogue-7k7", "polylogue-2qx"),  # research-tooling export (keeps fs1.3 dep)
    ("polylogue-tf0e", "polylogue-2qx"),  # bug fix in EXISTING generic-messages parser
    ("polylogue-da1", "polylogue-2qx"),  # drift sentinel monitors existing origins
    ("polylogue-ox0", "polylogue-2qx"),  # existing codex family; state-DB lane bypasses file dispatch
    ("polylogue-t0p", "polylogue-2qx"),  # extends existing claude-code origin sidecars
    ("polylogue-fs1.3", "polylogue-2qx"),  # fidelity declaration for existing Hermes origin
    ("polylogue-fs1.4", "polylogue-2qx"),  # analysis report
    ("polylogue-fs1.5", "polylogue-2qx"),  # eval JSONL export
    ("polylogue-fs1.6", "polylogue-2qx"),  # demo
    ("polylogue-fs1.7", "polylogue-2qx"),  # upstream/outbound integration
    ("polylogue-fs1.9", "polylogue-2qx"),  # Polylogue->Sinex emitter (export)
    ("polylogue-fs1.10", "polylogue-2qx"),  # spec-cards export
    ("polylogue-7xv", "polylogue-2qx"),  # git correlation analysis over existing data
    ("polylogue-l4kf.3", "polylogue-2qx"),  # outbound git-notes/SARIF export
    ("polylogue-h6r", "polylogue-2qx"),  # identity tuple uses core/sources.py vocab that exists today
    ("polylogue-rii.2", "polylogue-2qx"),  # materializes already-captured raw_hook_events/otlp
]

fails = []
for dependent, blocker in REMOVALS:
    proc = subprocess.run(
        ["bd", "dep", "remove", dependent, blocker],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(f"{status} {dependent} -x-> {blocker}: {out[:120]}")
    if proc.returncode != 0:
        fails.append((dependent, blocker, out))

if fails:
    print(f"\n{len(fails)} failures")
    sys.exit(1)
print(f"\nRemoved {len(REMOVALS)} edges")
