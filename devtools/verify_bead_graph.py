"""Bead-graph invariant lint (polylogue variant).

Run before shipping bead-state deltas. Universal invariants: no cycles; no
open bead without acceptance criteria; no duplicate ``wave:``/``area:``
labels; no wave inversions where both sides carry waves.

INTENTIONAL DIVERGENCE from sinex: multi-area labels are polylogue
convention; only duplicate ``wave:`` labels are flagged (the
``lane:``/``delivery:``/``horizon:`` taxonomy is local to this repo).

This checks *live* ``bd`` query state (``bd dep cycles`` / ``bd list --all
--json``), not the exported ``.beads/issues.jsonl`` snapshot that
``devtools lab policy backlog-hygiene`` reads — run it right before shipping
a bead-state delta, so it sees whatever is live in the local Dolt DB even if
not yet re-exported.

NB: ``bd list --json`` dependency objects use ``type``/``depends_on_id``
(``bd show --json`` differs).

This module supersedes the standalone ``.agent/scripts/bd-graph-lint`` shell
script (ported so it runs through the standard devtools command surface
instead of requiring a manually-invoked path).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    bead_id: str
    detail: str


def _wave(issue: dict[str, Any]) -> int | None:
    for label in issue.get("labels") or []:
        if label.startswith("wave:"):
            try:
                return int(label[len("wave:") :])
            except ValueError:
                return None
    return None


def _run_bd_dep_cycles() -> tuple[bool, str]:
    result = subprocess.run(["bd", "dep", "cycles"], capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _run_bd_list_all() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["bd", "list", "--all", "-n", "0", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    return payload if isinstance(payload, list) else []


def collect_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    by_id = {issue["id"]: issue for issue in issues}
    findings: list[Finding] = []
    for issue in issues:
        if issue.get("status") == "closed":
            continue
        labels = issue.get("labels") or []
        wave_labels = [label for label in labels if label.startswith("wave:")]
        if len(wave_labels) > 1:
            findings.append(Finding("duplicate-wave", issue["id"], f"labels={wave_labels}"))
        if not (issue.get("acceptance_criteria") or "").strip():
            findings.append(Finding("missing-ac", issue["id"], str(issue.get("title", ""))[:60]))
        wv = _wave(issue)
        for dep in issue.get("dependencies") or []:
            if dep.get("type") != "blocks":
                continue
            dep_issue = by_id.get(dep.get("depends_on_id"))
            if dep_issue is None or dep_issue.get("status") == "closed":
                continue
            dw = _wave(dep_issue)
            if wv is not None and dw is not None and dw > wv:
                findings.append(
                    Finding(
                        "wave-inversion",
                        issue["id"],
                        f"(wave:{wv}) <- {dep_issue['id']} (wave:{dw})",
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    del argv  # no options; mirrors the original script's zero-argument shape

    cycles_ok, cycles_output = _run_bd_dep_cycles()
    if cycles_output:
        print(cycles_output)
    if not cycles_ok:
        print("bead-graph: `bd dep cycles` reported a failure (see above)", file=sys.stderr)
        return 1

    issues = _run_bd_list_all()
    findings = collect_findings(issues)

    dup = sum(1 for f in findings if f.kind == "duplicate-wave")
    inv = sum(1 for f in findings if f.kind == "wave-inversion")
    noac = sum(1 for f in findings if f.kind == "missing-ac")
    for f in findings:
        if f.kind == "duplicate-wave":
            print(f"duplicate-wave: {f.bead_id} {f.detail}")
        elif f.kind == "missing-ac":
            print(f"missing-AC: {f.bead_id} {f.detail}")
        elif f.kind == "wave-inversion":
            print(f"wave-inversion: {f.bead_id} {f.detail}")

    print(f"violations: dup_labels={dup} inversions={inv} missing_ac={noac}")
    return 1 if (dup or inv or noac) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
