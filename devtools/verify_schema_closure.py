"""Ratchet the derived-schema identity closure: it may shrink, never grow.

Gate classification: **blocking ratchet over a checked-in baseline**.

``derived_schema_identity`` digests the lowering, materializer and
replay-routing fingerprints, and each is an AST closure over *imported* source.
Membership therefore follows the live import graph, not directory boundaries,
and a pure-performance edit inside the closure moves the identity exactly as a
new column does. A moved identity costs one full archive reconvergence, and it
also invalidates every seeded test archive, which
``tests/infra/workload_artifacts.py`` keys on that same identity.

Nothing made that boundary visible in review, so it drifted: the closure grew
by importing its way into new modules without anyone deciding to.

The ratchet is asymmetric on purpose, mirroring ``devtools gate layering``:

* **growth is blocking.** A module the baseline does not name is a new file
  whose ordinary edits now force reconvergence. Admitting it means editing
  ``docs/plans/schema-closure-baseline.json`` in the same PR, which is the
  artifact review reads.
* **shrinking is free.** Narrowing an import is the wanted direction and never
  fails the gate. The stale baseline entries are reported so the baseline can
  be lowered to hold the ground.

Usage:
  devtools gate schema-closure
  devtools gate schema-closure --json
  devtools gate schema-closure --update   # rewrite the baseline deliberately
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from devtools.required_gate import evidence_gate_result
from polylogue.core.json import dumps

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path("docs/plans/schema-closure-baseline.json")
BASELINE_KIND = "polylogue.derived-identity-closure-baseline"

__all__ = ["BASELINE_PATH", "load_baseline", "measure_closure", "main", "write_baseline"]


def measure_closure(*, root: Path = ROOT) -> tuple[str, ...]:
    """Return the repo-relative closure members, sorted."""
    from polylogue.sources.origin_specs import derived_identity_source_closure

    members: set[str] = set()
    for member in derived_identity_source_closure():
        path = Path(member)
        try:
            members.add(str(path.relative_to(root)))
        except ValueError:
            members.add(str(path))
    return tuple(sorted(members))


def load_baseline(path: Path) -> tuple[str, ...]:
    """Load the checked-in closure baseline, or an empty inventory if absent."""
    if not path.exists():
        return ()
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    members = payload.get("members") if isinstance(payload, dict) else None
    if not isinstance(members, list):
        return ()
    return tuple(sorted(str(member) for member in members))


def write_baseline(path: Path, members: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": BASELINE_KIND,
        "note": (
            "Modules feeding the derived schema identity. Growth is a blocking finding in "
            "`devtools gate schema-closure`; shrinking this list is always allowed. Regenerate with "
            "`devtools gate schema-closure --update` and explain the growth in the PR."
        ),
        "count": len(members),
        "members": list(members),
    }
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools gate schema-closure",
        description="Ratchet the derived schema identity closure against its checked-in baseline.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the measured closure (a deliberate, reviewable diff)",
    )
    args = parser.parse_args(list(argv or []))

    baseline_path = ROOT / BASELINE_PATH
    observed = measure_closure(root=ROOT)

    if args.update:
        write_baseline(baseline_path, observed)
        print(f"  wrote {BASELINE_PATH} with {len(observed)} closure member(s).")
        return 0

    baseline = load_baseline(baseline_path)
    added = tuple(sorted(set(observed) - set(baseline)))
    removed = tuple(sorted(set(baseline) - set(observed)))

    gate = evidence_gate_result(
        gate="schema-closure",
        executable=sys.executable,
        executable_available=True,
        required_count=1,
        inspected_count=1 if baseline_path.exists() else 0,
        missing_count=0 if baseline_path.exists() else 1,
        semantic_violation_count=len(added),
        details=added[:8] if added else (),
    )

    if args.json:
        print(
            dumps(
                {
                    "ok": not added and gate.ok,
                    "baseline": str(BASELINE_PATH),
                    "observed_count": len(observed),
                    "baseline_count": len(baseline),
                    "added": list(added),
                    "removed": list(removed),
                    "required_gate": gate.to_payload(),
                }
            )
        )
    else:
        if not baseline_path.exists():
            print(f"  error: {BASELINE_PATH} not found -- run `devtools gate schema-closure --update`", file=sys.stderr)
        for member in added:
            print(f"  ✗ {member}: newly feeds the derived schema identity")
        if added:
            print(
                f"  {len(added)} module(s) entered the closure ({len(baseline)} -> {len(observed)}). Editing any of "
                "them now forces a full archive reconvergence, and the change must land before a rebuild starts. "
                "Narrow the import, or record the growth with `devtools gate schema-closure --update` and say why "
                "in the PR.",
                file=sys.stderr,
            )
        else:
            print(f"  Closure holds at {len(observed)} module(s); no new member.")
        if removed:
            print(
                f"  {len(removed)} baseline entr(y/ies) no longer feed the identity -- run "
                "`devtools gate schema-closure --update` to ratchet the baseline down."
            )
    return 1 if added or not gate.ok else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
