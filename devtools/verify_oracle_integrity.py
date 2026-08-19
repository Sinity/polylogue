"""CLI entry for the oracle-integrity lint (polylogue-4v2d3, polylogue-jdesf).

Two sweeps over the test corpus, both members of the family "the test
exercises something other than what it claims": a test whose entire production
target set is unreachable certifies dead code, and a test that reads a real
user or archive path is reading the developer's machine rather than a fixture.

The analysis lives in :mod:`devtools.oracle_integrity`; this module is the thin
entrypoint the command catalog dispatches to.
"""

from __future__ import annotations

import argparse
import json

from devtools import repo_root
from devtools.oracle_integrity import BASELINE_PATH, check_oracle_integrity, load_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify tests certify live code, hermetically.")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable report.")
    parser.add_argument(
        "--ignore-baseline",
        action="store_true",
        help="Report pre-existing findings too (the WS-F deletion worklist).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the ratchet baseline from the current findings.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    baseline: frozenset[tuple[str, str, str]] | None = (
        frozenset() if (args.ignore_baseline or args.write_baseline) else None
    )
    report = check_oracle_integrity(root, baseline=baseline)

    if args.write_baseline:
        entries = [
            {"code": finding.code, "path": finding.path, "detail": finding.detail}
            for finding in sorted(report.findings, key=lambda item: (item.code, item.path, item.detail))
        ]
        payload = {
            "schema_version": 1,
            "purpose": (
                "Pre-existing oracle-integrity findings, exempted as a ratchet so the gate is "
                "adoptable today. Each entry is a WS-F deletion-sweep worklist item, not an "
                "endorsement. Removing entries is the goal."
            ),
            "generated_by": "devtools lab policy oracle-integrity --write-baseline",
            "entries": entries,
        }
        target = root / BASELINE_PATH
        previous = load_baseline(root)
        current = {(entry["code"], entry["path"], entry["detail"]) for entry in entries}
        added = sorted(current - previous)
        removed = sorted(previous - current)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {len(entries)} baseline entries to {BASELINE_PATH} (was {len(previous)})")
        # Regenerating a ratchet must never be a silent widening.
        for code, path_value, detail in added:
            print(f"  + {code} {path_value}: {detail}")
        for code, path_value, detail in removed:
            print(f"  - {code} {path_value}: {detail}")
        if not added and not removed:
            print("  (no change)")
        return 0

    if args.json:
        print(report.to_json())
    else:
        print(
            f"oracle-integrity: scanned {report.scanned_modules} test module(s); "
            f"{report.reachable_modules} reach a production entrypoint; "
            f"{len(report.type_only_modules)} reach one only through TYPE_CHECKING; "
            f"{report.baselined} baselined."
        )
        for module in report.type_only_modules:
            print(f"  [type-only] {module}  (needs a human; not proof of dead code)")
        for finding in report.findings:
            print(f"  [{finding.code}] {finding.path}: {finding.detail}")
        if report.ok:
            print("Oracle integrity intact.")
        else:
            print(f"Policy violation: {len(report.findings)} new oracle-integrity finding(s).")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
