"""Apply an exact, already-reconciled acceptance wave to a file copy.

This is a local file actuator for testing and coordinator dry runs. It never
invokes ``bd`` and never writes a Beads database.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from devtools import beads_acceptance_reconciliation as reconciliation
from polylogue.core.json import dumps as json_dumps


def apply_guarded_wave(*, repository: Path, before: Path, wave: Path, report: Path, output: Path) -> dict[str, Any]:
    """Write the exact guarded result, or accept an identical prior result."""
    _, before_rows, wave_rows, report_value = reconciliation._validate_report_and_wave(
        repository=repository,
        before=before,
        wave=wave,
        report_path=report,
    )
    expected_rows = dict(before_rows)
    expected_rows.update(wave_rows)
    expected_order = list(before_rows)
    expected_digest = reconciliation.equality_digest(expected_rows)
    if output.exists():
        existing = reconciliation.load_jsonl(output)
        if list(existing) == expected_order and reconciliation.equality_digest(existing) == expected_digest:
            return {
                "ok": True,
                "idempotent": True,
                "output_population_digest": expected_digest,
                "report_digest": reconciliation.report_digest(report_value),
                "targeted_ids": report_value["targeted_ids"],
            }
        raise reconciliation.ReconciliationError("output already exists with a different guarded population")
    reconciliation._write_jsonl(output, (expected_rows[bead_id] for bead_id in expected_order))
    return {
        "ok": True,
        "idempotent": False,
        "output_population_digest": expected_digest,
        "report_digest": reconciliation.report_digest(report_value),
        "targeted_ids": report_value["targeted_ids"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = apply_guarded_wave(
            repository=args.repository,
            before=args.before,
            wave=args.wave,
            report=args.report,
            output=args.output,
        )
    except reconciliation.ReconciliationError as exc:
        print(str(exc))
        return 1
    print(json_dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
