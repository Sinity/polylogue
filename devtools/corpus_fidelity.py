"""Run the production corpus-fidelity acceptance gate over an archive root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    CORPUS_FIDELITY_CHECKS,
    ArchiveVerificationReport,
    verify_archive,
)
from polylogue.paths import archive_root as default_archive_root

_STATUS_LABELS = {
    OutcomeStatus.OK: "OK",
    OutcomeStatus.WARNING: "WARN",
    OutcomeStatus.ERROR: "FAIL",
    OutcomeStatus.SKIP: "SKIP",
}


def _is_blocking(report: ArchiveVerificationReport) -> bool:
    """Treat skipped corpus checks as an incomplete acceptance run."""
    return report.blocking or any(check.status is OutcomeStatus.SKIP for check in report.checks)


def _render_plain(report: ArchiveVerificationReport, *, blocking: bool, stdout: TextIO) -> None:
    counts = report.summary_counts(include_skip=True)
    print(f"Corpus fidelity: {report.archive_root}", file=stdout)
    print(
        f"  {counts['ok']} ok, {counts['warning']} warning, {counts['error']} error, {counts.get('skip', 0)} skipped",
        file=stdout,
    )
    print("", file=stdout)
    for check in report.checks:
        label = _STATUS_LABELS.get(check.status, check.status.value.upper())
        print(f"  [{label}] {check.name}: {check.summary}", file=stdout)
        for detail in check.details[:5]:
            print(f"      {detail}", file=stdout)
        evidence = getattr(check, "evidence", {})
        for key in ("absent_by_origin_cause", "breakdown", "unexplained_by_origin"):
            values = evidence.get(key)
            if isinstance(values, dict):
                for bucket, value in values.items():
                    print(f"      {value:>8}  {bucket}", file=stdout)
    print("", file=stdout)
    print("BLOCKING" if blocking else "clear", file=stdout)


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to inspect; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum number of representative evidence samples per check.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the registry report as JSON.")
    args = parser.parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    root = args.archive_root if args.archive_root is not None else default_archive_root()

    report = verify_archive(
        root,
        checks=CORPUS_FIDELITY_CHECKS,
        sample_limit=args.sample_limit,
    )
    blocking = _is_blocking(report)
    if args.json:
        payload = dict(report.to_json())
        payload["blocking"] = blocking
        json.dump(payload, output, indent=2, sort_keys=True)
        print(file=output)
    else:
        _render_plain(report, blocking=blocking, stdout=output)
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
