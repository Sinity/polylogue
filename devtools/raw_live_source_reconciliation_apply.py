"""Actuator: promote quarantined raw evidence proven correct by live-source verification.

polylogue-u19l: the "act" half of live-source-verification-based quarantine
reconciliation. ``devtools workspace raw-live-source-reconciliation`` only
classifies and reports; this command promotes the safe subset (exact byte
match, or exact match after stripping the historical Codex synthetic header)
out of ``revision_authority='quarantined'`` -- never the ``diverged`` or
``source_missing`` verdicts.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually promote rows, which additionally requires ``--backup-manifest``
pointing at a verified backup manifest for the ``source`` tier (see
``polylogue backup --output-dir <dir> --verify``). This never runs blob GC or
``VACUUM`` -- that is a separate, later, operator-invoked step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.raw_live_source_reconciliation_apply import (
    RawLiveSourceReconciliationApplyError,
    apply_raw_live_source_reconciliation,
)
from polylogue.paths import archive_root as default_archive_root


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to operate on; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of quarantined rows classified/promoted (unbounded by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually promote rows. Without this flag, nothing is mutated.",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        default=None,
        help="Verified backup manifest for the source tier. Required with --apply.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()

    try:
        report = apply_raw_live_source_reconciliation(
            root,
            backup_manifest=args.backup_manifest,
            limit=args.limit,
            dry_run=not args.apply,
        )
    except (RawLiveSourceReconciliationApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        payload = {
            "applied": report.applied,
            "scanned_count": report.scanned_count,
            "promoted_count": report.promoted_count,
            "promoted_bytes": report.promoted_bytes,
            "codex_header_strip_match_count": report.codex_header_strip_match_count,
            "diverged_count": report.diverged_count,
            "diverged_bytes": report.diverged_bytes,
            "source_missing_count": report.source_missing_count,
            "source_missing_bytes": report.source_missing_bytes,
            "promoted_raw_ids": list(report.promoted_raw_ids),
            "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.1f} MB"

    mode = "APPLIED" if report.applied else "dry-run (no mutation performed -- pass --apply to promote)"
    print(f"mode: {mode}", file=stdout)
    print(f"quarantined rows scanned: {report.scanned_count}", file=stdout)
    print(
        f"{'promoted' if report.applied else 'promotable'}:"
        f"     {report.promoted_count:>7}  ({_mb(report.promoted_bytes)})"
        f"  -- {report.codex_header_strip_match_count} matched only after stripping the historical Codex header",
        file=stdout,
    )
    print(
        f"left quarantined (diverged):        {report.diverged_count:>7}  ({_mb(report.diverged_bytes)})", file=stdout
    )
    print(
        f"left quarantined (source missing):  {report.source_missing_count:>7}  ({_mb(report.source_missing_bytes)})",
        file=stdout,
    )
    if report.applied:
        print(f"backup manifest used: {report.backup_manifest}", file=stdout)
        print(
            "Each promoted row has an immutable receipt in "
            "raw_live_source_reconciliation_receipts. No blob GC or VACUUM was run -- "
            "that is a separate, later step.",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
