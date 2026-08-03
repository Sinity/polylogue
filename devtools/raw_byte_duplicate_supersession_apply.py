"""Actuator: promote quarantined, logical-key-less raws proven byte-identical to an indexed raw.

polylogue-6753s (corrected finding, 2026-08-03): of 7,200 unindexed logical-
source heads, 4,305 (17.6 of 22.9 GiB) are ``revision_authority='quarantined'``
with ``logical_source_key IS NULL`` -- invisible to every reconciliation path,
since all of them key off ``logical_source_key`` -- and are byte-identical
(same ``blob_hash``) to some OTHER raw that already has a materialized session
in ``index.db``. These are re-acquisitions/re-syncs of already-archived
content, not missing content.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually promote rows, which additionally requires ``--backup-manifest``
pointing at a verified backup manifest for the ``source`` tier (see
``polylogue backup --output-dir <dir> --verify``). This never runs blob GC or
``VACUUM`` -- that is a separate, later, operator-invoked step. ``index.db``
is only ever opened read-only; nothing here writes to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.raw_byte_duplicate_supersession_apply import (
    RawByteDuplicateSupersessionApplyError,
    apply_raw_byte_duplicate_supersession,
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
        help="Cap the number of quarantined, logical-key-less rows classified/promoted (unbounded by default).",
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
        report = apply_raw_byte_duplicate_supersession(
            root,
            backup_manifest=args.backup_manifest,
            limit=args.limit,
            dry_run=not args.apply,
        )
    except (RawByteDuplicateSupersessionApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        payload = {
            "applied": report.applied,
            "scanned_count": report.scanned_count,
            "promoted_count": report.promoted_count,
            "promoted_bytes": report.promoted_bytes,
            "novel_count": report.novel_count,
            "promoted_raw_ids": list(report.promoted_raw_ids),
            "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _gib(byte_count: int) -> str:
        return f"{byte_count / (1024**3):.2f} GiB"

    mode = "APPLIED" if report.applied else "dry-run (no mutation performed -- pass --apply to promote)"
    print(f"mode: {mode}", file=stdout)
    print(f"quarantined, logical-key-less rows scanned: {report.scanned_count}", file=stdout)
    print(
        f"{'promoted' if report.applied else 'promotable'} (byte-identical to an indexed raw): "
        f"{report.promoted_count:>7}  ({_gib(report.promoted_bytes)})",
        file=stdout,
    )
    print(
        f"novel (no indexed byte-identical twin, left untouched -- lkrc/hjpx scope): {report.novel_count}",
        file=stdout,
    )
    if report.applied:
        print(f"backup manifest used: {report.backup_manifest}", file=stdout)
        print(
            "Each promoted row has an immutable receipt in "
            "raw_byte_duplicate_supersession_receipts. No blob GC or VACUUM was run -- "
            "that is a separate, later step. index.db was never written to.",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
