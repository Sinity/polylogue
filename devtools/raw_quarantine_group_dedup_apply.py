"""Actuator: promote one representative raw per fully-quarantined byte-identical group.

polylogue-zm4w8: 1,777 raw_sessions rows (22.2 GiB, measured 2026-08-03) among
the codex-session quarantine backlog are pure redundant duplicates -- same
``source_path`` AND same ``blob_hash`` as another raw_sessions row, with
every member of the group still quarantined (no indexed twin anywhere) --
invisible to ``raw-byte-duplicate-supersession-apply``, which only matches a
quarantined raw against an already-INDEXED twin.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually promote a representative per group and mark the rest, which
additionally requires ``--backup-manifest`` pointing at a verified backup
manifest for the ``source`` tier (see ``polylogue backup --output-dir <dir>
--verify``). This never runs blob GC or ``VACUUM`` -- that is a separate,
later, operator-invoked step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.raw_quarantine_group_dedup_apply import (
    RawQuarantineGroupDedupApplyError,
    apply_raw_quarantine_group_dedup,
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
        help="Cap the number of (source_path, blob_hash) groups classified/promoted (unbounded by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually promote representatives and mark duplicates. Without this flag, nothing is mutated.",
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
        report = asyncio.run(
            apply_raw_quarantine_group_dedup(
                root,
                backup_manifest=args.backup_manifest,
                limit=args.limit,
                dry_run=not args.apply,
            )
        )
    except (RawQuarantineGroupDedupApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        payload = {
            "applied": report.applied,
            "scanned_count": report.scanned_count,
            "group_count": report.group_count,
            "already_resolved_group_count": report.already_resolved_group_count,
            "promoted_count": report.promoted_count,
            "marked_duplicate_count": report.marked_duplicate_count,
            "marked_duplicate_bytes": report.marked_duplicate_bytes,
            "promotions": [
                {
                    "source_path": promotion.source_path,
                    "blob_size": promotion.blob_size,
                    "representative_raw_id": promotion.representative_raw_id,
                    "representative_session_id": promotion.representative_session_id or None,
                    "duplicate_raw_ids": list(promotion.duplicate_raw_ids),
                }
                for promotion in report.promotions
            ],
            "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _gib(byte_count: int) -> str:
        return f"{byte_count / (1024**3):.2f} GiB"

    mode = "APPLIED" if report.applied else "dry-run (no mutation performed -- pass --apply to promote)"
    print(f"mode: {mode}", file=stdout)
    print(f"quarantined, source_path-bearing rows scanned: {report.scanned_count}", file=stdout)
    print(
        f"fully-quarantined duplicate groups {'processed' if report.applied else 'found'}: {report.group_count}",
        file=stdout,
    )
    print(
        f"already-resolved groups skipped (indexed twin or non-quarantined member elsewhere): "
        f"{report.already_resolved_group_count}",
        file=stdout,
    )
    print(
        f"{'materialized' if report.applied else 'would materialize'} representative(s): {report.promoted_count}",
        file=stdout,
    )
    print(
        f"{'marked' if report.applied else 'would mark'} duplicate(s) byte_proven: "
        f"{report.marked_duplicate_count:>7}  ({_gib(report.marked_duplicate_bytes)})",
        file=stdout,
    )
    if report.applied:
        print(f"backup manifest used: {report.backup_manifest}", file=stdout)
        print(
            "Each marked duplicate has an immutable receipt in "
            "raw_quarantine_group_dedup_receipts. No blob GC or VACUUM was run -- "
            "that is a separate, later step.",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
