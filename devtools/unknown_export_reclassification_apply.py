"""Maintenance command for the durable ChatGPT unknown-export reclassification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.unknown_export_reclassification_apply import (
    UnknownExportReclassificationApplyError,
    apply_unknown_export_reclassification,
)
from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.unknown_export_reclassification import CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to operate on; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--source-path-like",
        default=CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE,
        help=(
            "SQL LIKE pattern restricting the durable repair to source_path "
            "(default: the browser-capture/chatgpt spool). Pass an empty string to scan all sources."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of unknown-export rows scanned.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually reclassify proven ChatGPT rows. Without this flag, nothing is mutated.",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        default=None,
        help="Verified source-tier backup manifest. Required with --apply.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the receipt-shaped report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    source_path_like = args.source_path_like if args.source_path_like else None
    try:
        report = apply_unknown_export_reclassification(
            root,
            backup_manifest=args.backup_manifest,
            source_path_like=source_path_like,
            limit=args.limit,
            dry_run=not args.apply,
        )
    except (UnknownExportReclassificationApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        payload = {
            "applied": report.applied,
            "scanned_count": report.scanned_count,
            "reclassifiable_count": report.reclassifiable_count,
            "reclassifiable_bytes": report.reclassifiable_bytes,
            "chatgpt_reclassifiable_count": report.chatgpt_reclassifiable_count,
            "chatgpt_reclassifiable_bytes": report.chatgpt_reclassifiable_bytes,
            "non_chatgpt_reclassifiable_count": report.non_chatgpt_reclassifiable_count,
            "still_unknown_count": report.still_unknown_count,
            "blob_missing_count": report.blob_missing_count,
            "reclassified_count": report.reclassified_count,
            "reclassified_bytes": report.reclassified_bytes,
            "reclassified_raw_ids": list(report.reclassified_raw_ids),
            "source_path_like": report.source_path_like,
            "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
            "index_reparse_required": report.index_reparse_required,
            "index_rows_touched": report.index_rows_touched,
            "receipt_table": "raw_unknown_export_reclassification_receipts",
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    mode = "APPLIED" if report.applied else "dry-run (no mutation performed -- pass --apply to reclassify)"
    print(f"mode: {mode}", file=stdout)
    print(f"unknown-export rows scanned: {report.scanned_count}", file=stdout)
    print(
        f"reclassifiable: {report.reclassifiable_count:>7}  ({report.reclassifiable_bytes / (1024 * 1024):.2f} MB)"
        f"  -- ChatGPT eligible: {report.chatgpt_reclassifiable_count}",
        file=stdout,
    )
    print(f"non-ChatGPT envelopes left unchanged: {report.non_chatgpt_reclassifiable_count}", file=stdout)
    print(f"still unknown: {report.still_unknown_count}", file=stdout)
    print(f"blob missing: {report.blob_missing_count}", file=stdout)
    print(f"{'reclassified' if report.applied else 'would reclassify'}: {report.reclassified_count}", file=stdout)
    print(
        "Generated index session identity is untouched; run the normal reparse/materialization route afterward.",
        file=stdout,
    )
    if report.applied:
        print(f"backup manifest used: {report.backup_manifest}", file=stdout)
        print(
            "Each changed row has an immutable receipt in "
            "raw_unknown_export_reclassification_receipts. No parser, index, blob GC, or VACUUM action was run.",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
