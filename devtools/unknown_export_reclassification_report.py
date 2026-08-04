"""Read-only report: re-detect provider for browser captures stamped ``unknown-export``.

polylogue-mvq8: captures larger than 8MiB used to detect their provider via a
1MiB byte-prefix probe that could miss ``session.provider`` entirely when a
capture's ``raw_provider_payload`` (sorted before ``session`` in the
receiver's key-sorted output) exceeded that window -- such rows were
durably stamped ``unknown-export`` at acquisition time. This command runs
``polylogue.storage.unknown_export_reclassification.plan_unknown_export_reclassification``
against the live archive's ``source.db`` (opened strictly read-only) and
reports, per stored ``unknown-export`` row, whether the now-fixed detection
logic recovers a real provider.

This is a dry-run classifier only. It never mutates ``source.db``, never
rewrites a row's ``origin``, and never triggers re-parse/re-index -- acting
on a "reclassifiable" verdict is a deliberately separate, explicitly
operator-authorized follow-up.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.unknown_export_reclassification import (
    DEFAULT_SOURCE_PATH_LIKE,
    UnknownExportReclassificationCandidate,
    plan_unknown_export_reclassification,
)


def _candidate_payload(candidate: UnknownExportReclassificationCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    payload["recovered_provider"] = candidate.recovered_provider.value if candidate.recovered_provider else None
    payload["recovered_origin"] = candidate.recovered_origin.value if candidate.recovered_origin else None
    payload["previous_capture_mode"] = (
        candidate.previous_capture_mode.value if candidate.previous_capture_mode else None
    )
    return payload


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to inspect; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--source-path-like",
        default=DEFAULT_SOURCE_PATH_LIKE,
        help=(
            "SQL LIKE pattern restricting which unknown-export rows to scan by "
            "source_path (default: the browser-capture spool shape). Pass an "
            "empty string to scan every unknown-export row regardless of source."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of unknown-export rows scanned (unbounded by default).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="How many rows per bucket to print in the human-readable report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full classified plan as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    source_db = root / "source.db"
    if not source_db.exists():
        print(f"no source.db at {source_db}", file=stdout)
        return 1

    source_path_like = args.source_path_like if args.source_path_like else None
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        blob_store = BlobStore(root / "blob")
        plan = plan_unknown_export_reclassification(
            conn,
            blob_store=blob_store,
            source_path_like=source_path_like,
            limit=args.limit,
        )
    finally:
        conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "reclassifiable_count": len(plan.reclassifiable),
            "reclassifiable_bytes": plan.reclassifiable_bytes,
            "reclassifiable_by_origin": plan.reclassifiable_by_origin,
            "still_unknown_count": len(plan.still_unknown),
            "still_unknown_bytes": plan.still_unknown_bytes,
            "blob_missing_count": len(plan.blob_missing),
            "blob_missing_bytes": plan.blob_missing_bytes,
            "reclassifiable_sample": [_candidate_payload(c) for c in plan.reclassifiable[: args.sample_limit]],
            "still_unknown_sample": [_candidate_payload(c) for c in plan.still_unknown[: args.sample_limit]],
            "blob_missing_sample": [_candidate_payload(c) for c in plan.blob_missing[: args.sample_limit]],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.1f} MB"

    print(f"unknown-export rows scanned: {plan.scanned_count}", file=stdout)
    print(
        f"reclassifiable:  {len(plan.reclassifiable):>7}  ({_mb(plan.reclassifiable_bytes)})"
        f"  -- by origin: {plan.reclassifiable_by_origin}",
        file=stdout,
    )
    print(f"still unknown:   {len(plan.still_unknown):>7}  ({_mb(plan.still_unknown_bytes)})", file=stdout)
    print(f"blob missing:    {len(plan.blob_missing):>7}  ({_mb(plan.blob_missing_bytes)})", file=stdout)
    print("", file=stdout)
    print(
        "This is a read-only classification. No origin or blob state was mutated.",
        file=stdout,
    )
    for label, bucket in (
        ("reclassifiable", plan.reclassifiable),
        ("still unknown", plan.still_unknown),
        ("blob missing", plan.blob_missing),
    ):
        if not bucket:
            continue
        print(f"\n{label} sample (up to {args.sample_limit}):", file=stdout)
        for candidate in bucket[: args.sample_limit]:
            origin_note = f" -> {candidate.recovered_origin.value}" if candidate.recovered_origin else ""
            print(
                f"  {candidate.raw_id}  {candidate.blob_size}B  {candidate.source_path}{origin_note}",
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
