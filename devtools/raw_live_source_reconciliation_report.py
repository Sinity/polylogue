"""Read-only report: classify quarantined raw evidence against live source files.

polylogue-u19l: ~59GB of ``raw_sessions`` rows sit at
``revision_authority='quarantined'``, an absorbing state the existing
revision-graph reconciler never resolves for most of them. This command runs
``polylogue.storage.live_source_reconciliation.plan_quarantined_live_source_reconciliation``
against the live archive's ``source.db`` (opened strictly read-only) and
reports, per row, whether the recorded ``source_path`` still contains the
exact archived bytes, has diverged, or is gone.

This is a dry-run classifier only. It never mutates ``source.db``, never
marks a row's authority, and never deletes a blob -- acting on a
"reconcilable" classification is a deliberately separate, explicitly
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
from polylogue.storage.live_source_reconciliation import (
    LiveSourceReconciliationCandidate,
    plan_quarantined_live_source_reconciliation,
)


def _candidate_payload(candidate: LiveSourceReconciliationCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    payload["provider"] = candidate.provider.value
    payload["revision_kind"] = candidate.revision_kind.value
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
        "--limit",
        type=int,
        default=None,
        help="Cap the number of quarantined rows scanned (unbounded by default).",
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

    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        blob_store = BlobStore(root / "blob")
        plan = plan_quarantined_live_source_reconciliation(conn, blob_store=blob_store, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "exact_match_count": len(plan.exact_match),
            "exact_match_bytes": plan.exact_match_bytes,
            "codex_header_strip_match_count": plan.codex_header_strip_match_count,
            "diverged_count": len(plan.diverged),
            "diverged_bytes": plan.diverged_bytes,
            "source_missing_count": len(plan.source_missing),
            "source_missing_bytes": plan.source_missing_bytes,
            "exact_match_sample": [_candidate_payload(c) for c in plan.exact_match[: args.sample_limit]],
            "diverged_sample": [_candidate_payload(c) for c in plan.diverged[: args.sample_limit]],
            "source_missing_sample": [_candidate_payload(c) for c in plan.source_missing[: args.sample_limit]],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.1f} MB"

    print(f"quarantined rows scanned: {plan.scanned_count}", file=stdout)
    print(
        f"exact match:     {len(plan.exact_match):>7}  ({_mb(plan.exact_match_bytes)})"
        f"  -- {plan.codex_header_strip_match_count} matched only after stripping the historical Codex header",
        file=stdout,
    )
    print(f"diverged:        {len(plan.diverged):>7}  ({_mb(plan.diverged_bytes)})", file=stdout)
    print(f"source missing:  {len(plan.source_missing):>7}  ({_mb(plan.source_missing_bytes)})", file=stdout)
    print("", file=stdout)
    print(
        "This is a read-only classification. No authority state or blob was mutated.",
        file=stdout,
    )
    for label, bucket in (
        ("exact match", plan.exact_match),
        ("diverged", plan.diverged),
        ("source missing", plan.source_missing),
    ):
        if not bucket:
            continue
        print(f"\n{label} sample (up to {args.sample_limit}):", file=stdout)
        for candidate in bucket[: args.sample_limit]:
            detail = candidate.comparison.detail or ""
            header_note = " [header-strip]" if candidate.comparison.matched_after_codex_header_strip else ""
            print(
                f"  {candidate.raw_id}  {candidate.provider.value}/{candidate.revision_kind.value}"
                f"  {candidate.blob_size}B  {candidate.source_path}{header_note} {detail}".rstrip(),
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
