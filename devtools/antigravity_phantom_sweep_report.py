"""Read-only report: list ``antigravity-session`` brain-metadata phantom sessions.

polylogue-eo81 / polylogue-msia: every ``antigravity-session`` row (116/116
in the live archive as of the forensic audit) is a 1-message fragment
materialized from a per-artifact ``*.md.metadata.json`` sidecar, tagged
``degraded:brain-metadata-fragment`` at ingest time (PR #1856). This is a
pure classification pass over ``index.db``/``source.db`` -- it never
mutates either tier. Pair with
``devtools/antigravity_phantom_purge_apply.py`` (``--apply``-gated) to
actually delete the flagged sessions and reclassify their raw rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.antigravity_phantom_sweep import (
    AntigravityPhantomCandidate,
    scan_antigravity_phantom_sessions,
)


def _candidate_payload(candidate: AntigravityPhantomCandidate) -> dict[str, object]:
    return {
        "session_id": candidate.session_id,
        "raw_id": candidate.raw_id,
        "message_count": candidate.message_count,
        "source_path": candidate.source_path,
        "blob_size": candidate.blob_size,
    }


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
        help="Cap the number of candidate sessions scanned (unbounded by default).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=25,
        help="How many candidates to print in the human-readable report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full classified plan as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        print(f"no index.db at {index_db}", file=stdout)
        return 1

    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) if source_db.exists() else None
    try:
        plan = scan_antigravity_phantom_sessions(index_conn, source_conn, limit=args.limit)
    finally:
        index_conn.close()
        if source_conn is not None:
            source_conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "candidate_count": len(plan.candidates),
            "raw_bytes": plan.raw_bytes,
            "missing_raw_row_count": plan.missing_raw_row_count,
            "candidate_sample": [_candidate_payload(c) for c in plan.candidates[: args.sample_limit]],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    print(f"antigravity-session rows scanned: {plan.scanned_count}", file=stdout)
    print(
        f"brain-metadata phantom candidates: {len(plan.candidates):>6}  "
        f"({plan.raw_bytes / 1024:.2f} KB of raw sidecar payload)",
        file=stdout,
    )
    if plan.missing_raw_row_count:
        print(
            f"  (of which {plan.missing_raw_row_count} have no matching raw_sessions row -- "
            "still purge-eligible on session-tag evidence alone)",
            file=stdout,
        )
    print("", file=stdout)
    print("This is a read-only classification. No row was mutated.", file=stdout)
    if plan.candidates:
        print(f"\nsample (up to {args.sample_limit}):", file=stdout)
        for candidate in plan.candidates[: args.sample_limit]:
            print(
                f"  {candidate.session_id}  raw={candidate.raw_id}  "
                f"messages={candidate.message_count}  {candidate.source_path}",
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
