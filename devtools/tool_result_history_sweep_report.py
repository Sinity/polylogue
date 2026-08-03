"""Read-only report: find raw rows that should reclassify as sidecar artifacts.

polylogue-omsw: reports every ``claude-code-session`` ``raw_sessions`` row
that ``classify_artifact`` now recognizes as a ``tool-results/<name>``
sidecar or a file-history-snapshot-only stream, rather than an independent
session. Both classification gaps are closed for fresh acquisition
(``archive.artifact_taxonomy``); this report finds already-ingested rows
acquired before that fix landed. Pure classification pass -- never mutates
``source.db``. Pair with
``devtools/tool_result_history_reclassify_apply.py`` (``--apply``-gated) to
actually persist ``raw_artifacts`` rows for the flagged content.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.tool_result_history_sweep import (
    ToolResultHistoryCandidate,
    scan_tool_result_and_file_history_artifacts,
)


def _candidate_payload(candidate: ToolResultHistoryCandidate) -> dict[str, object]:
    return {
        "raw_id": candidate.raw_id,
        "origin": candidate.origin,
        "source_path": candidate.source_path,
        "blob_size": candidate.blob_size,
        "artifact_kind": candidate.artifact_kind,
        "classification_reason": candidate.classification_reason,
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
        help="Cap the number of raw_sessions rows scanned (unbounded by default).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=25,
        help="How many rows to print in the human-readable report.",
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
        plan = scan_tool_result_and_file_history_artifacts(conn, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "candidate_count": len(plan.candidates),
            "candidate_bytes": plan.candidate_bytes,
            "candidates_by_kind": plan.by_kind(),
            "sample": [_candidate_payload(c) for c in plan.candidates[: args.sample_limit]],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    print(f"claude-code-session raw_sessions rows scanned: {plan.scanned_count}", file=stdout)
    print(
        f"tool-result/file-history-shaped rows found: {len(plan.candidates):>6}"
        f"  ({plan.candidate_bytes / (1024 * 1024):.2f} MB)",
        file=stdout,
    )
    for kind, count in sorted(plan.by_kind().items()):
        print(f"  kind={kind:<24} {count:>6}", file=stdout)
    print("", file=stdout)
    print("This is a read-only classification. No row was mutated.", file=stdout)
    if plan.candidates:
        print(f"\nsample (up to {args.sample_limit}):", file=stdout)
        for candidate in plan.candidates[: args.sample_limit]:
            print(
                f"  {candidate.raw_id}  {candidate.origin}  {candidate.blob_size}B  "
                f"{candidate.source_path}  [{candidate.artifact_kind}] {candidate.classification_reason}",
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
