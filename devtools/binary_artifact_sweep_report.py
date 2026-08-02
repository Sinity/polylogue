"""Read-only report: find raw rows whose bytes are a non-session binary format.

polylogue-hbtj2: reports every ``raw_sessions`` row whose content matches a
recognized non-conversational binary signature (SQLite being the concrete
finding of the 2026-08-02 authority-dataflow audit; PNG/JPEG/gzip/PDF also
checked, see ``core.binary_signatures``). This is a pure classification
pass -- it never mutates ``source.db``. Pair with
``devtools/binary_artifact_reclassify_apply.py`` (``--apply``-gated) to
actually persist ``raw_artifacts`` rows for the flagged content.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.binary_artifact_sweep import BinaryArtifactCandidate, scan_binary_artifacts


def _candidate_payload(candidate: BinaryArtifactCandidate) -> dict[str, object]:
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
        plan = scan_binary_artifacts(conn, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "binary_count": len(plan.binary_candidates),
            "binary_bytes": plan.binary_bytes,
            "binary_by_kind": plan.by_kind(),
            "binary_by_origin": plan.by_origin(),
            "recognized_session_count": len(plan.recognized_session_candidates),
            "recognized_session_bytes": plan.recognized_session_bytes,
            "binary_sample": [_candidate_payload(c) for c in plan.binary_candidates[: args.sample_limit]],
            "recognized_session_sample": [
                _candidate_payload(c) for c in plan.recognized_session_candidates[: args.sample_limit]
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.2f} MB"

    print(f"raw_sessions rows scanned: {plan.scanned_count}", file=stdout)
    print(
        f"unrecognized binary artifacts:  {len(plan.binary_candidates):>6}  ({_mb(plan.binary_bytes)})",
        file=stdout,
    )
    for kind, count in sorted(plan.by_kind().items()):
        print(f"  kind={kind:<18} {count:>6}", file=stdout)
    for origin, count in sorted(plan.by_origin().items()):
        print(f"  origin={origin:<18} {count:>6}", file=stdout)
    print(
        f"recognized-session binary artifacts (Hermes state.db/verification_evidence.db, "
        f"left untouched by this bead -- see PR notes): {len(plan.recognized_session_candidates):>6}"
        f"  ({_mb(plan.recognized_session_bytes)})",
        file=stdout,
    )
    print("", file=stdout)
    print("This is a read-only classification. No row was mutated.", file=stdout)
    for label, bucket in (
        ("unrecognized binary", plan.binary_candidates),
        ("recognized-session (informational only)", plan.recognized_session_candidates),
    ):
        if not bucket:
            continue
        print(f"\n{label} sample (up to {args.sample_limit}):", file=stdout)
        for candidate in bucket[: args.sample_limit]:
            print(
                f"  {candidate.raw_id}  {candidate.origin}  {candidate.blob_size}B  "
                f"{candidate.source_path}  [{candidate.artifact_kind}] {candidate.classification_reason}",
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
