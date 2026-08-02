"""Read-only report: find ``agent-*.meta.json`` subagent-sidecar phantom sessions.

polylogue-ioz7: reports every ``index.db`` session whose join to
``source.raw_sessions`` matches the bead's repro predicate
(``message_count=0 AND source_path LIKE '%.meta.json'``) -- pre-2026-07-28
residue from a fixed producer bug that promoted a per-subagent metadata
sidecar file into its own empty session. This is a pure classification pass
-- it never mutates ``index.db`` or ``source.db``. Pair with
``devtools/agent_meta_sidecar_purge_apply.py`` (``--apply``-gated) to
actually delete the flagged ``sessions`` rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.agent_meta_sidecar_sweep import AgentMetaSidecarCandidate, scan_agent_meta_sidecar_sessions


def _candidate_payload(candidate: AgentMetaSidecarCandidate) -> dict[str, object]:
    return {
        "session_id": candidate.session_id,
        "origin": candidate.origin,
        "native_id": candidate.native_id,
        "raw_id": candidate.raw_id,
        "source_path": candidate.source_path,
        "blob_size": candidate.blob_size,
        "native_id_matches_agent_meta_shape": candidate.native_id_matches_agent_meta_shape,
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
        help="Cap the number of matching sessions returned (unbounded by default).",
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
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        print(f"no index.db at {index_db}", file=stdout)
        return 1
    if not source_db.exists():
        print(f"no source.db at {source_db}", file=stdout)
        return 1

    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        plan = scan_agent_meta_sidecar_sessions(conn, source_db, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        payload = {
            "scanned_count": plan.scanned_count,
            "candidate_count": plan.candidate_count,
            "candidate_bytes": plan.candidate_bytes,
            "shape_mismatch_count": plan.shape_mismatch_count,
            "by_origin": plan.by_origin(),
            "sample": [_candidate_payload(c) for c in plan.candidates[: args.sample_limit]],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.2f} MB"

    print(
        f"agent-*.meta.json sidecar phantom sessions: {plan.candidate_count:>6}  ({_mb(plan.candidate_bytes)} raw bytes)",
        file=stdout,
    )
    for origin, count in sorted(plan.by_origin().items()):
        print(f"  origin={origin:<24} {count:>6}", file=stdout)
    if plan.shape_mismatch_count:
        print(
            f"WARNING: {plan.shape_mismatch_count} candidates match source_path but NOT the "
            "'agent-<hash>.meta' native_id shape -- inspect before applying.",
            file=stdout,
        )
    print("", file=stdout)
    print("This is a read-only classification. No row was mutated.", file=stdout)
    if plan.candidates:
        print(f"\nsample (up to {args.sample_limit}):", file=stdout)
        for candidate in plan.candidates[: args.sample_limit]:
            print(
                f"  {candidate.session_id}  raw={candidate.raw_id}  {candidate.blob_size}B  {candidate.source_path}",
                file=stdout,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
