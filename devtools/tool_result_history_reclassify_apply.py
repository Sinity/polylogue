"""Actuator: persist raw_artifacts classification for tool-results/file-history rows.

polylogue-omsw: ``devtools/tool_result_history_sweep_report.py`` only
classifies and reports. This command actually writes ``raw_artifacts`` rows
(via ``polylogue.storage.artifacts.persistence.materialize_artifact_observations``,
the same durable-observation machinery ``devtools/binary_artifact_reclassify_apply.py``
already uses for the earlier binary-database miscapture) so the flagged rows
carry an explicit, queryable "this is a tool-result sidecar" / "this is a
file-history-snapshot-only stream, not a session" classification instead of
sitting unclassified or misclassified as ``coordinator_session_stream``.

This is scoped deliberately narrow: it upserts ``raw_artifacts`` rows for
every raw row (idempotent, safe to re-run) but never touches
``raw_sessions.revision_authority``, the ``sessions``/``messages`` tables in
``index.db``, or deletes anything. Per this repo's durable-tier evidence
retention precedent (the 2026-07-22 hook-inflation postmortem retained
genuinely-empty sessions rather than deleting them), the already-acquired
``raw_sessions`` rows for these families are real acquired artifacts, just
mislabelled -- reclassifying them, never deleting them, is the answer.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually write ``raw_artifacts`` rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.artifacts.persistence import materialize_artifact_observations
from polylogue.storage.tool_result_history_sweep import scan_tool_result_and_file_history_artifacts


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
        help="Cap the number of raw_sessions rows classified (unbounded by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write raw_artifacts rows. Without this flag, nothing is mutated.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    source_db = root / "source.db"
    if not source_db.exists():
        print(f"no source.db at {source_db}", file=stdout)
        return 1

    # Read-only classification pass first, regardless of --apply, so the
    # operator always sees exactly what would change before any write.
    ro_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        plan = scan_tool_result_and_file_history_artifacts(ro_conn, limit=args.limit)
    finally:
        ro_conn.close()

    applied = False
    observation_count = 0
    if args.apply:
        rw_conn = sqlite3.connect(source_db)
        rw_conn.row_factory = sqlite3.Row
        try:
            observations = materialize_artifact_observations(rw_conn)
        finally:
            rw_conn.close()
        observation_count = len(observations)
        applied = True

    if args.json:
        payload = {
            "applied": applied,
            "candidate_count": len(plan.candidates),
            "candidate_bytes": plan.candidate_bytes,
            "candidates_by_kind": plan.by_kind(),
            "observations_written": observation_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    mode = "APPLIED" if applied else "dry-run (no mutation performed -- pass --apply to write raw_artifacts rows)"
    print(f"mode: {mode}", file=stdout)
    print(
        f"tool-result/file-history-shaped rows found: {len(plan.candidates)}"
        f"  ({plan.candidate_bytes / (1024 * 1024):.2f} MB)  by kind: {plan.by_kind()}",
        file=stdout,
    )
    if applied:
        print(f"raw_artifacts observations written (full census, idempotent): {observation_count}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
