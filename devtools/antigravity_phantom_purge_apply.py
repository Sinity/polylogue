"""Actuator: purge ``antigravity-session`` brain-metadata phantom sessions.

polylogue-eo81 / polylogue-msia: ``devtools/antigravity_phantom_sweep_report.py``
only classifies and reports. This command acts on that report:

1. Deletes the flagged sessions via ``ArchiveStore.delete_sessions`` (the
   same low-level primitive CLI ``delete`` / MCP ``write(operation=
   'delete_session')`` already reach) -- a rebuild-safe removal, since the
   ingest-side content classifier already refuses ``*.md.metadata.json`` as
   session content (``AGENT_SIDECAR_META``, PR #3441) and will not
   resurrect them on re-ingest.
2. Persists an explicit ``raw_artifacts`` classification for each purged
   session's raw row (``materialize_artifact_observations_for_raw_ids``),
   scoped to exactly the affected raw ids rather than a full census, so the
   raw payload carries a durable "this is a sidecar, not a session" record
   instead of sitting unclassified once its session is gone.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually delete sessions and write ``raw_artifacts`` rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.antigravity_phantom_sweep import scan_antigravity_phantom_sessions
from polylogue.storage.artifacts.persistence import materialize_artifact_observations_for_raw_ids
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


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
        help="Cap the number of candidate sessions considered (unbounded by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete sessions and write raw_artifacts rows. Without this flag, nothing is mutated.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        print(f"no index.db at {index_db}", file=stdout)
        return 1

    # Read-only classification pass first, regardless of --apply, so the
    # operator always sees exactly what would change before any write.
    index_ro = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    source_ro = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) if source_db.exists() else None
    try:
        plan = scan_antigravity_phantom_sessions(index_ro, source_ro, limit=args.limit)
    finally:
        index_ro.close()
        if source_ro is not None:
            source_ro.close()

    applied = False
    deleted_count = 0
    observation_count = 0
    if args.apply and plan.candidates:
        session_ids = plan.session_ids()
        raw_ids = plan.raw_ids()

        with ArchiveStore(root) as archive:
            deleted_count = archive.delete_sessions(session_ids)

        if raw_ids and source_db.exists():
            source_rw = sqlite3.connect(source_db)
            try:
                observations = materialize_artifact_observations_for_raw_ids(source_rw, raw_ids)
            finally:
                source_rw.close()
            observation_count = len(observations)
        applied = True

    if args.json:
        payload = {
            "applied": applied,
            "candidate_count": len(plan.candidates),
            "deleted_count": deleted_count,
            "raw_artifacts_observations_written": observation_count,
            "missing_raw_row_count": plan.missing_raw_row_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    mode = "APPLIED" if applied else "dry-run (no mutation performed -- pass --apply to write)"
    print(f"mode: {mode}", file=stdout)
    print(f"brain-metadata phantom candidates found: {len(plan.candidates)}", file=stdout)
    if applied:
        print(f"sessions deleted: {deleted_count}", file=stdout)
        print(f"raw_artifacts observations written (scoped to purged raw ids): {observation_count}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
