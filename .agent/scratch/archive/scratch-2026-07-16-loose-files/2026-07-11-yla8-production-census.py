#!/usr/bin/env python3
"""Read-only append-frontier/cursor census for the yla8.6 production repair."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _rows(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql).fetchall()]


def _file_state(path_text: str) -> dict[str, Any]:
    try:
        stat = os.stat(path_text)
    except OSError as exc:
        return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def build_census(root: Path, *, quick_check: bool = False) -> dict[str, Any]:
    source_path = root / "source.db"
    index_path = root / "index.db"
    ops_path = root / "ops.db"
    with _ro(source_path) as source, _ro(index_path) as index, _ro(ops_path) as ops:
        missing_rows = _rows(
            source,
            """
            SELECT child.logical_source_key, child.raw_id,
                   child.predecessor_raw_id, child.source_path,
                   child.append_start_offset, child.append_end_offset,
                   child.acquisition_generation,
                   child.parsed_at_ms IS NOT NULL AS parsed
            FROM raw_sessions AS child
            LEFT JOIN raw_sessions AS parent
              ON parent.raw_id = child.predecessor_raw_id
            WHERE child.revision_kind = 'append'
              AND child.revision_authority = 'byte_proven'
              AND child.predecessor_raw_id IS NOT NULL
              AND parent.raw_id IS NULL
            ORDER BY child.logical_source_key, child.acquired_at_ms
            """,
        )

        index.execute("ATTACH DATABASE ? AS src", (str(source_path),))
        broken_heads = _rows(
            index,
            """
            SELECT head.logical_source_key, head.session_id,
                   head.accepted_raw_id, head.accepted_source_revision,
                   head.accepted_frontier_kind, head.accepted_frontier,
                   head.acquisition_generation, head.append_end_offset,
                   child.source_path, child.predecessor_raw_id,
                   child.append_start_offset,
                   child.append_end_offset AS source_append_end_offset
            FROM raw_revision_heads AS head
            JOIN src.raw_sessions AS child
              ON child.raw_id = head.accepted_raw_id
            LEFT JOIN src.raw_sessions AS parent
              ON parent.raw_id = child.predecessor_raw_id
            WHERE child.revision_kind = 'append'
              AND child.predecessor_raw_id IS NOT NULL
              AND parent.raw_id IS NULL
            ORDER BY head.logical_source_key
            """,
        )

        accepted_by_path = _rows(
            index,
            """
            SELECT child.source_path,
                   MAX(CASE
                         WHEN child.revision_kind = 'append'
                           THEN child.append_end_offset
                         ELSE child.blob_size
                       END) AS accepted_material_end,
                   COUNT(*) AS head_count
            FROM raw_revision_heads AS head
            JOIN src.raw_sessions AS child
              ON child.raw_id = head.accepted_raw_id
            GROUP BY child.source_path
            """,
        )
        accepted_end_by_path = {
            str(row["source_path"]): int(row["accepted_material_end"])
            for row in accepted_by_path
            if row["accepted_material_end"] is not None
        }
        cursor_rows = _rows(
            ops,
            """
            SELECT source_path, stat_size, byte_offset,
                   last_complete_newline, content_fingerprint,
                   failure_count, next_retry_at, excluded, updated_at_ms
            FROM ingest_cursor
            ORDER BY source_path
            """,
        )

        cursor_ahead: list[dict[str, Any]] = []
        broken_paths = {str(row["source_path"]) for row in broken_heads}
        reviewed_cursors: list[dict[str, Any]] = []
        for row in cursor_rows:
            source_file = str(row["source_path"])
            accepted_end = accepted_end_by_path.get(source_file)
            enriched = {
                **row,
                "accepted_material_end": accepted_end,
                "file": _file_state(source_file),
            }
            if accepted_end is not None and int(row["byte_offset"]) > accepted_end:
                cursor_ahead.append(enriched)
            if source_file in broken_paths:
                reviewed_cursors.append(enriched)

        integrity: dict[str, Any] = {"quick_check_run": quick_check}
        if quick_check:
            integrity.update(
                {
                    "source_quick_check": source.execute("PRAGMA quick_check").fetchone()[0],
                    "index_quick_check": index.execute("PRAGMA quick_check").fetchone()[0],
                    "ops_quick_check": ops.execute("PRAGMA quick_check").fetchone()[0],
                }
            )
        counts = {
            "source_raw_rows": int(source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]),
            "index_heads": int(index.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone()[0]),
            "ops_cursors": len(cursor_rows),
            "missing_parent_append_rows": len(missing_rows),
            "current_heads_with_missing_parent": len(broken_heads),
            "cursor_ahead_rows": len(cursor_ahead),
        }

    return {
        "schema": "polylogue.yla8-production-census.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "archive_root": str(root.resolve()),
        "counts": counts,
        "integrity": integrity,
        "missing_parent_append_rows": missing_rows,
        "current_heads_with_missing_parent": [
            {**row, "file": _file_state(str(row["source_path"]))} for row in broken_heads
        ],
        "cursor_ahead_rows": cursor_ahead,
        "broken_head_cursors": reviewed_cursors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path.home() / ".local/share/polylogue",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="also scan every SQLite B-tree; expensive on the production index",
    )
    args = parser.parse_args()
    print(json.dumps(build_census(args.archive_root, quick_check=args.quick_check), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
