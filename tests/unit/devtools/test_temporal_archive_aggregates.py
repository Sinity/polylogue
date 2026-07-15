from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from devtools import temporal_archive_aggregates


def _make_index_db(root: Path) -> Path:
    root.mkdir()
    db = root / "index.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            PRAGMA user_version = 18;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                root_session_id TEXT,
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            CREATE TABLE session_profiles (session_id TEXT PRIMARY KEY);
            INSERT INTO sessions VALUES
                ('s1', NULL, 's1', 'codex-session', NULL, 't1', NULL, 'n1', 1, 0, 1780308000000, 1780308000000),
                ('s2', NULL, 's1', 'codex-session', NULL, 't2', NULL, 'n2', 1, 0, 1780394400000, 1780394400000),
                ('s3', 's1', 's1', 'claude-code-session', 'subagent', 't3', NULL, 'n3', 1, 0, 1782900000000, 1782900000000);
            INSERT INTO blocks VALUES
                ('s1::m1::0', 's1', 'tool_use', 'm1', 0, NULL, NULL, 'tool-1', 'Bash', NULL, NULL, 'run tests'),
                ('s1::m1::1', 's1', 'tool_result', 'm1', 1, NULL, NULL, 'tool-1', NULL, 0, 0, 'ok');
            INSERT INTO session_profiles VALUES ('s1');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db


def test_temporal_archive_aggregates_report_and_files(tmp_path: Path) -> None:
    """polylogue-dab/itvd: runs/observed_events/context_snapshots are now
    source-derived CTE relations (run_projection_relations.py), computed
    from `sessions`/`blocks`, not standalone materialized tables. Every
    session unconditionally produces exactly one run row and one
    context-snapshot row, plus one 'session_started' observed-event row
    (more if it has tool_use/tool_result block pairs).
    """
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(archive_root=archive_root, out_dir=out_dir, json=True)

    report = temporal_archive_aggregates.build_report(args)

    assert report["archive_root"] == str(archive_root.resolve())
    assert report["index_schema_version"] == 18
    assert report["cardinality"] == {
        "sessions": 3,
        "physical_sessions": 3,
        "logical_root_sessions": 1,
        "session_profiles": 1,
        "session_profile_coverage_exact": True,
        "runs": 3,
        "observed_events": 4,
        "context_snapshots": 3,
    }
    assert report["monthly_runs_by_harness_role_status"] == [
        {"month": "2026-06", "harness": "codex", "role": "main", "status": "completed", "runs": 2},
        {"month": "2026-07", "harness": "claude-code", "role": "subagent", "status": "completed", "runs": 1},
    ]
    # The tool_finished event's source_updated_at is NULL (position-ordered,
    # not time-ordered) and buckets to the 'unknown' month fallback.
    assert {"month": "unknown", "kind": "tool_finished", "delivery_state": "observed", "events": 1} in report[
        "monthly_observed_events_by_kind"
    ]

    cardinality = json.loads((out_dir / "archive-cardinality.json").read_text(encoding="utf-8"))
    assert cardinality == [report["cardinality"]]
    with (out_dir / "monthly-observed-events-by-kind.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"month": "2026-06", "kind": "session_started", "delivery_state": "observed", "events": "2"} in rows
    written_report = json.loads((out_dir / "temporal-archive-aggregates.report.json").read_text(encoding="utf-8"))
    assert written_report["cardinality"] == report["cardinality"]
