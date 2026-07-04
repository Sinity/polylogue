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
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, root_session_id TEXT);
            CREATE TABLE session_profiles (session_id TEXT PRIMARY KEY);
            CREATE TABLE session_runs (
                source_updated_at TEXT,
                harness TEXT,
                role TEXT,
                status TEXT
            );
            CREATE TABLE session_observed_events (
                source_updated_at TEXT,
                kind TEXT,
                delivery_state TEXT
            );
            CREATE TABLE session_context_snapshots (
                source_updated_at TEXT,
                boundary TEXT,
                inheritance_mode TEXT
            );
            INSERT INTO sessions VALUES ('s1', 's1'), ('s2', 's1');
            INSERT INTO session_profiles VALUES ('s1');
            INSERT INTO session_runs VALUES
                ('2026-06-01T10:00:00Z', 'codex', 'main', 'completed'),
                ('2026-06-02T10:00:00Z', 'codex', 'main', 'completed'),
                ('2026-07-01T10:00:00Z', 'claude', 'subagent', 'failed');
            INSERT INTO session_observed_events VALUES
                ('2026-06-01T10:00:00Z', 'tool', 'delivered'),
                ('2026-06-02T10:00:00Z', 'tool', 'delivered'),
                ('2026-07-01T10:00:00Z', 'message', 'candidate');
            INSERT INTO session_context_snapshots VALUES
                ('2026-06-01T10:00:00Z', 'session_start', 'prefix-sharing'),
                ('2026-07-01T10:00:00Z', 'subagent_start', 'spawned-fresh');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db


def test_temporal_archive_aggregates_report_and_files(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(archive_root=archive_root, out_dir=out_dir, json=True)

    report = temporal_archive_aggregates.build_report(args)

    assert report["archive_root"] == str(archive_root.resolve())
    assert report["index_schema_version"] == 18
    assert report["cardinality"] == {
        "sessions": 2,
        "physical_sessions": 2,
        "logical_root_sessions": 1,
        "session_profiles": 1,
        "session_profile_coverage_exact": True,
        "runs": 3,
        "observed_events": 3,
        "context_snapshots": 2,
    }
    assert report["monthly_runs_by_harness_role_status"][0] == {
        "month": "2026-06",
        "harness": "codex",
        "role": "main",
        "status": "completed",
        "runs": 2,
    }

    cardinality = json.loads((out_dir / "archive-cardinality.json").read_text(encoding="utf-8"))
    assert cardinality == [report["cardinality"]]
    with (out_dir / "monthly-observed-events-by-kind.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {
        "month": "2026-06",
        "kind": "tool",
        "delivery_state": "delivered",
        "events": "2",
    }
    written_report = json.loads((out_dir / "temporal-archive-aggregates.report.json").read_text(encoding="utf-8"))
    assert written_report["cardinality"] == report["cardinality"]
