"""Archive maintenance benchmark tests.

Covers read-only backup-boundary planning, blob-GC dry-run candidate scans, and
SQLite archive space reporting. These benchmarks intentionally avoid backup
copy/restore execution and destructive GC semantics.

Run with:
    pytest tests/benchmarks/test_archive_maintenance.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from devtools.archive_space_report import build_space_report
from polylogue.cli.commands.maintenance import _backup_plan_payload
from polylogue.storage.blob_gc import run_blob_gc
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from tests.benchmarks.helpers import BenchmarkFixture


def _seed_archive_tiers(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for spec in ARCHIVE_TIER_SPECS.values():
        db = root / spec.filename
        with sqlite3.connect(db) as conn:
            conn.execute("PRAGMA user_version = 1")
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_marker(id TEXT PRIMARY KEY)")


def _seed_gc_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, blob_hash BLOB);
            CREATE TABLE blob_refs(blob_hash BLOB PRIMARY KEY);
            CREATE TABLE pending_blob_refs(
                operation_id TEXT NOT NULL,
                blob_hash BLOB NOT NULL,
                acquired_at_ms INTEGER NOT NULL
            );
            CREATE TABLE gc_generations(
                generation_id TEXT PRIMARY KEY,
                started_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER,
                reclaimed_count INTEGER NOT NULL DEFAULT 0,
                reclaimed_bytes INTEGER NOT NULL DEFAULT 0
            );
            """
        )


def _seed_sharded_blobs(blob_root: Path, count: int) -> None:
    old_mtime = 1.0
    for index in range(count):
        blob_hash = f"{index:064x}"
        path = blob_root / blob_hash[:2] / blob_hash[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-{index}".encode())
        os.utime(path, (old_mtime, old_mtime))


def _sharded_blob_paths(blob_root: Path, count: int) -> list[Path]:
    return [blob_root / f"{index:064x}"[:2] / f"{index:064x}"[2:] for index in range(count)]


def _seed_space_report_db(path: Path, rows: int = 500) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, blob_size INTEGER NOT NULL)")
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY, provider_meta TEXT)")
        conn.execute("CREATE TABLE messages(message_id TEXT PRIMARY KEY, session_id TEXT, text TEXT)")
        conn.execute("CREATE INDEX idx_messages_session ON messages(session_id)")
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?)",
            ((f"raw-{index}", index) for index in range(rows)),
        )
        conn.executemany(
            "INSERT INTO sessions VALUES (?, ?)",
            ((f"session-{index}", '{"source":"benchmark"}') for index in range(rows)),
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?)",
            (
                (f"message-{index}", f"session-{index}", f"archive maintenance benchmark row {index}")
                for index in range(rows)
            ),
        )


@pytest.mark.benchmark
def test_bench_archive_backup_plan_payload(
    benchmark: BenchmarkFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build the backup-plan payload over present tier files without copying data."""
    archive_root = tmp_path / "archive"
    _seed_archive_tiers(archive_root)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    payload = benchmark(lambda: _backup_plan_payload(archive_root))

    assert payload["mutates"] is False
    assert payload["mode"] == "backup_plan"


@pytest.mark.benchmark
def test_bench_blob_gc_dry_run_candidate_scan(
    benchmark: BenchmarkFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scan sharded blob candidates through GC dry-run without deleting files."""
    monkeypatch.chdir(tmp_path)
    archive_db = tmp_path / "index.db"
    blob_root = tmp_path / "blob"
    blob_count = 256
    _seed_gc_db(archive_db)
    _seed_sharded_blobs(blob_root, count=blob_count)

    would_delete = benchmark(lambda: run_blob_gc(archive_db, blob_root, max_batch=blob_count, dry_run=True))

    assert would_delete == blob_count
    assert all(path.exists() for path in _sharded_blob_paths(blob_root, blob_count))


@pytest.mark.benchmark
def test_bench_archive_space_report_object_scan(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    """Run the read-only dbstat-backed archive space report over a synthetic DB."""
    db = tmp_path / "index.db"
    _seed_space_report_db(db)

    report = benchmark(lambda: build_space_report(db, limit=10, include_objects=True))

    assert report["ok"] is True
    assert report["dbstat_available"] is True
