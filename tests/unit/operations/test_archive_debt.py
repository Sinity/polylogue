"""Tests for the unified archive debt projection."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.operations import archive_debt as module
from polylogue.operations.archive_debt import archive_debt_list
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS


def _write_tier_version(path: Path, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    finally:
        conn.close()


def _write_current_tier_files(root: Path) -> None:
    for spec in ARCHIVE_TIER_SPECS.values():
        _write_tier_version(root / spec.filename, spec.version)


def test_archive_debt_reports_missing_required_tiers(tmp_path: Path) -> None:
    payload = archive_debt_list(archive_root=tmp_path, kinds=("archive-tier",))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:archive-tier:source:missing" in refs
    assert "debt:archive-tier:user:missing" in refs
    assert payload.totals.critical >= 2
    assert all(row.kind == "archive-tier" for row in payload.rows)


def test_archive_debt_reports_convergence_failures(tmp_path: Path) -> None:
    _write_current_tier_files(tmp_path)
    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        conn.execute(
            """
            CREATE TABLE convergence_debt (
                debt_id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL,
                last_error TEXT,
                next_retry_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO convergence_debt (
                stage, target_type, target_id, status, attempts, priority, updated_at_ms, last_error, next_retry_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session_insights",
                "session",
                "sess-1",
                "failed",
                2,
                10,
                int(datetime(2026, 6, 19, tzinfo=UTC).timestamp() * 1000),
                "boom",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    payload = archive_debt_list(archive_root=tmp_path, kinds=("convergence",))

    assert payload.totals.total == 1
    row = payload.rows[0]
    assert row.kind == "convergence"
    assert row.stage == "session_insights"
    assert row.subject_ref == "session:sess-1"
    assert row.status == "actionable"
    assert row.details == "boom"


def test_archive_debt_converts_embedding_and_fts_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_current_tier_files(tmp_path)

    monkeypatch.setattr(
        module,
        "embedding_readiness_info",
        lambda _path, detail=False: {
            "embedding_config_enabled": True,
            "embedding_enabled": True,
            "embedding_has_voyage_key": True,
            "embedding_pending_count": 3,
            "embedding_pending_message_count": 30,
            "embedding_stale_count": 1,
            "embedding_failure_count": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "fts_readiness_info",
        lambda _path, exact=False: {
            "invariant_ready": False,
            "surfaces": {
                "messages_fts": {
                    "source_exists": True,
                    "exists": True,
                    "triggers_present": False,
                    "ready": False,
                    "missing_rows": 7,
                    "excess_rows": 0,
                    "duplicate_rows": 0,
                }
            },
        },
    )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("embedding", "fts"))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:embedding:catchup:failures" in refs
    assert "debt:embedding:catchup:backlog" in refs
    assert "debt:fts:messages_fts" in refs
    assert payload.totals.total == 3
    assert payload.totals.critical == 2
