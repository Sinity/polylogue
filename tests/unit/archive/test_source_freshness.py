"""Exact-source freshness contracts for polylogue-1xc.13."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import polylogue.archive.query.source_freshness as source_freshness_module
from polylogue.archive.query.source_freshness import (
    SOURCE_CURSOR_BYTE_LAG_FAMILY,
    NamedSourceOperationalReason,
    NamedSourceOperationalState,
    NamedSourceStage,
    ProjectionLimits,
    SourceStatEvidence,
    _unsafe_plan_details,
    aggregate_named_source_byte_lag,
    project_named_source_freshness,
)
from polylogue.archive.query.source_freshness_surfaces import (
    make_source_freshness_mcp_handler,
    register_source_freshness_mcp_tool,
    render_source_freshness_status,
    source_freshness_cli,
    source_freshness_status_payload,
)

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "source_freshness"
_NOW = datetime(2026, 7, 15, 20, 30, tzinfo=UTC)


def _fixture(name: str) -> dict[str, Any]:
    payload = json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


async def _invoke_handler(
    handler: Callable[[str], Awaitable[dict[str, object]]], source_path: str
) -> dict[str, object]:
    return await handler(source_path)


def _create_schema(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "ops.db") as conn:
        conn.executescript(
            """
            CREATE TABLE ingest_cursor (
                source_path TEXT PRIMARY KEY,
                stat_size INTEGER,
                byte_offset INTEGER,
                failure_count INTEGER,
                excluded INTEGER,
                next_retry_at TEXT,
                updated_at_ms INTEGER
            );
            CREATE TABLE ingest_attempts (
                attempt_id TEXT PRIMARY KEY,
                source_path TEXT,
                status TEXT,
                phase TEXT,
                error_message TEXT,
                started_at_ms INTEGER,
                heartbeat_at_ms INTEGER,
                finished_at_ms INTEGER
            );
            CREATE INDEX ingest_attempts_source_path_idx
                ON ingest_attempts(source_path);
            CREATE TABLE convergence_debt (
                target_type TEXT,
                target_id TEXT,
                stage TEXT,
                status TEXT,
                last_error TEXT
            );
            CREATE INDEX convergence_debt_target_stage_idx
                ON convergence_debt(target_id, stage);
            """
        )
    with sqlite3.connect(root / "source.db") as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                source_index INTEGER,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER,
                revision_authority TEXT,
                acquired_at_ms INTEGER
            );
            CREATE INDEX raw_sessions_source_path_idx
                ON raw_sessions(source_path);
            """
        )
    with sqlite3.connect(root / "index.db") as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                raw_id TEXT,
                sort_key_ms INTEGER
            );
            CREATE INDEX sessions_raw_id_idx ON sessions(raw_id);
            CREATE TABLE raw_revision_applications (
                raw_id TEXT,
                decision TEXT,
                detail TEXT,
                applied_at_ms INTEGER
            );
            CREATE INDEX raw_revision_applications_raw_id_idx
                ON raw_revision_applications(raw_id);
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT
            );
            CREATE INDEX messages_session_id_idx ON messages(session_id);
            CREATE TABLE blocks (
                message_id TEXT,
                search_text TEXT
            );
            CREATE INDEX blocks_message_id_idx ON blocks(message_id);
            CREATE VIRTUAL TABLE messages_fts
                USING fts5(search_text, content='');
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN
                SELECT 1;
            END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN
                SELECT 1;
            END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN
                SELECT 1;
            END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT,
                checked_at TEXT,
                source_rows INTEGER,
                indexed_rows INTEGER,
                missing_rows INTEGER,
                excess_rows INTEGER,
                duplicate_rows INTEGER
            );
            """
        )


def _source(root: Path, *, size: int) -> Path:
    path = root / "codex" / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _seed_cursor(
    root: Path,
    source: Path,
    *,
    observed_size: int,
    offset: int,
    failures: int = 0,
    excluded: bool = False,
    error: str | None = None,
) -> None:
    with sqlite3.connect(root / "ops.db") as conn:
        conn.execute(
            "INSERT INTO ingest_cursor VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(source),
                observed_size,
                offset,
                failures,
                int(excluded),
                None,
                int((_NOW.timestamp() - 600) * 1000),
            ),
        )
        if error is not None:
            conn.execute(
                "INSERT INTO ingest_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "attempt-five",
                    str(source),
                    "failed",
                    "parse",
                    error,
                    int((_NOW.timestamp() - 700) * 1000),
                    int((_NOW.timestamp() - 600) * 1000),
                    int((_NOW.timestamp() - 600) * 1000),
                ),
            )


def _seed_raw(
    root: Path,
    source: Path,
    *,
    raw_id: str,
    parsed: bool,
    acquired_at_ms: int,
    parse_error: str | None = None,
) -> None:
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                raw_id,
                "codex-session",
                "session-native",
                str(source),
                -1,
                bytes.fromhex("ab" * 32),
                "valid",
                parse_error,
                acquired_at_ms + 100 if parsed else None,
                "byte_proven",
                acquired_at_ms,
            ),
        )


def _seed_searchable(
    root: Path,
    source: Path,
    *,
    raw_id: str = "raw-current",
    acquired_at_ms: int = 1_000,
) -> None:
    _seed_raw(
        root,
        source,
        raw_id=raw_id,
        parsed=True,
        acquired_at_ms=acquired_at_ms,
    )
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", ("session-1", raw_id, 4_000))
        conn.execute(
            "INSERT INTO raw_revision_applications VALUES (?, ?, ?, ?)",
            (raw_id, "applied_append", "fixture application", 3_000),
        )
        conn.execute("INSERT INTO messages VALUES (?, ?)", ("message-1", "session-1"))
        conn.execute(
            "INSERT INTO blocks(rowid, message_id, search_text) VALUES (?, ?, ?)",
            (1, "message-1", "searchable fixture text"),
        )
        conn.execute(
            "INSERT INTO messages_fts(rowid, search_text) VALUES (?, ?)",
            (1, "searchable fixture text"),
        )
        conn.execute(
            "INSERT INTO fts_freshness_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("messages_fts", "ready", _NOW.isoformat(), 1, 1, 0, 0, 0),
        )


def _seed_stage(root: Path, source: Path, stage: str) -> None:
    _seed_cursor(root, source, observed_size=source.stat().st_size, offset=source.stat().st_size)
    if stage == "unseen":
        return
    _seed_raw(
        root,
        source,
        raw_id="raw-stage",
        parsed=stage != "acquired-unparsed",
        acquired_at_ms=1_000,
    )
    if stage == "acquired-unparsed":
        return
    if stage == "parsed-unindexed":
        return
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", ("session-stage", "raw-stage", 4_000))
        conn.execute("INSERT INTO messages VALUES (?, ?)", ("message-stage", "session-stage"))
        conn.execute(
            "INSERT INTO blocks(rowid, message_id, search_text) VALUES (?, ?, ?)",
            (2, "message-stage", "stage fixture text"),
        )
        state = "stale" if stage == "indexed-unconverged" else "ready"
        if stage == "searchable":
            conn.execute(
                "INSERT INTO messages_fts(rowid, search_text) VALUES (?, ?)",
                (2, "stage fixture text"),
            )
        conn.execute(
            "INSERT INTO fts_freshness_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("messages_fts", state, _NOW.isoformat(), 1, int(stage == "searchable"), 0, 0, 0),
        )


@pytest.mark.parametrize(
    "stage",
    [
        "unseen",
        "acquired-unparsed",
        "parsed-unindexed",
        "indexed-unconverged",
        "searchable",
    ],
)
def test_named_source_stage_ladder(tmp_path: Path, stage: str) -> None:
    root = tmp_path / stage
    _create_schema(root)
    source = _source(root, size=128)
    _seed_stage(root, source, stage)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.stage.value == stage
    assert projection.receipt.exact_source is True
    assert projection.receipt.archive_wide_aggregates is False
    assert projection.receipt.unsafe_scan_rejections == ()


def test_excluded_growing_source_is_degraded_before_idle_and_retains_reason(tmp_path: Path) -> None:
    fixture = _fixture("excluded-growing")
    expected = fixture["expected"]
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=fixture["file_size_bytes"])
    _seed_cursor(
        root,
        source,
        observed_size=fixture["cursor_observed_size_bytes"],
        offset=fixture["cursor_offset_bytes"],
        failures=fixture["failure_count"],
        excluded=fixture["excluded"],
        error=fixture["attempt_error"],
    )
    _seed_searchable(root, source, raw_id="raw-older-indexed", acquired_at_ms=1_000)
    _seed_raw(
        root,
        source,
        raw_id="raw-later-unparsed",
        parsed=False,
        acquired_at_ms=5_000,
    )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.operational_state.value == expected["operational_state"]
    assert projection.operational_reason is NamedSourceOperationalReason.CURSOR_EXCLUDED
    assert projection.stage.value == expected["stage"]
    assert projection.cursor.state == "excluded"
    assert projection.cursor.excluded is True
    assert projection.cursor.pending_bytes == expected["pending_bytes"]
    assert projection.cursor.unobserved_growth_bytes == expected["unobserved_growth_bytes"]
    assert projection.byte_lag.value_state == "known"
    assert projection.byte_lag.value == expected["pending_bytes"]
    assert projection.byte_lag.freshness.state == "degraded"
    assert projection.byte_lag.freshness.cause == "cursor-excluded"
    assert projection.byte_lag.freshness.last_good_evidence_refs
    assert projection.byte_lag.coverage.complete is False
    assert projection.byte_lag.coverage.exclusions[0].reason == "ingest-cursor-excluded"
    assert SOURCE_CURSOR_BYTE_LAG_FAMILY.validate(projection.byte_lag) == ()
    assert projection.retry.reason == fixture["attempt_error"]
    assert projection.accepted_raw_revision is not None
    assert projection.accepted_raw_revision.raw_id == expected["accepted_raw_id"]
    assert projection.parse.state == "pending"
    assert projection.index.accepted_raw_indexed is False
    assert projection.index.broken_head is True
    assert projection.index.source_session_ids == ("session-1",)
    assert projection.index.high_water_ms == 4_000
    assert projection.receipt.unsafe_scan_rejections == ()


def test_broken_accepted_head_degrades_an_otherwise_quiet_cursor(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source, raw_id="raw-older-indexed", acquired_at_ms=1_000)
    _seed_raw(
        root,
        source,
        raw_id="raw-current-unparsed",
        parsed=False,
        acquired_at_ms=5_000,
    )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.cursor.state == "idle"
    assert projection.index.broken_head is True
    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.operational_reason is NamedSourceOperationalReason.BROKEN_HEAD
    assert projection.stage is NamedSourceStage.ACQUIRED_UNPARSED


def test_healthy_quiet_source_reports_every_checkpoint(tmp_path: Path) -> None:
    fixture = _fixture("healthy-quiet")
    expected = fixture["expected"]
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=fixture["file_size_bytes"])
    _seed_cursor(
        root,
        source,
        observed_size=fixture["cursor_observed_size_bytes"],
        offset=fixture["cursor_offset_bytes"],
    )
    _seed_searchable(root, source)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.operational_state is NamedSourceOperationalState.IDLE
    assert projection.operational_reason is NamedSourceOperationalReason.CAUGHT_UP
    assert projection.operational_state.value == expected["operational_state"]
    assert projection.stage is NamedSourceStage.SEARCHABLE
    assert projection.stage.value == expected["stage"]
    assert projection.cursor.pending_bytes == expected["pending_bytes"]
    assert projection.cursor.unobserved_growth_bytes == expected["unobserved_growth_bytes"]
    assert projection.byte_lag.value_state == "known"
    assert projection.byte_lag.value == expected["pending_bytes"]
    assert projection.byte_lag.freshness.state == "fresh"
    assert projection.byte_lag.freshness.cause is None
    assert projection.byte_lag.coverage.exclusions == ()
    assert SOURCE_CURSOR_BYTE_LAG_FAMILY.validate(projection.byte_lag) == ()
    assert projection.accepted_raw_revision is not None
    assert projection.accepted_raw_revision.raw_id == expected["accepted_raw_id"]
    assert projection.accepted_raw_revision.authority_owner == "polylogue-lkrc"
    assert projection.accepted_raw_revision.revision_authority == "byte_proven"
    assert projection.parse.state == "parsed"
    assert projection.revision_applications[0].owner == "polylogue-yla8"
    assert projection.index.accepted_raw_indexed is True
    assert projection.index.broken_head is False
    assert projection.index.high_water_ms == 4_000
    assert projection.fts.converged is True
    assert projection.fts.triggers_present is True
    fts_plan = next(plan for plan in projection.receipt.query_plans if plan.label == "fts-rowids-by-exact-block-rowids")
    assert fts_plan.safe is True
    assert any("VIRTUAL TABLE INDEX" in detail for detail in fts_plan.details)
    assert projection.insights.converged is True
    assert projection.receipt.unsafe_scan_rejections == ()
    assert projection.receipt.max_rows_per_relation == 513
    assert json.loads(json.dumps(projection.to_dict()))["stage"] == "searchable"


def test_missing_source_with_persisted_search_evidence_is_degraded(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = root / "codex" / "missing.jsonl"
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.source_stat.exists is False
    assert projection.stage is NamedSourceStage.SEARCHABLE
    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.operational_reason is NamedSourceOperationalReason.SOURCE_MISSING


def test_source_stat_error_is_degraded_and_receipt_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source)
    monkeypatch.setattr(
        source_freshness_module,
        "_stat_source",
        lambda _path: SourceStatEvidence(
            exists=False,
            error="PermissionError: source metadata denied",
        ),
    )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.operational_reason is NamedSourceOperationalReason.SOURCE_STAT_ERROR
    assert projection.source_stat.error == "PermissionError: source metadata denied"
    assert "source stat failed: PermissionError: source metadata denied" in projection.errors


def test_index_without_high_water_column_remains_bounded_and_indexed(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_raw(root, source, raw_id="raw-no-high-water", parsed=True, acquired_at_ms=1_000)
    with sqlite3.connect(root / "index.db") as conn:
        conn.executescript(
            """
            DROP TABLE sessions;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                raw_id TEXT
            );
            CREATE INDEX sessions_raw_id_idx ON sessions(raw_id);
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?)",
            ("session-no-high-water", "raw-no-high-water"),
        )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.index.available is True
    assert projection.index.accepted_raw_indexed is True
    assert projection.index.accepted_session_ids == ("session-no-high-water",)
    assert projection.index.high_water_ms is None
    assert not any("ORDER BY term out of range" in error for error in projection.errors)
    assert projection.receipt.unsafe_scan_rejections == ()


def test_growing_nonexcluded_source_is_active_before_idle(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=128)
    _seed_cursor(root, source, observed_size=64, offset=64)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.cursor.state == "behind"
    assert projection.cursor.pending_bytes == 64
    assert projection.cursor.unobserved_growth_bytes == 64
    assert projection.operational_state is NamedSourceOperationalState.ACTIVE
    assert projection.operational_reason is NamedSourceOperationalReason.PENDING_BYTES


def test_existing_source_without_cursor_is_active(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = _source(root, size=64)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.stage is NamedSourceStage.UNSEEN
    assert projection.operational_state is NamedSourceOperationalState.ACTIVE
    assert projection.operational_reason is NamedSourceOperationalReason.CURSOR_MISSING
    assert projection.byte_lag.value_state == "unknown"
    assert projection.byte_lag.value is None
    assert projection.byte_lag.freshness.state == "unavailable"


def test_source_without_filesystem_or_archive_evidence_is_unseen(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = root / "codex" / "never-seen.jsonl"

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.stage is NamedSourceStage.UNSEEN
    assert projection.operational_state is NamedSourceOperationalState.UNSEEN
    assert projection.operational_reason is NamedSourceOperationalReason.NO_EVIDENCE


def test_cursor_ahead_is_degraded_before_idle(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=80, offset=96)

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.cursor.state == "ahead"
    assert projection.cursor.pending_bytes == 0
    assert projection.cursor.cursor_ahead_bytes == 32
    assert projection.cursor.observed_size_ahead_bytes == 16
    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.operational_reason is NamedSourceOperationalReason.CURSOR_AHEAD


def test_retrying_cursor_is_degraded_with_current_attempt_reason(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(
        root,
        source,
        observed_size=64,
        offset=64,
        failures=2,
        error="transient decoder failure",
    )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.cursor.state == "retrying"
    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.operational_reason is NamedSourceOperationalReason.CURSOR_RETRYING
    assert projection.retry.reason == "transient decoder failure"
    assert projection.retry.reason_source == "ops.ingest_attempts"


def test_healthy_cursor_does_not_replay_stale_attempt_reason(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(
        root,
        source,
        observed_size=64,
        offset=64,
        failures=0,
        excluded=False,
        error="historical decoder failure",
    )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.cursor.state == "idle"
    assert projection.retry.reason is None
    assert projection.retry.reason_source is None


def test_unindexed_fts_freshness_ledger_is_rejected_not_scanned(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source)
    with sqlite3.connect(root / "index.db") as conn:
        conn.executescript(
            """
            DROP TABLE fts_freshness_state;
            CREATE TABLE fts_freshness_state (
                surface TEXT,
                state TEXT,
                checked_at TEXT,
                source_rows INTEGER,
                indexed_rows INTEGER,
                missing_rows INTEGER,
                excess_rows INTEGER,
                duplicate_rows INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO fts_freshness_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("messages_fts", "ready", _NOW.isoformat(), 1, 1, 0, 0, 0),
        )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.stage is NamedSourceStage.INDEXED_UNCONVERGED
    assert projection.fts.converged is False
    assert any("fts-freshness-state-by-surface" in item for item in projection.receipt.unsafe_scan_rejections)


def test_insight_debt_keeps_target_type_namespaces_distinct(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source)
    with sqlite3.connect(root / "ops.db") as conn:
        conn.execute(
            "INSERT INTO convergence_debt VALUES (?, ?, ?, ?, ?)",
            ("source_path", "session-1", "insights", "pending", "wrong namespace"),
        )

    collision_only = project_named_source_freshness(root, source, now=_NOW)

    assert collision_only.stage is NamedSourceStage.SEARCHABLE
    assert collision_only.insights.converged is True

    with sqlite3.connect(root / "ops.db") as conn:
        conn.execute(
            "INSERT INTO convergence_debt VALUES (?, ?, ?, ?, ?)",
            ("session_id", "session-1", "insights", "pending", "real debt"),
        )

    with_real_debt = project_named_source_freshness(root, source, now=_NOW)

    assert with_real_debt.stage is NamedSourceStage.INDEXED_UNCONVERGED
    assert with_real_debt.insights.converged is False
    assert with_real_debt.insights.debt_errors == ("real debt",)


def test_projection_limits_reject_non_positive_bounds() -> None:
    with pytest.raises(ValueError, match="max_sessions must be a positive integer"):
        ProjectionLimits(max_sessions=0)


def test_sqlite_value_limit_rejects_oversized_exact_row(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_raw(
        root,
        source,
        raw_id="oversized-row",
        parsed=False,
        acquired_at_ms=1_000,
        parse_error="x" * 4_096,
    )

    projection = project_named_source_freshness(
        root,
        source,
        now=_NOW,
        limits=ProjectionLimits(sqlite_value_bytes=1_024),
    )

    assert projection.stage is NamedSourceStage.UNSEEN
    assert projection.receipt.sqlite_value_bytes == 1_024
    assert any("string or blob too big" in error.lower() for error in projection.errors)


def test_exact_source_reader_rejects_unindexed_archive_scan(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = _source(root, size=64)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                source_path TEXT,
                validation_status TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?, ?, ?)",
            [(f"decoy-{index}", f"/decoy/{index}", "valid", index) for index in range(200)],
        )
        conn.execute(
            "INSERT INTO raw_sessions VALUES (?, ?, ?, ?)",
            ("target", str(source), "valid", 1_000),
        )

    projection = project_named_source_freshness(root, source, now=_NOW)

    assert projection.stage is NamedSourceStage.UNSEEN
    assert projection.receipt.unsafe_scan_rejections
    assert any("raw-revisions-by-source-path" in item for item in projection.receipt.unsafe_scan_rejections)
    assert any("rejected unsafe query plan" in error for error in projection.errors)


def test_scan_guard_rejects_full_index_scan_but_allows_constrained_fts_lookup() -> None:
    assert _unsafe_plan_details(
        ("SCAN raw_sessions USING INDEX raw_sessions_acquired_idx",),
        ("raw_sessions",),
    ) == ("SCAN raw_sessions USING INDEX raw_sessions_acquired_idx",)
    assert (
        _unsafe_plan_details(
            ("SCAN messages_fts VIRTUAL TABLE INDEX 0:=",),
            ("messages_fts",),
        )
        == ()
    )


def test_accepted_raw_is_not_hidden_by_bounded_newer_skipped_rows(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_raw(
        root,
        source,
        raw_id="accepted-old",
        parsed=True,
        acquired_at_ms=1_000,
    )
    with sqlite3.connect(root / "source.db") as conn:
        for index in range(12):
            conn.execute(
                "INSERT INTO raw_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"skipped-new-{index}",
                    "codex-session",
                    f"skipped-{index}",
                    str(source),
                    index,
                    bytes.fromhex("cd" * 32),
                    "skipped",
                    None,
                    None,
                    "quarantined",
                    2_000 + index,
                ),
            )

    projection = project_named_source_freshness(
        root,
        source,
        now=_NOW,
        limits=ProjectionLimits(max_raw_revisions=4),
    )

    assert projection.raw_revisions_truncated is True
    assert len(projection.raw_revisions) == 4
    assert all(revision.validation_status == "skipped" for revision in projection.raw_revisions)
    assert projection.accepted_raw_revision is not None
    assert projection.accepted_raw_revision.raw_id == "accepted-old"
    assert projection.accepted_raw_revision.accepted_by_acquisition is True
    assert projection.stage is NamedSourceStage.PARSED_UNINDEXED


def test_bounded_attempt_tail_and_cursor_export_supply_exclusion_reason(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = _source(root, size=256)
    cursor_export = tmp_path / "cursor.json"
    cursor_export.write_text(
        json.dumps(
            {
                str(source): {
                    "source_path": str(source),
                    "stat_size": 128,
                    "byte_offset": 128,
                    "failure_count": 5,
                    "excluded": True,
                    "updated_at_ms": int(_NOW.timestamp() * 1000),
                }
            }
        ),
        encoding="utf-8",
    )
    attempt_log = tmp_path / "attempts.jsonl"
    filler = [json.dumps({"source_path": f"/decoy/{index}", "error": "x" * 40}) for index in range(100)]
    target_reason = "retained exact-source decoder failure"
    filler.append(json.dumps({"source_path": str(source), "error": target_reason, "attempt_id": "tail"}))
    attempt_log.write_text("\n".join(filler) + "\n", encoding="utf-8")
    limits = ProjectionLimits(attempt_tail_bytes=512)

    projection = project_named_source_freshness(
        root,
        source,
        now=_NOW,
        cursor_export=cursor_export,
        attempt_log=attempt_log,
        limits=limits,
    )

    assert projection.operational_state is NamedSourceOperationalState.DEGRADED
    assert projection.cursor.excluded is True
    assert projection.retry.reason == target_reason
    assert projection.retry.reason_source == "attempt-log-tail"
    assert projection.receipt.attempt_tail_bytes_read <= 512
    assert projection.receipt.cursor_export_bytes_read == cursor_export.stat().st_size


def test_cursor_export_read_rejects_content_beyond_hard_bound(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = _source(root, size=64)
    cursor_export = tmp_path / "oversized-cursor.json"
    cursor_export.write_bytes(b"{" + (b"x" * 128))

    projection = project_named_source_freshness(
        root,
        source,
        now=_NOW,
        cursor_export=cursor_export,
        limits=ProjectionLimits(cursor_export_bytes=32),
    )

    assert projection.cursor.present is False
    assert projection.receipt.cursor_export_bytes_read == 33
    assert "cursor export rejected: content exceeds 32-byte bound" in projection.errors


def test_status_cli_and_mcp_adapters_preserve_typed_projection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=64, offset=64)
    _seed_searchable(root, source)
    projection = project_named_source_freshness(root, source, now=_NOW)

    status = source_freshness_status_payload(projection)
    assert status["stage"] == "searchable"
    assert status["operational_state"] == "idle"
    assert status["operational_reason"] == "caught-up"
    assert status["source_exists"] is True
    assert status["source_size_bytes"] == 64
    assert status["cursor_observed_size_bytes"] == 64
    assert status["cursor_byte_offset"] == 64
    assert status["pending_bytes"] == 0
    assert status["byte_lag"] == projection.byte_lag.to_dict()
    assert status["byte_lag"]["value_state"] == "known"
    assert status["byte_lag"]["value"] == 0
    assert status["revision_authority_owner"] == "polylogue-lkrc"
    assert status["replay_prevention_owner"] == "polylogue-yla8"
    assert status["revision_application_count"] == 1
    assert status["broken_head"] is False
    assert status["cursor_ahead_bytes"] == 0
    assert status["source_raw_scope_truncated"] is False
    assert status["projection_error_count"] == 0
    assert "archive_session_count" not in status
    rendered = render_source_freshness_status(projection)
    assert "stage=searchable" in rendered
    assert "operational_reason=caught-up" in rendered
    assert "file_size=64" in rendered
    assert "cursor_offset=64" in rendered
    assert "pending_bytes=0" in rendered
    assert "byte_lag_evidence=fresh" in rendered
    assert "errors=0" in rendered

    exit_code = source_freshness_cli(["--archive-root", str(root), "--source", str(source), "--format", "json"])
    assert exit_code == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["stage"] == "searchable"
    assert cli_payload["byte_lag"]["value_state"] == "known"
    assert cli_payload["byte_lag"]["value"] == 0

    handler = make_source_freshness_mcp_handler(root)
    assert tuple(inspect.signature(handler).parameters) == ("source_path",)
    payload: dict[str, object] = asyncio.run(_invoke_handler(handler, str(source)))
    assert payload["stage"] == "searchable"
    mcp_byte_lag = payload["byte_lag"]
    assert isinstance(mcp_byte_lag, dict)
    assert mcp_byte_lag["value_state"] == "known"
    assert mcp_byte_lag["value"] == 0

    class FakeMcp:
        registered_name: str | None = None
        registered: object | None = None

        def tool(self, **kwargs: object) -> Callable[[object], object]:
            self.registered_name = str(kwargs["name"])

            def decorate(function: object) -> object:
                self.registered = function
                return function

            return decorate

    server = FakeMcp()
    registered = register_source_freshness_mcp_tool(server, handler)
    assert server.registered_name == "named_source_freshness"
    assert registered is handler


def test_named_source_byte_lag_aggregate_conserves_disjoint_sources_and_deduplicates_identity(
    tmp_path: Path,
) -> None:
    left_root = tmp_path / "left-archive"
    right_root = tmp_path / "right-archive"
    _create_schema(left_root)
    _create_schema(right_root)
    left_source = _source(left_root, size=100)
    right_source = _source(right_root, size=250)
    _seed_cursor(left_root, left_source, observed_size=70, offset=70)
    _seed_cursor(right_root, right_source, observed_size=200, offset=200)
    left = project_named_source_freshness(left_root, left_source, now=_NOW)
    right = project_named_source_freshness(right_root, right_source, now=_NOW)
    expected = (left_source, right_source)

    direct = aggregate_named_source_byte_lag((left, right), expected_source_paths=expected, now=_NOW)
    reordered_duplicate = aggregate_named_source_byte_lag(
        (right, left, left),
        expected_source_paths=(right_source, left_source, left_source),
        now=_NOW,
    )

    assert direct.value_state == "known"
    assert direct.value == 80
    assert direct.to_dict() == reordered_duplicate.to_dict()
    assert direct.coverage.intended_count == 2
    assert direct.coverage.observed_count == 2
    assert direct.coverage.supported_count == 2
    assert len(direct.contributions) == 2


def test_excluded_source_aggregate_retains_known_lag_but_degrades_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=96)
    _seed_cursor(root, source, observed_size=64, offset=64, excluded=True)
    projection = project_named_source_freshness(root, source, now=_NOW)

    total = aggregate_named_source_byte_lag(
        (projection,),
        expected_source_paths=(source,),
        now=_NOW,
    )

    assert total.value_state == "known"
    assert total.value == 32
    assert total.coverage.complete is False
    assert {item.reason for item in total.coverage.exclusions} == {
        "ingest-cursor-excluded",
        "contribution-coverage-incomplete",
    }
    assert total.freshness.state == "degraded"
    assert total.freshness.last_good_evidence_refs


def test_named_source_byte_lag_aggregate_dropped_source_and_unknown_lag_never_become_zero(
    tmp_path: Path,
) -> None:
    known_root = tmp_path / "known-archive"
    unknown_root = tmp_path / "unknown-archive"
    _create_schema(known_root)
    unknown_root.mkdir()
    known_source = _source(known_root, size=64)
    unknown_source = _source(unknown_root, size=64)
    _seed_cursor(known_root, known_source, observed_size=32, offset=32)
    known = project_named_source_freshness(known_root, known_source, now=_NOW)
    unknown = project_named_source_freshness(unknown_root, unknown_source, now=_NOW)
    expected = (known_source, unknown_source)

    with_unknown = aggregate_named_source_byte_lag((known, unknown), expected_source_paths=expected, now=_NOW)
    dropped = aggregate_named_source_byte_lag((known,), expected_source_paths=expected, now=_NOW)

    assert with_unknown.value_state == "unknown"
    assert with_unknown.value is None
    assert with_unknown.coverage.observed_count == 2
    assert with_unknown.coverage.supported_count == 1
    assert dropped.value_state == "unknown"
    assert dropped.value is None
    assert dropped.coverage.observed_count == 1
    assert dropped.coverage.exclusions[0].reason == "missing-contribution"


def test_named_source_byte_lag_aggregate_rejects_contradictory_duplicate_instead_of_order_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    _create_schema(root)
    source = _source(root, size=64)
    _seed_cursor(root, source, observed_size=32, offset=32)
    projection = project_named_source_freshness(root, source, now=_NOW)
    changed_evidence = replace(
        projection.byte_lag,
        value=12,
        evidence_refs=(*projection.byte_lag.evidence_refs, projection.byte_lag.definition_ref),
    )
    contradictory = replace(projection, byte_lag=changed_evidence)

    total = aggregate_named_source_byte_lag(
        (projection, contradictory),
        expected_source_paths=(source,),
        now=_NOW,
    )

    assert total.value_state == "unknown"
    assert total.value is None
    assert len(total.conflicts) == 1
    assert {observation.value for observation in total.conflicts[0].observations} == {12, 32}
