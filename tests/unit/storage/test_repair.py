from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.config import Config
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage import repair as repair_mod
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.fts.dangling_repair import DanglingFtsRepairOutcome
from polylogue.storage.insights.session.repair_assessment import assess_session_insight_repairs
from polylogue.storage.insights.session.runtime import SessionInsightCounts, SessionInsightStatusSnapshot
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _config(tmp_path: Path) -> Config:
    return Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=tmp_path / "archive.db")


def _status(
    *,
    source_documents: int = 0,
    materialized_documents: int = 0,
    materialized_rows: int = 0,
    pending_documents: int = 0,
    pending_rows: int = 0,
    stale_rows: int = 0,
    orphan_rows: int = 0,
) -> DerivedModelStatus:
    return DerivedModelStatus(
        name="test",
        ready=pending_documents == 0 and pending_rows == 0 and stale_rows == 0 and orphan_rows == 0,
        detail="",
        source_documents=source_documents,
        materialized_documents=materialized_documents,
        materialized_rows=materialized_rows,
        pending_documents=pending_documents,
        pending_rows=pending_rows,
        stale_rows=stale_rows,
        orphan_rows=orphan_rows,
    )


def test_session_insight_repair_count_uses_public_phase_status_key() -> None:
    statuses = {
        "session_profile_rows": _status(),
        "session_work_events": _status(),
        "session_work_events_fts": _status(),
        "session_phases": _status(pending_rows=2),
        "threads": _status(),
        "threads_fts": _status(),
        "session_tag_rollups": _status(),
    }

    assert repair_mod.session_insight_repair_count(statuses) == 2

    legacy_statuses = dict(statuses)
    legacy_statuses["session_phase_inference"] = legacy_statuses.pop("session_phases")
    assert repair_mod.session_insight_repair_count(legacy_statuses) == 0

    legacy_statuses = dict(statuses)
    legacy_statuses["session_work_event_inference"] = legacy_statuses.pop("session_work_events")
    assert repair_mod.session_insight_repair_count(legacy_statuses) == 0


def test_preview_counts_from_archive_debt_include_healthy_preview_targets_only() -> None:
    statuses = {
        "session_insights": repair_mod.ArchiveDebtStatus(
            name="session_insights",
            category=repair_mod._maintenance_target_spec("session_insights").category,
            destructive=False,
            issue_count=0,
            detail="ready",
            maintenance_target="session_insights",
        ),
        "dangling_fts": repair_mod.ArchiveDebtStatus(
            name="dangling_fts",
            category=repair_mod._maintenance_target_spec("dangling_fts").category,
            destructive=False,
            issue_count=0,
            detail="ready",
            maintenance_target="dangling_fts",
        ),
        "orphaned_messages": repair_mod.ArchiveDebtStatus(
            name="orphaned_messages",
            category=repair_mod._maintenance_target_spec("orphaned_messages").category,
            destructive=True,
            issue_count=0,
            detail="clean",
            maintenance_target="orphaned_messages",
        ),
        "empty_sessions": repair_mod.ArchiveDebtStatus(
            name="empty_sessions",
            category=repair_mod._maintenance_target_spec("empty_sessions").category,
            destructive=True,
            issue_count=4,
            detail="needs cleanup",
            maintenance_target="empty_sessions",
        ),
    }

    assert repair_mod.preview_counts_from_archive_debt(statuses) == {
        "session_insights": 0,
        "dangling_fts": 0,
        "empty_sessions": 4,
    }


def test_probe_only_archive_debt_skips_large_message_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    class Conn:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("large probe mode should not run exact SQL scans")

    statuses = {
        "messages_fts": _status(),
    }
    monkeypatch.setattr(repair_mod, "_table_has_more_than", lambda *_args: True)
    monkeypatch.setattr(repair_mod, "count_orphaned_messages_sync", lambda _conn: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(repair_mod, "count_empty_sessions_sync", lambda _conn: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        repair_mod, "count_unclassified_message_type_sync", lambda _conn: (_ for _ in ()).throw(AssertionError)
    )
    monkeypatch.setattr(repair_mod, "count_orphaned_attachments_sync", lambda _conn: 0)

    debt = repair_mod.collect_archive_debt_statuses_sync(
        cast(Any, Conn()), derived_statuses=statuses, include_expensive=False, probe_only=True
    )

    assert debt["orphaned_messages"].skipped is True
    assert debt["empty_sessions"].skipped is True
    assert debt["message_type_backfill"].skipped is True
    assert debt["orphaned_attachments"].skipped is False


def test_raw_materialization_preview_counts_replayable_rows_without_erasing_missing_blobs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    replayable_raw_id, replayable_size = blob_store.write_from_bytes(b'{"mapping":{}}')
    materialized_raw_id, materialized_size = blob_store.write_from_bytes(b'{"mapping":{"done":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replayable_raw_id,
                "chatgpt-export",
                "native-replay",
                "replay.json",
                0,
                bytes.fromhex(replayable_raw_id),
                replayable_size,
                1,
            ),
        )
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "missing-raw",
                "chatgpt-export",
                "native-missing",
                "missing.json",
                0,
                bytes.fromhex("f" * 64),
                9,
                2,
            ),
        )
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                materialized_raw_id,
                "chatgpt-export",
                "native-done",
                "done.json",
                0,
                bytes.fromhex(materialized_raw_id),
                materialized_size,
                3,
            ),
        )
        source_conn.commit()

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("native-done", "chatgpt-export", materialized_raw_id, "done", bytes(32)),
        )
        index_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.repaired_count == 1
    assert result.success is True
    assert result.metrics == {
        "raw_materialization_candidate_count": 1.0,
        "raw_materialization_selected_count": 1.0,
        "raw_materialization_missing_blob_count": 1.0,
        "raw_materialization_already_parsed_count": 0.0,
        "raw_materialization_total_blob_bytes": float(replayable_size),
        "raw_materialization_max_blob_bytes": float(replayable_size),
        "raw_materialization_selected_total_blob_bytes": float(replayable_size),
        "raw_materialization_selected_max_blob_bytes": float(replayable_size),
    }
    assert "selected raw payload bytes total=" in result.detail
    assert "largest=" in result.detail
    assert "1 raw rows blocked by missing blobs" in result.detail


def test_raw_materialization_replays_parsed_rows_when_index_is_empty(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    raw_id, blob_size = blob_store.write_from_bytes(b'{"mapping":{"already-parsed":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                "native-reset-replay",
                "reset-replay.json",
                0,
                bytes.fromhex(raw_id),
                blob_size,
                1,
                2,
            ),
        )
        source_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.repaired_count == 1
    assert result.success is True
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_already_parsed_count"] == 1.0
    assert "already parsed but not materialized" in result.detail


def test_raw_materialization_replays_parsed_rows_after_interrupted_index_rebuild(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    remaining_raw_id, remaining_size = blob_store.write_from_bytes(b'{"mapping":{"remaining":{}}}')
    done_raw_id, done_size = blob_store.write_from_bytes(b'{"mapping":{"done":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    remaining_raw_id,
                    "chatgpt-export",
                    "native-remaining",
                    "remaining.json",
                    0,
                    bytes.fromhex(remaining_raw_id),
                    remaining_size,
                    1,
                    2,
                ),
                (
                    done_raw_id,
                    "chatgpt-export",
                    "native-done",
                    "done.json",
                    0,
                    bytes.fromhex(done_raw_id),
                    done_size,
                    3,
                    4,
                ),
            ),
        )
        source_conn.commit()

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("native-done", "chatgpt-export", done_raw_id, "done", bytes(32)),
        )
        index_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.repaired_count == 1
    assert result.success is True
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_already_parsed_count"] == 1.0
    assert "already parsed but not materialized" in result.detail


def test_raw_materialization_replay_uses_batch_parse_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    first_raw_id, first_size = blob_store.write_from_bytes(b'{"mapping":{"first":{}}}')
    second_raw_id, second_size = blob_store.write_from_bytes(b'{"mapping":{"second":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    first_raw_id,
                    "chatgpt-export",
                    "native-first",
                    "first.json",
                    0,
                    bytes.fromhex(first_raw_id),
                    first_size,
                    1,
                ),
                (
                    second_raw_id,
                    "chatgpt-export",
                    "native-second",
                    "second.json",
                    0,
                    bytes.fromhex(second_raw_id),
                    second_size,
                    2,
                ),
            ),
        )
        source_conn.commit()

    calls: list[list[str]] = []

    class FakeParsingService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, *, raw_ids: list[str], **_kwargs: object) -> object:
            calls.append(list(raw_ids))
            return SimpleNamespace(processed_ids=set(raw_ids), parse_failures=0)

    import polylogue.pipeline.services.parsing as parsing_module

    monkeypatch.setattr(parsing_module, "ParsingService", FakeParsingService)

    result = repair_mod.repair_raw_materialization(config)

    assert result.success is True
    assert result.repaired_count == 2
    assert calls == [[second_raw_id, first_raw_id]]


def test_raw_materialization_dry_run_reports_limited_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(
            ["raw-slow", "raw-2", "raw-3", "raw-4"],
            0,
            4,
            {
                "raw-slow": 512,
                "raw-2": 1024,
                "raw-3": 2048,
                "raw-4": 4096,
            },
        ),
    )

    result = repair_mod.repair_raw_materialization(
        config,
        dry_run=True,
        raw_artifact_limit=2,
    )

    assert result.success is True
    assert result.repaired_count == 2
    assert "Would: replay 2 of 4 raw rows into index.db" in result.detail
    assert result.metrics["raw_materialization_candidate_count"] == 4.0
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert result.metrics["raw_materialization_limit"] == 2.0
    assert result.metrics["raw_materialization_total_blob_bytes"] == 7680.0
    assert result.metrics["raw_materialization_selected_total_blob_bytes"] == 1536.0
    assert result.metrics["raw_materialization_selected_max_blob_bytes"] == 1024.0


def test_raw_materialization_execute_replays_only_limited_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, *, db_path: Path) -> None:
            calls["db_path"] = db_path

    class FakeRepository:
        def __init__(self, *, backend: FakeBackend, archive_root: Path) -> None:
            calls["archive_root"] = archive_root

        async def close(self) -> None:
            calls["closed"] = True

    class FakeParseResult:
        processed_ids = {"session-2", "session-3"}
        parse_failures = 0

    class FakeParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, **kwargs: object) -> FakeParseResult:
            calls["parse_kwargs"] = kwargs
            return FakeParseResult()

    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(
            ["raw-slow", "raw-2", "raw-3", "raw-4"],
            0,
            4,
            {
                "raw-slow": 512,
                "raw-2": 1024,
                "raw-3": 2048,
                "raw-4": 4096,
            },
        ),
    )
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", FakeParsingService)
    monkeypatch.setattr("polylogue.storage.repository.SessionRepository", FakeRepository)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", FakeBackend)

    result = repair_mod.repair_raw_materialization(
        config,
        raw_artifact_limit=2,
    )

    assert result.success is True
    assert result.repaired_count == 2
    assert calls["parse_kwargs"] == {
        "raw_ids": ["raw-slow", "raw-2"],
        "progress_callback": None,
        "force_write": False,
        "repair_message_fts": False,
    }
    assert result.metrics["raw_materialization_candidate_count"] == 4.0
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert "Replayed 2 of 4 raw rows" in result.detail


def test_raw_materialization_raw_artifact_filter_counts_only_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    target_raw_id, target_size = blob_store.write_from_bytes(b'{"mapping":{"target":{}}}')
    other_raw_id, other_size = blob_store.write_from_bytes(b'{"mapping":{"other":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    target_raw_id,
                    "chatgpt-export",
                    "native-target",
                    "target.json",
                    0,
                    bytes.fromhex(target_raw_id),
                    target_size,
                    1,
                ),
                (
                    other_raw_id,
                    "chatgpt-export",
                    "native-other",
                    "other.json",
                    0,
                    bytes.fromhex(other_raw_id),
                    other_size,
                    2,
                ),
            ),
        )
        source_conn.commit()

    broad = repair_mod.repair_raw_materialization(config, dry_run=True)
    scoped = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_id=target_raw_id)

    assert broad.repaired_count == 2
    assert scoped.repaired_count == 1
    assert f"replay {1:,} of {1:,} acquired-but-unparsed raw rows" in scoped.detail


def test_raw_materialization_excludes_already_parsed_non_materialized_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    replayable_raw_id, replayable_size = blob_store.write_from_bytes(b'{"mapping":{"pending":{}}}')
    parsed_raw_id, parsed_size = blob_store.write_from_bytes(b'{"mapping":{"parsed":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    replayable_raw_id,
                    "chatgpt-export",
                    "native-pending",
                    "pending.json",
                    0,
                    bytes.fromhex(replayable_raw_id),
                    replayable_size,
                    1,
                    None,
                ),
                (
                    parsed_raw_id,
                    "chatgpt-export",
                    "native-parsed",
                    "parsed.json",
                    0,
                    bytes.fromhex(parsed_raw_id),
                    parsed_size,
                    2,
                    123,
                ),
            ),
        )
        source_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.repaired_count == 2
    assert "1 already parsed but not materialized" in result.detail

    scoped = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_id=parsed_raw_id)

    assert scoped.repaired_count == 1
    assert "already parsed but not materialized" in scoped.detail


def test_raw_materialization_excludes_parsed_non_session_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    raw_id, raw_size = blob_store.write_from_bytes(
        b'{"sessionId":"sidecar","projectHash":"abc","startTime":"now","lastUpdated":"now","kind":"metadata"}\n'
    )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "claude-code-session",
                "sidecar",
                "/captures/claude/sidecar.jsonl",
                0,
                bytes.fromhex(raw_id),
                raw_size,
                1,
                123,
                "passed",
            ),
        )
        source_conn.commit()

    broad = repair_mod.repair_raw_materialization(config, dry_run=True)
    scoped = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_id=raw_id)

    assert broad.repaired_count == 0
    assert scoped.repaired_count == 0
    assert broad.metrics["raw_materialization_candidate_count"] == 0.0
    assert scoped.metrics["raw_materialization_candidate_count"] == 0.0


def test_raw_materialization_explicit_scope_includes_already_parsed_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    parsed_raw_id, parsed_size = blob_store.write_from_bytes(b'{"items":[]}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parsed_raw_id,
                "gemini-cli-session",
                "gemini-parsed",
                "/captures/gemini/session.json",
                0,
                bytes.fromhex(parsed_raw_id),
                parsed_size,
                1,
                123,
            ),
        )
        source_conn.commit()

    broad = repair_mod.repair_raw_materialization(config, dry_run=True)
    by_family = repair_mod.repair_raw_materialization(config, dry_run=True, source_family="gemini-cli-session")
    by_root = repair_mod.repair_raw_materialization(config, dry_run=True, source_root=Path("/captures/gemini"))

    assert broad.repaired_count == 1
    assert by_family.repaired_count == 1
    assert "already parsed but not materialized" in by_family.detail
    assert by_root.repaired_count == 1


def test_raw_materialization_scope_filters_count_only_matching_raw_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    claude_raw_id, claude_size = blob_store.write_from_bytes(b'{"parentUuid":null,"sessionId":"claude-a"}')
    codex_raw_id, codex_size = blob_store.write_from_bytes(b'{"items":[]}')
    other_root_raw_id, other_root_size = blob_store.write_from_bytes(b'{"parentUuid":null,"sessionId":"claude-b"}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    claude_raw_id,
                    "claude-code-session",
                    "claude-a",
                    "/captures/claude/a.jsonl",
                    0,
                    bytes.fromhex(claude_raw_id),
                    claude_size,
                    1,
                ),
                (
                    codex_raw_id,
                    "codex-session",
                    "codex-a",
                    "/captures/codex/a.jsonl",
                    0,
                    bytes.fromhex(codex_raw_id),
                    codex_size,
                    2,
                ),
                (
                    other_root_raw_id,
                    "claude-code-session",
                    "claude-b",
                    "/elsewhere/claude/b.jsonl",
                    0,
                    bytes.fromhex(other_root_raw_id),
                    other_root_size,
                    3,
                ),
            ),
        )
        source_conn.commit()

    by_provider = repair_mod.repair_raw_materialization(config, dry_run=True, provider="claude-code")
    by_family = repair_mod.repair_raw_materialization(config, dry_run=True, source_family="codex-session")
    by_root = repair_mod.repair_raw_materialization(config, dry_run=True, source_root=Path("/captures/claude"))

    assert by_provider.repaired_count == 2
    assert by_family.repaired_count == 1
    assert by_root.repaired_count == 1
    assert by_provider.metrics["raw_materialization_candidate_count"] == 2.0
    assert by_provider.metrics["raw_materialization_total_blob_bytes"] == float(claude_size + other_root_size)
    assert by_provider.metrics["raw_materialization_max_blob_bytes"] == float(max(claude_size, other_root_size))


def test_raw_materialization_leaves_fts_to_ingest_or_fts_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, *, db_path: Path) -> None:
            calls["db_path"] = db_path

    class FakeRepository:
        def __init__(self, *, backend: FakeBackend, archive_root: Path) -> None:
            calls["archive_root"] = archive_root

        async def close(self) -> None:
            calls["closed"] = True

    class FakeParseResult:
        processed_ids = {"session-1", "session-2"}
        parse_failures = 0

    class FakeParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, **kwargs: object) -> FakeParseResult:
            calls["parse_kwargs"] = kwargs
            return FakeParseResult()

    def fake_candidate_ids(
        _config: Config,
        *,
        raw_artifact_id: str | None = None,
        provider: str | None = None,
        source_family: str | None = None,
        source_root: Path | None = None,
    ) -> repair_mod.RawMaterializationCandidates:
        calls["raw_artifact_id"] = raw_artifact_id
        calls["provider"] = provider
        calls["source_family"] = source_family
        calls["source_root"] = source_root
        return repair_mod.RawMaterializationCandidates(["raw-1"], 0, 0)

    monkeypatch.setattr(repair_mod, "_raw_materialization_candidate_ids", fake_candidate_ids)
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", FakeParsingService)
    monkeypatch.setattr("polylogue.storage.repository.SessionRepository", FakeRepository)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", FakeBackend)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is True
    assert result.repaired_count == 2
    assert "message FTS left to ingest triggers or the FTS maintenance stage" in result.detail
    assert calls["parse_kwargs"] == {
        "raw_ids": ["raw-1"],
        "progress_callback": None,
        "force_write": False,
        "repair_message_fts": False,
    }
    assert calls["raw_artifact_id"] is None
    assert calls["closed"] is True


def test_raw_materialization_progress_reports_raw_payload_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    progress: list[str] = []

    class FakeBackend:
        def __init__(self, *, db_path: Path) -> None:
            self.db_path = db_path

    class FakeRepository:
        def __init__(self, *, backend: FakeBackend, archive_root: Path) -> None:
            self.backend = backend
            self.archive_root = archive_root

        async def close(self) -> None:
            pass

    class FakeParseResult:
        processed_ids = {"session-1"}
        parse_failures = 0

    class FakeParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, **_kwargs: object) -> FakeParseResult:
            return FakeParseResult()

    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(
            ["raw-1"],
            0,
            0,
            {"raw-1": 256 * 1024 * 1024},
        ),
    )
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", FakeParsingService)
    monkeypatch.setattr("polylogue.storage.repository.SessionRepository", FakeRepository)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", FakeBackend)

    result = repair_mod.repair_raw_materialization(
        config,
        dry_run=False,
        progress_callback=lambda _amount, desc=None: progress.append(desc or ""),
    )

    assert result.success is True
    assert any("raw_materialization: parsing raw 1/1 raw-1 size=256.0 MiB" in line for line in progress)
    assert result.metrics["raw_materialization_total_blob_bytes"] == float(256 * 1024 * 1024)
    assert result.metrics["raw_materialization_max_blob_bytes"] == float(256 * 1024 * 1024)
    assert result.metrics["raw_materialization_session_change_count"] == 1.0
    assert result.metrics["raw_materialization_parse_failure_count"] == 0.0


def test_raw_materialization_blocks_oversized_actual_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class UnexpectedParsingService:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("oversized raw rows should be blocked before parsing")

    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(
            ["raw-1"],
            0,
            0,
            {"raw-1": 2 * 1024 * 1024 * 1024},
        ),
    )
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is False
    assert result.repaired_count == 0
    assert "exceed actual replay limit 1.0 GiB" in result.detail
    assert "largest=2.0 GiB" in result.detail
    assert result.metrics["raw_materialization_oversized_count"] == 1.0
    assert result.metrics["raw_materialization_execute_blob_limit_bytes"] == float(1024 * 1024 * 1024)


def test_raw_materialization_allows_oversized_stream_record_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, *, db_path: Path) -> None:
            calls["db_path"] = db_path

    class FakeRepository:
        def __init__(self, *, backend: FakeBackend, archive_root: Path) -> None:
            calls["archive_root"] = archive_root

        async def close(self) -> None:
            calls["closed"] = True

    class FakeParseResult:
        processed_ids = {"session-1"}
        parse_failures = 0

    class FakeParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, **kwargs: object) -> FakeParseResult:
            calls["parse_kwargs"] = kwargs
            return FakeParseResult()

    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(
            ["raw-1"],
            0,
            0,
            {"raw-1": 2 * 1024 * 1024 * 1024},
            {"raw-1": "claude-code-session"},
            {"raw-1": "/captures/claude/session.jsonl"},
        ),
    )
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", FakeParsingService)
    monkeypatch.setattr("polylogue.storage.repository.SessionRepository", FakeRepository)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", FakeBackend)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert result.metrics["raw_materialization_stream_oversized_count"] == 1.0
    assert "oversized stream-record raw rows used streaming replay" in result.detail
    assert calls["parse_kwargs"] == {
        "raw_ids": ["raw-1"],
        "progress_callback": None,
        "force_write": False,
        "repair_message_fts": False,
    }
    assert calls["closed"] is True


def test_raw_materialization_reports_parse_write_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class FakeBackend:
        def __init__(self, *, db_path: Path) -> None:
            self.db_path = db_path

    class FakeRepository:
        def __init__(self, *, backend: FakeBackend, archive_root: Path) -> None:
            self.backend = backend
            self.archive_root = archive_root

        async def close(self) -> None:
            pass

    class FakeParseResult:
        processed_ids: set[str] = set()
        parse_failures = 1

    class FakeParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, **_kwargs: object) -> FakeParseResult:
            return FakeParseResult()

    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_candidate_ids",
        lambda *_args, **_kwargs: repair_mod.RawMaterializationCandidates(["raw-1"], 0, 1),
    )
    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", FakeParsingService)
    monkeypatch.setattr("polylogue.storage.repository.SessionRepository", FakeRepository)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", FakeBackend)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is False
    assert result.repaired_count == 0
    assert "1 raw rows failed during parse/write" in result.detail
    assert "already parsed but not materialized" in result.detail
    assert result.metrics["raw_materialization_parse_failure_count"] == 1.0
    assert result.metrics["raw_materialization_already_parsed_count"] == 1.0


def _ready_session_insight_status() -> SessionInsightStatusSnapshot:
    return SessionInsightStatusSnapshot(
        profile_rows_ready=True,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        run_rows_ready=True,
        observed_event_rows_ready=True,
        context_snapshot_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
    )


def test_repair_session_insights_noops_when_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    @contextmanager
    def fake_connection_context(_path: Path) -> Iterator[object]:
        yield object()

    def fail_rebuild(*args: object, **kwargs: object) -> int:
        raise AssertionError("ready session insights must not run a full rebuild")

    monkeypatch.setattr("polylogue.storage.sqlite.connection.connection_context", fake_connection_context)
    monkeypatch.setattr(
        "polylogue.storage.insights.session.status.session_insight_status_sync",
        lambda _conn: _ready_session_insight_status(),
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
        fail_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 0
    assert result.detail == "Session insights already ready"


def test_repair_session_insights_dry_run_reports_archive_wide_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeArchive:
        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return SessionInsightStatusSnapshot(
                total_sessions=16_358,
                profile_rows_ready=False,
                latency_profile_rows_ready=True,
                work_event_inference_rows_ready=True,
                work_event_inference_fts_ready=True,
                phase_inference_rows_ready=True,
                threads_ready=True,
                threads_fts_ready=True,
                tag_rollups_ready=True,
                missing_profile_row_count=103,
            )

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=True)

    assert result.success is True
    assert result.repaired_count == 16_358
    assert result.detail == (
        "Would: rebuild archive-wide session insights for 16,358 session(s) to repair 103 debt row(s)"
    )


def test_repair_session_insights_dry_run_reports_scoped_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeArchive:
        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return SessionInsightStatusSnapshot(
                total_sessions=16_358,
                profile_rows_ready=False,
                latency_profile_rows_ready=True,
                work_event_inference_rows_ready=True,
                work_event_inference_fts_ready=True,
                phase_inference_rows_ready=True,
                threads_ready=True,
                threads_fts_ready=True,
                tag_rollups_ready=True,
                missing_profile_row_count=103,
            )

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )

    result = repair_mod.repair_session_insights(
        _config(tmp_path),
        dry_run=True,
        session_ids=("a", "b", "c"),
    )

    assert result.success is True
    assert result.repaired_count == 3
    assert result.detail == "Would: rebuild session insights for 3 scoped session(s)"


def test_repair_session_insights_clears_scoped_convergence_debt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            """
            CREATE TABLE convergence_debt (
                debt_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'failed' CHECK(status IN ('failed', 'deferred')),
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                next_retry_at TEXT,
                materializer_version TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                UNIQUE(stage, target_type, target_id)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO convergence_debt (
                debt_id, stage, target_type, target_id, status, priority,
                attempts, last_error, next_retry_at, materializer_version,
                created_at_ms, updated_at_ms
            )
            VALUES (?, ?, 'session_id', ?, 'deferred', 0, 1, 'quiet window', NULL, NULL, 1, 1)
            """,
            (
                ("debt-1", "insights", "codex-session:target"),
                ("debt-2", "insights", "codex-session:other"),
                ("debt-3", "fts", "codex-session:target"),
            ),
        )

    class FakeArchive:
        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return _ready_session_insight_status()

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.api.archive._rebuild_archive_session_insights",
        lambda _archive, **_kwargs: SessionInsightCounts(profiles=1),
    )

    result = repair_mod.repair_session_insights(
        _config(tmp_path),
        dry_run=False,
        session_ids=("codex-session:target",),
    )

    assert result.success is True
    assert result.repaired_count == 1
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute(
            """
            SELECT stage, target_id
            FROM convergence_debt
            ORDER BY debt_id
            """
        ).fetchall()

    assert rows == [
        ("insights", "codex-session:other"),
        ("fts", "codex-session:target"),
    ]


def test_repair_session_insights_uses_candidate_session_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, sort_key_ms REAL);
        CREATE TABLE session_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            work_event_count INTEGER,
            phase_count INTEGER
        );
        CREATE TABLE session_latency_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL
        );
        CREATE TABLE session_work_events (session_id TEXT);
        CREATE TABLE session_phases (session_id TEXT);
        CREATE TABLE insight_materialization (
            insight_type TEXT,
            session_id TEXT,
            materializer_version INTEGER,
            source_sort_key_ms INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO sessions(session_id, sort_key_ms) VALUES (?, ?)",
        (("ready", 1_000.0), ("missing", 2_000.0)),
    )
    conn.execute(
        """
        INSERT INTO session_profiles(
            session_id, materializer_version, source_sort_key, work_event_count, phase_count
        )
        VALUES ('ready', ?, 1.0, 0, 0)
        """,
        (repair_mod._session_insight_materializer_version(),),
    )
    conn.execute(
        """
        INSERT INTO session_latency_profiles(session_id, materializer_version, source_sort_key)
        VALUES ('ready', ?, 1.0)
        """,
        (repair_mod._session_insight_materializer_version(),),
    )
    conn.executemany(
        """
        INSERT INTO insight_materialization(
            insight_type, session_id, materializer_version, source_sort_key_ms
        ) VALUES (?, 'ready', ?, 1000)
        """,
        (
            ("session_profile", repair_mod._session_insight_materializer_version()),
            ("latency", repair_mod._session_insight_materializer_version()),
            ("work_events", repair_mod._session_insight_materializer_version()),
            ("phases", repair_mod._session_insight_materializer_version()),
            ("thread", repair_mod._session_insight_materializer_version()),
            ("runs", repair_mod._session_insight_materializer_version()),
            ("observed_events", repair_mod._session_insight_materializer_version()),
            ("context_snapshots", repair_mod._session_insight_materializer_version()),
        ),
    )

    calls: list[tuple[str, ...] | None] = []

    class FakeArchive:
        _conn = conn

        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return next(statuses)

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    stale_status = SessionInsightStatusSnapshot(
        total_sessions=2,
        profile_rows_ready=False,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        missing_profile_row_count=1,
    )
    statuses = iter((stale_status, _ready_session_insight_status()))

    def fake_rebuild(_archive: FakeArchive, *, session_ids: tuple[str, ...] | None, **_kwargs: object) -> Any:
        calls.append(session_ids)
        return SessionInsightCounts(profiles=1)

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.api.archive._rebuild_archive_session_insights",
        fake_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert calls == [("missing",)]


def test_repair_session_insights_refreshes_stale_thread_materialization_as_aggregate_debt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, sort_key_ms REAL);
        CREATE TABLE session_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            work_event_count INTEGER,
            phase_count INTEGER
        );
        CREATE TABLE session_latency_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL
        );
        CREATE TABLE session_work_events (session_id TEXT);
        CREATE TABLE session_phases (session_id TEXT);
        CREATE TABLE insight_materialization (
            insight_type TEXT,
            session_id TEXT,
            materializer_version INTEGER,
            source_sort_key_ms INTEGER
        );
        """
    )
    conn.execute("INSERT INTO sessions(session_id, sort_key_ms) VALUES ('stale-thread-marker', 1000)")
    current_version = repair_mod._session_insight_materializer_version()
    conn.execute(
        """
        INSERT INTO session_profiles(
            session_id, materializer_version, source_sort_key, work_event_count, phase_count
        )
        VALUES ('stale-thread-marker', ?, 1.0, 0, 0)
        """,
        (current_version,),
    )
    conn.execute(
        """
        INSERT INTO session_latency_profiles(session_id, materializer_version, source_sort_key)
        VALUES ('stale-thread-marker', ?, 1.0)
        """,
        (current_version,),
    )
    conn.executemany(
        """
        INSERT INTO insight_materialization(
            insight_type, session_id, materializer_version, source_sort_key_ms
        ) VALUES (?, 'stale-thread-marker', ?, 1000)
        """,
        (
            ("session_profile", current_version),
            ("latency", current_version),
            ("work_events", current_version),
            ("phases", current_version),
            ("runs", current_version),
            ("observed_events", current_version),
            ("context_snapshots", current_version),
            ("thread", current_version - 1),
        ),
    )

    calls: list[tuple[str, tuple[str, ...] | None]] = []

    class FakeArchive:
        _conn = conn

        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return next(statuses)

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    stale_status = SessionInsightStatusSnapshot(
        total_sessions=1,
        profile_rows_ready=True,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=False,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        missing_thread_materialization_count=1,
    )
    statuses = iter((stale_status, stale_status, _ready_session_insight_status()))

    def fake_rebuild(_archive: FakeArchive, *, session_ids: tuple[str, ...] | None, **_kwargs: object) -> Any:
        calls.append(("rebuild", session_ids))
        return SessionInsightCounts()

    def fake_aggregate_refresh(_conn: sqlite3.Connection, **_kwargs: object) -> SessionInsightCounts:
        calls.append(("aggregate", None))
        return SessionInsightCounts(threads=1)

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.api.archive._rebuild_archive_session_insights",
        fake_rebuild,
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.refresh_session_insight_aggregates_sync",
        fake_aggregate_refresh,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert calls == [("rebuild", ()), ("aggregate", None)]


def test_repair_assessment_ignores_optional_run_projection_cache_gaps() -> None:
    status = SessionInsightStatusSnapshot(
        total_sessions=1,
        profile_rows_ready=True,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        run_rows_ready=True,
        observed_event_rows_ready=True,
        context_snapshot_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        missing_run_materialization_count=1,
        missing_context_snapshot_materialization_count=1,
    )

    assessment = assess_session_insight_repairs(status)

    assert assessment.row_debt == 0


def test_repair_session_insights_uses_stale_profile_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    class FakeArchive:
        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return next(statuses)

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    stale_status = SessionInsightStatusSnapshot(
        profile_rows_ready=False,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=False,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        stale_profile_row_count=2,
        stale_work_event_inference_count=2,
        work_event_inference_fts_count=4,
        work_event_inference_count=4,
        thread_fts_count=1,
        thread_count=1,
    )
    statuses = iter((stale_status, _ready_session_insight_status()))

    def fake_rebuild(_archive: FakeArchive, *, session_ids: tuple[str, ...] | None, **_kwargs: object) -> Any:
        calls.append(("rebuild", session_ids))
        return SessionInsightCounts(profiles=2, work_events=2)

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.api.archive._rebuild_archive_session_insights",
        fake_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 4
    assert ("rebuild", None) in calls


def test_offline_maintenance_refuses_live_daemon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 1234)

    results = repair_mod.run_selected_maintenance(
        _config(tmp_path),
        repair=True,
        cleanup=False,
        targets=("session_insights",),
    )

    assert len(results) == 1
    assert results[0].name == "session_insights"
    assert results[0].success is False
    assert "polylogued PID 1234 is running" in results[0].detail


def test_offline_maintenance_preview_allowed_with_live_daemon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 1234)

    results = repair_mod.run_selected_maintenance(
        _config(tmp_path),
        repair=True,
        cleanup=False,
        dry_run=True,
        preview_counts={"session_insights": 2},
        targets=("session_insights",),
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].repaired_count == 2


def test_repair_dangling_fts_uses_targeted_missing_row_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class _Cursor:
        def __init__(self, value: object) -> None:
            self.value = value

        def fetchone(self) -> tuple[object, ...]:
            return (self.value,)

    class FakeConn:
        def execute(self, sql: str, params: object = ()) -> _Cursor:
            if sql.startswith("PRAGMA "):
                return _Cursor(None)
            if "sqlite_master" in sql and "messages_fts" in sql:
                return _Cursor("messages_fts")
            raise AssertionError(f"unexpected SQL: {sql}")

        def commit(self) -> None:
            calls.append(("commit", ()))

    @contextmanager
    def fake_connection_context() -> Iterator[FakeConn]:
        yield FakeConn()

    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        conn.execute(
            """
            CREATE TABLE convergence_debt (
                debt_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'failed' CHECK(status IN ('failed', 'deferred')),
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                next_retry_at TEXT,
                materializer_version TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                UNIQUE(stage, target_type, target_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO convergence_debt (
                debt_id, stage, target_type, target_id, status, priority,
                attempts, last_error, next_retry_at, materializer_version,
                created_at_ms, updated_at_ms
            )
            VALUES (
                'debt-1', 'fts', 'fts_surface', 'messages_fts', 'failed', 0,
                1, 'stale ledger', '1970-01-01T00:00:00+00:00', NULL, 1, 1
            )
            """
        )

    monkeypatch.setattr(repair_mod, "_open_archive_index_connection", fake_connection_context)

    def fail_full_rebuild(_conn: FakeConn) -> None:
        raise AssertionError("repair_dangling_fts must not run the full FTS rebuild path")

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
        fail_full_rebuild,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_missing_fts_rows",
        lambda _conn: DanglingFtsRepairOutcome(
            repaired_count=2,
            success=True,
            detail="FTS sync: repaired index",
        ),
    )

    result = repair_mod.repair_dangling_fts(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 2
    assert result.detail == "FTS sync: repaired index"
    assert ("commit", ()) in calls
    with sqlite3.connect(ops_db) as conn:
        row = conn.execute(
            """
            SELECT status, last_error, next_retry_at, updated_at_ms
            FROM convergence_debt
            WHERE stage = 'fts' AND target_type = 'fts_surface' AND target_id = 'messages_fts'
            """
        ).fetchone()
    assert row is None
