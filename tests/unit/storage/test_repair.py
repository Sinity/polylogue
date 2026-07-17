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
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage import repair as repair_mod
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session.repair_assessment import assess_session_insight_repairs
from polylogue.storage.insights.session.runtime import SessionInsightCounts, SessionInsightStatusSnapshot
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _config(tmp_path: Path) -> Config:
    return Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=tmp_path / "archive.db")


def _complete_bounded_raw_census(config: Config, *, limit: int) -> tuple[repair_mod.RepairResult, list[str]]:
    """Advance census-only passes until a quiescent preview can publish plans."""
    incomplete_census_ids: list[str] = []
    for _pass in range(1_000):
        result = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_limit=limit)
        assert result.census_receipt is not None
        if result.census_receipt.quiescent:
            return result, incomplete_census_ids
        assert result.census_receipt.plan_count == 0
        attempted = result.metrics["raw_materialization_census_components_attempted"]
        assert 1.0 <= attempted <= float(limit)
        incomplete_census_ids.append(result.census_receipt.census_id)
    raise AssertionError("bounded raw census did not quiesce")


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


def test_archive_debt_collection_honors_target_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = {
        "session_profile_rows": _status(pending_rows=3),
        "session_work_events": _status(),
        "session_work_events_fts": _status(),
        "session_phases": _status(),
        "threads": _status(),
        "threads_fts": _status(),
        "session_tag_rollups": _status(),
    }

    def fail_unrelated(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("target-scoped session_insights preview must not scan unrelated maintenance debt")

    monkeypatch.setattr(repair_mod, "count_orphaned_messages_sync", fail_unrelated)
    monkeypatch.setattr(repair_mod, "count_empty_sessions_sync", fail_unrelated)
    monkeypatch.setattr(repair_mod, "count_orphaned_attachments_sync", fail_unrelated)
    monkeypatch.setattr(repair_mod, "count_unclassified_message_type_sync", fail_unrelated)
    monkeypatch.setattr(repair_mod, "count_orphaned_blobs_sync", fail_unrelated)
    monkeypatch.setattr(repair_mod, "count_superseded_raw_snapshots_sync", fail_unrelated)

    with sqlite3.connect(":memory:") as conn:
        debt = repair_mod.collect_archive_debt_statuses_sync(
            conn,
            derived_statuses=statuses,
            target_names=("session_insights",),
        )

    assert tuple(debt) == ("session_insights",)
    assert debt["session_insights"].issue_count == 3


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

    assert result.repaired_count == 0
    assert result.success is False
    assert result.metrics == {
        "raw_materialization_candidate_count": 1.0,
        "raw_materialization_selected_count": 1.0,
        "raw_materialization_missing_blob_count": 1.0,
        "raw_materialization_missing_blob_source_available_count": 0.0,
        "raw_materialization_missing_blob_source_missing_count": 1.0,
        "raw_materialization_already_parsed_count": 0.0,
        "raw_materialization_total_blob_bytes": float(replayable_size),
        "raw_materialization_max_blob_bytes": float(replayable_size),
        "raw_materialization_selected_total_blob_bytes": float(replayable_size),
        "raw_materialization_selected_max_blob_bytes": float(replayable_size),
        "raw_materialization_adoption_deferred_count": 0.0,
        "raw_materialization_authority_quarantined_count": 0.0,
        "raw_materialization_byte_authority_fragment_count": 0.0,
        "raw_materialization_byte_authority_pending_count": 0.0,
        "raw_materialization_byte_authority_quarantined_count": 0.0,
        "raw_materialization_before_component_count": 1.0,
        "raw_materialization_selected_executable_component_count": 1.0,
        "raw_materialization_selected_blocked_component_count": 0.0,
        "raw_materialization_census_sequence": 1.0,
        "raw_materialization_census_fixed_point": 0.0,
    }
    assert "per-session revision authority" in result.detail
    assert "selected raw payload bytes total=" in result.detail
    assert "largest=" in result.detail
    assert "1 raw rows remain blocked by missing blobs (1 with source paths missing)" in result.detail


def test_raw_materialization_replays_same_native_when_index_raw_link_is_dangling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    replacement_raw_id, replacement_size = blob_store.write_from_bytes(b'{"mapping":{"replacement":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replacement_raw_id,
                "chatgpt-export",
                "native-dangling",
                "replacement.json",
                0,
                bytes.fromhex(replacement_raw_id),
                replacement_size,
                10,
            ),
        )
        source_conn.commit()

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("native-dangling", "chatgpt-export", "old-missing-raw", "dangling", bytes(32)),
        )
        index_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.success is False
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 1.0


def test_raw_materialization_split_root_routes_authority_replay(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    routed_root.mkdir()
    initialize_archive_database(routed_root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(routed_root / "index.db", ArchiveTier.INDEX)
    raw_id, raw_size = BlobStore(routed_root / "blob").write_from_bytes(b'{"mapping":{"routed":{}}}')
    with sqlite3.connect(routed_root / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                "routed-session",
                "routed.json",
                0,
                bytes.fromhex(raw_id),
                raw_size,
                1,
            ),
        )
        source_conn.commit()
    config = Config(
        archive_root=configured_root,
        render_root=tmp_path / "render",
        sources=[],
        db_path=routed_root / "index.db",
    )

    backlog = repair_mod.raw_materialization_replay_backlog(config)
    result = repair_mod.repair_raw_materialization(config)

    assert backlog["execution_blocked"] is False
    assert backlog["execution_block_reason"] is None
    assert backlog["blocked_candidate_count"] == 0
    assert backlog["candidate_count"] == 1
    assert result.success is True
    assert result.repaired_count == 1
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_selected_count"] == 1.0


def test_raw_materialization_retries_typed_transient_lock_failure(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"lock-retry","timestamp":"2026-07-11T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"survives retry"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="lock-retry.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parse_error = 'OperationalError: database is locked' WHERE raw_id = ?",
            (raw_id,),
        )
        source_conn.commit()

    result = repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == (1, None)


def test_raw_materialization_split_root_classifies_parsed_sidecar_from_routed_blob(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    routed_root.mkdir()
    initialize_archive_database(routed_root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(routed_root / "index.db", ArchiveTier.INDEX)
    raw_id, raw_size = BlobStore(routed_root / "blob").write_from_bytes(b'{"type":"session_meta"}\n')
    with sqlite3.connect(routed_root / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                "metadata-only",
                "rollout.jsonl",
                0,
                bytes.fromhex(raw_id),
                raw_size,
                1,
                2,
            ),
        )
        source_conn.commit()
    config = Config(
        archive_root=configured_root,
        render_root=tmp_path / "render",
        sources=[],
        db_path=routed_root / "index.db",
    )

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.success is True
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 0.0


def test_superseded_raw_cleanup_protects_split_index_referenced_raw_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    source_file = tmp_path / "source.jsonl"
    source_file.write_text("{}", encoding="utf-8")

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "raw-referenced-old",
                    "chatgpt-export",
                    "native-old",
                    str(source_file),
                    0,
                    bytes.fromhex("11" * 32),
                    10,
                    1,
                ),
                (
                    "raw-newer",
                    "chatgpt-export",
                    "native-newer",
                    str(source_file),
                    0,
                    bytes.fromhex("22" * 32),
                    11,
                    2,
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
            ("native-old", "chatgpt-export", "raw-referenced-old", "old", bytes(32)),
        )
        index_conn.commit()

    result = repair_mod.repair_superseded_raw_snapshots(config, dry_run=True)

    assert result.repaired_count == 0
    assert "skipped 1 active revision raw rows" in result.detail


def test_superseded_raw_cleanup_allows_history_before_active_full(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    source_file = tmp_path / "source.jsonl"
    source_file.write_text("{}", encoding="utf-8")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, logical_source_key, revision_kind,
                source_revision, acquisition_generation, revision_authority
            ) VALUES (?, 'codex-session', 'session-1', ?, 0, ?, ?, ?,
                      'codex:session-1', 'full', ?, ?, 'byte_proven')
            """,
            (
                ("raw-old-full", str(source_file), bytes.fromhex("11" * 32), 10, 1, "revision-old", 0),
                ("raw-new-full", str(source_file), bytes.fromhex("22" * 32), 20, 2, "revision-new", 1),
            ),
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
               VALUES ('session-1', 'codex-session', 'raw-new-full', 'session', ?)""",
            (bytes(32),),
        )
        conn.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id,
                accepted_source_revision, accepted_content_hash,
                accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES ('codex:session-1', 'codex-session:session-1', 'raw-new-full',
                      'revision-new', ?, 'byte', 20, 1, NULL, 1)
            """,
            (bytes(32),),
        )
        conn.execute(
            """
            INSERT INTO raw_revision_applications (
                decision_id, raw_id, session_id, logical_source_key,
                source_revision, acquisition_generation, decision,
                accepted_raw_id, accepted_source_revision, accepted_content_hash,
                detail, decided_at_ms
            ) VALUES ('old-superseded', 'raw-old-full', 'codex-session:session-1',
                      'codex:session-1', 'revision-old', 1, 'superseded',
                      'raw-new-full', 'revision-new', ?, 'superseded by accepted full', 1)
            """,
            (bytes(32),),
        )

    result = repair_mod.repair_superseded_raw_snapshots(config, dry_run=True)

    # Anti-vacuity: traversing a full raw's historical cohort would protect
    # raw-old-full and reduce this production repair preview to zero.
    assert result.success is True
    assert result.repaired_count == 1


def test_superseded_raw_cleanup_fails_closed_without_index(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    # This valid but unrelated legacy anchor must never authorize deletion
    # from the split archive_root/source.db file set.
    initialize_archive_database(config.db_path, ArchiveTier.INDEX)
    source_file = tmp_path / "source.jsonl"
    source_file.write_text("{}", encoding="utf-8")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'chatgpt-export', ?, 0, ?, 10, ?)
            """,
            (
                ("raw-old", str(source_file), bytes.fromhex("11" * 32), 1),
                ("raw-new", str(source_file), bytes.fromhex("22" * 32), 2),
            ),
        )

    result = repair_mod.repair_superseded_raw_snapshots(config, dry_run=False)

    # Anti-vacuity: the old fail-open empty-set fallback would delete raw-old.
    assert result.success is False
    assert result.repaired_count == 0
    assert "index tier is unavailable" in result.detail
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)


def test_raw_materialization_retries_restored_missing_blob_parse_errors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    replayable_raw_id, replayable_size = blob_store.write_from_bytes(b'{"mapping":{}}')
    bad_raw_id, bad_size = blob_store.write_from_bytes(b'{"mapping":{"bad":{}}}')

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, parse_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    replayable_raw_id,
                    "chatgpt-export",
                    "native-retry",
                    "retry.json",
                    0,
                    bytes.fromhex(replayable_raw_id),
                    replayable_size,
                    2,
                    3,
                    "decode: [Errno 2] No such file or directory: '/old/blob/path'",
                ),
                (
                    bad_raw_id,
                    "chatgpt-export",
                    "native-bad",
                    "bad.json",
                    0,
                    bytes.fromhex(bad_raw_id),
                    bad_size,
                    1,
                    4,
                    "parse: malformed provider payload",
                ),
            ],
        )
        source_conn.commit()

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_missing_blob_count"] == 0.0
    assert result.metrics["raw_materialization_total_blob_bytes"] == float(replayable_size)


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

    assert result.repaired_count == 0
    assert result.success is False
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

    assert result.repaired_count == 0
    assert result.success is False
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_already_parsed_count"] == 1.0
    assert "already parsed but not materialized" in result.detail


def test_raw_materialization_receipts_partition_terminal_deferred_and_executable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    decisions = (
        "selected_baseline",
        "applied_append",
        "superseded",
        "ambiguous",
        "deferred",
        None,
    )
    raw_rows: list[tuple[object, ...]] = []
    receipt_rows: list[tuple[object, ...]] = []
    executable_raw_id = ""
    for generation, decision in enumerate(decisions, start=1):
        payload = f'{{"mapping":{{"receipt-{generation}":{{}}}}}}'.encode()
        raw_id, blob_size = blob_store.write_from_bytes(payload)
        raw_rows.append(
            (
                raw_id,
                "chatgpt-export",
                f"receipt-{generation}",
                f"receipt-{generation}.json",
                0,
                bytes.fromhex(raw_id),
                blob_size,
                generation,
            )
        )
        if decision is None:
            executable_raw_id = raw_id
            continue
        detail = "ordinary_replay:incomparable_existing_index_state" if decision == "deferred" else "test:terminal"
        receipt_rows.append(
            (
                f"decision-{generation}",
                raw_id,
                f"chatgpt-export:receipt-{generation}",
                f"logical-{generation}",
                f"revision-{generation}",
                generation,
                decision,
                detail,
                generation,
            )
        )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            raw_rows,
        )
        source_conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.executemany(
            """
            INSERT INTO raw_revision_applications (
                decision_id, raw_id, session_id, logical_source_key,
                source_revision, acquisition_generation, decision, detail,
                decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            receipt_rows,
        )
        index_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(config)

    assert candidates.raw_ids == [executable_raw_id]
    assert candidates.adoption_deferred == 1


def test_raw_materialization_retires_only_complete_governed_bundle_membership(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    raw_ids: list[str] = []
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        for position, decision in enumerate(("applied", "ambiguous", None, "applied"), start=1):
            raw_id, blob_size = blob_store.write_from_bytes(f'{{"bundle":{position}}}'.encode())
            raw_ids.append(raw_id)
            source_conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms
                ) VALUES (?, 'chatgpt-export', ?, 0, ?, ?, ?)
                """,
                (raw_id, f"bundle-{position}.json", bytes.fromhex(raw_id), blob_size, position),
            )
            source_conn.execute(
                """
                INSERT INTO raw_membership_census (
                    raw_id, parser_fingerprint, status, member_count, censused_at_ms
                ) VALUES (?, 'test', 'complete', ?, 1)
                """,
                (raw_id, 2 if position in (1, 4) else 1),
            )
            source_conn.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count, decision, decided_at_ms
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    raw_id,
                    f"bundle:{position}",
                    f"session-{position}",
                    f"revision-{position}",
                    bytes.fromhex(raw_id),
                    decision,
                    1 if decision is not None else None,
                ),
            )
            if position == 4:
                source_conn.execute(
                    """
                    INSERT INTO raw_session_memberships (
                        raw_id, logical_source_key, provider_session_id, source_revision,
                        normalized_content_hash, message_count, decision, decided_at_ms
                    ) VALUES (?, 'bundle:4:second', 'session-4-second', 'revision-4-second', ?, 1,
                              'superseded_equivalent', 1)
                    """,
                    (raw_id, bytes.fromhex(raw_id)),
                )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(config)

    assert set(candidates.raw_ids) == {raw_ids[0], raw_ids[2]}
    assert candidates.authority_quarantined == 1


def test_raw_materialization_reports_uncensused_append_fragments_as_pending_debt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_store = BlobStore(tmp_path / "blob")
    raw_id, blob_size = blob_store.write_from_bytes(b'{"fragment":true}')
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', 'session.jsonl', -1, ?, ?, 1)
            """,
            (raw_id, bytes.fromhex(raw_id), blob_size),
        )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(config)
    backlog = repair_mod.raw_materialization_replay_backlog(config)
    targeted = repair_mod.repair_raw_materialization(config, raw_artifact_id=raw_id)

    assert candidates.raw_ids == []
    assert candidates.byte_authority_pending == 1
    assert backlog["candidate_count"] == 0
    assert backlog["execution_blocked"] is True
    assert backlog["durable_authority_debt_count"] == 1
    assert backlog["byte_authority_pending_count"] == 1
    assert targeted.success is False
    assert "pending byte-authority adjudication" in targeted.detail

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, 'test', 'failed', 0, 2,
                      'append fragments are governed by byte revision authority')
            """,
            (raw_id,),
        )
        source_conn.commit()

    governed = repair_mod._raw_materialization_candidate_ids(config)
    governed_target = repair_mod.repair_raw_materialization(config, raw_artifact_id=raw_id)

    assert governed.byte_authority_pending == 0
    assert governed.byte_authority_quarantined == 1
    assert governed_target.success is False

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET revision_authority = 'byte_proven' WHERE raw_id = ?",
            (raw_id,),
        )
        source_conn.commit()

    proven = repair_mod._raw_materialization_candidate_ids(config)
    assert proven.byte_authority_quarantined == 0
    assert proven.byte_authority_fragments == 1


def test_raw_materialization_ordinary_replay_reaches_two_call_fixed_point(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    payload = b"""{
      "id": "fixed-point",
      "title": "fixed point",
      "create_time": 1,
      "update_time": 2,
      "mapping": {
        "message-1": {
          "id": "message-1",
          "parent": null,
          "children": [],
          "message": {
            "id": "message-1",
            "author": {"role": "user"},
            "create_time": 2,
            "content": {"content_type": "text", "parts": ["fixed"]}
          }
        }
      },
      "current_node": "message-1"
    }"""
    raw_id, blob_size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                "fixed-point",
                "fixed-point.json",
                0,
                bytes.fromhex(raw_id),
                blob_size,
                1,
            ),
        )
        source_conn.commit()

    first = repair_mod.repair_raw_materialization(config)
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        receipts_after_first = index_conn.execute(
            "SELECT decision_id, raw_id, decision FROM raw_revision_applications ORDER BY decision_id"
        ).fetchall()
    second = repair_mod.repair_raw_materialization(config)
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        receipts_after_second = index_conn.execute(
            "SELECT decision_id, raw_id, decision FROM raw_revision_applications ORDER BY decision_id"
        ).fetchall()

    assert first.success is True
    assert first.repaired_count == 1
    assert first.metrics["raw_materialization_remaining_candidate_count"] == 0.0
    assert second.success is True
    assert second.repaired_count == 0
    assert second.metrics["raw_materialization_candidate_count"] == 0.0
    assert receipts_after_second == receipts_after_first
    assert receipts_after_first


def test_raw_materialization_uses_authority_replay_not_legacy_batch_parser(
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

    calls: list[tuple[list[str], bool | None]] = []

    class FakeParsingService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def parse_from_raw(self, *, raw_ids: list[str], **kwargs: object) -> object:
            calls.append((list(raw_ids), cast(bool | None, kwargs.get("force_write"))))
            return SimpleNamespace(processed_ids=set(raw_ids), parse_failures=0)

    import polylogue.pipeline.services.parsing as parsing_module

    monkeypatch.setattr(parsing_module, "ParsingService", FakeParsingService)

    result = repair_mod.repair_raw_materialization(config)

    assert result.success is True
    assert result.repaired_count == 2
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert calls == []


def test_raw_materialization_ordinary_repair_preserves_newer_index_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    older_payload = b"""{
      "id": "logical-session",
      "title": "older raw snapshot",
      "create_time": 1,
      "update_time": 2,
      "mapping": {
        "old-message": {
          "id": "old-message",
          "parent": null,
          "children": [],
          "message": {
            "id": "old-message",
            "author": {"role": "user"},
            "create_time": 2,
            "content": {"content_type": "text", "parts": ["old content"]}
          }
        }
      },
      "current_node": "old-message"
    }"""
    raw_id, raw_size = BlobStore(tmp_path / "blob").write_from_bytes(older_payload)
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                "logical-session",
                "older.json",
                0,
                bytes.fromhex(raw_id),
                raw_size,
                1,
            ),
        )
        source_conn.commit()
    newer_hash = bytes.fromhex("ab" * 32)
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, message_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("logical-session", "chatgpt-export", "newer-index-raw", "newer indexed state", newer_hash, 1),
        )
        session_id = "chatgpt-export:logical-session"
        index_conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, "newer-message", 0, "user", "message", newer_hash),
        )
        index_conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (f"{session_id}:newer-message", session_id, 0, "text", "newer content"),
        )
        index_conn.commit()
        fts_hits_before = index_conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'newer' ORDER BY rowid"
        ).fetchall()
    assert len(fts_hits_before) == 1

    class UnexpectedParsingService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("authority-blocked repair must not construct ParsingService")

    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is False
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_selected_count"] == 1.0
    assert "typed revision authority" in result.detail
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        row = index_conn.execute(
            "SELECT raw_id, title, content_hash, message_count FROM sessions WHERE native_id = 'logical-session'"
        ).fetchone()
        message_ids = [
            str(message_id)
            for (message_id,) in index_conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position",
                ("chatgpt-export:logical-session",),
            ).fetchall()
        ]
        fts_hits_after = index_conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'newer' ORDER BY rowid"
        ).fetchall()
    assert row == ("newer-index-raw", "newer indexed state", newer_hash, 1)
    assert message_ids == ["chatgpt-export:logical-session:newer-message"]
    assert fts_hits_after == fts_hits_before
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        raw_state = source_conn.execute(
            "SELECT parsed_at_ms, parse_error, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    # The source-only census now completes before replay planning.  It may
    # establish byte-proven source authority, while the incomparable index
    # state still remains untouched and receives a deferred application.
    assert raw_state == (None, None, "byte_proven")
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        deferred = index_conn.execute(
            "SELECT decision, detail FROM raw_revision_applications WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    assert deferred == ("deferred", "ordinary_replay:incomparable_existing_index_state")
    assert result.metrics["raw_materialization_adoption_deferred_count"] == 1.0
    from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["blocked"] == 1
    assert readiness["affected_blocked"] == 1
    readiness_categories = cast(dict[str, int], readiness["category_counts"])
    assert readiness_categories["adoption_deferred"] == 1

    retry = repair_mod.repair_raw_materialization(config, dry_run=False)
    assert retry.success is False
    assert retry.metrics["raw_materialization_candidate_count"] == 0.0
    assert retry.metrics["raw_materialization_adoption_deferred_count"] == 1.0
    assert "remain deferred" in retry.detail


def test_raw_materialization_dry_run_reports_limited_selection(
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    sizes = [512, 1024, 2048, 4096]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"dry-{index}"}}}}\n'.encode(),
                source_path=f"dry-{index}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index in range(4)
        ]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", zip(sizes, raw_ids, strict=True))
        conn.commit()
    config = _config(tmp_path)

    result, incomplete_censuses = _complete_bounded_raw_census(config, limit=2)

    assert len(incomplete_censuses) == 1
    assert result.success is False
    assert result.repaired_count == 0
    assert "Would: classify and replay" in result.detail
    assert result.metrics["raw_materialization_candidate_count"] == 4.0
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert result.metrics["raw_materialization_limit"] == 2.0
    assert result.metrics["raw_materialization_total_blob_bytes"] == 7680.0
    assert result.metrics["raw_materialization_selected_total_blob_bytes"] == 1536.0
    assert result.metrics["raw_materialization_selected_max_blob_bytes"] == 1024.0


def test_raw_materialization_execute_limits_authority_selection(
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"execute-{index}"}}}}\n'.encode(),
                source_path=f"execute-{index}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index in range(4)
        ]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            zip((512, 1024, 2048, 4096), raw_ids, strict=True),
        )
        conn.commit()
    config = _config(tmp_path)

    preview, incomplete_censuses = _complete_bounded_raw_census(config, limit=2)
    result = repair_mod.repair_raw_materialization(config, raw_artifact_limit=2)

    assert len(incomplete_censuses) == 1
    assert len(preview.plan_outcomes) == 2
    assert result.success is False
    assert result.repaired_count == 2
    assert result.metrics["raw_materialization_candidate_count"] == 4.0
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert result.metrics["raw_materialization_executed_count"] == 2.0


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

    assert broad.repaired_count == 0
    assert scoped.repaired_count == 0
    assert broad.metrics["raw_materialization_candidate_count"] == 2.0
    assert scoped.metrics["raw_materialization_candidate_count"] == 1.0


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

    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 2.0
    assert "1 already parsed but not materialized" in result.detail

    scoped = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_id=parsed_raw_id)

    assert scoped.repaired_count == 0
    assert scoped.metrics["raw_materialization_candidate_count"] == 1.0
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

    assert broad.repaired_count == 0
    assert by_family.repaired_count == 0
    assert "already parsed but not materialized" in by_family.detail
    assert by_root.repaired_count == 0
    assert by_family.metrics["raw_materialization_candidate_count"] == 1.0
    assert by_root.metrics["raw_materialization_candidate_count"] == 1.0


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

    assert by_provider.repaired_count == 0
    assert by_family.repaired_count == 0
    assert by_root.repaired_count == 0
    assert by_provider.metrics["raw_materialization_candidate_count"] == 2.0
    assert by_provider.metrics["raw_materialization_total_blob_bytes"] == float(claude_size + other_root_size)
    assert by_provider.metrics["raw_materialization_max_blob_bytes"] == float(max(claude_size, other_root_size))


def test_raw_materialization_uses_authority_substrate_not_legacy_ingest_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"authority-substrate"}}\n',
            source_path="authority-substrate.jsonl",
            acquired_at_ms=1,
        )
    config = _config(tmp_path)

    class UnexpectedParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("raw authority repair must not construct the legacy ParsingService")

    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert "typed revision authority" in result.detail


def test_raw_materialization_reports_authority_progress_and_payload_size(
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"progress"}}\n',
            source_path="progress.jsonl",
            acquired_at_ms=1,
        )
    declared_size = 256 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (declared_size, raw_id))
        conn.commit()
    config = _config(tmp_path)
    progress: list[str] = []

    result = repair_mod.repair_raw_materialization(
        config,
        dry_run=False,
        progress_callback=lambda _amount, desc=None: progress.append(desc or ""),
    )

    assert result.success is True
    assert len(progress) == 1
    assert "typed revision authority" in progress[0]
    assert result.metrics["raw_materialization_total_blob_bytes"] == float(declared_size)
    assert result.metrics["raw_materialization_max_blob_bytes"] == float(declared_size)
    assert result.metrics["raw_materialization_selected_count"] == 1.0


def test_raw_materialization_blocks_oversized_actual_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"{}",
            source_path="oversized.json",
            acquired_at_ms=1,
        )
    oversized = 2 * 1024 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (oversized, raw_id))
        conn.commit()
    config = _config(tmp_path)

    class UnexpectedParsingService:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("oversized raw rows should be blocked before parsing")

    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is False
    assert result.repaired_count == 0
    assert "planning paused" in result.detail
    assert result.metrics["raw_materialization_oversized_count"] == 1.0
    assert result.metrics["raw_materialization_resource_blocked_count"] == 1.0
    assert result.metrics["raw_materialization_executed_count"] == 0.0
    assert result.metrics["raw_materialization_execute_blob_limit_bytes"] == float(1024 * 1024 * 1024)


def test_raw_materialization_classifies_oversized_stream_record_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"oversized-stream"}}\n',
            source_path="/captures/codex/session.jsonl",
            acquired_at_ms=1,
        )
    oversized = 2 * 1024 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (oversized, raw_id))
        conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: pytest.fail("stream-safe oversized replay must not eagerly read a blob"),
    )

    result = repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert result.metrics["raw_materialization_stream_oversized_count"] == 1.0
    assert result.metrics.get("raw_materialization_resource_blocked_count", 0.0) == 0.0


def test_raw_materialization_blocks_oversized_expanded_cohort_before_blob_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"expanded-size","timestamp":"2026-07-11T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","id":"two","role":"assistant","content":'
        b'[{"type":"output_text","text":"two"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        small_raw = archive.write_raw_payload(
            provider=Provider.CODEX, payload=baseline, source_path="expanded.json", acquired_at_ms=1
        )
        oversized_raw = archive.write_raw_payload(
            provider=Provider.CODEX, payload=newest, source_path="expanded.json", acquired_at_ms=2
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES + 1, oversized_raw),
        )
        source_conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[small_raw, oversized_raw])

    monkeypatch.setattr(
        "polylogue.sources.revision_backfill._parse_retained_raw",
        lambda *_args, **_kwargs: pytest.fail("expanded cohort size must be checked before opening any blob"),
    )
    result = repair_mod.repair_raw_materialization(
        _config(tmp_path),
        raw_artifact_id=small_raw,
        dry_run=False,
    )

    assert result.success is False
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_resource_blocked_count"] == 2.0
    assert "authority components" in result.detail


def test_raw_materialization_backlog_expands_to_oversized_materialized_sibling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    small_payload = b'{"type":"session_meta","payload":{"id":"small-gap"}}\n'
    large_payload = b'{"type":"session_meta","payload":{"id":"large-done"}}\n'
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        small_raw = archive.write_raw_payload(
            provider=Provider.CODEX, payload=small_payload, source_path="shared.json", acquired_at_ms=1
        )
        large_raw = archive.write_raw_payload(
            provider=Provider.CODEX, payload=large_payload, source_path="shared.json", acquired_at_ms=2
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES + 1, large_raw),
        )
        source_conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        index_conn.execute(
            "INSERT INTO sessions(native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("large-done", "codex-session", large_raw, "done", bytes(32)),
        )
        index_conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[small_raw, large_raw])

    backlog = repair_mod.raw_materialization_replay_backlog(_config(tmp_path))
    assert backlog["candidate_count"] == 1
    assert backlog["expanded_candidate_count"] == 2
    assert backlog["execution_blocked"] is True
    assert backlog["blocked_candidate_count"] == 2

    monkeypatch.setattr(
        "polylogue.sources.revision_backfill._parse_retained_raw",
        lambda *_args, **_kwargs: pytest.fail("oversized materialized sibling must block before blob open"),
    )
    result = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_id=small_raw)
    assert result.success is False
    assert "authority components" in result.detail


def test_raw_materialization_blocks_aggregate_sub_limit_cohort_before_blob_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"aggregate-{index}"}}}}\n'.encode(),
                source_path="aggregate.json",
                acquired_at_ms=index,
            )
            for index in range(2)
        ]
    per_raw_size = 600 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            ((per_raw_size, raw_id) for raw_id in raw_ids),
        )
        source_conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=raw_ids)

    backlog = repair_mod.raw_materialization_replay_backlog(_config(tmp_path))
    assert backlog["oversized_count"] == 0
    assert backlog["expanded_aggregate_blocked"] is True
    assert backlog["execution_blocked"] is True

    monkeypatch.setattr(
        "polylogue.sources.revision_backfill._parse_retained_raw",
        lambda *_args, **_kwargs: pytest.fail("aggregate cohort limit must be checked before blob open"),
    )
    result = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    repeated = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert result.success is False
    assert result.metrics["raw_materialization_resource_blocked_count"] == 2.0
    assert len(result.plan_outcomes) == 1
    assert result.plan_outcomes[0].status.value == "retryable"
    assert result.plan_outcomes[0].plan_id == repeated.plan_outcomes[0].plan_id
    assert "aggregate payload exceeds 1.0 GiB" in result.detail


def test_raw_materialization_processes_independent_components_across_bounded_passes(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    raw_count = 25
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"independent-{index}"}}}}\n'.encode(),
                source_path=f"independent-{index}.jsonl",
                acquired_at_ms=index,
            )
            for index in range(raw_count)
        ]
    per_raw_size = 50 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            ((per_raw_size, raw_id) for raw_id in raw_ids),
        )
        source_conn.commit()

    config = _config(tmp_path)
    backlog = repair_mod.raw_materialization_replay_backlog(config)
    assert backlog["candidate_count"] == raw_count
    assert backlog["authority_component_count"] == raw_count
    assert (
        int(cast(int, backlog["expanded_total_blob_bytes"])) > repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    )
    assert backlog["execution_blocked"] is False
    assert backlog["executable_authority_component_count"] == raw_count

    preview, incomplete_censuses = _complete_bounded_raw_census(config, limit=5)
    assert len(incomplete_censuses) == 4
    assert len(preview.plan_outcomes) == 5
    repaired_per_pass: list[int] = []
    for _pass in range(5):
        result = repair_mod.repair_raw_materialization(config, raw_artifact_limit=5)
        repaired_per_pass.append(result.repaired_count)
    assert repaired_per_pass == [5, 5, 5, 5, 5]
    assert repair_mod.repair_raw_materialization(config, raw_artifact_limit=5).success is True


def test_raw_materialization_durable_ledger_survives_ops_reset_for_fairness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retryable oldest component must not monopolize a slot after ops reset."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"session-{index}",'
                    '"timestamp":"2026-07-15T00:00:00Z"}}}}\n'
                ).encode(),
                source_path=f"session-{index}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index in range(3)
        ]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES + 1, raw_ids[0]),
        )
        conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=raw_ids)

    original_stream_safe = repair_mod._raw_materialization_stream_safe
    monkeypatch.setattr(
        repair_mod,
        "_raw_materialization_stream_safe",
        lambda candidates, raw_id: raw_id != raw_ids[0] and original_stream_safe(candidates, raw_id),
    )

    first = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert first.plan_outcomes[0].status.value == "retryable"
    (tmp_path / "ops.db").unlink()

    second = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert second.repaired_count == 1
    assert second.plan_outcomes[0].input_raw_ids == (raw_ids[1],)


def test_raw_materialization_isolates_failed_component_and_continues_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One runtime failure must produce a receipt without starving peers."""
    from polylogue.core.enums import Provider
    from polylogue.sources import revision_backfill
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"session-{index}"}}}}\n'.encode(),
                source_path=f"session-{index}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index in range(3)
        ]

    original = revision_backfill.backfill_historical_revision_evidence

    def fail_oldest(*args: Any, selected_raw_ids: list[str] | None = None, **kwargs: Any) -> Any:
        if selected_raw_ids == [raw_ids[0]]:
            raise RuntimeError("injected component failure")
        return original(*args, selected_raw_ids=selected_raw_ids, **kwargs)

    monkeypatch.setattr(revision_backfill, "backfill_historical_revision_evidence", fail_oldest)
    result = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=3)

    assert result.repaired_count == 2
    assert [outcome.status.value for outcome in result.plan_outcomes].count("retryable") == 1
    assert [outcome.status.value for outcome in result.plan_outcomes].count("executed") == 2


def test_raw_materialization_batch_limit_counts_authority_components(tmp_path: Path) -> None:
    """One revision-heavy source must not consume the whole daemon batch."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    session_meta = b'{"type":"session_meta","payload":{"id":"shared-session","timestamp":"2026-07-15T00:00:00Z"}}\n'
    shared_raw_ids: list[str] = []
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for revision in range(5):
            messages = b"".join(
                (
                    b'{"type":"response_item","payload":{"type":"message",'
                    b'"role":"user","content":[{"type":"input_text","text":"revision-'
                    + str(index).encode()
                    + b'"}]}}\n'
                )
                for index in range(revision + 1)
            )
            shared_raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=session_meta + messages,
                    source_path="shared-session.jsonl",
                    acquired_at_ms=revision + 1,
                )
            )
        independent_raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"independent-{index}",'
                    '"timestamp":"2026-07-15T00:00:00Z"}}}}\n'
                ).encode(),
                source_path=f"independent-{index}.jsonl",
                acquired_at_ms=100 + index,
            )
            for index in range(4)
        ]

    # The previous scheduler sorted individual raws by size before applying
    # the batch limit. Force that old ordering deterministically: the fixed
    # scheduler must still select three complete, oldest-first components.
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            [
                *((index + 1, raw_id) for index, raw_id in enumerate(shared_raw_ids)),
                *((100 + index, raw_id) for index, raw_id in enumerate(independent_raw_ids)),
            ],
        )
        source_conn.commit()

    config = _config(tmp_path)
    before = repair_mod.raw_materialization_replay_backlog(config)
    assert before["candidate_count"] == 9
    assert before["authority_component_count"] == 5

    preview, incomplete_censuses = _complete_bounded_raw_census(config, limit=3)
    first = repair_mod.repair_raw_materialization(config, raw_artifact_limit=3)
    after = repair_mod.raw_materialization_replay_backlog(config)

    # The first bounded attempt discovers the five-revision shared component
    # transitively; the next pass handles the remaining independent components
    # and publishes the complete plan inventory.
    assert len(incomplete_censuses) == 1
    assert first.repaired_count == 3, (first.detail, first.metrics, after)
    assert first.metrics["raw_materialization_selected_component_count"] == 3.0
    assert first.metrics["raw_materialization_plan_outcome_count"] == 5.0
    assert first.metrics["raw_materialization_plan_carried_forward_count"] == 2.0
    assert first.metrics["raw_materialization_plan_executed_count"] == 3.0
    assert {outcome.plan_id for outcome in first.plan_outcomes} == {
        outcome.plan_id for outcome in preview.plan_outcomes
    }
    assert {outcome.status.value for outcome in first.plan_outcomes} == {"executed"}
    assert after["candidate_count"] == 2


def test_raw_materialization_quarantines_parse_failures_without_legacy_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"\xff\n",
            source_path="broken.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = 1 WHERE raw_id = ?", (raw_id,))
        conn.commit()
    config = _config(tmp_path)

    class UnexpectedParsingService:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("parse failures must remain inside the authority census route")

    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)
    monkeypatch.setattr(
        "polylogue.sources.revision_backfill._parse_retained_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic retained-byte decode failure")),
    )

    result = repair_mod.repair_raw_materialization(config, dry_run=False)

    assert result.success is False
    assert result.repaired_count == 0
    assert "parser census completes" in result.detail
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
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, sort_key_ms REAL, updated_at_ms INTEGER);
        CREATE TABLE session_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            source_updated_at TEXT,
            work_event_count INTEGER,
            phase_count INTEGER
        );
        CREATE TABLE session_latency_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            source_updated_at TEXT
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
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, sort_key_ms REAL, updated_at_ms INTEGER);
        CREATE TABLE session_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            source_updated_at TEXT,
            work_event_count INTEGER,
            phase_count INTEGER
        );
        CREATE TABLE session_latency_profiles (
            session_id TEXT PRIMARY KEY,
            materializer_version INTEGER,
            source_sort_key REAL,
            source_updated_at TEXT
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
