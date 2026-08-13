from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.config import Config
from polylogue.core.enums import ArtifactSupportStatus
from polylogue.core.errors import RawCASFrontierError
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.daemon.status import raw_failure_info_for_root
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage import repair as repair_mod
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session.repair_assessment import assess_session_insight_repairs
from polylogue.storage.insights.session.runtime import SessionInsightCounts, SessionInsightStatusSnapshot
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_authority import RawReplayPlan, RawReplayPlanOutcome
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _config(tmp_path: Path) -> Config:
    return Config(archive_root=tmp_path, render_root=tmp_path, sources=[])


def test_raw_materialization_binds_current_generation_under_writer_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion cannot race generation resolution, replay, and postconditions."""
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

    initialize_active_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path, sources=[])
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    class InnerReachedError(RuntimeError):
        pass

    def inspect_inner(*_args: object, **_kwargs: object) -> Any:
        assert config.current_db_path() == active_index
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        raise InnerReachedError

    monkeypatch.setattr(repair_mod, "_repair_raw_materialization", inspect_inner)

    with pytest.raises(InnerReachedError):
        repair_mod.repair_raw_materialization(config)
    with RebuildLease(tmp_path):
        pass


def test_raw_snapshot_cleanup_binds_authority_and_delete_under_writer_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion cannot change the protected generation during destructive raw cleanup."""

    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

    initialize_active_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path, sources=[])

    class InnerReachedError(RuntimeError):
        pass

    def inspect_inner(*_args: object, **_kwargs: object) -> Any:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        raise InnerReachedError

    monkeypatch.setattr(repair_mod, "_repair_superseded_raw_snapshots", inspect_inner)

    with pytest.raises(InnerReachedError):
        repair_mod.repair_superseded_raw_snapshots(config)
    with RebuildLease(tmp_path):
        pass


def test_raw_materialization_reparses_legacy_indexed_raw_before_receipting(tmp_path: Path) -> None:
    """The daemon reopens legacy bytes instead of certifying old durable bindings."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="legacy-indexed-receipt",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="legacy receipt")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id, _session_id = archive.write_raw_and_parsed(
            session,
            payload=(
                b'{"type":"session_meta","payload":{"id":"legacy-indexed-receipt"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
                b'"content":[{"type":"input_text","text":"legacy receipt"}]}}\n'
            ),
            source_path="legacy/codex.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            UPDATE raw_authority_parser_census
            SET detail = 'current parser established durable authority identity'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        conn.commit()

    repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=True)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT parser_fingerprint, status, detail FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()

    assert receipt is not None
    assert receipt[:2] == (RAW_AUTHORITY_PARSER_FINGERPRINT, "complete")
    assert str(receipt[2]).startswith("parser-observed:")


def test_raw_materialization_parser_census_respects_raw_scope(tmp_path: Path) -> None:
    """A one-raw repair census does not scan or receipt unrelated raw evidence."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids: list[str] = []
        for provider_session_id in ("scope-selected", "scope-unselected"):
            session = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text=provider_session_id)],
            )
            raw_id, _session_id = archive.write_raw_and_parsed(
                session,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"{provider_session_id}"}}}}\n'
                    f'{{"type":"response_item","payload":{{"type":"message","id":"m1","role":"user",'
                    f'"content":[{{"type":"input_text","text":"{provider_session_id}"}}]}}}}\n'
                ).encode(),
                source_path=f"scope/{provider_session_id}.jsonl",
                acquired_at_ms=1,
            )
            raw_ids.append(raw_id)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id IN (?, ?)", raw_ids)
        conn.commit()

    repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=True, raw_artifact_id=raw_ids[0])

    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipts = dict(
            conn.execute("SELECT raw_id, detail FROM raw_authority_parser_census WHERE raw_id IN (?, ?)", raw_ids)
        )

    assert str(receipts[raw_ids[0]]).startswith("parser-observed:")
    assert raw_ids[1] not in receipts


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


def _repair_after_persisted_census(
    config: Config,
    *,
    dry_run: bool = False,
    raw_artifact_id: str | None = None,
) -> repair_mod.RepairResult:
    """Exercise replay only after the durable parser census reaches quiescence."""
    _complete_bounded_raw_census(config, limit=1_000)
    return repair_mod.repair_raw_materialization(config, dry_run=dry_run, raw_artifact_id=raw_artifact_id)


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
        "session_tag_rollups": _status(),
    }

    assert repair_mod.session_insight_repair_count(statuses) == 2

    legacy_statuses = dict(statuses)
    legacy_statuses["session_phase_inference"] = legacy_statuses.pop("session_phases")
    assert repair_mod.session_insight_repair_count(legacy_statuses) == 0

    legacy_statuses = dict(statuses)
    legacy_statuses["session_work_event_inference"] = legacy_statuses.pop("session_work_events")
    assert repair_mod.session_insight_repair_count(legacy_statuses) == 0


def test_deleted_orphan_repairs_are_unreachable() -> None:
    """The schema FK/CASCADE guarantee replaces these manual repair paths."""
    from polylogue.maintenance.targets import build_maintenance_target_catalog

    catalog = build_maintenance_target_catalog()
    assert not hasattr(repair_mod, "repair_orphaned_messages")
    assert not hasattr(repair_mod, "repair_orphaned_attachments")
    assert not hasattr(repair_mod, "preview_orphaned_messages")
    assert not hasattr(repair_mod, "preview_orphaned_attachments")
    assert "orphaned_messages" not in repair_mod.REPAIR_HANDLERS
    assert "orphaned_attachments" not in repair_mod.REPAIR_HANDLERS
    assert "orphaned_messages" not in repair_mod.PREVIEW_HANDLERS
    assert "orphaned_attachments" not in repair_mod.PREVIEW_HANDLERS
    assert catalog.resolve_name("orphaned_messages") is None
    assert catalog.resolve_name("orphaned_attachments") is None


def test_session_insights_convergence_matches_repair_archive_route(tmp_path: Path) -> None:
    """The daemon's real archive route repairs the same session rows as repair."""
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.daemon.convergence_stages import make_insights_stage
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    source_path = tmp_path / "codex-session.jsonl"
    source_path.write_bytes(b"real archive source\n")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="convergence-parity",
        title="Convergence parity",
        messages=[
            ParsedMessage(
                provider_message_id="message-1",
                role=Role.USER,
                text="Exercise the real archive route.",
                position=0,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="Exercise the real archive route.",
                    )
                ],
            )
        ],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _raw_id, session_id = archive.write_raw_and_parsed(
            session,
            payload=source_path.read_bytes(),
            source_path=str(source_path),
            acquired_at_ms=1,
        )

    def insight_facts() -> tuple[tuple[object, ...], ...]:
        with sqlite3.connect(tmp_path / "index.db") as conn:
            profile = conn.execute(
                """
                SELECT materializer_version, input_row_count, message_count, word_count
                FROM session_profiles WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            materialization = conn.execute(
                """
                SELECT insight_type, materializer_version, input_row_count
                FROM insight_materialization WHERE session_id = ? ORDER BY insight_type
                """,
                (session_id,),
            ).fetchall()
            work_events = conn.execute(
                """
                SELECT position, work_event_type, summary
                FROM session_work_events WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
            phases = conn.execute(
                """
                SELECT position, start_index, end_index, word_count
                FROM session_phases WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
        return (
            tuple(profile) if profile is not None else (),
            *map(tuple, materialization),
            *map(tuple, work_events),
            *map(tuple, phases),
        )

    manual_result = repair_mod.repair_session_insights(
        _config(tmp_path),
        archive_root_override=tmp_path,
    )
    assert manual_result.success is True
    manual_facts = insight_facts()

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM insight_materialization WHERE session_id = ?", (session_id,))
        conn.commit()

    stage = make_insights_stage(tmp_path / "index.db")
    assert stage.check(source_path) is True
    stage_result = stage.execute(source_path)
    assert getattr(stage_result, "success", stage_result) is True
    assert stage.check(source_path) is False
    assert insight_facts() == manual_facts


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
    monkeypatch.setattr(repair_mod, "count_empty_sessions_sync", lambda _conn: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        repair_mod, "count_unclassified_message_type_sync", lambda _conn: (_ for _ in ()).throw(AssertionError)
    )

    debt = repair_mod.collect_archive_debt_statuses_sync(
        cast(Any, Conn()), derived_statuses=statuses, include_expensive=False, probe_only=True
    )

    assert debt["empty_sessions"].skipped is True
    assert debt["message_type_backfill"].skipped is True


def test_archive_debt_collection_honors_target_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = {
        "session_profile_rows": _status(pending_rows=3),
        "session_work_events": _status(),
        "session_work_events_fts": _status(),
        "session_phases": _status(),
        "threads": _status(),
        "session_tag_rollups": _status(),
    }

    def fail_unrelated(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("target-scoped session_insights preview must not scan unrelated maintenance debt")

    monkeypatch.setattr(repair_mod, "count_empty_sessions_sync", fail_unrelated)
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
    initialize_active_archive_root(tmp_path)
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
    assert result.census_receipt is not None
    assert result.census_receipt.quiescent is False
    assert result.metrics["raw_materialization_census_incomplete_raw_count"] == 1.0
    assert result.metrics["raw_materialization_missing_blob_count"] == 1.0
    assert "persisted parser census" in result.detail


def test_raw_materialization_replays_same_native_when_index_raw_link_is_dangling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
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

    result = _repair_after_persisted_census(config, dry_run=True)

    assert result.success is True
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 1.0


def test_raw_materialization_split_root_routes_authority_replay(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    initialize_active_archive_root(routed_root)
    raw_id, raw_size = BlobStore(routed_root / "blob").write_from_bytes(
        b'{"mapping":{"routed":{"id":"routed","message":{"id":"m1","author":{"role":"user"},'
        b'"content":{"content_type":"text","parts":["hi"]}},"parent":null,"children":[]}},'
        b'"current_node":"routed"}'
    )
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
    result = _repair_after_persisted_census(config)

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


def test_raw_materialization_retries_only_with_deferred_frontier_evidence(tmp_path: Path) -> None:
    """CAS retryability comes from durable evidence, never parse-error prose."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
    payloads = {
        "retryable": b'{"mapping":{}}',
        "cas": b'{"mapping":{"cas":{}}}',
        "membership": b'{"mapping":{"membership":{}}}',
        "stale": b'{"mapping":{"stale":{}}}',
        "sibling": b'{"mapping":{"sibling":{}}}',
    }
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            name: archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"{name}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index, (name, payload) in enumerate(payloads.items())
        }
    sizes = {name: len(payload) for name, payload in payloads.items()}

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            UPDATE raw_sessions SET parsed_at_ms = ?, parse_error = ? WHERE raw_id = ?
            """,
            [
                (3, "changed wording for retryable frontier", raw_ids["retryable"]),
                (4, "another changed wording", raw_ids["cas"]),
                (4, "third changed wording", raw_ids["membership"]),
                (4, "RuntimeError: unrelated parser failure", raw_ids["stale"]),
                (4, "RuntimeError: unrelated parser failure", raw_ids["sibling"]),
            ],
        )
        for name in ("retryable", "cas", "membership"):
            upsert_raw_artifact(
                source_conn,
                raw_ids[name],
                ArchiveSourceArtifact(
                    artifact_id=f"deferred-{name}",
                    origin="codex-session",
                    source_path=f"{name}.jsonl",
                    source_index=0,
                    artifact_kind="deferred_cas_frontier",
                    classification_reason="deferred_cas_frontier",
                    support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                    parse_as_session=True,
                    schema_eligible=True,
                    first_observed_at_ms=100,
                    last_observed_at_ms=100,
                ),
            )
        upsert_raw_artifact(
            source_conn,
            raw_ids["cas"],
            ArchiveSourceArtifact(
                artifact_id="newer-unrelated-cas-artifact",
                origin="codex-session",
                source_path="cas.sqlite",
                source_index=0,
                artifact_kind="sqlite_state_database",
                classification_reason="sqlite_state_database",
                support_status=ArtifactSupportStatus.UNKNOWN,
                first_observed_at_ms=200,
                last_observed_at_ms=200,
            ),
        )
        upsert_raw_artifact(
            source_conn,
            raw_ids["sibling"],
            ArchiveSourceArtifact(
                artifact_id="deferred-on-sibling-coordinate",
                origin="codex-session",
                source_path="sibling-other.jsonl",
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
                first_observed_at_ms=100,
                last_observed_at_ms=100,
            ),
        )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(config)
    assert raw_ids["cas"] in candidates.raw_ids
    assert raw_ids["sibling"] not in candidates.raw_ids

    result = repair_mod.repair_raw_materialization(config, dry_run=True)

    assert result.metrics["raw_materialization_candidate_count"] == 3.0
    assert result.metrics["raw_materialization_total_blob_bytes"] == float(
        sizes["retryable"] + sizes["cas"] + sizes["membership"]
    )


def test_raw_materialization_rejects_contradictory_deferred_evidence(tmp_path: Path) -> None:
    """A deferred kind with terminal support cannot authorize replay."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"mapping":{"contradictory":true}}',
            source_path="contradictory.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("changed wording", raw_id),
        )
        upsert_raw_artifact(
            source_conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="contradictory-deferred-evidence",
                origin="codex-session",
                source_path="contradictory.jsonl",
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(_config(tmp_path))

    assert raw_id not in candidates.raw_ids


def test_raw_materialization_requires_exact_failed_artifact_coordinate(tmp_path: Path) -> None:
    """A deferred neighbor cannot authorize replay for another raw coordinate."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"neighbor":true}',
            source_path="target.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("deferred failure", raw_id),
        )
        for suffix, origin, source_path, source_index in (
            ("origin", "claude-code-session", "target.jsonl", 0),
            ("path", "codex-session", "neighbor.jsonl", 0),
            ("index", "codex-session", "target.jsonl", 1),
        ):
            upsert_raw_artifact(
                source_conn,
                raw_id,
                ArchiveSourceArtifact(
                    artifact_id=f"neighbor-{suffix}",
                    origin=origin,
                    source_path=source_path,
                    source_index=source_index,
                    artifact_kind="deferred_cas_frontier",
                    classification_reason="deferred_cas_frontier",
                    support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                    parse_as_session=True,
                    schema_eligible=True,
                ),
            )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(_config(tmp_path))

    assert raw_id not in candidates.raw_ids


def test_raw_materialization_validation_failure_cannot_reuse_deferred_authority(tmp_path: Path) -> None:
    """Repair and its public backlog report share the worker validation gate."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"deferred":true}',
            source_path="validation-failed.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = NULL, parse_error = ?, validation_status = 'failed' WHERE raw_id = ?",
            ("decode: malformed input", raw_id),
        )
        upsert_raw_artifact(
            source_conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="validation-failed-deferred",
                origin="codex-session",
                source_path="validation-failed.jsonl",
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        source_conn.commit()

    config = _config(tmp_path)
    assert raw_id not in repair_mod._raw_materialization_candidate_ids(config).raw_ids
    backlog = repair_mod.raw_materialization_replay_backlog(config)
    assert backlog["candidate_count"] == 0


def test_raw_materialization_replays_successful_raw_with_historical_validation_failure(tmp_path: Path) -> None:
    """Index reset replays a successful raw while retaining its failed-validation history."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"historical-validation"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
                b'"content":[{"type":"input_text","text":"repair retained validation"}]}}\n'
            ),
            source_path="historical-validation.jsonl",
            acquired_at_ms=1,
        )

    assert repair_mod.repair_raw_materialization(_config(tmp_path)).success is True

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.finalize_raw_parse_state(
            raw_id,
            state=RawSessionStateUpdate(
                parsed_at=None,
                parse_error=None,
                validation_status="failed",
                validation_error="validator rejected an earlier observation",
            ),
        )
        archive.record_raw_failure_evidence(
            raw_id,
            provider=Provider.CODEX,
            source_path="historical-validation.jsonl",
            source_index=0,
            acquired_at_ms=1,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CODEX)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_state = conn.execute(
            "SELECT parsed_at_ms, parse_error, validation_status, validation_error FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        assert raw_state is not None
        assert raw_state[0] is not None
        assert raw_state[1:] == (None, "failed", "validator rejected an earlier observation")
        assert conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("terminal_corrupt_input", "decode_failed")

    # A reset removes only the derived projection; durable raw evidence and
    # its historical validation diagnosis remain available to replay.  Leave
    # the populated conventional index as a stale shadow: the production
    # planner and replay postcondition must use this promoted empty generation.
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")

    replay = repair_mod.repair_raw_materialization(_config(tmp_path))

    assert replay.success is True
    assert replay.repaired_count == 1
    with sqlite3.connect(active_index) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)


def test_raw_materialization_refuses_validation_failure_newer_than_parse(tmp_path: Path) -> None:
    """A later validation failure remains current authority after an earlier parse."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"later-validation-failure"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
                b'"content":[{"type":"input_text","text":"current failure"}]}}\n'
            ),
            source_path="later-validation-failure.jsonl",
            acquired_at_ms=1,
        )

    config = _config(tmp_path)
    assert repair_mod.repair_raw_materialization(config).success is True
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms = int(
            conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        )
        conn.execute(
            """
            UPDATE raw_sessions
            SET validation_status = 'failed', validation_error = ?, validated_at_ms = ?
            WHERE raw_id = ?
            """,
            ("strict validation rejected the later observation", parsed_at_ms + 1, raw_id),
        )
        conn.commit()

    active_index = tmp_path / "generations" / "after-validation" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")

    assert raw_id not in repair_mod._raw_materialization_candidate_ids(config).raw_ids
    assert repair_mod.raw_materialization_replay_backlog(config)["candidate_count"] == 0


@pytest.mark.parametrize("artifact_kind", ["deferred_hot_jsonl_capture", "deferred_claude_code_partial_jsonl"])
def test_raw_materialization_does_not_replay_hot_partial_capture(tmp_path: Path, artifact_kind: str) -> None:
    """Hot partial evidence stays deferred until a complete source observation arrives."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"type":"message_start"}\n',
            source_path="rollout.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("partial JSONL capture", raw_id),
        )
        upsert_raw_artifact(
            source_conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id=f"{artifact_kind}-{raw_id}",
                origin="claude-code-session",
                source_path="rollout.jsonl",
                source_index=0,
                artifact_kind=artifact_kind,
                classification_reason=artifact_kind,
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
            ),
        )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(_config(tmp_path))

    assert raw_id not in candidates.raw_ids


def test_raw_materialization_repairs_deferred_stale_frontier_failure(tmp_path: Path) -> None:
    """Durable frontier evidence reaches the real replay actuator."""
    from polylogue.core.enums import Provider
    from polylogue.storage.raw_retention import raw_frontier_integrity_projection
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"legacy-frontier-repair"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"repair durable frontier evidence"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="legacy-frontier-repair.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            ("RuntimeError: raw revision CAS rejected an older accepted frontier", raw_id),
        )
        source_conn.commit()

    result = repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        assert index_conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (1,)
    frontier = raw_frontier_integrity_projection(
        tmp_path,
        {"available": True, "lost_source_evidence_count": 0, "lost_source_evidence_samples": []},
    )
    assert frontier.overall_status == "healthy"
    assert frontier.broken_head_count == 0


def test_raw_materialization_preserves_bounded_historical_cas_retry_authority(tmp_path: Path) -> None:
    """Historical CAS rows remain selectable, while arbitrary prose stays terminal."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    errors = {
        "prefix": "MembershipReplayConflictError: old guard wording",
        "frontier": "RuntimeError: raw revision CAS rejected an older accepted frontier",
        "byte": "RuntimeError: membership replay cannot replace an unconvertible byte head",
        "unrelated": "RuntimeError: parser failed while decoding a session",
    }
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            name: archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"name":"{name}"}}'.encode(),
                source_path=f"{name}.jsonl",
                acquired_at_ms=index + 1,
            )
            for index, name in enumerate(errors)
        }
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET parsed_at_ms = 2, parse_error = ? WHERE raw_id = ?",
            [(error, raw_ids[name]) for name, error in errors.items()],
        )
        source_conn.commit()

    candidates = repair_mod._raw_materialization_candidate_ids(_config(tmp_path))

    assert set(candidates.raw_ids) == {raw_ids["prefix"], raw_ids["frontier"], raw_ids["byte"]}


def test_raw_materialization_terminal_carrier_overrides_legacy_cas_marker(tmp_path: Path) -> None:
    """A reviewed terminal carrier blocks legacy-marker replay authority."""
    from polylogue.core.enums import Origin, Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"reviewed-terminal"}',
            source_path="reviewed-terminal.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = ?",
            ("MembershipReplayConflictError: historical marker", raw_id),
        )
        upsert_raw_artifact(
            source_conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="reviewed-terminal-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="reviewed-terminal.jsonl",
                source_index=0,
                artifact_kind=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                classification_reason="reviewed terminal disposition",
                support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                parse_as_session=False,
                schema_eligible=False,
                first_observed_at_ms=2,
                last_observed_at_ms=2,
            ),
        )
        source_conn.commit()

    assert repair_mod._raw_materialization_candidate_ids(_config(tmp_path)).raw_ids == []


def test_raw_cas_frontier_error_is_typed_transient() -> None:
    error = RawCASFrontierError("frontier changed")

    assert error.is_transient is True


def test_non_codex_cas_frontier_failure_persists_provider_neutral_evidence(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"type":"session_meta","payload":{"id":"cas-frontier"}}\n',
            source_path="rollout.jsonl",
            acquired_at_ms=1,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CLAUDE_CODE,
            error=RawCASFrontierError("frontier changed"),
        )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute(
            "SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("deferred_cas_frontier", "partial_decode", 1)


def test_generic_parse_state_failure_retires_prior_failure_authority(tmp_path: Path) -> None:
    """An untyped retained-raw failure cannot reuse an earlier replay carrier."""
    from polylogue.core.enums import Provider
    from polylogue.storage.raw.models import RawSessionStateUpdate
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"stale-authority"}}\n',
            source_path="stale-authority.jsonl",
            acquired_at_ms=1,
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CODEX,
            error=RawCASFrontierError("first frontier"),
        )
        archive.finalize_raw_parse_state(
            raw_id,
            state=RawSessionStateUpdate(
                parse_error="ValueError: later parser failure",
                payload_provider=Provider.CODEX,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute(
            "SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == (
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
            0,
        )

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.deferred == 0
    assert lifecycle.unexplained == 1
    assert repair_mod._raw_materialization_candidate_ids(_config(tmp_path)).raw_ids == []


def test_failed_raw_lifecycle_preserves_exact_evidence_for_same_coordinate(
    tmp_path: Path,
) -> None:
    """Two retained failures at one coordinate keep independent replay authority."""
    from polylogue.core.enums import Provider
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        old_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"old"}',
            source_path="same-coordinate.jsonl",
            source_index=0,
            acquired_at_ms=1,
        )
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"new"}',
            source_path="same-coordinate.jsonl",
            source_index=0,
            acquired_at_ms=2,
        )
        archive.mark_raw_parse_failed(
            old_raw_id,
            provider=Provider.CODEX,
            error=RawCASFrontierError("old frontier"),
        )
        archive.mark_raw_parse_failed(
            new_raw_id,
            provider=Provider.CODEX,
            error=RawCASFrontierError("new frontier"),
        )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        rows = source_conn.execute(
            """
            SELECT raw_id, origin, source_path, source_index, artifact_kind, support_status
            FROM raw_artifacts
            WHERE source_path = 'same-coordinate.jsonl'
            ORDER BY raw_id
            """
        ).fetchall()
    assert {tuple(row) for row in rows} == {
        (
            old_raw_id,
            "codex-session",
            "same-coordinate.jsonl",
            0,
            RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
            "partial_decode",
        ),
        (
            new_raw_id,
            "codex-session",
            "same-coordinate.jsonl",
            0,
            RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
            "partial_decode",
        ),
    }

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.deferred == 2
    assert lifecycle.unexplained == 0
    assert {sample["raw_id"] for sample in lifecycle.samples} == {old_raw_id, new_raw_id}
    candidates = repair_mod._raw_materialization_candidate_ids(_config(tmp_path))
    assert set(candidates.raw_ids) == {old_raw_id, new_raw_id}


def test_failed_raw_lifecycle_ignores_newer_ordinary_artifact_at_same_coordinate(
    tmp_path: Path,
) -> None:
    """A newer ordinary observation cannot hide a valid closed failure carrier."""
    from polylogue.core.enums import Origin
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="coexisting.jsonl",
            source_index=4,
            payload=b"unsupported",
            acquired_at_ms=1,
            parse_error="worker rejected shape",
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="failure-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="coexisting.jsonl",
                source_index=4,
                artifact_kind=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                classification_reason=RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
                support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                first_observed_at_ms=10,
                last_observed_at_ms=10,
            ),
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="ordinary-carrier",
                origin=Origin.CODEX_SESSION,
                source_path="coexisting.jsonl",
                source_index=4,
                artifact_kind="session_export",
                classification_reason="ordinary re-observation",
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
                first_observed_at_ms=20,
                last_observed_at_ms=20,
            ),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (2,)

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False
    assert lifecycle.state == "degraded"
    assert lifecycle.samples[0]["artifact_kind"] == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value


def test_cas_failure_evidence_rolls_back_with_parse_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed parse-state write cannot leave a committed CAS evidence receipt."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers import revision_governance
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"revision":"atomic"}',
            source_path="atomic.jsonl",
            acquired_at_ms=1,
        )

        def fail_state_update(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("state update failed")

        monkeypatch.setattr(revision_governance, "apply_source_raw_state_update", fail_state_update)
        with pytest.raises(RuntimeError, match="state update failed"):
            archive.mark_raw_parse_failed(
                raw_id,
                provider=Provider.CODEX,
                error=RawCASFrontierError("frontier"),
            )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            None,
        )
        assert source_conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)


def test_deferred_cas_evidence_is_superseded_after_resolution_and_non_cas_failure(tmp_path: Path) -> None:
    """Resolved deferred evidence cannot authorize a later replay attempt."""
    from polylogue.core.enums import Provider
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_success = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"success"}',
            source_path="success.jsonl",
            acquired_at_ms=1,
        )
        raw_failure = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"name":"failure"}',
            source_path="failure.jsonl",
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        for raw_id, source_path, neighbor_path in (
            (raw_success, "success.jsonl", "success-neighbor.jsonl"),
            (raw_failure, "failure.jsonl", "failure-neighbor.jsonl"),
        ):
            for artifact_id, path in ((f"deferred-{raw_id}", source_path), (f"neighbor-{raw_id}", neighbor_path)):
                upsert_raw_artifact(
                    source_conn,
                    raw_id,
                    ArchiveSourceArtifact(
                        artifact_id=artifact_id,
                        origin="codex-session",
                        source_path=path,
                        source_index=0,
                        artifact_kind="deferred_cas_frontier",
                        classification_reason="deferred_cas_frontier",
                        support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                        parse_as_session=True,
                        schema_eligible=True,
                    ),
                )
        source_conn.commit()

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.mark_raw_parse_succeeded(raw_success, provider=Provider.CODEX)
        archive.mark_raw_parse_failed(
            raw_failure,
            provider=Provider.CODEX,
            error=ValueError("unrelated parser failure"),
        )

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET parsed_at_ms = 3, parse_error = ? WHERE raw_id = ?",
            ("later unrelated parser failure", raw_success),
        )
        source_conn.commit()
        observations = source_conn.execute(
            """
            SELECT raw_id, source_path, artifact_kind, support_status
            FROM raw_artifacts
            WHERE raw_id IN (?, ?)
            ORDER BY raw_id, source_path
            """,
            (raw_success, raw_failure),
        ).fetchall()

    assert {tuple(row) for row in observations if row[1].endswith("neighbor.jsonl")} == {
        (raw_failure, "failure-neighbor.jsonl", "deferred_cas_frontier", "partial_decode"),
        (raw_success, "success-neighbor.jsonl", "deferred_cas_frontier", "partial_decode"),
    }
    assert {(row[0], row[1], row[2], row[3]) for row in observations if not row[1].endswith("neighbor.jsonl")} == {
        (
            raw_failure,
            "failure.jsonl",
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        ),
        (
            raw_success,
            "success.jsonl",
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        ),
    }

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    # The two CAS replacement receipts are bound, non-failure resolutions.
    # The later unrelated parser failures therefore remain unexplained rather
    # than being hidden behind a stale success carrier.
    assert lifecycle.terminal == 0
    assert lifecycle.deferred == 0
    assert lifecycle.unexplained == 2
    status = raw_failure_info_for_root(tmp_path)
    assert status["terminal_rejections"] == 0
    assert status["unexplained_failures"] == 2
    assert repair_mod._raw_materialization_candidate_ids(_config(tmp_path)).raw_ids == []
    assert repair_mod.raw_materialization_replay_backlog(_config(tmp_path))["candidate_count"] == 0


def test_raw_materialization_split_root_classifies_parsed_sidecar_from_routed_blob(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    routed_root = tmp_path / "routed"
    configured_root.mkdir()
    initialize_active_archive_root(routed_root)
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

    result = _repair_after_persisted_census(config, dry_run=True)

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


def test_superseded_raw_cleanup_follows_active_index_pointer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    shadow_index = tmp_path / "index.db"
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(shadow_index, ArchiveTier.INDEX)
    initialize_archive_database(active_index, ArchiveTier.INDEX)
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
    with sqlite3.connect(active_index) as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("native-old", "chatgpt-export", "raw-referenced-old", "old", bytes(32)),
        )
        index_conn.commit()
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")

    result = repair_mod.repair_superseded_raw_snapshots(config, dry_run=True)

    assert result.success is True
    assert result.repaired_count == 0
    assert "skipped 1 active revision raw rows" in result.detail


def test_superseded_raw_cleanup_preserves_explicit_index_override(tmp_path: Path) -> None:
    """An explicit generation remains cleanup authority even when a pointer differs."""

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    pointer_index = tmp_path / "generations" / "pointer" / "index.db"
    explicit_index = tmp_path / "generations" / "explicit" / "index.db"
    initialize_archive_database(pointer_index, ArchiveTier.INDEX)
    initialize_archive_database(explicit_index, ArchiveTier.INDEX)
    source_file = tmp_path / "source.jsonl"
    source_file.write_text("{}", encoding="utf-8")
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'chatgpt-export', ?, ?, 0, ?, ?, ?)
            """,
            (
                ("raw-explicit", "native-explicit", str(source_file), bytes.fromhex("11" * 32), 10, 1),
                ("raw-newer", "native-newer", str(source_file), bytes.fromhex("22" * 32), 11, 2),
            ),
        )
        source_conn.commit()
    with sqlite3.connect(explicit_index) as index_conn:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('native-explicit', 'chatgpt-export', 'raw-explicit', 'explicit', ?)
            """,
            (bytes(32),),
        )
        index_conn.commit()
    (tmp_path / ".index-active-pointer").write_text(f"{pointer_index}\n", encoding="utf-8")
    config = Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=explicit_index)

    result = repair_mod.repair_superseded_raw_snapshots(config, dry_run=True)

    assert result.success is True
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
    initialize_archive_database(tmp_path / "archive.db", ArchiveTier.INDEX)
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
    initialize_active_archive_root(tmp_path)
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

    result = _repair_after_persisted_census(config, dry_run=True)

    assert result.repaired_count == 0
    assert result.success is True
    assert result.metrics["raw_materialization_candidate_count"] == 1.0
    assert result.metrics["raw_materialization_already_parsed_count"] == 1.0
    assert "already parsed but not materialized" in result.detail


def test_raw_materialization_replays_parsed_rows_after_interrupted_index_rebuild(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
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

    result = _repair_after_persisted_census(config, dry_run=True)

    assert result.repaired_count == 0
    assert result.success is True
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


def test_raw_materialization_replays_complete_governed_bundle_membership_after_index_loss(tmp_path: Path) -> None:
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

    assert set(candidates.raw_ids) == {raw_ids[0], raw_ids[2], raw_ids[3]}
    assert candidates.authority_quarantined == 1


def test_raw_materialization_replays_governed_bundle_after_index_reset(tmp_path: Path) -> None:
    """A complete durable census governs replay; it never substitutes for index rows."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    def conversation(session_id: str) -> dict[str, object]:
        return {
            "id": session_id,
            "title": session_id,
            "create_time": 1,
            "update_time": 2,
            "mapping": {
                "message-1": {
                    "id": "message-1",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message-1",
                        "author": {"role": "user"},
                        "create_time": 2,
                        "content": {"content_type": "text", "parts": [f"durable {session_id}"]},
                    },
                }
            },
            "current_node": "message-1",
        }

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation("bundle-one"), conversation("bundle-two")]).encode(),
            source_path="bundle.json",
            acquired_at_ms=1,
        )

    first = repair_mod.repair_raw_materialization(_config(tmp_path))
    assert first.success is True
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "complete",
        )
        assert conn.execute(
            "SELECT status FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete",)

    # Model the normal derived-tier reset: source authority and its already
    # complete parser census survive while every index projection is rebuilt.
    (tmp_path / "index.db").unlink()
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    replay = repair_mod.repair_raw_materialization(_config(tmp_path))

    assert replay.success is True
    assert replay.repaired_count == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (2,)


def test_raw_materialization_reports_uncensused_append_fragments_as_pending_debt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
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
    assert targeted.census_receipt is not None
    assert targeted.census_receipt.quiescent is False
    assert "persisted parser census" in targeted.detail

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            UPDATE raw_membership_census
            SET parser_fingerprint = 'test', status = 'failed', member_count = 0,
                censused_at_ms = 2, detail = 'append fragments are governed by byte revision authority'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        assert source_conn.total_changes == 1
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
    initialize_active_archive_root(tmp_path)
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

    first = _repair_after_persisted_census(config)
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


def test_raw_materialization_no_progress_component_terminalizes_instead_of_looping(tmp_path: Path) -> None:
    """polylogue-hjpx AC1: an accepted plan that executes with zero typed
    progress must not be silently re-selected forever.

    Reproduces the exact production-shaped defect this bead names: a raw is
    classified as a replayable/selected authority component by
    ``repair_raw_materialization``, but its logical cohort has no unique
    byte-proven full baseline (a genuinely orphaned ``append``-kind row with
    no sibling ``full`` row for the same ``logical_source_key``) and no
    membership evidence either -- so ``backfill_historical_revision_evidence``
    runs its full census/replay pipeline without raising, yet returns
    ``replayed_logical_sources=0`` with zero quarantine or adoption-deferral
    too. Before this fix that raw was typed ``RETRYABLE`` forever: identical
    selection, identical zero-progress execution, every pass, with no durable
    signal distinguishing it from a plausible transient retry. It must
    instead terminalize on the first no-progress execution and stop being
    automatically reselected on the next pass.
    """
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
    payload = b"""{
      "id": "orphan-append",
      "title": "orphan append",
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
            "content": {"content_type": "text", "parts": ["orphan"]}
          }
        }
      },
      "current_node": "message-1"
    }"""
    raw_id, blob_size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    logical_source_key = "chatgpt-export:orphan-append"
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, logical_source_key, revision_kind,
                source_revision, revision_authority, acquisition_generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                "orphan-append",
                "orphan-append.json",
                0,
                bytes.fromhex(raw_id),
                blob_size,
                1,
                logical_source_key,
                # An 'append'-kind row with no sibling 'full' row for the same
                # logical_source_key: classify_raw_revision_cohort's full_rows
                # query is empty (no 'full' rows at all), so plan_revision_
                # replay's proven_full list is also empty regardless of this
                # row's own authority -- accepted_raw_ids is (). The fallback
                # to membership governance (convertible_full_revision_raw_ids)
                # then refuses too, since not every row for this key is
                # 'full'. No branch ever touches this raw.
                "append",
                "orphan-append-rev-1",
                "quarantined",
                0,
            ),
        )
        source_conn.commit()

    first = _repair_after_persisted_census(config)
    assert first.success is False
    assert first.repaired_count == 0
    assert first.metrics.get("raw_materialization_no_progress_count") == 1.0
    assert "zero typed progress" in first.detail
    assert first.metrics["raw_materialization_remaining_candidate_count"] == 1.0

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        terminal_rows_after_first = source_conn.execute(
            "SELECT COUNT(*) FROM raw_authority_census_plans WHERE outcome_status = 'terminal'"
        ).fetchone()[0]
    assert terminal_rows_after_first == 1

    second = repair_mod.repair_raw_materialization(config)
    # The stalled plan must not be reselected: no second execution attempt,
    # so the metric that only appears when a component is actually selected
    # for this pass is absent, and the terminal receipt count is unchanged
    # (not doubled by a second no-progress execution of the same plan).
    assert second.metrics.get("raw_materialization_selected_executable_component_count", 0.0) == 0.0
    assert second.metrics.get("raw_materialization_no_progress_plan_count") == 1.0
    # The raw remains honestly visible as unresolved debt, not silently
    # reported as converged (this pass takes the "nothing newly admissible"
    # early-return branch, which reports the base candidate count rather
    # than a post-execution "remaining" count).
    assert second.metrics["raw_materialization_candidate_count"] == 1.0
    assert second.success is False

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        terminal_rows_after_second = source_conn.execute(
            "SELECT COUNT(*) FROM raw_authority_census_plans WHERE outcome_status = 'terminal'"
        ).fetchone()[0]
    # Exactly one terminal receipt total: the plan was not re-executed and
    # re-terminalized a second time.
    assert terminal_rows_after_second == terminal_rows_after_first


def test_raw_materialization_uses_authority_replay_not_legacy_batch_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
    blob_store = BlobStore(tmp_path / "blob")
    first_raw_id, first_size = blob_store.write_from_bytes(
        b'{"mapping":{"first":{"id":"first","message":{"id":"m1","author":{"role":"user"},'
        b'"content":{"content_type":"text","parts":["hi"]}},"parent":null,"children":[]}},'
        b'"current_node":"first"}'
    )
    second_raw_id, second_size = blob_store.write_from_bytes(
        b'{"mapping":{"second":{"id":"second","message":{"id":"m1","author":{"role":"user"},'
        b'"content":{"content_type":"text","parts":["hi"]}},"parent":null,"children":[]}},'
        b'"current_node":"second"}'
    )

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

    result = _repair_after_persisted_census(config)

    assert result.success is True
    assert result.repaired_count == 2
    assert result.metrics["raw_materialization_selected_count"] == 2.0
    assert calls == []


def test_raw_materialization_ordinary_repair_preserves_newer_index_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    initialize_active_archive_root(tmp_path)
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
        message_ids_before = [
            str(message_id)
            for (message_id,) in index_conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        ]
    assert len(fts_hits_before) == 1

    class UnexpectedParsingService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("authority-blocked repair must not construct ParsingService")

    monkeypatch.setattr("polylogue.pipeline.services.parsing.ParsingService", UnexpectedParsingService)

    result = _repair_after_persisted_census(config)

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
    assert message_ids == message_ids_before
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
    assert result.success is True
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
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"execute-{index}"}}}}\n'
                    '{"type":"response_item","payload":{"type":"message","role":"user",'
                    '"content":[{"type":"input_text","text":"hi"}]}}\n'
                ).encode(),
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
    initialize_active_archive_root(tmp_path)
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

    result = _repair_after_persisted_census(config, dry_run=True)

    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_candidate_count"] == 2.0
    assert "1 already parsed but not materialized" in result.detail

    scoped = _repair_after_persisted_census(config, dry_run=True, raw_artifact_id=parsed_raw_id)

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
    initialize_active_archive_root(tmp_path)
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

    _complete_bounded_raw_census(config, limit=1_000)
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
            payload=(
                b'{"type":"session_meta","payload":{"id":"authority-substrate"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"hi"}]}}\n'
            ),
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
            payload=(
                b'{"type":"session_meta","payload":{"id":"progress"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"hi"}]}}\n'
            ),
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
    with sqlite3.connect(tmp_path / "source.db") as conn:
        first_census_count = int(conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0])
        parser_fingerprint = str(
            conn.execute(
                "SELECT parser_fingerprint FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
            ).fetchone()[0]
        )
    repeated = repair_mod.repair_raw_materialization(config, dry_run=False)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        repeated_census_count = int(conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0])

    assert result.success is False
    assert result.repaired_count == 0
    assert "replay candidate(s) remain" in result.detail
    assert result.metrics["raw_materialization_oversized_count"] == 1.0
    assert result.metrics["raw_materialization_resource_blocked_count"] == 1.0
    assert result.metrics["raw_materialization_executed_count"] == 0.0
    assert result.metrics["raw_materialization_execute_blob_limit_bytes"] == float(1024 * 1024 * 1024)
    assert parser_fingerprint.endswith(":resource-blocked:1073741824")
    assert repeated.success is False
    assert len(repeated.plan_outcomes) == 1
    assert repeated.plan_outcomes[0].status.value == "terminal"
    assert repeated_census_count == first_census_count + 1


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

    assert result.success is False
    assert result.repaired_count == 0
    assert result.metrics["raw_materialization_stream_oversized_count"] == 1.0
    assert result.metrics["raw_materialization_resource_blocked_count"] == 1.0
    assert "1 replay candidate(s) remain" in result.detail


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
    assert result.plan_outcomes[0].status.value == "terminal"
    assert repeated.success is False
    assert len(repeated.plan_outcomes) == 1
    assert repeated.plan_outcomes[0].status.value == "terminal"
    assert "aggregate payload exceeds 1.0 GiB" in repeated.detail
    assert "aggregate payload exceeds 1.0 GiB" in result.detail


def test_raw_materialization_reuses_pre_envelope_deferred_receipt(tmp_path: Path) -> None:
    """Upgrading does not strand a completed deferred receipt without envelope identity."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"legacy-deferred"}}\n',
            source_path="legacy-deferred.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES + 1, raw_id),
        )
        conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    first = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert first.census_receipt is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        scope = json.loads(
            str(
                conn.execute(
                    "SELECT scope_json FROM raw_authority_censuses WHERE census_id = ?",
                    (first.census_receipt.census_id,),
                ).fetchone()[0]
            )
        )
        scope.pop("max_payload_bytes")
        conn.execute(
            "UPDATE raw_authority_censuses SET scope_json = ? WHERE census_id = ?",
            (json.dumps(scope, sort_keys=True, separators=(",", ":")), first.census_receipt.census_id),
        )
        conn.commit()

    repeated = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)

    assert repeated.success is True
    assert repeated.census_receipt is not None
    assert repeated.census_receipt.census_id == first.census_receipt.census_id


def test_raw_materialization_reports_the_active_custom_payload_envelope(tmp_path: Path) -> None:
    """A bounded dry run must describe the envelope that governed its plan."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    max_payload_bytes = 100
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"custom-envelope"}}\n',
            source_path="custom-envelope.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (max_payload_bytes + 1, raw_id))
        conn.commit()
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])

    result = repair_mod.repair_raw_materialization(_config(tmp_path), dry_run=True, max_payload_bytes=max_payload_bytes)

    assert result.success is True
    assert result.metrics["raw_materialization_execute_blob_limit_bytes"] == float(max_payload_bytes)
    assert "100 B" in result.detail
    assert "1.0 GiB" not in result.detail


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
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"independent-{index}"}}}}\n'
                    '{"type":"response_item","payload":{"type":"message","role":"user",'
                    '"content":[{"type":"input_text","text":"hi"}]}}\n'
                ).encode(),
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


def test_raw_materialization_max_pass_seconds_bounds_one_pass_and_preserves_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """polylogue-de2a: a per-pass wall-clock budget must stop a single call
    from replaying every selected component, even when ``raw_artifact_limit``
    alone would admit them all in one call -- live evidence showed the
    component count limit did not bound hold time (188s+ holds at a 16-64
    component cap). The budget must be checked only *between* components (so
    at least one always completes, guaranteeing forward progress) and must
    leave the rest as ordinary candidates for the next call, not lost or
    corrupted work.
    """
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    raw_count = 3
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = [
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"budget-{index}",'
                    '"timestamp":"2026-07-11T00:00:00Z"}}}}\n'
                    '{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
                    f'[{{"type":"input_text","text":"budget {index}"}}]}}}}\n'
                ).encode(),
                source_path=f"budget-{index}.jsonl",
                acquired_at_ms=index,
            )
            for index in range(raw_count)
        ]
    assert raw_ids

    config = _config(tmp_path)
    _complete_bounded_raw_census(config, limit=raw_count)

    # A monotonic clock that always advances well past any small budget after
    # its very first read -- deterministic regardless of exactly how many
    # ``time.monotonic()`` calls happen elsewhere in the pass, since the
    # first call establishes the pass start and every later call already
    # clears a 1-second budget.
    elapsed = iter(float(step) * 100.0 for step in range(1000))
    monkeypatch.setattr(time, "monotonic", lambda: next(elapsed))

    bounded = repair_mod.repair_raw_materialization(
        config,
        raw_artifact_limit=raw_count,
        max_pass_seconds=1.0,
    )
    assert bounded.repaired_count == 1
    assert bounded.metrics["raw_materialization_executed_count"] == 1.0
    assert bounded.metrics["raw_materialization_time_budget_exceeded"] == 1.0
    assert bounded.metrics["raw_materialization_remaining_candidate_count"] == float(raw_count - 1)
    assert bounded.success is False

    monkeypatch.undo()
    remainder = repair_mod.repair_raw_materialization(config, raw_artifact_limit=raw_count)
    assert remainder.repaired_count == raw_count - 1
    assert remainder.success is True


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
                    '{"type":"response_item","payload":{"type":"message","role":"user",'
                    '"content":[{"type":"input_text","text":"hi"}]}}\n'
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
    assert first.plan_outcomes[0].status.value == "terminal"
    (tmp_path / "ops.db").unlink()

    second = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert second.repaired_count == 1
    assert second.plan_outcomes[0].input_raw_ids == (raw_ids[1],)


def test_raw_materialization_fair_rotation_mutation_recreates_starvation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing durable attempt-age ordering lets one retryable component monopolize a slot."""
    from polylogue.core.enums import Provider
    from polylogue.sources import revision_backfill
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    def run(*, remove_fair_rotation: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
        root = tmp_path / ("unfair" if remove_fair_rotation else "fair")
        initialize_active_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_ids = [
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=f'{{"type":"session_meta","payload":{{"id":"fair-{index}"}}}}\n'.encode(),
                    source_path=f"fair-{index}.jsonl",
                    acquired_at_ms=index + 1,
                )
                for index in range(3)
            ]

        original_backfill = revision_backfill.backfill_historical_revision_evidence
        _complete_bounded_raw_census(_config(root), limit=1)

        def retry_oldest(*args: Any, selected_raw_ids: list[str] | None = None, **kwargs: Any) -> Any:
            if selected_raw_ids == [raw_ids[0]]:
                raise RuntimeError("injected retryable oldest component")
            return original_backfill(*args, selected_raw_ids=selected_raw_ids, **kwargs)

        with monkeypatch.context() as mutation:
            mutation.setattr(revision_backfill, "backfill_historical_revision_evidence", retry_oldest)
            if remove_fair_rotation:

                def acquisition_only_order(
                    candidates: Any, *, archive_root: Path, index_db_path: Path
                ) -> list[tuple[str, ...]]:
                    del index_db_path
                    return sorted(
                        candidates.authority_components,
                        key=lambda component: min(candidates.raw_acquired_at_ms[raw_id] for raw_id in component),
                    )

                mutation.setattr(repair_mod, "_raw_materialization_ordered_components", acquisition_only_order)
            first = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=1)
            second = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=1)

        assert first.plan_outcomes[0].input_raw_ids == (raw_ids[0],)
        return first.plan_outcomes[0].input_raw_ids, second.plan_outcomes[0].input_raw_ids

    fair_first, fair_second = run(remove_fair_rotation=False)
    unfair_first, unfair_second = run(remove_fair_rotation=True)

    assert fair_first == unfair_first
    assert fair_second != fair_first
    assert unfair_second == unfair_first


def test_raw_materialization_ordering_is_size_agnostic_and_does_not_starve_large_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hjpx AC3: bounded scheduling must pick components by stable
    fairness/age, not by cheapness -- a size-preferring order recreates the
    exact starvation failure mode AC3 names ("repeatedly selecting the same
    cheap components... starving large valid work"), even though every
    component here is independently executable in one pass.
    """
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    def run(*, prefer_cheap: bool) -> tuple[tuple[str, ...], str]:
        root = tmp_path / ("cheap-first" if prefer_cheap else "fair-order")
        initialize_active_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            # The large valid component is acquired FIRST (oldest), so fair
            # age-based ordering must select it on the very first pass.
            large_raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"large-valid"}}\n',
                source_path="large-valid.jsonl",
                acquired_at_ms=1,
            )
            for index in range(5):
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=f'{{"type":"session_meta","payload":{{"id":"cheap-{index}"}}}}\n'.encode(),
                    source_path=f"cheap-{index}.jsonl",
                    acquired_at_ms=index + 2,
                )
        with sqlite3.connect(root / "source.db") as source_conn:
            # blob_size is scheduling metadata only (parsing reads the tiny
            # real payload); this makes the large component "expensive but
            # still executable" (well under the execute limit) without
            # generating megabytes of fixture bytes.
            source_conn.execute(
                "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
                (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES // 2, large_raw_id),
            )
            source_conn.commit()

        config = _config(root)
        _complete_bounded_raw_census(config, limit=1)
        with monkeypatch.context() as mutation:
            if prefer_cheap:

                def cheap_first_order(
                    candidates: Any, *, archive_root: Path, index_db_path: Path
                ) -> list[tuple[str, ...]]:
                    del index_db_path
                    candidate_ids = set(candidates.raw_ids)
                    source_components = candidates.authority_components or tuple(
                        (raw_id,) for raw_id in candidates.raw_ids
                    )
                    components = [c for c in source_components if candidate_ids.intersection(c)]
                    return sorted(
                        components,
                        key=lambda component: sum(
                            candidates.expanded_blob_bytes.get(rid, candidates.raw_blob_bytes.get(rid, 0))
                            for rid in component
                        ),
                    )

                mutation.setattr(repair_mod, "_raw_materialization_ordered_components", cheap_first_order)
            result = repair_mod.repair_raw_materialization(config, raw_artifact_limit=1)
        return result.plan_outcomes[0].input_raw_ids, large_raw_id

    fair_selected, fair_large_id = run(prefer_cheap=False)
    cheap_selected, cheap_large_id = run(prefer_cheap=True)

    assert fair_selected == (fair_large_id,)
    assert cheap_selected != (cheap_large_id,)


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
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"session-{index}"}}}}\n'
                    '{"type":"response_item","payload":{"type":"message","role":"user",'
                    '"content":[{"type":"input_text","text":"hi"}]}}\n'
                ).encode(),
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


def test_raw_materialization_replay_scopes_derived_rebuild_to_touched_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-qsagp: replaying ONE authority component must refresh
    FTS/trigram/action_pairs/delegation_facts only for the session(s) that
    component touched -- never the archive-wide ``rebuild_fts_index_sync`` /
    ``rebuild_command_trigram_index_sync`` / ``rebuild_all_action_pairs_sync`` /
    ``rebuild_all_delegation_facts_sync`` quartet ``maintenance/rebuild_index.py``'s
    terminal blue-green pass owns. Proven at fixture scale with N pre-existing,
    fully materialized sessions already holding real ``action_pairs`` rows
    (each carries one Codex ``function_call``/``function_call_output`` pair):
    patching all four archive-wide rebuild functions to raise, then replaying
    exactly one NEW single-session component, must still succeed without
    tripping any of them. Reverting ``bulk_build=False`` back to ``True`` in
    ``repair_raw_materialization`` (the exact regression this guards) makes
    this test fail immediately on the patched raise, not on some indirect
    symptom.
    """
    from polylogue.core.enums import Provider
    from polylogue.storage.fts import fts_lifecycle as fts_lifecycle_mod
    from polylogue.storage.sqlite import action_pairs as action_pairs_mod
    from polylogue.storage.sqlite import delegation_facts as delegation_facts_mod
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    def _tool_call_payload(native_id: str) -> bytes:
        return (
            f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'.encode()
            + b'{"type":"response_item","payload":{"type":"message","role":"user",'
            b'"content":[{"type":"input_text","text":"run a command"}]}}\n'
            b'{"type":"response_item","payload":{"type":"function_call","id":"fc_1",'
            b'"call_id":"call_abc","name":"exec_command","arguments":"{\\"cmd\\": \\"ls\\"}"}}\n'
            b'{"type":"response_item","payload":{"type":"function_call_output",'
            b'"call_id":"call_abc","output":"file1.txt"}}\n'
        )

    initialize_active_archive_root(tmp_path)
    existing_native_ids = [f"existing-{index}" for index in range(8)]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index, native_id in enumerate(existing_native_ids):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_tool_call_payload(native_id),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=index + 1,
            )

    config = _config(tmp_path)
    baseline = repair_mod.repair_raw_materialization(config)
    assert baseline.success is True
    assert baseline.repaired_count == len(existing_native_ids)

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        watched_session_id = index_conn.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE ? ORDER BY session_id LIMIT 1",
            (f"%{existing_native_ids[0]}",),
        ).fetchone()[0]
        pre_action_pair_rowids = {
            row[0]
            for row in index_conn.execute("SELECT rowid FROM action_pairs WHERE session_id = ?", (watched_session_id,))
        }
        pre_delegation_fact_count = index_conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0]
    assert pre_action_pair_rowids, "fixture must actually populate action_pairs for the watched session"

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("archive-wide derived rebuild must not run for a single-component replay")

    monkeypatch.setattr(fts_lifecycle_mod, "rebuild_fts_index_sync", _fail)
    monkeypatch.setattr(fts_lifecycle_mod, "rebuild_command_trigram_index_sync", _fail)
    monkeypatch.setattr(action_pairs_mod, "rebuild_all_action_pairs_sync", _fail)
    monkeypatch.setattr(delegation_facts_mod, "rebuild_all_delegation_facts_sync", _fail)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_tool_call_payload("new-component"),
            source_path="new-component.jsonl",
            acquired_at_ms=1000,
        )

    result = repair_mod.repair_raw_materialization(config)

    assert result.success is True
    assert result.repaired_count == 1

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        post_action_pair_rowids = {
            row[0]
            for row in index_conn.execute("SELECT rowid FROM action_pairs WHERE session_id = ?", (watched_session_id,))
        }
        post_delegation_fact_count = index_conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0]
        new_action_pair_rows = index_conn.execute(
            "SELECT COUNT(*) FROM action_pairs WHERE session_id LIKE '%new-component'"
        ).fetchone()[0]

    # An archive-wide ``DELETE FROM action_pairs; INSERT ...`` rebuild
    # reassigns every row (including the untouched watched session's) a fresh
    # rowid; a session-scoped refresh never touches rows for a session
    # outside this component at all. Same rowids proves this component's
    # replay left the unrelated session's action_pairs rows alone.
    assert post_action_pair_rowids == pre_action_pair_rowids
    assert new_action_pair_rows > 0
    # delegation_facts has no rows to rowid-compare in this fixture (no
    # delegation links), but the count must grow by exactly what the new
    # component's own session-scoped refresh contributes -- zero here --
    # not shrink to zero and be rebuilt, which the patched raise already
    # rules out.
    assert post_delegation_fact_count == pre_delegation_fact_count


def test_raw_materialization_transient_failure_retries_with_same_plan_id_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hjpx AC4: a transient interruption must remain retryable under the
    *same* plan id and later succeed once -- not spawn a fresh plan id, and
    not silently mutate anything before the retry lands.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources import revision_backfill
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"transient-target"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"hi"}]}}\n'
            ),
            source_path="transient-target.jsonl",
            acquired_at_ms=1,
        )

    original = revision_backfill.backfill_historical_revision_evidence
    should_fail = True

    def fail_once(*args: Any, selected_raw_ids: list[str] | None = None, **kwargs: Any) -> Any:
        if should_fail and selected_raw_ids == [raw_id]:
            raise RuntimeError("OperationalError: database is locked")
        return original(*args, selected_raw_ids=selected_raw_ids, **kwargs)

    monkeypatch.setattr(revision_backfill, "backfill_historical_revision_evidence", fail_once)

    config = _config(tmp_path)
    first = repair_mod.repair_raw_materialization(config)
    assert first.plan_outcomes[0].status.value == "retryable"
    assert "database is locked" in first.plan_outcomes[0].reason
    first_plan_id = first.plan_outcomes[0].plan_id

    # Non-mutating: the injected failure must not have left any parse/apply
    # residue behind before the retry runs.
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            None,
        )

    should_fail = False
    second = repair_mod.repair_raw_materialization(config)

    assert second.plan_outcomes[0].status.value == "executed"
    assert second.plan_outcomes[0].plan_id == first_plan_id
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        assert source_conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)


def test_raw_materialization_cas_conflict_outcome_is_typed_durable_and_non_mutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hjpx AC4: a CAS conflict/incomparable-authority rejection from the
    revision-application layer must surface as a typed, durably-recorded,
    non-mutating outcome through the reconciler -- it must not silently
    vanish, apply partially, or lose its plan id.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources import revision_backfill
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"cas-conflict-target"}}\n',
            source_path="cas-conflict-target.jsonl",
            acquired_at_ms=1,
        )

    cas_message = (
        "raw revision CAS rejected a conflicting accepted head: "
        "logical_source_key='codex:cas-conflict-target' existing(session_id='cas-conflict-target', "
        "accepted_raw_id='other-raw') incoming(session_id='cas-conflict-target', accepted_raw_id='" + raw_id + "')"
    )

    def raise_cas_conflict(*args: Any, selected_raw_ids: list[str] | None = None, **kwargs: Any) -> Any:
        assert selected_raw_ids == [raw_id]
        raise RuntimeError(cas_message)

    monkeypatch.setattr(revision_backfill, "backfill_historical_revision_evidence", raise_cas_conflict)
    result = repair_mod.repair_raw_materialization(_config(tmp_path))

    outcome = result.plan_outcomes[0]
    assert outcome.status.value == "retryable"
    assert "CAS rejected a conflicting accepted head" in outcome.reason

    assert result.census_receipt is not None
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        # Durable: the typed outcome is recorded against the exact plan id in
        # the durable source-tier ledger, not only in the in-memory receipt.
        recorded = source_conn.execute(
            """
            SELECT outcome_status, reason
            FROM raw_authority_census_plans
            WHERE census_id = ? AND plan_id = ?
            """,
            (result.census_receipt.census_id, outcome.plan_id),
        ).fetchone()
        assert recorded == ("retryable", outcome.reason)
        # Non-mutating: no parse residue exists for the raw the CAS
        # rejection blocked.
        assert source_conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
            None,
        )
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        # Non-mutating: no application/head state exists in the (rebuildable
        # but still write-through) index tier either.
        assert (
            index_conn.execute("SELECT COUNT(*) FROM raw_revision_applications WHERE raw_id = ?", (raw_id,)).fetchone()[
                0
            ]
            == 0
        )
        assert index_conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone()[0] == 0


def test_raw_materialization_fails_closed_on_plan_conservation_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The success flag must not conceal a mutated before/after plan algebra."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"conservation"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"hi"}]}}\n'
            ),
            source_path="conservation.jsonl",
            acquired_at_ms=1,
        )

    original = repair_mod._raw_replay_conservation_metrics

    def corrupt_outcome_algebra(
        plans: Sequence[RawReplayPlan],
        selected_plan_ids: set[str],
        outcomes: Sequence[RawReplayPlanOutcome],
    ) -> tuple[int, int, int]:
        plan_count, carried_forward, _errors = original(plans, selected_plan_ids, outcomes)
        return plan_count, carried_forward, 1

    monkeypatch.setattr(repair_mod, "_raw_replay_conservation_metrics", corrupt_outcome_algebra)
    result = repair_mod.repair_raw_materialization(_config(tmp_path))

    assert result.repaired_count == 1
    assert result.metrics["raw_materialization_plan_conservation_error_count"] == 1.0
    assert result.success is False


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
                    '{"type":"response_item","payload":{"type":"message","role":"user",'
                    '"content":[{"type":"input_text","text":"hi"}]}}\n'
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
        "polylogue.storage.insights.session.rebuild.rebuild_archive_session_insights",
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
        "polylogue.storage.insights.session.rebuild.rebuild_archive_session_insights",
        fake_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert calls == [("missing",)]


def test_repair_session_insights_targets_stale_thread_materialization(
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
        tag_rollups_ready=True,
        missing_thread_materialization_count=1,
    )
    statuses = iter((stale_status, _ready_session_insight_status()))

    def fake_rebuild(_archive: FakeArchive, *, session_ids: tuple[str, ...] | None, **_kwargs: object) -> Any:
        calls.append(("rebuild", session_ids))
        return SessionInsightCounts(threads=1)

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.rebuild_archive_session_insights",
        fake_rebuild,
    )
    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert calls == [("rebuild", ("stale-thread-marker",))]


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
        tag_rollups_ready=True,
        stale_profile_row_count=2,
        stale_work_event_inference_count=2,
        work_event_inference_fts_count=4,
        work_event_inference_count=4,
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
        "polylogue.storage.insights.session.rebuild.rebuild_archive_session_insights",
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


# polylogue-t93b: the daemon whale pass. A component whose aggregate raw
# payload exceeds the ordinary fast-path envelope must not stay
# permanently resource-blocked when every member is stream-record-safe --
# ``raw_materialization_whale_pass_candidate`` selects it for a dedicated,
# single-component pass at a widened envelope, and the very same
# ``repair_raw_materialization`` entrypoint (now ``raw_artifact_id``-scoped)
# converges it. Non-stream-safe oversized components must never be
# selected and must carry a distinct typed reason instead.


def test_raw_materialization_whale_pass_candidate_selects_stream_safe_blocked_component(tmp_path: Path) -> None:
    from tests.infra.revision_backfill_benchmark import build_revision_chain_corpus

    raw_ids = build_revision_chain_corpus(tmp_path, superseded_count=9, final_payload_bytes=2_000)
    config = _config(tmp_path)

    # Envelope wide enough to admit the whole chain: nothing is oversized,
    # no escalation needed.
    assert (
        repair_mod.raw_materialization_whale_pass_candidate(
            config, ordinary_max_payload_bytes=10_000_000, whale_max_payload_bytes=10_000_000
        )
        is None
    )

    # Ordinary envelope too small for the chain, but the whale envelope is
    # wide enough and every member is stream-safe (Provider.CODEX, .jsonl):
    # the earliest-acquired raw (the fairness seed) is selected.
    seed = repair_mod.raw_materialization_whale_pass_candidate(
        config, ordinary_max_payload_bytes=500, whale_max_payload_bytes=10_000_000
    )
    assert seed == raw_ids[0]

    # Whale envelope also too small: genuinely still blocked, no candidate.
    assert (
        repair_mod.raw_materialization_whale_pass_candidate(
            config, ordinary_max_payload_bytes=500, whale_max_payload_bytes=500
        )
        is None
    )


def test_stream_safe_resolves_non_candidate_component_members_via_expanded_maps() -> None:
    """A component's already-materialized (non-candidate) members are absent from
    raw_origins/raw_source_paths (candidate-only) but present in the expanded
    maps. Stream-safety must resolve them there, not read origin=None and judge a
    fully stream-safe codex component non-safe -- the bug that made the daemon
    whale pass skip the 6.33GB codex witness component (polylogue-t93b)."""
    candidates = repair_mod.RawMaterializationCandidates(
        raw_ids=["cand"],
        missing_blobs=0,
        already_parsed=0,
        raw_origins={"cand": "codex-session"},
        raw_source_paths={"cand": "/c/rollout-2026-a.jsonl"},
        expanded_origins={"cand": "codex-session", "noncand": "codex-session"},
        expanded_source_paths={"cand": "/c/rollout-2026-a.jsonl", "noncand": "/c/rollout-2026-b.jsonl"},
    )
    # Candidate member: stream-safe as before.
    assert repair_mod._raw_materialization_stream_safe(candidates, "cand") is True
    # Non-candidate member present ONLY in the expanded maps must ALSO be
    # judged by its real codex origin -- True, not False from a missing lookup.
    assert repair_mod._raw_materialization_stream_safe(candidates, "noncand") is True


def test_raw_materialization_whale_pass_candidate_excludes_non_stream_safe_component(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    payload = json.dumps({"mapping": {}, "title": "non-stream-safe-whale"}).encode() + b"x" * 2_000
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="chatgpt-export/conversations.json",
            acquired_at_ms=1,
        )
    config = _config(tmp_path)

    # Oversized under the ordinary envelope, within the whale envelope, but
    # ChatGPT export + non-jsonl source path is not stream-record-safe --
    # must never be selected for escalation, at any whale envelope width.
    assert (
        repair_mod.raw_materialization_whale_pass_candidate(
            config, ordinary_max_payload_bytes=500, whale_max_payload_bytes=1_000_000_000
        )
        is None
    )
    del raw_id


def test_raw_materialization_ordinary_pass_census_detail_distinguishes_escalation_eligibility(
    tmp_path: Path,
) -> None:
    """The durable census detail must say which of the two blocked reasons
    applies, for one individually-oversized raw of each kind (Codex .jsonl
    -- stream-safe -- versus ChatGPT .json -- not)."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        stream_safe_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"t93b-stream-safe"}}\n',
            source_path="codex/t93b-stream-safe.jsonl",
            acquired_at_ms=1,
        )
        non_stream_safe_raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps({"mapping": {}, "title": "t93b-non-stream-safe"}).encode(),
            source_path="chatgpt-export/conversations.json",
            acquired_at_ms=2,
        )
    ordinary_limit = 500
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            ((ordinary_limit + 1, raw_id) for raw_id in (stream_safe_raw_id, non_stream_safe_raw_id)),
        )
        conn.commit()

    stream_safe_result = repair_mod.repair_raw_materialization(
        _config(tmp_path), raw_artifact_id=stream_safe_raw_id, max_payload_bytes=ordinary_limit
    )
    non_stream_safe_result = repair_mod.repair_raw_materialization(
        _config(tmp_path), raw_artifact_id=non_stream_safe_raw_id, max_payload_bytes=ordinary_limit
    )

    assert stream_safe_result.success is False
    assert non_stream_safe_result.success is False
    assert stream_safe_result.metrics.get("raw_materialization_stream_oversized_count", 0.0) >= 1.0
    assert non_stream_safe_result.metrics.get("raw_materialization_non_stream_safe_oversized_count", 0.0) >= 1.0

    with sqlite3.connect(tmp_path / "source.db") as conn:
        stream_safe_detail = conn.execute(
            "SELECT detail FROM raw_authority_parser_census WHERE raw_id = ?", (stream_safe_raw_id,)
        ).fetchone()[0]
        non_stream_safe_detail = conn.execute(
            "SELECT detail FROM raw_authority_parser_census WHERE raw_id = ?", (non_stream_safe_raw_id,)
        ).fetchone()[0]

    assert "escalation-eligible: stream-safe" in stream_safe_detail
    assert "escalation-blocked: non-stream-safe" in non_stream_safe_detail


def test_non_stream_safe_envelope_terminal_never_reports_deferred_success(tmp_path: Path) -> None:
    """A durable terminal envelope outcome remains failed on the next real pass."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps({"mapping": {}, "title": "manual-only-envelope"}).encode(),
            source_path="chatgpt-export/manual-only-envelope.json",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = 501 WHERE raw_id = ?", (raw_id,))
        conn.commit()

    first = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_id=raw_id, max_payload_bytes=500)
    second = repair_mod.repair_raw_materialization(_config(tmp_path), raw_artifact_id=raw_id, max_payload_bytes=500)

    assert first.success is False
    # Removing the terminal/deferred distinction from the envelope query makes
    # this second real maintenance pass take the all-deferred success branch.
    assert second.success is False


def test_raw_materialization_whale_pass_converges_blocked_component_to_resolved_head(tmp_path: Path) -> None:
    """The escalation-tier whale pass (raw_artifact_id-scoped, widened
    envelope) must converge a component the ordinary envelope permanently
    resource-blocks -- reusing the exact same convergence entrypoint, not a
    parallel/offline code path.

    Anti-vacuity: removing the ``raw_artifact_id`` escalation scoping (i.e.
    running the archive-wide ordinary pass at ``max_payload_bytes=whale_limit``
    without narrowing to one component -- simulated below by asserting the
    ordinary-envelope call fails first) reproduces the permanently-blocked
    bug this closes.
    """
    from tests.infra.revision_backfill_benchmark import build_revision_chain_corpus

    raw_ids = build_revision_chain_corpus(tmp_path, superseded_count=9, final_payload_bytes=2_000)
    config = _config(tmp_path)
    ordinary_limit = 500
    whale_limit = 10_000_000

    # The ordinary fast-path envelope permanently blocks the whole chain.
    blocked = repair_mod.repair_raw_materialization(
        config, raw_artifact_id=raw_ids[0], max_payload_bytes=ordinary_limit
    )
    assert blocked.success is False

    seed = repair_mod.raw_materialization_whale_pass_candidate(
        config, ordinary_max_payload_bytes=ordinary_limit, whale_max_payload_bytes=whale_limit
    )
    assert seed is not None

    converged = repair_mod.repair_raw_materialization(config, raw_artifact_id=seed, max_payload_bytes=whale_limit)

    assert converged.success is True
    assert converged.repaired_count >= 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        row = conn.execute(
            "SELECT native_id, origin FROM sessions WHERE native_id = ?", ("nh44-chain-session",)
        ).fetchone()
    assert row is not None
    assert row[1] == "codex-session"

    # Fully converged: no longer a whale-pass candidate at any envelope.
    assert (
        repair_mod.raw_materialization_whale_pass_candidate(
            config, ordinary_max_payload_bytes=ordinary_limit, whale_max_payload_bytes=whale_limit
        )
        is None
    )


def test_raw_materialization_whale_pass_commit_batches_bounded(tmp_path: Path) -> None:
    """The whale pass must honor ``commit_batch_size`` -- shrinking the batch
    must shrink writer-hold length within the escalated component (more,
    smaller commit-boundary transactions), not commit the whole hundreds-raw
    chain as one unbounded transaction regardless of the config knob.

    Anti-vacuity: if the whale-pass call path silently dropped
    ``commit_batch_size`` (e.g. always resolving the single-big-batch
    default no matter what the caller passed), ``fine_count`` and
    ``coarse_count`` below would come out equal -- verified by asserting
    they differ and that finer batching produces strictly more explicit
    commits.
    """
    import unittest.mock as mock

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.revision_backfill_benchmark import build_revision_chain_corpus

    whale_limit = 10_000_000

    def _run(root: Path, *, commit_batch_size: int) -> tuple[int, int]:
        raw_ids = build_revision_chain_corpus(root, superseded_count=11, final_payload_bytes=2_000)
        commit_count = 0
        original_commit = ArchiveStore.commit

        def counting_commit(self: ArchiveStore) -> None:
            nonlocal commit_count
            commit_count += 1
            original_commit(self)

        with mock.patch.object(ArchiveStore, "commit", counting_commit):
            result = repair_mod.repair_raw_materialization(
                _config(root),
                raw_artifact_id=raw_ids[0],
                max_payload_bytes=whale_limit,
                commit_batch_size=commit_batch_size,
            )
        assert result.success is True
        return commit_count, len(raw_ids)

    fine_count, raw_count = _run(tmp_path / "fine", commit_batch_size=3)
    coarse_count, _raw_count = _run(tmp_path / "coarse", commit_batch_size=1_000)

    assert coarse_count >= 1
    assert fine_count > coarse_count
    assert fine_count <= raw_count
