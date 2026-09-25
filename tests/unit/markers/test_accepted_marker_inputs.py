"""Accepted source inputs retain effects even when the current index advances."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.config import Config
from polylogue.core.enums import Origin, Provider, Role
from polylogue.markers.preparation import marker_candidates_for_prepared_write, marker_recipe_fingerprint
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.services.ingest_batch import _core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch._core import _marker_request_facts, _write_session_entry
from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sinex.models import PublicationMode, PublicationPayload
from polylogue.sinex.obligations import AsyncSqlConnection, stage_payload_async
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputRefusedError,
    append_accepted_marker_input,
    finalize_pending_accepted_marker_input,
    prepare_accepted_marker_input,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write, write_parsed_session_to_archive
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.durable_change_train import _runtime_consumer_results, validate_durable_migration_sidecars
from tests.infra.archive_templates import bootstrap_archive_root
from tests.unit.sinex.test_ingest_atomicity import _AsyncConnection


def _session(text: str, *, native_id: str = "session", message_id: str = "") -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        created_at="2026-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id=message_id, role=Role.ASSISTANT, text=text)],
    )


def _nested(candidate: Mapping[str, object], section: str, field: str) -> object:
    value = candidate[section]
    assert isinstance(value, Mapping)
    return value[field]


_SqlRows = tuple[tuple[object, ...], ...]


def _index_message_state(path: Path) -> tuple[_SqlRows, _SqlRows, _SqlRows]:
    """Snapshot materialized rows so a refused replay proves rollback."""
    with sqlite3.connect(path) as index:
        sessions = cast(_SqlRows, tuple(index.execute("SELECT session_id FROM sessions ORDER BY session_id")))
        messages = cast(
            _SqlRows,
            tuple(
                index.execute(
                    "SELECT session_id, message_id, content_hash FROM messages ORDER BY session_id, message_id"
                )
            ),
        )
        blocks = cast(_SqlRows, tuple(index.execute("SELECT block_id, search_text FROM blocks ORDER BY block_id")))
    return sessions, messages, blocks


def _accepted_marker_state(path: Path, raw_id: str) -> tuple[object, ...] | None:
    with sqlite3.connect(path) as source:
        return cast(
            tuple[object, ...] | None,
            source.execute(
                "SELECT sequence, payload, index_incarnation_id FROM accepted_marker_inputs WHERE raw_id = ?",
                (raw_id,),
            ).fetchone(),
        )


def test_batch_route_retains_r1_and_r2_empty_and_identical_replay(workspace_env: dict[str, Path]) -> None:
    """Finalization retries consume one pending carrier and keep its sequence."""
    path = workspace_env["archive_root"] / "source.db"
    batch = prepare_accepted_marker_input("r1", [{"session_id": "s", "candidates": []}], request_facts={"blob": "a"})
    with sqlite3.connect(path) as source:
        source.executescript(SOURCE_DDL)
        source.execute("BEGIN IMMEDIATE")
        from polylogue.storage.accepted_marker_inputs import persist_pending_marker_input_sync

        persist_pending_marker_input_sync(source, batch, expected_incarnation_id=str(uuid.uuid4()))
    with sqlite3.connect(path) as source:
        first = asyncio.run(finalize_pending_accepted_marker_input(_AsyncConnection(source), batch))
    with sqlite3.connect(path) as source:
        retry = asyncio.run(finalize_pending_accepted_marker_input(_AsyncConnection(source), batch))
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone()[0] == 1
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone()[0] == 0
    assert first == retry == 1


def test_request_facts_ignore_mutable_detected_provider_projection() -> None:
    raw = RawSessionRecord(
        raw_id="raw-provider-projection",
        source_name="unknown-export",
        source_path="capture.jsonl",
        source_index=0,
        blob_hash="a" * 64,
        blob_size=1,
        acquired_at="2026-01-01T00:00:00Z",
    )
    before = _marker_request_facts(raw, validation_mode="advisory")
    after_parse = _marker_request_facts(
        raw.model_copy(update={"payload_provider": Provider.CODEX}), validation_mode="advisory"
    )

    assert before == after_parse
    assert "payload_provider" not in before


def test_marker_recipe_fingerprint_tracks_parser_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.markers import parser

    before = marker_recipe_fingerprint()

    def changed_parse_markers(text: str, *, registry: object = None) -> tuple[object, ...]:
        del text, registry
        return ()

    monkeypatch.setattr(parser, "parse_markers", changed_parse_markers)
    assert marker_recipe_fingerprint() != before


def test_marker_recipe_fingerprint_tracks_marker_spec_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.markers import parser

    before = marker_recipe_fingerprint()

    def changed_marker_spec(registry: object, kind: str) -> None:
        del registry, kind
        return None

    monkeypatch.setattr(parser, "marker_spec", changed_marker_spec)
    assert marker_recipe_fingerprint() != before


def test_marker_recipe_fingerprint_tracks_registry_get(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.markers.registry import MarkerRegistry

    before = marker_recipe_fingerprint()

    def changed_get(self: object, kind: str) -> None:
        del self, kind
        return None

    monkeypatch.setattr(MarkerRegistry, "get", changed_get)
    assert marker_recipe_fingerprint() != before


def test_marker_recipe_fingerprint_tracks_registry_contains(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.markers.registry import MarkerRegistry

    before = marker_recipe_fingerprint()

    def changed_contains(self: object, kind: str) -> bool:
        del self, kind
        return False

    monkeypatch.setattr(MarkerRegistry, "__contains__", changed_contains)
    assert marker_recipe_fingerprint() != before


@pytest.mark.parametrize("name", ["_LINE", "_INLINE", "_INLINE_OPEN", "_MALFORMED"])
def test_marker_recipe_fingerprint_tracks_each_grammar_constant(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from polylogue.markers import parser

    before = marker_recipe_fingerprint()
    original = getattr(parser, name)
    monkeypatch.setattr(parser, name, re.compile(original.pattern + "|(?!)", original.flags))
    assert marker_recipe_fingerprint() != before


def test_marker_recipe_fingerprint_tracks_prepared_write_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.markers import preparation

    before = marker_recipe_fingerprint()

    def changed_adapter(prepared: object) -> list[dict[str, object]]:
        del prepared
        return []

    monkeypatch.setattr(preparation, "marker_candidates_for_prepared_write", changed_adapter)
    assert marker_recipe_fingerprint() != before


def test_marker_recipe_fingerprint_tracks_block_insert_column_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace as dataclass_replace

    from polylogue.markers import preparation
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC

    before = marker_recipe_fingerprint()
    columns = BLOCKS_SPEC.writable_columns
    altered_first = dataclass_replace(columns[0], name=f"{columns[0].name}_changed")
    monkeypatch.setattr(
        preparation,
        "BLOCKS_SPEC",
        dataclass_replace(BLOCKS_SPEC, writable_columns=(altered_first, *columns[1:])),
    )
    assert marker_recipe_fingerprint() != before


def test_batch_acceptance_conflict_rolls_back_raw_state(workspace_env: dict[str, Path]) -> None:
    """Changed carrier bytes under one request key refuse before index publication."""
    path = workspace_env["archive_root"] / "source.db"
    original = prepare_accepted_marker_input("r1", [{"session_id": "s", "candidates": []}], request_facts={"blob": "a"})
    conflict = prepare_accepted_marker_input(
        "r1", [{"session_id": "s", "candidates": [{"body": "changed"}]}], request_facts={"blob": "a"}
    )
    with sqlite3.connect(path) as source:
        source.executescript(SOURCE_DDL)
        from polylogue.storage.accepted_marker_inputs import persist_pending_marker_input_sync

        source.execute("BEGIN IMMEDIATE")
        persist_pending_marker_input_sync(source, original, expected_incarnation_id=str(uuid.uuid4()))
    with sqlite3.connect(path) as source:
        with pytest.raises(AcceptedMarkerInputRefusedError, match="pending marker request conflicts"):
            source.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(source, conflict, expected_incarnation_id=str(uuid.uuid4()))
        assert source.execute("SELECT carrier_digest FROM pending_accepted_marker_inputs").fetchone() == (
            original.payload_sha256,
        )


def test_conflicting_replay_rolls_back_acceptance_and_earlier_batch(workspace_env: dict[str, Path]) -> None:
    """A conflicting accepted identity must roll back the whole caller transaction."""
    path = workspace_env["archive_root"] / "source.db"
    original = prepare_accepted_marker_input("r1", [{"session_id": "s", "candidates": []}])
    conflict = prepare_accepted_marker_input("r1", [{"session_id": "s", "candidates": [{"body": "changed"}]}])
    with sqlite3.connect(path) as source:
        assert asyncio.run(append_accepted_marker_input(_AsyncConnection(source), original)) == 1
    with sqlite3.connect(path) as source:
        with pytest.raises(AcceptedMarkerInputRefusedError, match="conflicting replay"):
            with source:
                source.execute("CREATE TABLE IF NOT EXISTS acceptance_probe(raw_id TEXT)")
                source.execute("INSERT INTO acceptance_probe VALUES ('r2')")
                asyncio.run(
                    append_accepted_marker_input(_AsyncConnection(source), prepare_accepted_marker_input("r2", []))
                )
                asyncio.run(append_accepted_marker_input(_AsyncConnection(source), conflict))
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone()[0] == 1
        assert source.execute("SELECT COUNT(*) FROM acceptance_probe").fetchone()[0] == 0
        with pytest.raises(AcceptedMarkerInputRefusedError):
            asyncio.run(append_accepted_marker_input(_AsyncConnection(source), replace(original, payload=b"{}")))
        for table in ("accepted_marker_inputs", "accepted_marker_stream"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                source.execute(f"DELETE FROM {table}")


def test_malformed_carrier_session_is_refused_with_typed_error(workspace_env: dict[str, Path]) -> None:
    """Malformed persisted JSON shapes must not leak implementation exceptions."""
    payload = b'{"raw_id":"r1","sessions":[null]}'
    batch = prepare_accepted_marker_input("r1", [{"session_id": "s", "candidates": []}])
    malformed = replace(
        batch,
        payload=payload,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )

    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as source:
        with pytest.raises(AcceptedMarkerInputRefusedError, match="invalid accepted marker carrier"):
            asyncio.run(append_accepted_marker_input(_AsyncConnection(source), malformed))


def test_prepared_fallback_append_and_lineage_coordinates_match_writer(workspace_env: dict[str, Path]) -> None:
    """A synthetic ordinal ID or ignored append occurrence offset makes this red."""
    conn = sqlite3.connect(workspace_env["archive_root"] / "index.db")
    conn.row_factory = sqlite3.Row
    try:
        session = _session("::note: repeated")
        first = prepare_session_write(conn, session, merge_append=False)
        first_candidate = marker_candidates_for_prepared_write(first)[0]
        write_parsed_session_to_archive(
            conn, session, prepared_write=first, content_hash=first.input_content_hash.hex()
        )
        second = prepare_session_write(conn, session, merge_append=True)
        second_candidate = marker_candidates_for_prepared_write(second)[0]
        assert first_candidate["provenance"] != second_candidate["provenance"]
        write_parsed_session_to_archive(
            conn, session, prepared_write=second, merge_append=True, content_hash=second.input_content_hash.hex()
        )
        stored = {row[0] for row in conn.execute("SELECT block_id FROM blocks")}
        assert _nested(first_candidate, "provenance", "block_id") in stored
        assert _nested(second_candidate, "provenance", "block_id") in stored

        parent = _session("::note: parent", native_id="parent", message_id="p")
        write_parsed_session_to_archive(conn, parent)
        child = parent.model_copy(
            update={
                "provider_session_id": "child",
                "parent_session_provider_id": "parent",
                "messages": [
                    *parent.messages,
                    ParsedMessage(provider_message_id="tail", role=Role.ASSISTANT, text="::note: child"),
                ],
            }
        )
        prepared_child = prepare_session_write(conn, child, merge_append=False)
        candidates = marker_candidates_for_prepared_write(prepared_child)
        assert [_nested(candidate, "match", "body") for candidate in candidates] == ["child"]
        write_parsed_session_to_archive(
            conn, child, prepared_write=prepared_child, content_hash=prepared_child.input_content_hash.hex()
        )
        stored = {
            row[0] for row in conn.execute("SELECT block_id FROM blocks WHERE session_id = 'codex-session:child'")
        }
        assert {_nested(candidate, "provenance", "block_id") for candidate in candidates} == stored
    finally:
        conn.close()


def test_batch_append_retains_only_published_delta_with_canonical_occurrences(workspace_env: dict[str, Path]) -> None:
    """Preparing the full input before append selection retains prefix markers and occurrence zero."""
    conn = sqlite3.connect(workspace_env["archive_root"] / "index.db")
    conn.row_factory = sqlite3.Row
    parent = _session("::note: parent prefix", native_id="parent", message_id="parent-message")
    repeated = ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="::note: repeated child")
    child = parent.model_copy(
        update={
            "provider_session_id": "child",
            "parent_session_provider_id": "parent",
            "messages": [*parent.messages, repeated],
        }
    )

    def accept(session: ParsedSession, raw_id: str, *, append_only: bool = False) -> _IngestBatchSummary:
        summary = _IngestBatchSummary()
        payload = SessionWritePayload(
            session_id=f"codex-session:{session.provider_session_id}",
            content_hash=str(session_content_hash(session)),
            parsed_session=session,
            message_count=len(session.messages),
            attachment_count=0,
            raw_id=raw_id,
            append_only=append_only,
        )
        assert _write_session_entry(conn, raw_id, payload, summary=summary), summary.failed_raw_ids
        return summary

    try:
        accept(parent, "parent-raw")
        accept(child, "child-r1")
        before = {row[0] for row in conn.execute("SELECT block_id FROM blocks")}
        full_append = child.model_copy(update={"messages": [*child.messages, repeated]})
        appended = accept(full_append, "child-r2", append_only=True)
        candidates = cast(list[dict[str, object]], appended.marker_sessions_by_raw_id["child-r2"][0]["candidates"])
        assert [_nested(candidate, "match", "body") for candidate in candidates] == ["repeated child"]
        after = {row[0] for row in conn.execute("SELECT block_id FROM blocks")}
        retained_ids = {_nested(candidate, "provenance", "block_id") for candidate in candidates}
        assert retained_ids == after - before
        assert len(retained_ids) == 1
        assert str(next(iter(retained_ids))).endswith(".1:0")
    finally:
        conn.close()


def test_source_migration_matches_fresh_ddl_and_preserves_restart_sequence(tmp_path: Path) -> None:
    """The additive migration must preserve historical source rows and stream position."""
    migration = Path("polylogue/storage/sqlite/migrations/source/005_accepted_marker_inputs.sql").read_text()
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as source:
        source.execute("CREATE TABLE historical_source(value TEXT)")
        source.execute("INSERT INTO historical_source VALUES ('retained')")
        source.executescript(migration)
        assert (
            asyncio.run(append_accepted_marker_input(_AsyncConnection(source), prepare_accepted_marker_input("r1", [])))
            == 1
        )
        objects = source.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name IN ("
            "'pending_accepted_marker_inputs', 'accepted_marker_stream', 'accepted_marker_inputs', "
            "'excised_marker_inputs', "
            "'accepted_marker_stream_no_update', 'accepted_marker_stream_no_delete', "
            "'accepted_marker_inputs_no_update', 'accepted_marker_inputs_no_delete') ORDER BY name"
        ).fetchall()
    with sqlite3.connect(":memory:") as fresh:
        fresh.executescript(SOURCE_DDL)
        assert (
            fresh.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name IN ("
                "'pending_accepted_marker_inputs', 'accepted_marker_stream', 'accepted_marker_inputs', "
                "'excised_marker_inputs', "
                "'accepted_marker_stream_no_update', 'accepted_marker_stream_no_delete', "
                "'accepted_marker_inputs_no_update', 'accepted_marker_inputs_no_delete') ORDER BY name"
            ).fetchall()
            == objects
        )
    with sqlite3.connect(path) as source:
        assert source.execute("SELECT value FROM historical_source").fetchone()[0] == "retained"
        assert (
            asyncio.run(append_accepted_marker_input(_AsyncConnection(source), prepare_accepted_marker_input("r2", [])))
            == 2
        )


def test_marker_migration_train_has_runtime_proof_and_refuses_without_backup(tmp_path: Path) -> None:
    """An unregistered rider or an unbacked durable migration must fail."""
    steps = migration_runner._load_migrations(ArchiveTier.SOURCE)
    sidecars = validate_durable_migration_sidecars(ArchiveTier.SOURCE, tuple((step.name, step.sql) for step in steps))
    train = next(sidecar.train for sidecar in sidecars if sidecar.slot == 5)
    assert train.migration.requires_backup
    assert all(result.passed for result in _runtime_consumer_results(train, tmp_path))
    migration = next(step.sql for step in steps if step.version == 5)
    # Build a genuine v4 fixture: SOURCE_DDL is the current fresh schema and
    # includes v5's additive objects. MigrationStep.sql includes its safety
    # metadata comment, so strip that metadata before removing the SQL body.
    migration_body = migration.split("\n", 1)[1]
    with sqlite3.connect(tmp_path / "source.db") as source:
        v4_ddl, replacements = re.subn(re.escape(migration_body), "", SOURCE_DDL, count=1)
        assert replacements == 1
        source.executescript(v4_ddl)
        source.execute("PRAGMA user_version = 4")
        v5_objects = source.execute(
            "SELECT name FROM sqlite_master WHERE name IN ("
            "'pending_accepted_marker_inputs', 'accepted_marker_stream', 'accepted_marker_inputs', "
            "'excised_marker_inputs', "
            "'accepted_marker_stream_no_update', 'accepted_marker_stream_no_delete', "
            "'accepted_marker_inputs_no_update', 'accepted_marker_inputs_no_delete')"
        ).fetchall()
        assert v5_objects == []
        source.commit()
        with pytest.raises(migration_runner.MigrationError, match="requires a verified backup manifest"):
            migration_runner.migrate_archive_tier(source, ArchiveTier.SOURCE, backup_manifest=None)
        assert source.execute("PRAGMA user_version").fetchone() == (4,)


def test_request_identity_uses_full_parse_while_carrier_keeps_selected_delta() -> None:
    """A no-op retry has the same request identity but cannot replace delta bytes."""
    parsed: list[dict[str, object]] = [{"session_id": "child", "input_content_hash": "full-hash"}]
    original = prepare_accepted_marker_input(
        "raw",
        [{**parsed[0], "disposition": "append", "candidates": [{"block_id": "tail:1"}]}],
        request_facts={"recipe": "r1"},
        request_sessions=parsed,
    )
    retry = prepare_accepted_marker_input(
        "raw",
        [{**parsed[0], "disposition": "no-op", "candidates": []}],
        request_facts={"recipe": "r1"},
        request_sessions=parsed,
    )
    changed_recipe = prepare_accepted_marker_input(
        "raw",
        [{**parsed[0], "disposition": "no-op", "candidates": []}],
        request_facts={"recipe": "r2"},
        request_sessions=parsed,
    )
    assert original.identity == retry.identity
    assert original.payload != retry.payload
    assert changed_recipe.identity != original.identity


def test_index_witness_recovers_and_refreshes_exact_accepted_carrier(tmp_path: Path) -> None:
    """A current witness can be rebuilt from the exact retained accepted carrier."""
    from polylogue.pipeline.services.ingest_batch._core import _publish_marker_witnesses_before_index_commit

    root = tmp_path / "archive"
    root.mkdir()
    with sqlite3.connect(root / "source.db") as source:
        source.executescript(SOURCE_DDL)
    with sqlite3.connect(root / "index.db") as index:
        index.executescript(INDEX_DDL)
        ingest_batch_core._ensure_ingest_index_incarnation(index)
        full: list[dict[str, object]] = [{"session_id": "child", "input_content_hash": "full-hash"}]
        original: dict[str, object] = {
            **full[0],
            "disposition": "append",
            "candidates": [{"block_id": "tail:1"}],
        }
        summary = _IngestBatchSummary(
            marker_request_facts_by_raw_id={"raw": {"recipe": "r1"}},
            marker_request_sessions_by_raw_id={"raw": full},
            marker_session_dispositions_by_raw_id={"raw": [{"session_id": "child", "disposition": "append"}]},
            marker_sessions_by_raw_id={"raw": [original]},
        )
        index.execute("BEGIN IMMEDIATE")
        _publish_marker_witnesses_before_index_commit(index, archive_root=root, summary=summary)
        index.commit()
        first_batch = summary.marker_batches_by_raw_id["raw"]

        retry = _IngestBatchSummary(
            marker_request_facts_by_raw_id={"raw": {"recipe": "r1"}},
            marker_request_sessions_by_raw_id={"raw": full},
            marker_session_dispositions_by_raw_id={"raw": [{"session_id": "child", "disposition": "no-op"}]},
            # The public ingest route classifies this exact current-witness
            # retry before the session drain and reuses the retained carrier.
            marker_batches_by_raw_id={"raw": first_batch},
        )
        index.execute("BEGIN IMMEDIATE")
        _publish_marker_witnesses_before_index_commit(index, archive_root=root, summary=retry)
        index.rollback()
        assert retry.marker_batches_by_raw_id["raw"] == first_batch

        with sqlite3.connect(root / "source.db") as source:
            assert asyncio.run(finalize_pending_accepted_marker_input(_AsyncConnection(source), first_batch)) == 1
        witness = index.execute("SELECT carrier_digest, dispositions_json FROM ingest_marker_witnesses").fetchone()
        assert witness == (
            first_batch.payload_sha256,
            '[{"disposition":"append","session_id":"child"}]',
        )

    old_incarnation = index.execute(
        "SELECT incarnation_id FROM ingest_index_incarnation WHERE singleton = 1"
    ).fetchone()[0]
    index.close()
    for suffix in ("", "-wal", "-shm"):
        (root / f"index.db{suffix}").unlink(missing_ok=True)
    with sqlite3.connect(root / "index.db") as rebuilt_index:
        rebuilt_index.executescript(INDEX_DDL)
        ingest_batch_core._ensure_ingest_index_incarnation(rebuilt_index)
        new_incarnation = rebuilt_index.execute(
            "SELECT incarnation_id FROM ingest_index_incarnation WHERE singleton = 1"
        ).fetchone()[0]
        assert new_incarnation != old_incarnation
        same_accepted_input = _IngestBatchSummary(
            marker_request_facts_by_raw_id={"raw": {"recipe": "r1"}},
            marker_request_sessions_by_raw_id={"raw": full},
            marker_session_dispositions_by_raw_id={"raw": [{"session_id": "child", "disposition": "append"}]},
            marker_sessions_by_raw_id={"raw": [original]},
        )
        rebuilt_index.execute("BEGIN IMMEDIATE")
        _publish_marker_witnesses_before_index_commit(rebuilt_index, archive_root=root, summary=same_accepted_input)
        rebuilt_index.commit()
        assert same_accepted_input.marker_batches_by_raw_id["raw"] == first_batch
        assert rebuilt_index.execute(
            "SELECT carrier_digest, dispositions_json, incarnation_id FROM ingest_marker_witnesses"
        ).fetchone() == (
            first_batch.payload_sha256,
            '[{"disposition":"append","session_id":"child"}]',
            new_incarnation,
        )


def test_deferred_raw_does_not_publish_marker_witness_or_pending_carrier(tmp_path: Path) -> None:
    """A raw deferred before index publication is not marker-accepted."""
    from polylogue.pipeline.services.ingest_batch._core import _publish_marker_witnesses_before_index_commit

    root = tmp_path / "archive"
    root.mkdir()
    with sqlite3.connect(root / "source.db") as source:
        source.executescript(SOURCE_DDL)
    with sqlite3.connect(root / "index.db") as index:
        index.executescript(INDEX_DDL)
        ingest_batch_core._ensure_ingest_index_incarnation(index)
        index.execute("BEGIN IMMEDIATE")
        summary = _IngestBatchSummary(
            marker_request_facts_by_raw_id={"deferred-raw": {"recipe": "r1"}},
            marker_request_sessions_by_raw_id={"deferred-raw": [{"session_id": "s"}]},
            publication_deferred_raw_ids={"deferred-raw"},
        )
        _publish_marker_witnesses_before_index_commit(index, archive_root=root, summary=summary)
        assert summary.marker_batches_by_raw_id == {}
        assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (0,)

    with sqlite3.connect(root / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary",
    [
        "before-index-commit",
        "after-index-commit",
        "after-source-finalization",
        "after-index-generation-replacement",
    ],
)
async def test_public_process_ingest_batch_recovers_empty_marker_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_boundary: str
) -> None:
    """The public route recovers exact empty-marker bytes across tier commits."""
    bootstrap_archive_root(tmp_path)
    payload = (Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json").read_bytes()
    BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CHATGPT_EXPORT,
            source_path="accepted-marker-empty.json",
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )
        source.commit()
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    original_commit_boundary = ingest_batch_core._commit_sync_ingest_side_effects
    original_raw_state_boundary = ingest_batch_core._persist_batch_raw_state_updates
    try:
        if failure_boundary == "before-index-commit":

            def interrupt_before_index_commit(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("simulated loss before index commit")

            monkeypatch.setattr(ingest_batch_core, "_commit_sync_ingest_side_effects", interrupt_before_index_commit)
        else:

            async def interrupt_after_index_commit(*args: object, **kwargs: object) -> float:
                if failure_boundary == "after-source-finalization":
                    await cast(Callable[..., Awaitable[float]], original_raw_state_boundary)(*args, **kwargs)
                    raise RuntimeError("simulated loss after source finalization")
                raise RuntimeError("simulated loss after index commit")

            monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", interrupt_after_index_commit)
        with pytest.raises(RuntimeError, match="simulated loss"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
            )
        if failure_boundary == "before-index-commit":
            monkeypatch.setattr(ingest_batch_core, "_commit_sync_ingest_side_effects", original_commit_boundary)
        else:
            monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", original_raw_state_boundary)
        with sqlite3.connect(tmp_path / "source.db") as source:
            if failure_boundary == "after-source-finalization":
                assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
                assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (1,)
            else:
                assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (1,)
                assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (
                0 if failure_boundary == "before-index-commit" else 1,
            )
        if failure_boundary == "after-index-generation-replacement":
            await repository.close()
            for suffix in ("", "-wal", "-shm"):
                (tmp_path / f"index.db{suffix}").unlink(missing_ok=True)
            from polylogue.storage.sqlite.connection import open_connection

            with open_connection(tmp_path / "index.db"):
                pass
            repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
            service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
            with pytest.raises(AcceptedMarkerInputRefusedError, match="replaced index incarnation"):
                await ingest_batch_core.process_ingest_batch(
                    service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
                )
            with sqlite3.connect(tmp_path / "source.db") as source:
                assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (1,)
                assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
            with sqlite3.connect(tmp_path / "index.db") as index:
                assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
            return
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        row = source.execute(
            "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        assert row is not None and row[0] == 1
        carrier = json.loads(row[1])
        assert carrier["raw_id"] == raw_id
        assert all(not session["candidates"] for session in carrier["sessions"])
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (1,)


@pytest.mark.asyncio
async def test_public_marker_retry_reuses_pending_carrier_after_another_raw_publishes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback carrier survives a same-incarnation publication by another raw.

    Every parse returns a new object: the production batch deliberately clears
    parsed payloads after use, so reusing the fixture object would hide the
    retry that must retain the original marker bytes.
    """
    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    raw_ids: list[str] = []
    parsed_template = _session("::note: retained through retry")
    payload_template = SessionWritePayload(
        session_id="codex-session:session",
        content_hash=str(session_content_hash(parsed_template)),
        parsed_session=parsed_template,
        message_count=len(parsed_template.messages),
    )
    parsed_objects: list[ParsedSession] = []
    try:
        for suffix in ("first", "second"):
            payload_bytes = f"same-normalized-session-{suffix}".encode()
            BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
            with sqlite3.connect(tmp_path / "source.db") as source:
                raw_ids.append(
                    write_source_raw_session(
                        source,
                        origin=Origin.CODEX_SESSION,
                        source_path=f"pending-replay-{suffix}.jsonl",
                        source_index=0,
                        payload=payload_bytes,
                        acquired_at_ms=1,
                    )
                )

        def fresh_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
            payload = copy.deepcopy(payload_template)
            payload = replace(payload, raw_id=record.raw_id)
            parsed_objects.append(payload.parsed_session)
            return IngestRecordResult(
                raw_id=record.raw_id,
                payload_provider=Provider.CODEX.value,
                validation_status="passed",
                outcome_code="success",
                sessions=[payload],
            )

        monkeypatch.setattr(ingest_batch_core, "ingest_record", fresh_ingest)
        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
        )
        original_commit_boundary = ingest_batch_core._commit_sync_ingest_side_effects

        def interrupt_before_index_commit(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated loss before first index commit")

        monkeypatch.setattr(ingest_batch_core, "_commit_sync_ingest_side_effects", interrupt_before_index_commit)
        with pytest.raises(RuntimeError, match="before first index commit"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_ids[0]], ParseResult(), None, repair_message_fts=False
            )
        monkeypatch.setattr(ingest_batch_core, "_commit_sync_ingest_side_effects", original_commit_boundary)

        with sqlite3.connect(tmp_path / "source.db") as source:
            pending = source.execute(
                "SELECT request_key, carrier_digest, payload FROM pending_accepted_marker_inputs WHERE raw_id = ?",
                (raw_ids[0],),
            ).fetchone()
            assert pending is not None
            pending_payload = bytes(cast(bytes, pending[2]))
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
            assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (0,)

        # The second raw writes the same normalized session in the unchanged
        # index incarnation. It has a distinct request identity and cannot
        # replace the first raw's pending byte carrier.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[1]], ParseResult(), None, repair_message_fts=False
        )
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT raw_id FROM sessions").fetchone() == (raw_ids[1],)

        # Retry the first raw from a fresh parsed object. Its ordinary writer
        # now sees a no-op, but the existing session proves the retained
        # pending interpretation was successfully published in this
        # incarnation. Finalization must preserve those exact bytes.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[0]], ParseResult(), None, repair_message_fts=False
        )
    finally:
        await repository.close()

    assert len(parsed_objects) == 3
    assert len({id(parsed) for parsed in parsed_objects}) == 3
    with sqlite3.connect(tmp_path / "source.db") as source:
        accepted = source.execute(
            "SELECT payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_ids[0],)
        ).fetchone()
        assert accepted is not None
        assert bytes(cast(bytes, accepted[0])) == pending_payload
        assert source.execute(
            "SELECT COUNT(*) FROM pending_accepted_marker_inputs WHERE raw_id = ?", (raw_ids[0],)
        ).fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT carrier_digest FROM ingest_marker_witnesses WHERE request_key = ?", (pending[0],)
        ).fetchone() == (pending[1],)


@pytest.mark.asyncio
async def test_public_batch_append_retry_reuses_the_first_delta_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Append replay keeps the exact selected PreparedSessionWrite candidates."""
    from polylogue.pipeline.ids import session_content_hash

    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    sessions: dict[str, SessionWritePayload] = {}
    raw_ids: list[str] = []
    try:
        for suffix, text, append_only in (
            ("base", "::note: repeated lesson", False),
            ("tail", "::note: repeated lesson", True),
        ):
            payload_bytes = f"raw-{suffix}".encode()
            BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
            with sqlite3.connect(tmp_path / "source.db") as source:
                raw_id = write_source_raw_session(
                    source,
                    origin=Origin.CODEX_SESSION,
                    source_path=f"marker-{suffix}.jsonl",
                    source_index=0,
                    payload=payload_bytes,
                    acquired_at_ms=1,
                )
            raw_ids.append(raw_id)
            parsed = _session(text)
            if append_only:
                parsed = parsed.model_copy(
                    update={
                        "messages": [
                            ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text=text),
                            ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text=text),
                        ]
                    }
                )
            sessions[raw_id] = SessionWritePayload(
                session_id="codex-session:session",
                content_hash=str(session_content_hash(parsed)),
                parsed_session=parsed,
                message_count=len(parsed.messages),
                raw_id=raw_id,
                append_only=append_only,
            )

        parse_calls: list[tuple[str, tuple[str | None, ...]]] = []

        def fake_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
            raw_id = record.raw_id
            payload = copy.deepcopy(sessions[raw_id])
            parse_calls.append((raw_id, tuple(message.text for message in payload.parsed_session.messages)))
            return IngestRecordResult(
                raw_id=raw_id,
                payload_provider=Provider.CODEX.value,
                validation_status="passed",
                outcome_code="success",
                sessions=[payload],
            )

        monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
        )
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[0]], ParseResult(), None, repair_message_fts=False
        )
        raw_state_boundary = ingest_batch_core._persist_batch_raw_state_updates

        async def interrupt_after_index_commit(*_args: object, **_kwargs: object) -> float:
            raise RuntimeError("simulated loss after append index commit")

        monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", interrupt_after_index_commit)
        with pytest.raises(RuntimeError, match="after append index commit"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_ids[1]], ParseResult(), None, repair_message_fts=False
            )
        monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", raw_state_boundary)
        with sqlite3.connect(tmp_path / "source.db") as source:
            pending = source.execute(
                "SELECT request_key, carrier_digest, payload, expected_incarnation_id "
                "FROM pending_accepted_marker_inputs WHERE raw_id = ?",
                (raw_ids[1],),
            ).fetchone()
            assert pending is not None
            original_payload = bytes(cast(bytes, pending[2]))
            original = json.loads(original_payload)
            assert [item["match"]["body"] for item in original["sessions"][0]["candidates"]] == ["repeated lesson"]
            provenance = original["sessions"][0]["candidates"][0]["provenance"]
            assert provenance["block_id"].endswith(".1:0")
        original_index = _index_message_state(tmp_path / "index.db")
        assert original_index[0] == (("codex-session:session",),)
        assert len(original_index[1]) == 2
        assert any("repeated lesson" in str(row[1]) for row in original_index[2])
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute(
                "SELECT carrier_digest FROM ingest_marker_witnesses WHERE request_key = ?", (pending[0],)
            ).fetchone() == (pending[1],)
        assert _accepted_marker_state(tmp_path / "source.db", raw_ids[1]) is None

        # Force publication bypasses the ordinary exact-witness shortcut. A
        # changed append/replacement carrier must refuse despite the committed
        # historical witness, leaving both index rows and pending source bytes
        # intact.
        with pytest.raises(AcceptedMarkerInputRefusedError, match="immutable retained marker carrier"):
            await ingest_batch_core.process_ingest_batch(
                service,
                repository.backend,
                [raw_ids[1]],
                ParseResult(),
                None,
                force_write=True,
                repair_message_fts=False,
            )
        assert _index_message_state(tmp_path / "index.db") == original_index
        with sqlite3.connect(tmp_path / "source.db") as source:
            assert (
                source.execute(
                    "SELECT request_key, carrier_digest, payload, expected_incarnation_id "
                    "FROM pending_accepted_marker_inputs WHERE raw_id = ?",
                    (raw_ids[1],),
                ).fetchone()
                == pending
            )

        # Ordinary retry reuses the exact current-witness carrier and completes
        # source acceptance without rewriting the already committed append.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[1]], ParseResult(), None, repair_message_fts=False
        )
        assert _index_message_state(tmp_path / "index.db") == original_index
        original_accepted = _accepted_marker_state(tmp_path / "source.db", raw_ids[1])
        assert original_accepted is not None and original_accepted[0] == 2
        assert bytes(cast(bytes, original_accepted[1])) == original_payload

        with pytest.raises(AcceptedMarkerInputRefusedError, match="immutable retained marker carrier"):
            await ingest_batch_core.process_ingest_batch(
                service,
                repository.backend,
                [raw_ids[1]],
                ParseResult(),
                None,
                force_write=True,
                repair_message_fts=False,
            )
        assert _index_message_state(tmp_path / "index.db") == original_index
        assert _accepted_marker_state(tmp_path / "source.db", raw_ids[1]) == original_accepted
        assert [texts for raw_id, texts in parse_calls if raw_id == raw_ids[1]] == [
            ("::note: repeated lesson", "::note: repeated lesson"),
            ("::note: repeated lesson", "::note: repeated lesson"),
            ("::note: repeated lesson", "::note: repeated lesson"),
            ("::note: repeated lesson", "::note: repeated lesson"),
        ]
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_ids[1],)
        ).fetchone() == (2, original_payload)


@pytest.mark.asyncio
async def test_public_batch_rebuild_reingests_and_rewitnesses_exact_accepted_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real retained-raw parsing can republish the exact carrier after index replacement."""
    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    payload_bytes = (
        b'{"type":"session_meta","payload":{"id":"rebuild-replay","timestamp":"2026-01-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"assistant",'
        b'"timestamp":"2026-01-01T00:00:01Z","id":"m1","content":'
        b'[{"type":"output_text","text":"::note: durable marker"}]}}\n'
    )
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            source_path=".codex/sessions/rebuild-replay.jsonl",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    try:
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        with sqlite3.connect(tmp_path / "source.db") as source:
            original = source.execute(
                "SELECT sequence, payload, index_incarnation_id FROM accepted_marker_inputs WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
        assert original is not None and original[0] == 1
        original_payload = bytes(original[1])
        original_candidates = json.loads(original_payload)["sessions"][0]["candidates"]
        assert [item["match"]["body"] for item in original_candidates] == ["durable marker"]
        await repository.close()

        # Model a new derived index generation while leaving durable source
        # acceptance untouched.
        for suffix in ("", "-wal", "-shm"):
            (tmp_path / f"index.db{suffix}").unlink(missing_ok=True)
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(tmp_path / "index.db"):
            pass
        await repository.close()
        repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
        service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
            witness = index.execute("SELECT carrier_digest, incarnation_id FROM ingest_marker_witnesses").fetchone()
            assert witness is not None
            rebuilt_incarnation = witness[1]
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert (
            source.execute(
                "SELECT sequence, payload, index_incarnation_id FROM accepted_marker_inputs WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            == original
        )
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert index.execute("SELECT carrier_digest, incarnation_id FROM ingest_marker_witnesses").fetchone() == (
            hashlib.sha256(original_payload).hexdigest(),
            rebuilt_incarnation,
        )


@pytest.mark.asyncio
async def test_public_drive_marker_retry_keeps_identity_across_revision_binding_and_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive lineage refinement cannot strand a pending marker carrier after index rollback."""
    from polylogue.pipeline.ids import session_content_hash

    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    payload_bytes = b'{"id":"drive-replay","messages":["::note: drive marker"]}'
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.AISTUDIO_DRIVE,
            capture_mode=Provider.GEMINI,
            source_path="Drive/replay.json",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    parsed = _session("::note: drive marker", native_id="drive-replay").model_copy(
        update={"source_name": Provider.GEMINI}
    )
    template = SessionWritePayload(
        session_id="aistudio-drive:drive-replay",
        content_hash=str(session_content_hash(parsed)),
        parsed_session=parsed,
        message_count=len(parsed.messages),
        raw_id=raw_id,
    )
    parse_calls: list[tuple[str | None, ...]] = []

    def fake_ingest(_record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        payload = copy.deepcopy(template)
        parse_calls.append(tuple(message.text for message in payload.parsed_session.messages))
        return IngestRecordResult(
            raw_id=raw_id,
            payload_provider=Provider.GEMINI.value,
            validation_status="passed",
            outcome_code="success",
            sessions=[payload],
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    real_commit = cast(Callable[..., None], ingest_batch_core._commit_sync_ingest_side_effects)
    commit_calls = 0

    def fail_first_index_commit(*args: object, **kwargs: object) -> None:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            raise RuntimeError("injected crash before Drive index commit")
        real_commit(*args, **kwargs)

    monkeypatch.setattr(ingest_batch_core, "_commit_sync_ingest_side_effects", fail_first_index_commit)
    try:
        with pytest.raises(RuntimeError, match="injected crash before Drive index commit"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
            )
        with sqlite3.connect(tmp_path / "source.db") as source:
            pending = source.execute(
                "SELECT request_key, carrier_digest, payload FROM pending_accepted_marker_inputs"
            ).fetchone()
            assert pending is not None
            bound_revision = source.execute(
                "SELECT logical_source_key, revision_kind, source_revision, acquisition_generation "
                "FROM raw_sessions WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            assert bound_revision is not None and bound_revision[0] == "aistudio-drive:drive-replay"
            assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
        assert _index_message_state(tmp_path / "index.db")[0] == ()
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (0,)

        records = await repository.get_raw_sessions_batch([raw_id])
        retry_facts = _marker_request_facts(records[0], validation_mode="off")
        assert retry_facts["origin"] == Origin.AISTUDIO_DRIVE.value
        # The durable Drive envelope changed during the failed first attempt,
        # but the request identity uses only immutable acquisition evidence.
        assert retry_facts["revision"] is None
        assert retry_facts["native_id"] is None
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        state_after_retry = _index_message_state(tmp_path / "index.db")
        accepted_after_retry = _accepted_marker_state(tmp_path / "source.db", raw_id)
        assert accepted_after_retry is not None and accepted_after_retry[0] == 1
        accepted_payload = bytes(cast(bytes, accepted_after_retry[1]))
        candidates = json.loads(accepted_payload)["sessions"][0]["candidates"]
        assert [item["match"]["body"] for item in candidates] == ["drive marker"]

        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        assert _index_message_state(tmp_path / "index.db") == state_after_retry
        assert _accepted_marker_state(tmp_path / "source.db", raw_id) == accepted_after_retry
        assert parse_calls == [("::note: drive marker",)] * 3
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute(
            "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, accepted_payload)
    with sqlite3.connect(tmp_path / "index.db") as index:
        witness = index.execute("SELECT carrier_digest, incarnation_id FROM ingest_marker_witnesses").fetchone()
        assert witness is not None
        assert witness[0] == hashlib.sha256(accepted_payload).hexdigest()


@pytest.mark.asyncio
async def test_public_mirror_replay_restages_without_rewriting_witnessed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MIRROR exact retries preserve the carrier while restaging publication payloads."""
    from polylogue.pipeline.ids import session_content_hash

    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    payload_bytes = b"mirror accepted marker raw"
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            source_path="mirror-marker.jsonl",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    parsed = _session("::note: mirror marker")
    template = SessionWritePayload(
        session_id="codex-session:session",
        content_hash=str(session_content_hash(parsed)),
        parsed_session=parsed,
        message_count=len(parsed.messages),
        raw_id=raw_id,
    )
    parse_calls = 0

    def fake_ingest(_record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        nonlocal parse_calls
        parse_calls += 1
        return IngestRecordResult(
            raw_id=raw_id,
            payload_provider=Provider.CODEX.value,
            validation_status="passed",
            outcome_code="success",
            sessions=[copy.deepcopy(template)],
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "mirror"})(),
    )
    staged_object_ids: list[str] = []

    async def observe_stage_payload(
        conn: object, *, payload: PublicationPayload, mode: PublicationMode, now_ms: int
    ) -> None:
        staged_object_ids.append(payload.object_id)
        await stage_payload_async(cast(AsyncSqlConnection, conn), payload=payload, mode=mode, now_ms=now_ms)

    monkeypatch.setattr(ingest_batch_core, "stage_payload_async", observe_stage_payload)
    try:
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        state_after_first = _index_message_state(tmp_path / "index.db")
        accepted_after_first = _accepted_marker_state(tmp_path / "source.db", raw_id)
        assert accepted_after_first is not None and accepted_after_first[0] == 1

        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        assert _index_message_state(tmp_path / "index.db") == state_after_first
        assert _accepted_marker_state(tmp_path / "source.db", raw_id) == accepted_after_first
        assert parse_calls == 2
        assert len(staged_object_ids) >= 2
        assert set(staged_object_ids) == {"codex-session:session"}
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_public_primary_pending_witness_retry_preserves_defer_then_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRIMARY admission runs before exact-witness reuse and source finalization."""
    from polylogue.pipeline.ids import session_content_hash

    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    payload_bytes = b"primary accepted marker raw"
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            source_path="primary-marker.jsonl",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    parsed = _session("::note: primary marker")
    template = SessionWritePayload(
        session_id="codex-session:session",
        content_hash=str(session_content_hash(parsed)),
        parsed_session=parsed,
        message_count=len(parsed.messages),
        raw_id=raw_id,
    )
    parse_calls = 0

    def fake_ingest(_record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        nonlocal parse_calls
        parse_calls += 1
        return IngestRecordResult(
            raw_id=raw_id,
            payload_provider=Provider.CODEX.value,
            validation_status="passed",
            outcome_code="success",
            sessions=[copy.deepcopy(template)],
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "primary"})(),
    )
    projection_results = iter((False, True, False))
    publication_events: list[tuple[str, object]] = []

    class _PublicationProbe:
        def __init__(self, _db_path: Path, _mode: object, _transport: object) -> None:
            pass

        def stage_payload(self, payload: PublicationPayload) -> None:
            publication_events.append(("stage", payload.object_id))

        def drain_once(self, *, object_ids: list[str], limit: int) -> None:
            publication_events.append(("drain", (tuple(object_ids), limit)))

        def projection_blocked(self, _object_ids: list[str]) -> bool:
            return next(projection_results)

    monkeypatch.setattr(ingest_batch_core, "resolve_configured_transport", lambda: object())
    monkeypatch.setattr(ingest_batch_core, "PublicationService", _PublicationProbe)
    raw_state_boundary = ingest_batch_core._persist_batch_raw_state_updates

    async def interrupt_source_finalization(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("simulated loss after PRIMARY index commit")

    monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", interrupt_source_finalization)
    try:
        with pytest.raises(RuntimeError, match="after PRIMARY index commit"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
            )
        monkeypatch.setattr(ingest_batch_core, "_persist_batch_raw_state_updates", raw_state_boundary)
        with sqlite3.connect(tmp_path / "source.db") as source:
            pending = source.execute(
                "SELECT request_key, carrier_digest, payload, expected_incarnation_id "
                "FROM pending_accepted_marker_inputs WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()
            assert pending is not None
            assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
        first_index_state = _index_message_state(tmp_path / "index.db")
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute(
                "SELECT carrier_digest FROM ingest_marker_witnesses WHERE request_key = ?", (pending[0],)
            ).fetchone() == (pending[1],)

        # PRIMARY remains authoritative: an unconfirmed duplicate is deferred
        # before the marker reuse shortcut and leaves pending source state.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        assert _index_message_state(tmp_path / "index.db") == first_index_state
        with sqlite3.connect(tmp_path / "source.db") as source:
            assert (
                source.execute(
                    "SELECT request_key, carrier_digest, payload, expected_incarnation_id "
                    "FROM pending_accepted_marker_inputs WHERE raw_id = ?",
                    (raw_id,),
                ).fetchone()
                == pending
            )
            assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)

        # Once PRIMARY admission succeeds, current-witness reuse skips only
        # the session write; source finalization and Sinex restaging proceed.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
        )
        accepted = _accepted_marker_state(tmp_path / "source.db", raw_id)
        assert accepted is not None and accepted[0] == 1
        assert bytes(cast(bytes, accepted[1])) == bytes(cast(bytes, pending[2]))
        assert _index_message_state(tmp_path / "index.db") == first_index_state
        assert parse_calls == 3
        assert [event[0] for event in publication_events] == ["stage", "drain"] * 3
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute(
            "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, bytes(pending[2]))


@pytest.mark.asyncio
async def test_public_child_before_parent_retry_keeps_its_accepted_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later parent cannot rewrite an already accepted child marker batch."""
    from polylogue.pipeline.ids import session_content_hash

    bootstrap_archive_root(tmp_path)
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    inputs: dict[str, SessionWritePayload] = {}
    try:
        parent = _session("::note: parent marker", native_id="parent", message_id="parent-message")
        child = parent.model_copy(
            update={
                "provider_session_id": "child",
                "parent_session_provider_id": "parent",
                "messages": [
                    *parent.messages,
                    ParsedMessage(
                        provider_message_id="child-message",
                        role=Role.ASSISTANT,
                        text="::note: child marker",
                    ),
                ],
            }
        )
        raw_ids: dict[str, str] = {}
        for name, parsed in (("child", child), ("parent", parent)):
            payload_bytes = f"lineage-{name}".encode()
            BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
            with sqlite3.connect(tmp_path / "source.db") as source:
                raw_id = write_source_raw_session(
                    source,
                    origin=Origin.CODEX_SESSION,
                    source_path=f"lineage-{name}.jsonl",
                    source_index=0,
                    payload=payload_bytes,
                    acquired_at_ms=1,
                )
            raw_ids[name] = raw_id
            inputs[raw_id] = SessionWritePayload(
                session_id=f"codex-session:{parsed.provider_session_id}",
                content_hash=str(session_content_hash(parsed)),
                parsed_session=parsed,
                message_count=len(parsed.messages),
                raw_id=raw_id,
            )

        parse_calls: list[tuple[str, tuple[str | None, ...]]] = []

        def fake_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
            raw_id = record.raw_id
            payload = copy.deepcopy(inputs[raw_id])
            parse_calls.append((raw_id, tuple(message.text for message in payload.parsed_session.messages)))
            return IngestRecordResult(
                raw_id=raw_id,
                payload_provider=Provider.CODEX.value,
                validation_status="passed",
                outcome_code="success",
                sessions=[payload],
            )

        monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
        )
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids["child"]], ParseResult(), None, repair_message_fts=False
        )
        with sqlite3.connect(tmp_path / "source.db") as source:
            first = source.execute(
                "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_ids["child"],)
            ).fetchone()
            assert first is not None and first[0] == 1
            original_payload = bytes(first[1])
            original_candidates = json.loads(original_payload)["sessions"][0]["candidates"]
            assert [item["match"]["body"] for item in original_candidates] == ["parent marker", "child marker"]
        child_accepted = _accepted_marker_state(tmp_path / "source.db", raw_ids["child"])

        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids["parent"]], ParseResult(), None, repair_message_fts=False
        )
        index_after_parent = _index_message_state(tmp_path / "index.db")
        assert index_after_parent[0] == (("codex-session:child",), ("codex-session:parent",))
        assert len(index_after_parent[1]) == 2
        child_messages = {str(row[1]) for row in index_after_parent[1] if row[0] == "codex-session:child"}
        assert child_messages == {"codex-session:child:n:child-message"}
        assert any("parent marker" in str(row[1]) for row in index_after_parent[2])
        assert any("child marker" in str(row[1]) for row in index_after_parent[2])

        # Reparse into fresh nested objects. A current exact witness classifies
        # this as an ordinary retry before lineage preparation, preserving the
        # accepted bytes and existing physical child rows.
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids["child"]], ParseResult(), None, repair_message_fts=False
        )
        assert _index_message_state(tmp_path / "index.db") == index_after_parent
        assert _accepted_marker_state(tmp_path / "source.db", raw_ids["child"]) == child_accepted
        assert [texts for raw_id, texts in parse_calls if raw_id == raw_ids["child"]] == [
            ("::note: parent marker", "::note: child marker"),
            ("::note: parent marker", "::note: child marker"),
        ]
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert (
            source.execute(
                "SELECT sequence, payload, index_incarnation_id FROM accepted_marker_inputs WHERE raw_id = ?",
                (raw_ids["child"],),
            ).fetchone()
            == child_accepted
        )


@pytest.mark.asyncio
async def test_public_partial_multi_session_raw_rolls_back_before_marker_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful sibling session cannot commit without its raw's complete carrier."""
    bootstrap_archive_root(tmp_path)
    payload_bytes = b"one acquired raw producing two sessions"
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            source_path="partial-raw.jsonl",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    parsed_sessions = [_session("::note: first", native_id="a"), _session("::note: second", native_id="z")]
    payloads = [
        SessionWritePayload(
            session_id=f"codex-session:{parsed.provider_session_id}",
            content_hash=str(session_content_hash(parsed)),
            parsed_session=parsed,
            message_count=len(parsed.messages),
            raw_id=raw_id,
        )
        for parsed in parsed_sessions
    ]

    def fake_ingest(_record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        return IngestRecordResult(
            raw_id=raw_id,
            payload_provider=Provider.CODEX.value,
            outcome_code="success",
            sessions=payloads,
        )

    real_write = write_parsed_session_to_archive

    def fail_second_session(conn: sqlite3.Connection, session: ParsedSession, *args: Any, **kwargs: Any) -> Any:
        if session.provider_session_id == "z":
            raise RuntimeError("injected second-session failure")
        return real_write(conn, session, *args, **kwargs)

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
    monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", fail_second_session)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    try:
        with pytest.raises(AcceptedMarkerInputRefusedError, match="partially written raw marker input"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
            )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)


@pytest.mark.asyncio
async def test_public_duplicate_normalized_session_ids_refuse_before_index_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carrier map cannot collapse two interpreted sessions onto one ID."""
    bootstrap_archive_root(tmp_path)
    payload_bytes = b"one acquired raw with duplicate normalized session IDs"
    BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
    with sqlite3.connect(tmp_path / "source.db") as source:
        raw_id = write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            source_path="duplicate-sessions.jsonl",
            source_index=0,
            payload=payload_bytes,
            acquired_at_ms=1,
        )
    parsed_sessions = [
        _session("::note: first", native_id="duplicate"),
        _session("::note: second", native_id="duplicate"),
    ]
    payloads = [
        SessionWritePayload(
            session_id=f"codex-session:{parsed.provider_session_id}",
            content_hash=str(session_content_hash(parsed)),
            parsed_session=parsed,
            message_count=len(parsed.messages),
            raw_id=raw_id,
        )
        for parsed in parsed_sessions
    ]

    def fake_ingest(_record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        return IngestRecordResult(raw_id=raw_id, outcome_code="success", sessions=payloads)

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    try:
        with pytest.raises(AcceptedMarkerInputRefusedError, match="duplicate normalized session IDs"):
            await ingest_batch_core.process_ingest_batch(
                service, repository.backend, [raw_id], ParseResult(), None, repair_message_fts=False
            )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert index.execute("SELECT COUNT(*) FROM ingest_marker_witnesses").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)


@pytest.mark.asyncio
async def test_source_required_mode_refuses_before_index_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required durable acceptance cannot begin as an index-only write."""
    from polylogue.sinex.material_adapter import PublicationEncodingError
    from polylogue.storage.repository import SessionRepository

    async def raw_records(_batch_ids: list[str]) -> list[RawSessionRecord]:
        return [
            RawSessionRecord(
                raw_id="required-source",
                source_name="codex",
                source_path="required-source.jsonl",
                blob_size=2,
                acquired_at="2026-04-02T00:00:00Z",
            )
        ]

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"))
    monkeypatch.setattr(repository, "get_raw_sessions_batch", raw_records)
    service = ParsingService(
        repository=repository,
        archive_root=tmp_path,
        config=Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[]),
        ingest_workers=1,
    )
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "primary"})(),
    )

    def should_not_process(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("index processing started before source capability preflight")

    monkeypatch.setattr(ingest_batch_core, "_process_ingest_batch_sync", should_not_process)
    with pytest.raises(PublicationEncodingError, match="refusing before index publication"):
        await ingest_batch_core.process_ingest_batch(service, backend, ["required-source"], ParseResult(), None)
