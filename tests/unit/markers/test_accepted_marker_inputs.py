"""Accepted source inputs retain effects even when the current index advances."""

from __future__ import annotations

import asyncio
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
            "'accepted_marker_stream_no_update', 'accepted_marker_stream_no_delete', "
            "'accepted_marker_inputs_no_update', 'accepted_marker_inputs_no_delete') ORDER BY name"
        ).fetchall()
    with sqlite3.connect(":memory:") as fresh:
        fresh.executescript(SOURCE_DDL)
        assert (
            fresh.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name IN ("
                "'pending_accepted_marker_inputs', 'accepted_marker_stream', 'accepted_marker_inputs', "
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

        def fake_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
            raw_id = record.raw_id
            return IngestRecordResult(
                raw_id=raw_id,
                payload_provider=Provider.CODEX.value,
                validation_status="passed",
                outcome_code="success",
                sessions=[sessions[raw_id]],
            )

        monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest)
        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
        )
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[0]], ParseResult(), None, repair_message_fts=False
        )
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[1]], ParseResult(), None, repair_message_fts=False
        )
        with sqlite3.connect(tmp_path / "source.db") as source:
            accepted = source.execute(
                "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_ids[1],)
            ).fetchone()
            assert accepted is not None
            original_payload = bytes(accepted[1])
            original = json.loads(original_payload)
            assert accepted[0] == 2
            assert [item["match"]["body"] for item in original["sessions"][0]["candidates"]] == ["repeated lesson"]
            provenance = original["sessions"][0]["candidates"][0]["provenance"]
            assert provenance["block_id"].endswith(".1:0")

        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids[1]], ParseResult(), None, repair_message_fts=False
        )
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

        def fake_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
            raw_id = record.raw_id
            return IngestRecordResult(
                raw_id=raw_id,
                payload_provider=Provider.CODEX.value,
                validation_status="passed",
                outcome_code="success",
                sessions=[inputs[raw_id]],
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

        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids["parent"]], ParseResult(), None, repair_message_fts=False
        )
        await ingest_batch_core.process_ingest_batch(
            service, repository.backend, [raw_ids["child"]], ParseResult(), None, repair_message_fts=False
        )
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT sequence, payload FROM accepted_marker_inputs WHERE raw_id = ?", (raw_ids["child"],)
        ).fetchone() == (1, original_payload)


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
