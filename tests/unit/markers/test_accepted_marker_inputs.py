"""Accepted source inputs retain effects even when the current index advances."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.markers.preparation import marker_candidates_for_prepared_write
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.services.ingest_batch._core import _persist_batch_raw_state_updates, _write_session_entry
from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputRefusedError,
    append_accepted_marker_input,
    prepare_accepted_marker_input,
)
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write, write_parsed_session_to_archive
from polylogue.storage.sqlite.durable_change_train import _runtime_consumer_results, validate_durable_migration_sidecars
from tests.unit.sinex.test_ingest_atomicity import _AsyncConnection, _Repository, _SourceBackend


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
    """Removing source publication or rereading the current index loses R1 here."""
    root = workspace_env["archive_root"]
    backend = _SourceBackend(root / "source.db")
    service = SimpleNamespace(repository=_Repository(backend))
    with sqlite3.connect(root / "source.db") as source:
        source.execute("CREATE TABLE test_raw_acceptance(raw_id TEXT PRIMARY KEY, accepted INTEGER NOT NULL)")
    index = sqlite3.connect(root / "index.db")
    index.row_factory = sqlite3.Row
    summaries = []
    try:
        for raw_id, text in (("r1", "::note: first"), ("r2", "::note: second"), ("r3", "ordinary text")):
            session = _session(text)
            summary = _IngestBatchSummary()
            payload = SessionWritePayload(
                session_id="codex-session:session",
                content_hash=str(session_content_hash(session)),
                parsed_session=session,
                message_count=1,
                attachment_count=0,
                raw_id=raw_id,
            )
            assert _write_session_entry(index, raw_id, payload, summary=summary, force_write=True)
            summaries.append((raw_id, summary))
        # All index replacements happened before any source delivery.
        for raw_id, summary in summaries:
            asyncio.run(
                _persist_batch_raw_state_updates(
                    service,
                    backend,
                    outcomes={},
                    succeeded_raw_ids={raw_id},
                    skipped_raw_ids=set(),
                    failed_raw_ids={},
                    validation_mode="advisory",
                    marker_sessions_by_raw_id=summary.marker_sessions_by_raw_id,
                )
            )
        asyncio.run(
            _persist_batch_raw_state_updates(
                service,
                backend,
                outcomes={},
                succeeded_raw_ids=set(),
                skipped_raw_ids={"r1", "empty-parse"},
                failed_raw_ids={},
                validation_mode="advisory",
                marker_sessions_by_raw_id=summaries[0][1].marker_sessions_by_raw_id,
            )
        )
    finally:
        index.close()
    with sqlite3.connect(root / "source.db") as source:
        rows = source.execute("SELECT sequence, payload FROM accepted_marker_inputs ORDER BY sequence").fetchall()
        stream_id = source.execute("SELECT stream_id FROM accepted_marker_stream").fetchone()[0]
        assert [row[0] for row in rows] == [1, 2, 3, 4]
        assert json.loads(rows[3][1])["sessions"] == []
        candidates = [json.loads(row[1])["sessions"][0]["candidates"] for row in rows[:3]]
        assert [items[0]["match"]["body"] for items in candidates[:2]] == ["first", "second"]
        assert candidates[2] == []
    with sqlite3.connect(root / "source.db") as source:
        assert source.execute("SELECT stream_id FROM accepted_marker_stream").fetchone()[0] == stream_id


def test_batch_acceptance_conflict_rolls_back_raw_state(workspace_env: dict[str, Path]) -> None:
    """Committing raw acceptance before marker publication makes the counter advance."""

    class CountingRepository(_Repository):
        async def update_raw_state(self, raw_id: str, *, state: object) -> None:
            assert self.source_backend.active is not None
            self.source_backend.active.execute(
                "INSERT INTO test_raw_acceptance VALUES (?, 1) ON CONFLICT(raw_id) DO UPDATE SET accepted = accepted + 1",
                (raw_id,),
            )

    path = workspace_env["archive_root"] / "source.db"
    with sqlite3.connect(path) as source:
        source.execute("CREATE TABLE test_raw_acceptance(raw_id TEXT PRIMARY KEY, accepted INTEGER NOT NULL)")
    backend = _SourceBackend(path)
    service = SimpleNamespace(repository=CountingRepository(backend))

    def accept(candidates: list[dict[str, object]]) -> None:
        asyncio.run(
            _persist_batch_raw_state_updates(
                service,
                backend,
                outcomes={},
                succeeded_raw_ids={"r1"},
                skipped_raw_ids=set(),
                failed_raw_ids={},
                validation_mode="advisory",
                marker_sessions_by_raw_id={"r1": [{"session_id": "s", "candidates": candidates}]},
            )
        )

    accept([])
    with pytest.raises(AcceptedMarkerInputRefusedError, match="conflicting replay"):
        accept([{"body": "changed"}])
    with sqlite3.connect(path) as source:
        assert source.execute("SELECT accepted FROM test_raw_acceptance").fetchone() == (1,)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (1,)


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
            "SELECT type, name, sql FROM sqlite_master WHERE name LIKE 'accepted_marker_%' ORDER BY name"
        ).fetchall()
    with sqlite3.connect(":memory:") as fresh:
        fresh.executescript(SOURCE_DDL)
        assert (
            fresh.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name LIKE 'accepted_marker_%' ORDER BY name"
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
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.executescript(SOURCE_DDL.replace(migration.strip(), ""))
        source.execute("PRAGMA user_version = 4")
        source.commit()
        with pytest.raises(migration_runner.MigrationError, match="requires a verified backup manifest"):
            migration_runner.migrate_archive_tier(source, ArchiveTier.SOURCE, backup_manifest=None)
        assert source.execute("PRAGMA user_version").fetchone() == (4,)
