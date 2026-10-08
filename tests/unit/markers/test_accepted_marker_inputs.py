"""Accepted source inputs retain effects even when the current index advances."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.markers.preparation import marker_candidates_for_prepared_write, marker_recipe_fingerprint
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputRefusedError,
    append_accepted_marker_input,
    finalize_pending_accepted_marker_input,
    prepare_accepted_marker_input,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write
from tests.infra.index_writer import fixture_index_mutation_scope, write_fixture_index_session
from tests.infra.raw_owner_routes import replay_retained_raws
from tests.infra.retained_jsonl import acquire_full_revision
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


def test_prepared_identical_markers_in_one_block_keep_one_stable_candidate(
    workspace_env: dict[str, Path],
) -> None:
    """Repeated syntax with one durable identity is carried once, unchanged."""
    from polylogue.markers.lowering import assertion_id_for_marker, candidates_for_block

    conn = sqlite3.connect(workspace_env["archive_root"] / "index.db")
    conn.row_factory = sqlite3.Row
    try:
        prepared = prepare_session_write(
            conn,
            _session("::note: same lesson\n::note: same lesson", native_id="duplicate-marker"),
            merge_append=False,
        )
        candidates = marker_candidates_for_prepared_write(prepared)
        assert len(candidates) == 1
        provenance = candidates[0]["provenance"]
        assert isinstance(provenance, Mapping)
        match = candidates[0]["match"]
        assert isinstance(match, Mapping)
        expected = candidates_for_block(
            str(provenance["message_id"]),
            str(provenance["block_id"]),
            "::note: same lesson\n::note: same lesson",
        )[0]
        assert match["raw_text"] == expected.match.raw_text
        expected_id = assertion_id_for_marker(expected)
        assert expected_id is not None
        evidence_refs = (f"message:{provenance['message_id']}", f"block:{provenance['block_id']}")
        preserved_id = (
            "marker-"
            + hashlib.sha256(
                "\x1f".join((str(match["kind"]), str(match["raw_text"]), *evidence_refs)).encode()
            ).hexdigest()[:32]
        )
        assert preserved_id == expected_id
    finally:
        conn.close()


def test_prepared_fallback_append_and_lineage_coordinates_match_writer(workspace_env: dict[str, Path]) -> None:
    """A synthetic ordinal ID or ignored append occurrence offset makes this red."""
    from polylogue.storage.io_phase_metrics import connect_measured

    # The Index seal requires the writer's original measured physical creator.
    conn = connect_measured(workspace_env["archive_root"] / "index.db")
    conn.row_factory = sqlite3.Row
    try:
        session = _session("::note: repeated")
        first = prepare_session_write(conn, session, merge_append=False)
        first_candidate = marker_candidates_for_prepared_write(first)[0]
        with fixture_index_mutation_scope(conn):
            write_fixture_index_session(
                conn, session, prepared_write=first, content_hash=first.input_content_hash.hex()
            )
        second = prepare_session_write(conn, session, merge_append=True)
        second_candidate = marker_candidates_for_prepared_write(second)[0]
        assert first_candidate["provenance"] != second_candidate["provenance"]
        with fixture_index_mutation_scope(conn):
            write_fixture_index_session(
                conn, session, prepared_write=second, merge_append=True, content_hash=second.input_content_hash.hex()
            )
        stored = {row[0] for row in conn.execute("SELECT block_id FROM blocks")}
        assert _nested(first_candidate, "provenance", "block_id") in stored
        assert _nested(second_candidate, "provenance", "block_id") in stored

        parent = _session("::note: parent", native_id="parent", message_id="p")
        write_fixture_index_session(conn, parent)
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
        with fixture_index_mutation_scope(conn):
            write_fixture_index_session(
                conn, child, prepared_write=prepared_child, content_hash=prepared_child.input_content_hash.hex()
            )
        stored = {
            row[0] for row in conn.execute("SELECT block_id FROM blocks WHERE session_id = 'codex-session:child'")
        }
        assert {_nested(candidate, "provenance", "block_id") for candidate in candidates} == stored
    finally:
        conn.close()


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


def test_retained_owner_publishes_marker_carrier_from_the_accepted_raw(
    tmp_path: Path, workspace_env: dict[str, Path]
) -> None:
    """The production retained owner binds marker content and custody to its accepted raw."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "retained-marker.jsonl"
    native_id = "retained-marker"
    payload = (
        b"\n".join(
            (
                json.dumps(
                    {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-01-01T00:00:00Z"}},
                    sort_keys=True,
                ).encode(),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": "retained-marker-message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "::note: retained owner note"}],
                        },
                    },
                    sort_keys=True,
                ).encode(),
            )
        )
        + b"\n"
    )

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = acquire_full_revision(
            archive,
            provider=Provider.CODEX,
            source_path=source_path,
            payload=payload,
            native_id=native_id,
            generation=0,
            acquired_at_ms=1,
        )

    replay_retained_raws(archive_root, (raw_id,))

    state = _accepted_marker_state(archive_root / "source.db", raw_id)
    assert state is not None
    sequence, carrier_bytes, _index_incarnation_id = state
    assert sequence == 1
    carrier = json.loads(cast(bytes, carrier_bytes))
    assert carrier["raw_id"] == raw_id
    assert carrier["request_facts"]["source_path"] == str(source_path)
    assert carrier["request_facts"]["provider"] == Provider.CODEX.value
    assert [session["session_id"] for session in carrier["request_sessions"]] == [f"codex-session:{native_id}"]
    candidates = carrier["sessions"][0]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["match"]["body"] == "retained owner note"
    with sqlite3.connect(archive_root / "index.db") as index:
        retained_message = index.execute(
            "SELECT message_id FROM messages WHERE session_id = ?", (f"codex-session:{native_id}",)
        ).fetchone()
    assert retained_message is not None
    assert candidates[0]["provenance"]["message_id"] == retained_message[0]

    with sqlite3.connect(archive_root / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        parsed_at = source.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
        assert parsed_at is not None and parsed_at[0] is not None
