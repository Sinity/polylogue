"""Canonical preparation admission, append frontier and counted consumption."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    PREPARED_ACCEPTED_DISPOSITIONS,
    PreparedSessionWriteRefusedError,
    prepare_session_write,
    prepared_row_dispositions,
    reset_prepared_row_dispositions,
)
from tests.infra.index_writer import fixture_index_mutation_scope, write_fixture_index_session


@pytest.fixture(autouse=True)
def _fresh_counters() -> None:
    reset_prepared_row_dispositions()


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _message(native_id: str, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=Role.USER,
        text=text,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        position=position,
    )


def _session(provider_session_id: str, texts: list[str], *, id_offset: int = 0) -> ParsedSession:
    """``id_offset`` keeps an appended turn's native ids distinct from the stored ones."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=provider_session_id,
        title="dispositions",
        messages=[_message(f"m{index + id_offset}", text, index) for index, text in enumerate(texts)],
    )


def test_canonical_write_is_counted_and_consumed(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = _session("consumed", ["one", "two"])
    try:
        prepared = prepare_session_write(conn, session, merge_append=False)
        try:
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(
                    conn, session, prepared_write=prepared, content_hash=prepared.input_content_hash.hex()
                )
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        finally:
            prepared.close()
    finally:
        conn.close()
    assert prepared_row_dispositions() == {"prepared_write": 1}
    assert set(prepared_row_dispositions()) <= PREPARED_ACCEPTED_DISPOSITIONS


def test_changed_input_refuses_before_rows(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    original = _session("mutates", ["original"])
    mutated = original.model_copy(update={"messages": [_message("m0", "changed", 0)]})
    try:
        prepared = prepare_session_write(conn, original, merge_append=False)
        try:
            with pytest.raises(PreparedSessionWriteRefusedError):
                with fixture_index_mutation_scope(conn):
                    write_fixture_index_session(
                        conn,
                        mutated,
                        prepared_write=prepared,
                        content_hash=str(session_content_hash(mutated)),
                        pending_input_content_hash=str(session_content_hash(mutated)),
                    )
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        finally:
            prepared.close()
    finally:
        conn.close()


def test_append_matching_frontier_consumes_preparation(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    first = _session("append", ["one"])
    appended = _session("append", ["two"], id_offset=1)
    try:
        write_fixture_index_session(conn, first)
        reset_prepared_row_dispositions()
        prepared = prepare_session_write(conn, appended, merge_append=True)
        try:
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(
                    conn,
                    appended,
                    prepared_write=prepared,
                    merge_append=True,
                    content_hash=prepared.input_content_hash.hex(),
                )
            assert [row[0] for row in conn.execute("SELECT position FROM messages ORDER BY position")] == [0, 1]
        finally:
            prepared.close()
    finally:
        conn.close()
    assert prepared_row_dispositions() == {"prepared_write": 1}


@pytest.mark.parametrize("mutation", ["position", "occurrences"])
def test_append_changed_frontier_refuses_without_relowering(tmp_path: Path, mutation: str) -> None:
    conn = _connect(tmp_path / "index.db")
    first = _session("frontier", ["one"])
    appended = _session("frontier", ["two"], id_offset=1)
    try:
        write_fixture_index_session(conn, first)
        prepared = prepare_session_write(conn, appended, merge_append=True)
        rows = (
            replace(prepared.rows, position_offset=0)
            if mutation == "position"
            else replace(prepared.rows, occurrence_frontier=None)
        )
        stale = replace(prepared, rows=rows)
        try:
            with pytest.raises(PreparedSessionWriteRefusedError):
                with fixture_index_mutation_scope(conn):
                    write_fixture_index_session(
                        conn,
                        appended,
                        prepared_write=stale,
                        merge_append=True,
                        content_hash=prepared.input_content_hash.hex(),
                    )
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        finally:
            prepared.close()
    finally:
        conn.close()


def test_changed_predecessor_refuses_before_replacement(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = _session("predecessor", ["one"])
    try:
        prepared = prepare_session_write(conn, session, merge_append=False)
        write_fixture_index_session(conn, session)
        try:
            with pytest.raises(PreparedSessionWriteRefusedError):
                with fixture_index_mutation_scope(conn):
                    write_fixture_index_session(
                        conn, session, prepared_write=prepared, content_hash=prepared.input_content_hash.hex()
                    )
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        finally:
            prepared.close()
    finally:
        conn.close()


def test_dispositions_accumulate_for_published_preparations(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        for index in range(3):
            write_fixture_index_session(conn, _session(f"sum-{index}", ["body"]))
    finally:
        conn.close()
    assert prepared_row_dispositions() == {"prepared_write": 3}


def test_prefix_sharing_child_consumes_already_sliced_preparation(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent = _session("slice-parent", ["A", "B"])
    child = _session("slice-child", ["A", "B", "C"]).model_copy(update={"parent_session_provider_id": "slice-parent"})
    try:
        write_fixture_index_session(conn, parent)
        prepared = prepare_session_write(conn, child, merge_append=False)
        try:
            assert len(prepared.context.messages) == 1
            assert len(prepared.rows.content_identities) == 1
            reset_prepared_row_dispositions()
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(
                    conn, child, prepared_write=prepared, content_hash=prepared.input_content_hash.hex()
                )
            assert (
                conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (prepared.session_id,)).fetchone()[0]
                == 1
            )
            assert (
                conn.execute(
                    "SELECT branch_point_message_id FROM session_links WHERE src_session_id=?", (prepared.session_id,)
                ).fetchone()[0]
                is not None
            )
        finally:
            prepared.close()
    finally:
        conn.close()
    assert prepared_row_dispositions() == {"prepared_write": 1}


def test_append_after_late_parent_reextraction_preserves_occurrence_ids(tmp_path: Path) -> None:
    """Removing an inherited prefix leaves holes, not reusable message IDs."""
    conn = _connect(tmp_path / "index.db")
    repeated = [_message("", "same semantics", ordinal) for ordinal in range(3)]
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="occurrence-child",
        parent_session_provider_id="occurrence-parent",
        branch_type=BranchType.CONTINUATION,
        messages=repeated,
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="occurrence-parent",
        messages=repeated[:2],
    )
    try:
        child_id = write_fixture_index_session(conn, child)
        original_tail = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND content_occurrence = 2", (child_id,)
        ).fetchone()[0]
        write_fixture_index_session(conn, parent)
        assert conn.execute(
            "SELECT content_occurrence FROM messages WHERE session_id = ? ORDER BY position", (child_id,)
        ).fetchall() == [(2,)]
        for expected_occurrence in (3, 4):
            tail = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="occurrence-child",
                messages=[_message("", "same semantics", 0)],
            )
            prepared = archive_tier_write.prepare_session_write(conn, tail, merge_append=True)
            try:
                with fixture_index_mutation_scope(conn):
                    write_fixture_index_session(
                        conn,
                        tail,
                        merge_append=True,
                        prepared_write=prepared,
                    )
            finally:
                prepared.close()
            assert (
                conn.execute(
                    "SELECT MAX(content_occurrence) FROM messages WHERE session_id = ?", (child_id,)
                ).fetchone()[0]
                == expected_occurrence
            )
        assert (
            conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? AND content_occurrence = 2", (child_id,)
            ).fetchone()[0]
            == original_tail
        )
    finally:
        conn.close()


def test_append_occurrence_hole_keeps_existing_ids_and_uses_next_free_id(tmp_path: Path) -> None:
    """A retained 0/2 frontier cannot reuse the existing occurrence 2."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="occurrence-hole",
        messages=[_message("", "same semantics", ordinal) for ordinal in range(3)],
    )
    try:
        session_id = write_fixture_index_session(conn, session)
        # Model the row hole left by a legitimate removal; IDs of surviving
        # rows remain canonical and must survive the actual prepared append.
        with fixture_index_mutation_scope(conn):
            conn.execute("DELETE FROM messages WHERE session_id = ? AND content_occurrence = 1", (session_id,))
        before = conn.execute(
            "SELECT content_occurrence, message_id FROM messages WHERE session_id = ? ORDER BY content_occurrence",
            (session_id,),
        ).fetchall()
        assert [occurrence for occurrence, _message_id in before] == [0, 2]
        tail = session.model_copy(update={"messages": [_message("", "same semantics", 0)]})
        prepared = archive_tier_write.prepare_session_write(conn, tail, merge_append=True)
        try:
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(conn, tail, merge_append=True, prepared_write=prepared)
        finally:
            prepared.close()
        after = conn.execute(
            "SELECT content_occurrence, message_id FROM messages WHERE session_id = ? ORDER BY content_occurrence",
            (session_id,),
        ).fetchall()
        assert after[:2] == before
        assert after[2][0] == 3
    finally:
        conn.close()


def test_append_frontier_certificate_refuses_new_variant_at_same_position(tmp_path: Path) -> None:
    """A canonical replacement can move occurrences without moving max position."""
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="occurrence-stale",
        messages=[_message("", "same semantics", ordinal) for ordinal in range(3)],
    )
    prepared = None
    try:
        session_id = write_fixture_index_session(conn, session)
        tail = session.model_copy(update={"messages": [_message("", "same semantics", 0)]})
        prepared = archive_tier_write.prepare_session_write(conn, tail, merge_append=True)
        pinned_position = prepared.rows.position_offset
        gained_variant = session.model_copy(
            update={
                "messages": [
                    *session.messages,
                    _message("", "same semantics", 2).model_copy(update={"variant_index": 1}),
                ],
            }
        )
        write_fixture_index_session(conn, gained_variant)
        assert archive_tier_write._next_message_position(conn, session_id) == pinned_position
        with pytest.raises(archive_tier_write.PreparedSessionWriteRefusedError):
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(conn, tail, merge_append=True, prepared_write=prepared)
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 4
    finally:
        if prepared is not None:
            prepared.close()
        conn.close()


def test_prepared_tail_uses_indexed_incoming_keys_without_history_census(tmp_path: Path) -> None:
    """Both preparation and settlement avoid enumerating historical digests."""
    conn = _connect(tmp_path / "index.db")
    session = _session("indexed-tail", [f"history {ordinal}" for ordinal in range(128)])
    prepared = None
    historical_census = []
    try:
        session_id = write_fixture_index_session(conn, session)
        tail = _session("indexed-tail", ["new tail"], id_offset=128)
        conn.set_trace_callback(
            lambda sql: (
                historical_census.append(sql)
                if (
                    "COUNT(DISTINCT content_identity)" in sql
                    or "GROUP BY content_identity" in sql
                    or "SELECT DISTINCT content_identity" in sql
                )
                else None
            )
        )
        prepared = archive_tier_write.prepare_session_write(conn, tail, merge_append=True)
        assert historical_census == []
        assert prepared.rows.occurrence_frontier is not None
        # Full verification runs before writer admission. It must not repeat
        # when the canonical writer consumes this pinned evidence.
        prepared.rows.occurrence_frontier.verify(full=True)
        with fixture_index_mutation_scope(conn):
            write_fixture_index_session(conn, tail, merge_append=True, prepared_write=prepared)
        assert historical_census == []
        query_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT content_occurrence FROM messages "
            "WHERE session_id = ? AND content_identity = ? ORDER BY content_occurrence DESC LIMIT 1",
            (session_id, "neutral-digest"),
        ).fetchall()
        assert any("idx_messages_session_content_identity" in str(row) for row in query_plan)
    finally:
        if prepared is not None:
            prepared.close()
        conn.close()


def test_prepared_admission_verifies_append_frontier_bytes(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = _session("sealed-frontier", ["neutral"])
    prepared = prepare_session_write(conn, session, merge_append=True)
    try:
        frontier = prepared.rows.occurrence_frontier
        assert frontier is not None
        prepared.verify_files(full=True)
        forged = replace(frontier, seal=replace(frontier.seal, sha256="x" * 64))
        forged_write = replace(prepared, rows=replace(prepared.rows, occurrence_frontier=forged))
        forged_write.verify_files(full=False)
        with pytest.raises(PreparedSessionWriteRefusedError, match="frontier changed"):
            forged_write.verify_files(full=True)
    finally:
        prepared.close()
        conn.close()
