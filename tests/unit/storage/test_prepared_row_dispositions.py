"""Canonical preparation admission, append frontier and counted consumption."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
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
            else replace(prepared.rows, content_occurrence_offsets=(("unseen", 1),))
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
