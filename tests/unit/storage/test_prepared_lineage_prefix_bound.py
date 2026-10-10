"""Prepared publication consumes exactly its inherited parent signatures."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.index_writer import fixture_index_mutation_scope, write_fixture_index_session


def _session(native_id: str, count: int, *, parent: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        parent_session_provider_id=parent,
        messages=[
            ParsedMessage(provider_message_id=f"m-{position}", role=Role.USER, text=f"message {position}")
            for position in range(count)
        ],
    )


def _child(root: Path, *, disk: bool, attachment: ParsedAttachment | None = None) -> ParsedSession:
    session = _session("child", 11, parent="parent")
    session.messages[10] = ParsedMessage(provider_message_id="tail", role=Role.USER, text="child tail")
    if attachment is not None:
        session = session.model_copy(update={"attachments": [attachment]})
    if not disk:
        return session
    store = SqliteMessageStore(root / "child-messages.db")
    sink = store.new_sink()
    sink.extend(session.messages)
    sink.normalized_messages(session.session_events, origin=Origin.CODEX_SESSION)
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    return session.model_copy(update={"messages": sealed, "content_hash": str(session_content_hash(session))})


def _connection(root: Path) -> sqlite3.Connection:
    conn = connect_measured(root / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


@pytest.mark.parametrize("disk", (False, True))
def test_prepared_lineage_publication_consumes_only_inherited_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disk: bool
) -> None:
    with closing(_connection(tmp_path)) as conn:
        parent_id = write_fixture_index_session(conn, _session("parent", 128))
        child = _child(tmp_path, disk=disk)
        consumed: list[str] = []
        spooled: list[str] = []
        original_rows = write._iter_composed_rows
        original_append = write._DiskSignatureSequence.append
        original_validate = write.validate_prepared_session_lineage
        validating = False

        def rows(
            current: sqlite3.Connection,
            session_id: str,
            before_input: write.BeforeIndexInput | None = None,
        ) -> Generator[tuple[str, str, str], None, None]:
            with closing(original_rows(current, session_id, before_input)) as stream:
                for row in stream:
                    if validating and session_id == parent_id:
                        consumed.append(row[0])
                    yield row

        def append(sequence: write._DiskSignatureSequence, message_id: str, digest: str) -> None:
            if validating:
                spooled.append(message_id)
            original_append(sequence, message_id, digest)

        def validate(
            current: sqlite3.Connection,
            session: ParsedSession,
            prepared: write.PreparedSessionWrite,
            *,
            source_read: write.SessionSourceRead | None,
        ) -> None:
            nonlocal validating
            assert current.in_transaction
            assert len(session.messages) - len(prepared.context.messages) == 10
            validating = True
            try:
                original_validate(current, session, prepared, source_read=source_read)
            finally:
                validating = False

        monkeypatch.setattr(write, "_iter_composed_rows", rows)
        monkeypatch.setattr(write._DiskSignatureSequence, "append", append)
        monkeypatch.setattr(write, "validate_prepared_session_lineage", validate)
        child_id = write_fixture_index_session(conn, child)
        expected = [
            str(row[0])
            for row in conn.execute(
                "SELECT message_id FROM messages WHERE session_id=? ORDER BY position LIMIT 10", (parent_id,)
            )
        ]
        assert consumed == expected
        assert spooled == (expected if disk else [])
        assert conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (child_id,)).fetchone()[0] == 1
        assert conn.execute(
            "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id=?", (child_id,)
        ).fetchone()[:] == ("prefix-sharing", expected[-1])


@pytest.mark.parametrize("disk", (False, True))
@pytest.mark.parametrize("change", ("tail", "prefix", "shorter"))
def test_prepared_prefix_rechecks_current_parent_and_accepts_only_tail_changes(
    tmp_path: Path, disk: bool, change: str
) -> None:
    with closing(_connection(tmp_path)) as conn:
        parent = _session("parent", 16)
        write_fixture_index_session(conn, parent)
        child = _child(tmp_path, disk=disk)
        prepared = write.prepare_session_write(conn, child, merge_append=False)
        try:
            if change == "shorter":
                replacement = parent.model_copy(update={"messages": list(parent.messages[:9])})
            else:
                position = 15 if change == "tail" else 0
                messages = list(parent.messages)
                messages[position] = messages[position].model_copy(update={"text": "changed current parent"})
                replacement = parent.model_copy(update={"messages": messages})
            write_fixture_index_session(conn, replacement, force_replace=True)
            with fixture_index_mutation_scope(conn):
                if change == "tail":
                    child_id = write_fixture_index_session(conn, child, prepared_write=prepared)
                    assert (
                        conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (child_id,)).fetchone()[0] == 1
                    )
                else:
                    with pytest.raises(write.PreparedSessionWriteRefusedError, match="lineage prefix changed"):
                        write_fixture_index_session(conn, child, prepared_write=prepared)
                    assert conn.execute("SELECT 1 FROM sessions WHERE native_id='child'").fetchone() is None
        finally:
            prepared.close()


def test_bounded_prepared_prefix_rechecks_inherited_attachment_ownership(tmp_path: Path) -> None:
    attachment = ParsedAttachment(provider_attachment_id="shared-file", message_provider_id="m-9", name="shared.txt")
    with closing(_connection(tmp_path)) as conn:
        parent = _session("parent", 16).model_copy(update={"attachments": [attachment]})
        write_fixture_index_session(conn, parent)
        child = _child(tmp_path, disk=True, attachment=attachment)
        prepared = write.prepare_session_write(conn, child, merge_append=False)
        try:
            assert len(prepared.context.messages) == 1
            write_fixture_index_session(conn, parent.model_copy(update={"attachments": []}), force_replace=True)
            with fixture_index_mutation_scope(conn):
                with pytest.raises(write.PreparedSessionWriteRefusedError, match="lineage prefix attachments changed"):
                    write_fixture_index_session(conn, child, prepared_write=prepared)
                assert conn.execute("SELECT 1 FROM sessions WHERE native_id='child'").fetchone() is None
        finally:
            prepared.close()


@pytest.mark.parametrize("disk", (False, True))
def test_bounded_prepared_prefix_refuses_changed_parent_binding(tmp_path: Path, disk: bool) -> None:
    with closing(_connection(tmp_path)) as conn:
        parent_id = write_fixture_index_session(conn, _session("parent", 16))
        child = _child(tmp_path, disk=disk)
        prepared = write.prepare_session_write(conn, child, merge_append=False)
        try:
            with fixture_index_mutation_scope(conn):
                conn.execute("DELETE FROM sessions WHERE session_id=?", (parent_id,))
                with pytest.raises(write.PreparedSessionWriteRefusedError, match="lineage evidence changed"):
                    write_fixture_index_session(conn, child, prepared_write=prepared)
                assert conn.execute("SELECT 1 FROM sessions WHERE native_id='child'").fetchone() is None
        finally:
            prepared.close()
