"""Search admits and reads its hits from the snapshot its FTS verdict measured."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.search import runtime
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.snapshot_probe import CommitBetweenStatements


def _text_session(native_id: str, text: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-u1",
                role=Role.USER,
                text=text,
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            )
        ],
    )


def _session_ids(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}


def test_search_hits_come_from_the_snapshot_its_readiness_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit after the FTS admission count is invisible to the admitted query.

    ``search_messages_impl`` inspects FTS readiness with several COUNT
    statements and then runs the ranked query. A session committed right after
    the first readiness count was never part of the verdict that admitted the
    query, so it must not appear among the hits.

    Anti-vacuity: drop the ``BEGIN`` in ``search_messages_impl`` and the ranked
    query reads the later commit, returning the second session as well.
    """
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as facade:
        writer = facade._conn
        assert str(writer.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        write_parsed_session_to_archive(writer, _text_session("admitted-first", "quick fox before admission"))
        writer.commit()
        admitted = _session_ids(writer)
        db_path = facade.index_db_path

        def concurrent_commit() -> None:
            write_parsed_session_to_archive(writer, _text_session("admitted-second", "quick fox after admission"))
            writer.commit()

        real_open = runtime.open_read_connection
        probes: list[CommitBetweenStatements] = []

        @contextmanager
        def open_probe(path: Path | str | None = None) -> Iterator[CommitBetweenStatements]:
            with real_open(path) as conn:
                probe = CommitBetweenStatements(
                    conn,
                    trigger_sql="SELECT COUNT(*) FROM blocks WHERE search_text != ''",
                    commit=concurrent_commit,
                )
                probes.append(probe)
                yield probe

        monkeypatch.setattr(runtime, "open_read_connection", open_probe)
        result = runtime.search_messages_impl(
            query="quick fox",
            archive_root=archive_root,
            db_path=db_path,
            limit=100,
            source=None,
            since=None,
        )
        committed = _session_ids(writer)

    assert [probe.fired for probe in probes] == [True]
    assert committed > admitted
    assert result.hits
    assert {hit.session_id for hit in result.hits} == admitted
