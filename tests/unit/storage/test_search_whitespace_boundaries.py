"""Legal FTS separators preserve the selected relation on both search routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.storage.search.runtime import search_messages_impl
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.queries.sessions_search import search_session_evidence_hits
from tests.infra.identity import archive_message_id


@pytest.fixture
def searchable_archive(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        conn = archive._conn
        for native_id, text in (("separate", "alpha beta"), ("joined", "alphabeta")):
            session_id = f"unknown-export:{native_id}"
            conn.execute(
                "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, 'unknown-export', ?)",
                (native_id, bytes(32)),
            )
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash) "
                "VALUES (?, 'm0', 0, 'user', 'message', ?)",
                (session_id, bytes(32)),
            )
            conn.execute(
                "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
                (archive_message_id(session_id, "m0"), session_id, text),
            )
        conn.commit()
        return root, archive.index_db_path


@pytest.mark.parametrize("separator", [" ", "\t", "\n", "\r", "\f", "\v"])
def test_message_search_preserves_whitespace_boundaries(searchable_archive: tuple[Path, Path], separator: str) -> None:
    """Deleting the separator selects the distinct concatenated-token control."""
    root, index = searchable_archive
    result = search_messages_impl(f"alpha{separator}beta", root, index, 20, None, None)
    assert [hit.session_id for hit in result.hits] == ["unknown-export:separate"]


@pytest.mark.parametrize("separator", [" ", "\t", "\n", "\r", "\f", "\v"])
async def test_evidence_search_preserves_whitespace_boundaries(
    searchable_archive: tuple[Path, Path], separator: str
) -> None:
    _, index = searchable_archive
    async with aiosqlite.connect(index) as conn:
        conn.row_factory = sqlite3.Row
        hits = await search_session_evidence_hits(conn, f"alpha{separator}beta", 20)
        assert [hit.session_id for hit in hits] == ["unknown-export:separate"]
        assert hits[0].matched_terms == ("alpha", "beta")
        assert not conn.in_transaction
