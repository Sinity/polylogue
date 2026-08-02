"""Empty-session repair must delete positively-classified debris, never a
session that is merely message-less.

History (polylogue-ne6k): the original ``repair_empty_sessions`` predicate was
a blanket ``NOT EXISTS (messages)`` join, which cannot distinguish corruption
debris from a session that is legitimately empty -- the 2026-07-22
hook-inflation postmortem explicitly chose to RETAIN ~832 such sessions after
de-inflation.

A first fix attempt tried ``raw_id IS NULL`` as the discriminator ("no
acquired bytes behind it = illegitimate"). That was tried and REFUTED: measured
on the live archive, all 5,257 message-less sessions carry a non-empty
``raw_id`` -- 4,945 of them are ``<agent>.meta`` sidecar phantoms that were
genuinely acquired (the sidecar file really was read), so "acquisition
happened" cannot separate phantoms from legitimate stubs.

The real discriminator is WHAT THE ACQUIRED ARTIFACT IS, not whether bytes
were acquired: re-run each candidate's raw bytes through the same
``classify_artifact``/``inspect_raw_artifact`` pipeline live ingest uses, and
only delete the ones that pipeline still refuses to admit as a session. This
mirrors the ``looks_like_code`` fix in
``sources/parsers/claude/code_detection.py`` (polylogue-9ykn/gvgi): a genuine
positive marker is required, never a location- or absence-based guess.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.repair import _empty_session_debris_session_ids, count_empty_sessions_sync
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_ORIGIN = "claude-code-session"


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A real split source.db/index.db pair plus a blob store, all under tmp_path."""
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    blob_root = tmp_path / "blob"
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: blob_root, raising=False)
    reset_blob_store()
    yield tmp_path
    reset_blob_store()


def _insert_raw_session(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    native_id: str,
    source_path: str,
    blob_size: int,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
        ) VALUES (?, ?, ?, ?, 0, ?, ?, 1)
        """,
        (raw_id, _ORIGIN, native_id, source_path, bytes.fromhex(raw_id), blob_size),
    )


def _insert_session(conn: sqlite3.Connection, *, native_id: str, raw_id: str | None) -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
        (native_id, _ORIGIN, raw_id, "Test", bytes(32)),
    )
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE native_id = ? AND origin = ?",
        (native_id, _ORIGIN),
    ).fetchone()
    return str(row[0])


def _insert_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    native_id: str,
    position: int = 0,
    word_count: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
        "VALUES (?, ?, ?, 'user', ?, ?)",
        (session_id, native_id, position, word_count, bytes(32)),
    )


def _update_session_word_count(conn: sqlite3.Connection, *, session_id: str, word_count: int) -> None:
    conn.execute(
        "UPDATE sessions SET word_count = ? WHERE session_id = ?",
        (word_count, session_id),
    )


def _seed(archive: Path) -> sqlite3.Connection:
    """Seed one of every candidate shape and return an open index.db connection.

    - ``legit-empty``: raw bytes the current classifier still admits as a
      session (a genuine Claude Code record with a ``sessionId`` marker) --
      the shape the hook-inflation postmortem retained ~832 of.
    - ``phantom``: raw bytes an ``agent-*.meta.json`` sidecar path -- the
      4,945-row phantom class that is genuinely debris.
    - ``no-provenance``: no ``raw_id`` at all -- no evidence either way, so it
      must be retained, not treated as debris by default.
    - ``healthy``: has a message, excluded regardless of provenance.
    - ``all-empty-content-phantom``: many message rows (like a real
      transcript), but every one carries zero words, backed by a
      relationship-index-shaped raw artifact under an ``analysis/``
      directory (polylogue-21qj's ``conversation_relationships.jsonl``
      shape) -- debris, even though it is not literally message-less.
    - ``all-empty-content-legit``: the same "many messages, all zero words"
      shape, but backed by a genuine Claude Code record -- must be retained,
      proving the broadened candidate query does not sweep in a legitimate
      all-tool-use session with no text turns.
    """
    store = BlobStore(archive / "blob")
    legit_raw_id, legit_size = store.write_from_bytes(b'{"type":"summary","sessionId":"legit-empty-1"}\n')
    phantom_raw_id, phantom_size = store.write_from_bytes(b'{"agentType":"general-purpose"}')
    relationship_index_bytes = (
        b'{"conversation": "conv-1", "parent": "p-1", "child": "c-1", "type": "assistant", '
        b'"timestamp": "2026-05-01T00:00:00.000Z"}\n'
    )
    relationship_index_raw_id, relationship_index_size = store.write_from_bytes(relationship_index_bytes)
    legit_transcript_bytes = b'{"type":"summary","sessionId":"legit-all-empty-1"}\n'
    legit_transcript_raw_id, legit_transcript_size = store.write_from_bytes(legit_transcript_bytes)

    with sqlite3.connect(archive / "source.db") as source_conn:
        _insert_raw_session(
            source_conn,
            raw_id=legit_raw_id,
            native_id="legit-empty",
            source_path="conversation.jsonl",
            blob_size=legit_size,
        )
        _insert_raw_session(
            source_conn,
            raw_id=phantom_raw_id,
            native_id="phantom",
            source_path="agent-1234.meta.json",
            blob_size=phantom_size,
        )
        _insert_raw_session(
            source_conn,
            raw_id=relationship_index_raw_id,
            native_id="conversation_relationships",
            source_path="analysis/index/conversation_relationships.jsonl",
            blob_size=relationship_index_size,
        )
        _insert_raw_session(
            source_conn,
            raw_id=legit_transcript_raw_id,
            native_id="legit-all-empty",
            source_path="conversation-all-empty.jsonl",
            blob_size=legit_transcript_size,
        )
        source_conn.commit()

    conn = sqlite3.connect(archive / "index.db")
    conn.row_factory = sqlite3.Row
    _insert_session(conn, native_id="legit-empty", raw_id=legit_raw_id)
    _insert_session(conn, native_id="phantom", raw_id=phantom_raw_id)
    _insert_session(conn, native_id="no-provenance", raw_id=None)
    healthy_session_id = _insert_session(conn, native_id="healthy", raw_id=None)
    _insert_message(conn, session_id=healthy_session_id, native_id="m1")

    phantom_many_id = _insert_session(conn, native_id="conversation_relationships", raw_id=relationship_index_raw_id)
    for index in range(3):
        _insert_message(conn, session_id=phantom_many_id, native_id=f"cr-{index}", position=index, word_count=0)
    _update_session_word_count(conn, session_id=phantom_many_id, word_count=0)

    legit_many_id = _insert_session(conn, native_id="legit-all-empty", raw_id=legit_transcript_raw_id)
    for index in range(3):
        _insert_message(conn, session_id=legit_many_id, native_id=f"le-{index}", position=index, word_count=0)
    _update_session_word_count(conn, session_id=legit_many_id, word_count=0)

    conn.commit()
    return conn


def test_only_the_positively_refused_artifact_counts_as_debris(archive: Path) -> None:
    conn = _seed(archive)
    try:
        # "phantom" (message-less) and "conversation_relationships" (many
        # messages, all zero words) -- both positively refused by the
        # current classifier. "legit-empty" and "legit-all-empty" are the
        # sibling shapes backed by a genuine record and must be excluded.
        assert count_empty_sessions_sync(conn) == 2
    finally:
        conn.close()


def test_all_empty_content_session_with_phantom_raw_counts_as_debris(archive: Path) -> None:
    """polylogue-21qj: a session with real message rows that all carry zero
    words (the ``conversation_relationships`` shape -- 96,748 message rows on
    the live archive, none with any block/word content) must be treated as
    debris when its raw artifact is a relationship-index-shaped, ``analysis/``
    -directory file that the current classifier positively refuses. The prior
    predicate (``NOT EXISTS messages``) could never see this session at all,
    since it is not literally message-less.
    """
    conn = _seed(archive)
    try:
        debris_ids = _empty_session_debris_session_ids(conn)
        assert "claude-code-session:conversation_relationships" in debris_ids
    finally:
        conn.close()


def test_all_empty_content_session_with_legit_raw_is_retained(archive: Path) -> None:
    """Sibling of the test above: the identical 'many messages, all zero
    words' shape must NOT become debris when its raw artifact is a genuine
    Claude Code record (e.g. an all-tool-use session with no text turns) --
    the widened candidate query only ever proposes candidates; the classifier
    gate is what actually decides, and it must still retain a legitimate
    zero-word session.
    """
    conn = _seed(archive)
    try:
        debris_ids = _empty_session_debris_session_ids(conn)
        assert "claude-code-session:legit-all-empty" not in debris_ids
    finally:
        conn.close()


def test_blanket_predicate_would_have_swept_up_legitimate_sessions(archive: Path) -> None:
    """Pin the defect itself: a revert to the old blanket predicate must fail this."""
    conn = _seed(archive)
    try:
        blanket = int(
            conn.execute(
                "SELECT COUNT(*) FROM sessions s "
                "WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.session_id)"
            ).fetchone()[0]
        )
        # The blanket predicate sweeps in the legitimately-empty session and
        # the no-provenance session too -- three total, not two -- and it
        # cannot see "conversation_relationships"/"legit-all-empty" at all
        # (they are not message-less).
        assert blanket == 3
        assert count_empty_sessions_sync(conn) == 2, (
            "classifier-aware count must exclude the legitimate and no-provenance sessions"
        )
    finally:
        conn.close()


def test_raw_id_is_null_predicate_would_have_missed_the_phantom(archive: Path) -> None:
    """Pin the other refuted defect (polylogue-ne6k correction): the phantom
    row carries a non-empty raw_id, so 'debris = raw_id IS NULL' reports zero
    debris here too, exactly like the live-archive measurement that refuted
    it."""
    conn = _seed(archive)
    try:
        raw_id_null_predicate = int(
            conn.execute(
                "SELECT COUNT(*) FROM sessions s "
                "WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.session_id) "
                "AND (s.raw_id IS NULL OR s.raw_id = '')"
            ).fetchone()[0]
        )
        assert raw_id_null_predicate == 1  # only "no-provenance" -- the wrong row
        assert count_empty_sessions_sync(conn) == 2  # "phantom" + "conversation_relationships" -- the right rows
    finally:
        conn.close()
