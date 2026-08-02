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
from polylogue.storage.repair import count_empty_sessions_sync
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


def _insert_message(conn: sqlite3.Connection, *, session_id: str, native_id: str) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
        "VALUES (?, ?, 0, 'user', 1, ?)",
        (session_id, native_id, bytes(32)),
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
    """
    store = BlobStore(archive / "blob")
    legit_raw_id, legit_size = store.write_from_bytes(b'{"type":"summary","sessionId":"legit-empty-1"}\n')
    phantom_raw_id, phantom_size = store.write_from_bytes(b'{"agentType":"general-purpose"}')

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
        source_conn.commit()

    conn = sqlite3.connect(archive / "index.db")
    conn.row_factory = sqlite3.Row
    _insert_session(conn, native_id="legit-empty", raw_id=legit_raw_id)
    _insert_session(conn, native_id="phantom", raw_id=phantom_raw_id)
    _insert_session(conn, native_id="no-provenance", raw_id=None)
    healthy_session_id = _insert_session(conn, native_id="healthy", raw_id=None)
    _insert_message(conn, session_id=healthy_session_id, native_id="m1")
    conn.commit()
    return conn


def test_only_the_positively_refused_artifact_counts_as_debris(archive: Path) -> None:
    conn = _seed(archive)
    try:
        assert count_empty_sessions_sync(conn) == 1
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
        # the no-provenance session too -- three total, not one.
        assert blanket == 3
        assert count_empty_sessions_sync(conn) == 1, (
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
        assert count_empty_sessions_sync(conn) == 1  # only "phantom" -- the right row
    finally:
        conn.close()
