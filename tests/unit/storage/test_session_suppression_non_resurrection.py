"""The archive must not resurrect a session the operator tombstoned.

Identity-preserving reset keeps the acquired raw evidence in ``source.db``
and records the deletion as a durable ``user.db`` suppression assertion,
dropping only the rebuildable ``index.db`` row. A rebuild/replay reads that
retained raw evidence and writes it back through
``write_parsed_session_to_archive``, so the suppression has to be honoured
there.

Anti-vacuity: every test here fails if the suppression check in
``write_parsed_session_to_archive`` is removed (the session row reappears)
or if it is made silent by dropping ``record_suppression_refusal`` /
``ArchiveWriteOutcome.suppression_skipped`` (the refusal stops being
counted, which would trade the privacy defect for a silent-loss defect).
Deleting the write itself would not make these green: an unsuppressed
sibling session in the same replay is asserted to still be written.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.session_suppression import (
    reset_suppression_caches,
    suppression_refusal_scope,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_suppression
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveWriteOutcome,
    write_parsed_session_to_archive,
)


@pytest.fixture(autouse=True)
def _drop_pooled_user_handles() -> object:
    reset_suppression_caches()
    yield
    reset_suppression_caches()


def _parsed(native_id: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id=native_id,
        title=f"Session {native_id}",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-msg-1",
                role=Role.USER,
                text="content the operator deleted",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )


def _session_id(native_id: str) -> str:
    return str(archive_session_id(origin_from_provider(Provider.UNKNOWN).value, native_id))


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    index_db = ArchiveLocation.resolve(root).active_index_path
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    return root


def _index_path(archive_root: Path) -> Path:
    return ArchiveLocation.resolve(archive_root).active_index_path


def _tombstone(archive_root: Path, session_id: str) -> None:
    """Record the durable suppression a reset writes, without deleting raws."""
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        upsert_suppression(conn, session_id=session_id, reason="reset --session", mode="hide")


def _replay(archive_root: Path, session: ParsedSession) -> ArchiveWriteOutcome:
    """Write one parsed session exactly as a rebuild's replay does."""
    outcomes: list[ArchiveWriteOutcome] = []
    conn = sqlite3.connect(_index_path(archive_root))
    conn.row_factory = sqlite3.Row
    try:
        write_parsed_session_to_archive(conn, session, write_outcome=outcomes)
    finally:
        conn.close()
    assert len(outcomes) == 1
    return outcomes[0]


def _session_rows(archive_root: Path, session_id: str) -> int:
    with sqlite3.connect(_index_path(archive_root)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0])


def _message_rows(archive_root: Path, session_id: str) -> int:
    with sqlite3.connect(_index_path(archive_root)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0])


def test_rebuild_replay_does_not_resurrect_a_tombstoned_session(archive_root: Path) -> None:
    """A tombstoned session replayed from retained raw evidence stays gone.

    This is the rebuild blocker: ``ops maintenance rebuild-index`` replays
    accepted raw cohorts straight back through this writer.
    """
    session = _parsed("tombstoned-one")
    session_id = _session_id("tombstoned-one")

    # It was archived, then the operator deleted it.
    assert _replay(archive_root, session).wrote is True
    assert _session_rows(archive_root, session_id) == 1
    _tombstone(archive_root, session_id)
    with sqlite3.connect(_index_path(archive_root)) as conn:
        # Reset drops the rebuildable rows; foreign keys carry the cascade,
        # so they have to be on here as they are on every production handle.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    assert _message_rows(archive_root, session_id) == 0

    # The rebuild replays the raw evidence reset deliberately retained.
    outcome = _replay(archive_root, session)

    assert outcome.wrote is False
    assert outcome.suppression_skipped is True
    assert _session_rows(archive_root, session_id) == 0, "the tombstoned session was resurrected by replay"
    # Not readable, which for a fresh rebuild is exactly "no rows exist":
    # nothing can query content that never entered the rebuildable tier.
    assert _message_rows(archive_root, session_id) == 0


def test_suppression_refusal_is_counted_not_silently_dropped(archive_root: Path) -> None:
    """A skipped session is reported, so a refusal is never silent loss."""
    kept = _parsed("kept-one")
    dropped = _parsed("dropped-one")
    dropped_id = _session_id("dropped-one")
    _tombstone(archive_root, dropped_id)

    with suppression_refusal_scope() as totals:
        kept_outcome = _replay(archive_root, kept)
        dropped_outcome = _replay(archive_root, dropped)

    assert kept_outcome.wrote is True, "an unsuppressed session must still be written"
    assert _session_rows(archive_root, _session_id("kept-one")) == 1
    assert dropped_outcome.suppression_skipped is True
    assert totals.count == 1
    assert totals.session_ids == (dropped_id,)
    assert dropped_id in totals.describe()


def test_a_revoked_suppression_lets_the_session_be_archived_again(archive_root: Path) -> None:
    """The guard reads live durable state, not a one-time snapshot."""
    session = _parsed("revoked-one")
    session_id = _session_id("revoked-one")
    _tombstone(archive_root, session_id)
    assert _replay(archive_root, session).suppression_skipped is True

    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.execute(
            "UPDATE assertions SET status = 'deleted' WHERE kind = 'suppression' AND target_ref = ?",
            (f"session:{session_id}",),
        )

    assert _replay(archive_root, session).wrote is True
    assert _session_rows(archive_root, session_id) == 1


def test_an_archive_without_a_user_tier_is_not_blocked(tmp_path: Path) -> None:
    """A bare index with no durable user.db has no tombstone to honour."""
    index_db = tmp_path / "bare" / "index.db"
    index_db.parent.mkdir(parents=True)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    try:
        outcomes: list[ArchiveWriteOutcome] = []
        write_parsed_session_to_archive(conn, _parsed("bare-one"), write_outcome=outcomes)
    finally:
        conn.close()
    assert outcomes[0].wrote is True
    assert outcomes[0].suppression_skipped is False
