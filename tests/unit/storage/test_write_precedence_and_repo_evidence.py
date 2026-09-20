"""Untrusted timestamps and unrooted commits (polylogue-1pzmq)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ingest_precedence import (
    UNTRUSTED_FUTURE_FRESHNESS_TOLERANCE_MS,
    should_skip_stale_replace,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows, write_parsed_session_to_archive

_NOW_MS = 1_760_000_000_000


def test_implausibly_future_stored_timestamp_stops_pinning_the_session() -> None:
    """A year-3000 ``updated_at_ms`` must not block every later genuine replay.

    Anti-vacuity: delete the ``_freshness_is_untrusted`` guard and the second
    assert flips to True -- the crafted timestamp wins forever.
    """
    far_future_ms = _NOW_MS + 1000 * 365 * 24 * 60 * 60 * 1000
    assert should_skip_stale_replace(incoming_freshness_ms=100, existing_updated_at_ms=200, now_ms=_NOW_MS) is True
    assert (
        should_skip_stale_replace(incoming_freshness_ms=_NOW_MS, existing_updated_at_ms=far_future_ms, now_ms=_NOW_MS)
        is False
    )


def test_ordinary_clock_skew_still_decides_the_comparison() -> None:
    """The bound is a tolerance, not a clamp: honest skew keeps its authority.

    Anti-vacuity: shrink ``UNTRUSTED_FUTURE_FRESHNESS_TOLERANCE_MS`` below a
    day (or replace it with "any future timestamp is untrusted") and this is
    red.
    """
    skewed_ms = _NOW_MS + UNTRUSTED_FUTURE_FRESHNESS_TOLERANCE_MS - 1
    assert (
        should_skip_stale_replace(incoming_freshness_ms=_NOW_MS, existing_updated_at_ms=skewed_ms, now_ms=_NOW_MS)
        is True
    )


def test_commit_evidence_survives_a_checkout_this_machine_cannot_resolve(tmp_path: Path) -> None:
    """Codex's branch+commit-without-remote shape reaches ``session_commits``.

    The declared cwd resolves to no git root here and there is no remote, so
    ``_write_repo_edges`` used to return before its ``session_commits``
    insert and the commit the export stated was dropped.

    Anti-vacuity: restore the bare ``return`` and the row count goes to 0.
    """
    unrooted = tmp_path / "gone" / "checkout"
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="unrooted-commit",
        working_directories=[str(unrooted)],
        git_commit_hash="0123456789abcdef0123456789abcdef01234567",
        git_branch="feature/offline",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
    )
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = write_parsed_session_to_archive(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared=prepare_session_rows(session),
        )
        rows = list(conn.execute("SELECT * FROM session_commits WHERE session_id = ?", (session_id,)))
        assert len(rows) == 1
        assert rows[0]["commit_sha"] == "0123456789abcdef0123456789abcdef01234567"
        assert rows[0]["repo_id"] is None
        evidence = json.loads(rows[0]["evidence_json"])
        assert evidence["git_branch"] == "feature/offline"
        assert evidence["unresolved_working_directories"] == [str(unrooted)]
        # No repository was invented for it.
        assert list(conn.execute("SELECT repo_id FROM repos")) == []
    finally:
        conn.close()
