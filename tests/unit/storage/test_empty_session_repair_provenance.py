"""Empty-session repair must delete debris, not legitimately-empty sessions.

`repair_empty_sessions` used a blanket `NOT EXISTS (messages)` predicate, which
cannot distinguish corruption debris from a session that is legitimately empty.
That distinction is load-bearing here: the 2026-07-22 hook-inflation postmortem
explicitly chose to RETAIN the genuinely-empty sessions that survived
de-inflation.

Measured on the live archive 2026-07-29: 874 message-less sessions, and all 874
carry a non-empty `raw_id` -- every one has real acquired bytes behind it, zero
lack provenance. So the old predicate would have deleted 874 legitimately
acquired sessions and zero pieces of debris. It was one explicit `--cleanup`
away from running against the live archive.

`raw_id` is the right signal because it names the retained source bytes: a
session carrying one can always be re-materialized by replay, which is exactly
what makes it not debris.
"""

from __future__ import annotations

import sqlite3

from polylogue.storage.repair import count_empty_sessions_sync


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            origin     TEXT NOT NULL,
            raw_id     TEXT
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO sessions (session_id, origin, raw_id) VALUES (?, ?, ?)",
        [
            # Legitimately empty: acquired bytes exist, replay can rebuild it.
            ("claude-code-session:acquired-empty", "claude-code-session", "rawhash-aaa"),
            # Debris: no messages and no provenance at all.
            ("claude-code-session:debris", "claude-code-session", None),
            ("claude-code-session:debris-blank", "claude-code-session", ""),
            # Healthy: has messages.
            ("claude-code-session:healthy", "claude-code-session", "rawhash-bbb"),
        ],
    )
    conn.execute(
        "INSERT INTO messages (message_id, session_id) VALUES (?, ?)",
        ("m1", "claude-code-session:healthy"),
    )
    conn.commit()


def test_only_sessions_without_provenance_count_as_debris() -> None:
    """The exact live shape: an acquired-but-empty session is NOT debt."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed(conn)

    # Two debris rows (NULL and empty-string raw_id); the acquired-empty and
    # the healthy session are both excluded.
    assert count_empty_sessions_sync(conn) == 2


def test_blanket_predicate_would_have_reported_the_retained_session() -> None:
    """Pin the defect itself, so a revert is caught rather than silently re-shipped."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed(conn)

    blanket = int(
        conn.execute(
            "SELECT COUNT(*) FROM sessions s "
            "WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.session_id)"
        ).fetchone()[0]
    )

    # The old predicate sweeps in the acquired-but-empty session too.
    assert blanket == 3
    assert count_empty_sessions_sync(conn) == 2, "provenance-aware count must exclude the acquired session"
