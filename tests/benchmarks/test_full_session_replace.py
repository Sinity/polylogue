"""Regression benchmark for polylogue-rgbj: full-session message replacement.

Covers the exact production hot path from the bead's py-spy evidence:
``_replace_full_session_messages_and_blocks``'s bare
``DELETE FROM messages WHERE session_id = ?``
(``polylogue/storage/sqlite/archive_tiers/write.py``, near line 1795).

Root cause: ``web_content_constructs`` was the only table with a FK to
``messages(message_id)`` that lacked a leading index on the FK column.
SQLite's ``ON DELETE CASCADE`` enforcement does not require a child-key
index — without one it falls back to a full table scan of the child table
for *every* deleted row. A session-scoped delete of N messages therefore
cost O(N x web_content_constructs_rows) instead of O(N x log(rows)), which
is what turned a 15k-message production Codex session replacement into a
10+ minute stall while the writer held the transaction (live archive had
132,796 ``web_content_constructs`` rows at the time).

This test proves the fix at the SQL level: it builds two structurally
identical index-tier databases from the canonical DDL, drops the new
``idx_web_constructs_message`` index on one to reproduce the pre-fix shape,
and asserts the with-index delete is dramatically faster.

Run with:
    pytest tests/benchmarks/test_full_session_replace.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_INDEX_DDL = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]

# Sized to reproduce the production shape (many unrelated web_content_constructs
# rows scattered across other sessions) while keeping wall time bounded for a
# benchmark test. The live archive ratio (132,796 web_content_constructs rows /
# 4,525,143 messages) is not what matters here -- it's the absolute background
# table size the FK-cascade scan must walk per deleted message.
_TARGET_MESSAGES = 500
_BACKGROUND_WEB_CONSTRUCTS = 30_000


def _build_replace_fixture(db_path: Path, *, drop_leading_index: bool) -> tuple[sqlite3.Connection, str]:
    """Seed a fresh index-tier DB with one large target session plus many
    unrelated background sessions carrying web_content_constructs rows.

    Uses raw SQL (not the production writer) because this test proves a
    schema/query-plan property, not writer correctness -- the writer-level
    behavior is already covered by tests/unit/storage/test_crud.py and
    tests/unit/storage/test_archive_tiers_write.py.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_INDEX_DDL)
    if drop_leading_index:
        conn.execute("DROP INDEX IF EXISTS idx_web_constructs_message")

    target_session_id = "codex-session:target"
    conn.execute(
        "INSERT INTO sessions (origin, native_id, title, content_hash) VALUES ('codex-session', 'target', 't', ?)",
        (b"t" * 32,),
    )
    conn.executemany(
        "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
        (
            (target_session_id, f"m{i}", i, "user" if i % 2 == 0 else "assistant", bytes([i % 256]) * 32)
            for i in range(_TARGET_MESSAGES)
        ),
    )

    # Background sessions: 3 messages/blocks/web_content_constructs rows each,
    # none related to the target session, simulating archive-wide footprint.
    n_background_sessions = max(1, _BACKGROUND_WEB_CONSTRUCTS // 3)
    background_messages = []
    background_blocks = []
    background_constructs = []
    for session_index in range(n_background_sessions):
        session_id = f"chatgpt-export:bg{session_index}"
        conn.execute(
            "INSERT INTO sessions (origin, native_id, title, content_hash) VALUES ('chatgpt-export', ?, 't', ?)",
            (f"bg{session_index}", bytes([session_index % 256]) * 32),
        )
        for position in range(3):
            native_id = f"m{position}"
            message_id = f"{session_id}:{native_id}"
            background_messages.append((session_id, native_id, position, "user", bytes([position]) * 32))
            background_blocks.append((message_id, session_id, 0, "text", "hi"))
            block_id = f"{message_id}:0"
            background_constructs.append((session_id, message_id, block_id, 0, "chatgpt", "canvas"))
    conn.executemany(
        "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
        background_messages,
    )
    conn.executemany(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
        background_blocks,
    )
    conn.executemany(
        "INSERT INTO web_content_constructs "
        "(session_id, message_id, block_id, position, provider, construct_type) VALUES (?, ?, ?, ?, ?, ?)",
        background_constructs,
    )
    conn.commit()
    return conn, target_session_id


@pytest.mark.benchmark
def test_full_session_message_delete_uses_indexed_fk_cascade(tmp_path: Path) -> None:
    """DELETE FROM messages WHERE session_id = ? must not full-scan
    web_content_constructs (polylogue-rgbj).

    Builds two identical fixtures (with/without the leading FK index) and
    times the exact statement from
    `_replace_full_session_messages_and_blocks` on each. The with-index
    delete must be at least 20x faster -- comfortably below the ~300x
    measured locally, leaving headroom for slower CI hardware while still
    catching a regression if the index is ever dropped from canonical DDL.
    """
    without_index_conn, without_index_session_id = _build_replace_fixture(
        tmp_path / "without_index.db", drop_leading_index=True
    )
    with_index_conn, with_index_session_id = _build_replace_fixture(
        tmp_path / "with_index.db", drop_leading_index=False
    )

    try:
        started_at = time.perf_counter()
        without_index_conn.execute("DELETE FROM messages WHERE session_id = ?", (without_index_session_id,))
        without_index_conn.commit()
        without_index_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        with_index_conn.execute("DELETE FROM messages WHERE session_id = ?", (with_index_session_id,))
        with_index_conn.commit()
        with_index_seconds = time.perf_counter() - started_at
    finally:
        without_index_conn.close()
        with_index_conn.close()

    print(
        f"\nfull-session DELETE FROM messages ({_TARGET_MESSAGES} messages, "
        f"{_BACKGROUND_WEB_CONSTRUCTS} background web_content_constructs rows): "
        f"without idx_web_constructs_message={without_index_seconds:.4f}s, "
        f"with idx_web_constructs_message={with_index_seconds:.4f}s, "
        f"speedup={without_index_seconds / max(with_index_seconds, 1e-9):.0f}x"
    )

    assert with_index_seconds * 20 < without_index_seconds, (
        f"Expected the indexed delete to be at least 20x faster; got "
        f"without_index={without_index_seconds:.4f}s with_index={with_index_seconds:.4f}s "
        f"(only {without_index_seconds / max(with_index_seconds, 1e-9):.1f}x)"
    )
    # Absolute guard: catches a future regression even if the "before" fixture
    # also regressed for an unrelated reason.
    assert with_index_seconds < 1.0, f"Indexed delete took {with_index_seconds:.4f}s, expected well under 1s"
