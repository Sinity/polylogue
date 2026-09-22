"""``stuck_sessions`` must page over stuck profiles, not over all sessions.

``stuck_tool_count`` is materialized into ``session_latency_profiles`` and read
from nowhere else, so the stuck verdict is a SQL-expressible predicate. Applying
it in Python *after* ``LIMIT/OFFSET`` made the page a window over every session
and the filter a post-pass inside that window: the default 50-row request
answered "no stuck sessions" whenever the newest 50 sessions happened not to be
stuck, while qualifying profiles sat further down the archive.

Anti-vacuity: ``test_stuck_page_reaches_older_stuck_profile`` is
red if the predicate moves back out of the WHERE clause -- the newest session is
not stuck, so a ``limit=1`` page over all sessions selects it and the post-filter
returns nothing. The opposite direction (a predicate that admits everything, or
one that drops the filter) is pinned by
``test_stuck_page_excludes_non_stuck_profile``, which fails if a non-stuck
session is ever returned.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path) -> Path:
    """Two sessions: the newer one is not stuck, the older one is."""
    initialize_active_archive_root(root)
    index_db = root / "index.db"
    for name, updated_at in (("older", "2026-01-01T00:00:00+00:00"), ("newer", "2026-06-01T00:00:00+00:00")):
        (
            SessionBuilder(index_db, name)
            .provider("claude-code")
            .title(name)
            .updated_at(updated_at)
            .add_message("m-0", role="user", text="hello")
            .save()
        )
    conn = sqlite3.connect(index_db)
    try:
        with conn:
            for name, stuck in (("older", 3), ("newer", 0)):
                conn.execute(
                    """
                    INSERT INTO session_latency_profiles (
                        session_id, source_name, median_tool_call_ms, p90_tool_call_ms,
                        max_tool_call_ms, stuck_tool_count, materialized_at
                    ) VALUES (?, 'claude-code-session', 1, 2, 3, ?, '2026-06-01T00:00:00+00:00')
                    """,
                    (f"claude-code-session:ext-{name}", stuck),
                )
    finally:
        conn.close()
    return index_db


def test_stuck_page_reaches_older_stuck_profile(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        page = archive.find_stuck_session_latency_profile_insights(limit=1)
    assert [insight.session_id for insight in page] == ["claude-code-session:ext-older"]
    assert page[0].latency.stuck_tool_count == 3


def test_stuck_page_excludes_non_stuck_profile(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        page = archive.find_stuck_session_latency_profile_insights(limit=50)
        unfiltered = archive.list_session_latency_profile_insights(limit=50)
    assert [insight.session_id for insight in page] == ["claude-code-session:ext-older"]
    assert {insight.session_id for insight in unfiltered} == {
        "claude-code-session:ext-older",
        "claude-code-session:ext-newer",
    }
