"""A watched saved view's ``query_names`` binding follows its lifecycle.

``queries``/``query_names`` is an independent durable substrate keyed by
*name*, while a saved view is an assertion keyed by ``view_id``. Deleting a
view only tombstoned its assertion and renaming one only registered the new
name, so ``list_watched_queries`` kept handing the standing-query convergence
stage definitions the product no longer holds -- and that stage persists result
sets and emits findings for whatever it is handed.

Anti-vacuity: drop either ``clear_query_watch`` call in
``ArchiveStore.save_view``/``delete_view`` and the matching test goes red with
the stale name still watched. ``test_an_unrelated_watch_survives_...`` pins the
opposite direction so a blanket "clear every watch" cannot pass.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.query_objects import list_watched_queries

_CODEX = json.dumps({"query": "sessions where origin:codex-session"})
_CLAUDE = json.dumps({"query": "sessions where origin:claude-code-session"})


def _watched_names(root: Path) -> list[str]:
    with sqlite3.connect(root / "user.db") as conn:
        return [str(row[0]) for row in conn.execute("SELECT name FROM query_names WHERE watch = 1 ORDER BY name")]


def _watched_definitions(root: Path) -> int:
    with sqlite3.connect(root / "user.db") as conn:
        return len(list_watched_queries(conn))


class TestWatchBindingLifecycle:
    def test_deleting_a_watched_view_retires_its_watch(self, tmp_path: Path) -> None:
        with ArchiveStore(tmp_path) as archive:
            archive.save_view("view-a", "alpha", _CODEX, watch=True)
            assert _watched_names(tmp_path) == ["alpha"]
            archive.delete_view("view-a")
            assert [row["name"] for row in archive.list_views()] == []
        assert _watched_names(tmp_path) == [], "a deleted view's name is still watched"
        assert _watched_definitions(tmp_path) == 0

    def test_renaming_a_watched_view_retires_the_old_name(self, tmp_path: Path) -> None:
        with ArchiveStore(tmp_path) as archive:
            archive.save_view("view-a", "old", _CODEX, watch=True)
            archive.save_view("view-a", "new", _CLAUDE, watch=True)
            assert [row["name"] for row in archive.list_views()] == ["new"]
        assert _watched_names(tmp_path) == ["new"], "the previous name is still watched beside the new one"
        assert _watched_definitions(tmp_path) == 1, "one saved view must not evaluate two definitions"

    def test_renaming_an_unwatched_view_leaves_no_binding(self, tmp_path: Path) -> None:
        with ArchiveStore(tmp_path) as archive:
            archive.save_view("view-a", "old", _CODEX, watch=False)
            archive.save_view("view-a", "new", _CLAUDE, watch=False)
        assert _watched_names(tmp_path) == []

    def test_an_unrelated_watch_survives_a_delete(self, tmp_path: Path) -> None:
        """Opposite direction: clearing every watch on any lifecycle event fails."""
        with ArchiveStore(tmp_path) as archive:
            archive.save_view("view-a", "alpha", _CODEX, watch=True)
            archive.save_view("view-b", "beta", _CLAUDE, watch=True)
            archive.delete_view("view-a")
        assert _watched_names(tmp_path) == ["beta"]
        assert _watched_definitions(tmp_path) == 1

    def test_an_unrelated_watch_survives_a_rename(self, tmp_path: Path) -> None:
        """Opposite direction: a rename must retire only its own prior name."""
        with ArchiveStore(tmp_path) as archive:
            archive.save_view("view-a", "alpha", _CODEX, watch=True)
            archive.save_view("view-b", "beta", _CLAUDE, watch=True)
            archive.save_view("view-a", "alpha-renamed", _CODEX, watch=True)
        assert _watched_names(tmp_path) == ["alpha-renamed", "beta"]
