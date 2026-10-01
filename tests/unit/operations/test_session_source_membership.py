"""Retained source joins keep their acquisition and failure contracts."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.operations import session_source_membership as membership


def test_source_join_returns_every_matching_session_and_preserves_query_fault() -> None:
    """Dropping the raw/session join or swallowing a broken join makes this red."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT, raw_id TEXT)")
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT, source_path TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?, ?)", [("second", "raw"), ("first", "raw")])
        conn.execute("INSERT INTO raw_sessions VALUES ('raw', 'synthetic/session.jsonl')")
        path = Path("synthetic/session.jsonl")
        absent = Path("synthetic/absent.jsonl")
        assert membership.session_ids_for_source_paths(conn, (path, absent, path)) == {
            path: ["first", "second"],
            absent: [],
        }
        conn.execute("ALTER TABLE raw_sessions RENAME COLUMN source_path TO broken_path")
        with pytest.raises(sqlite3.OperationalError):
            membership.session_ids_for_source_paths(conn, (path,))


def test_hot_probe_only_attachment_fault_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capturing the later session join would conceal a broken required relation."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT, raw_id TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('session', 'raw')")
        events: list[str] = []
        monkeypatch.setattr(membership, "emit", lambda event, **_fields: events.append(event))

        def unavailable(*_args: object, **_kwargs: object) -> bool:
            raise sqlite3.OperationalError("synthetic attachment unavailable")

        monkeypatch.setattr(membership, "_ensure_source_tier_attached", unavailable)
        assert membership.hot_insight_session_ids(conn, ("session",), archive_root=tmp_path) == set()
        assert events == ["daemon.archive.source_tier_attach_failed"]
        with pytest.raises(sqlite3.OperationalError):
            membership.session_ids_for_source_paths(conn, (Path("synthetic/session.jsonl"),), archive_root=tmp_path)

        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT, wrong_column TEXT)")
        with pytest.raises(sqlite3.OperationalError):
            membership.hot_insight_session_ids(conn, ("session",), archive_root=tmp_path)


def test_unreadable_promoted_index_cannot_fall_back_to_conventional_membership(tmp_path: Path) -> None:
    """Serving the readable shadow after active acquisition fails makes this red."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    generation = tmp_path / ".index-generations" / "promoted"
    generation.mkdir(parents=True)
    index = generation / "index.db"
    index.write_bytes(b"synthetic unreadable index")
    (tmp_path / ".index-active-pointer").write_text(str(index), encoding="utf-8")
    # The conventional initialized index remains valid and empty. It cannot
    # supply an answer for the generation the archive has actually selected.
    with pytest.raises(sqlite3.DatabaseError):
        membership.session_ids_for_paths(tmp_path / "index.db", (Path("synthetic/session.jsonl"),))


def test_absent_active_index_retains_explicit_empty_membership(tmp_path: Path) -> None:
    """A genuinely absent tier remains distinct from a failed acquisition."""
    path = Path("synthetic/session.jsonl")
    assert membership.session_ids_for_paths(tmp_path / "index.db", (path,)) == {path: []}
