from __future__ import annotations

import sqlite3

import pytest


def test_rebuild_index_rebuilds_message_fts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage import index as index_mod

    conn = sqlite3.connect(":memory:")
    fts_calls: list[sqlite3.Connection] = []

    monkeypatch.setattr(index_mod, "rebuild_fts_index_sync", lambda db_conn: fts_calls.append(db_conn))
    monkeypatch.setattr(index_mod, "invalidate_search_cache", lambda: None)

    index_mod.rebuild_index(conn)

    assert fts_calls == [conn]


def test_update_index_repairs_message_fts_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage import index as index_mod

    conn = sqlite3.connect(":memory:")
    repaired_fts_targets: list[list[str]] = []

    monkeypatch.setattr(
        index_mod,
        "repair_fts_index_sync",
        lambda db_conn, session_ids: repaired_fts_targets.append(list(session_ids)),
    )
    monkeypatch.setattr(index_mod, "invalidate_search_cache", lambda: None)

    index_mod.update_index_for_sessions(["conv-a", "conv-b"], conn)

    assert repaired_fts_targets == [["conv-a", "conv-b"]]
