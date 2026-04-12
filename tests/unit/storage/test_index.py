from __future__ import annotations

import sqlite3


def test_rebuild_index_skips_action_rebuild_when_candidates_are_current(monkeypatch) -> None:
    from polylogue.storage import index as index_mod

    conn = sqlite3.connect(":memory:")
    fts_calls: list[sqlite3.Connection] = []

    monkeypatch.setattr(index_mod, "action_event_repair_candidates_sync", lambda db_conn: [])

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("action-event rebuild should have been skipped")

    monkeypatch.setattr(index_mod, "rebuild_action_event_read_model_sync", fail_rebuild)
    monkeypatch.setattr(index_mod, "rebuild_fts_index_sync", lambda db_conn: fts_calls.append(db_conn))
    monkeypatch.setattr(index_mod, "invalidate_search_cache", lambda: None)

    index_mod.rebuild_index(conn)

    assert fts_calls == [conn]


def test_update_index_repairs_only_candidate_subset(monkeypatch) -> None:
    from polylogue.storage import index as index_mod

    conn = sqlite3.connect(":memory:")
    repaired_targets: list[list[str]] = []
    repaired_fts_targets: list[list[str]] = []

    monkeypatch.setattr(index_mod, "action_event_repair_candidates_sync", lambda db_conn: ["conv-a", "conv-c"])
    monkeypatch.setattr(
        index_mod,
        "rebuild_action_event_read_model_sync",
        lambda db_conn, *, conversation_ids: repaired_targets.append(list(conversation_ids)),
    )
    monkeypatch.setattr(
        index_mod,
        "repair_fts_index_sync",
        lambda db_conn, conversation_ids: repaired_fts_targets.append(list(conversation_ids)),
    )
    monkeypatch.setattr(index_mod, "invalidate_search_cache", lambda: None)

    index_mod.update_index_for_conversations(["conv-a", "conv-b"], conn)

    assert repaired_targets == [["conv-a"]]
    assert repaired_fts_targets == [["conv-a", "conv-b"]]
