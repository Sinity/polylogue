from __future__ import annotations

import json
import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import (
    migrate_saved_query_assertions,
    put_query,
    put_query_edge,
    put_result_set,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def test_result_sets_distinguish_membership_from_rank_and_reject_routine_members() -> None:
    conn = _conn()
    query = put_query(
        conn,
        {"field": "origin", "value": "codex-session"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    first = put_result_set(
        conn,
        result_set_id="rs-first",
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="epoch-1",
        member_refs=("session:a", "session:b"),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )
    second = put_result_set(
        conn,
        result_set_id="rs-second",
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="epoch-1",
        member_refs=("session:b", "session:a"),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=3,
    )

    assert first.membership_merkle_root == second.membership_merkle_root
    assert first.ordered_rank_hash != second.ordered_rank_hash
    with pytest.raises(ValueError, match="routine"):
        put_result_set(
            conn,
            result_set_id="rs-routine",
            query_hash=query.query_hash,
            grain="session",
            corpus_epoch="epoch-1",
            member_refs=("session:a",),
            exactness="exact",
            persistence_class="routine",
            created_at_ms=4,
        )


def test_query_edge_rejects_derived_from_cycle() -> None:
    conn = _conn()
    first = put_query(
        conn, {"field": "title", "value": "one"}, grain="session", lane="dialogue", rank_policy="mixed", created_at_ms=1
    )
    second = put_query(
        conn, {"field": "title", "value": "two"}, grain="session", lane="dialogue", rank_policy="mixed", created_at_ms=2
    )
    put_query_edge(
        conn,
        src_query_hash=first.query_hash,
        dst_query_hash=second.query_hash,
        edge_kind="derived-from",
        created_at_ms=3,
    )

    with pytest.raises(ValueError, match="cycle"):
        put_query_edge(
            conn,
            src_query_hash=second.query_hash,
            dst_query_hash=first.query_hash,
            edge_kind="derived-from",
            created_at_ms=4,
        )


def test_saved_query_migration_preserves_all_assertions_and_repoints_targets() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, value_json, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?, ?)",
        ("saved", "saved_view:one", "saved_query", json.dumps({"origin": "codex-session"}), 1, 1),
    )
    conn.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?, ?)",
        ("note", "session:one", "note", "must survive", 1, 1),
    )

    assert migrate_saved_query_assertions(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 2
    target_ref = conn.execute("SELECT target_ref FROM assertions WHERE assertion_id = 'saved'").fetchone()[0]
    assert str(target_ref).startswith("query:")
    assert conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
