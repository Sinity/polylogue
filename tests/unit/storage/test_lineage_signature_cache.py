"""Bounded semantic lineage-prefix cache coverage.

The cache is deliberately tested at the writer boundary: raw-byte identity is
not a reliable key for provider revisions, while canonical message signatures
and archive message ids are the identity/provenance evidence the writer
already trusts for branch extraction.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    LineageSignatureCache,
    write_parsed_session_to_archive,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _msg(native_id: str, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
        text=text,
        position=position,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _session(native_id: str, messages: list[ParsedMessage], *, parent: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        messages=messages,
    )


def test_lineage_signature_cache_is_weight_bounded_and_recency_ordered() -> None:
    one = [("message-a", "a" * 64)]
    weight = LineageSignatureCache._weight("a", one)
    cache = LineageSignatureCache(max_bytes=weight * 2)
    cache["a"] = one
    cache["b"] = [("message-b", "b" * 64)]
    assert cache.resident_bytes <= cache.max_bytes

    # Touching a moves it behind b; adding c therefore evicts b first. This
    # proves the cache is recency-bounded rather than an ever-growing dict.
    assert cache.get("a") == one
    cache["c"] = [("message-c", "c" * 64)]
    assert cache.get("a") == one
    assert cache.get("b") is None
    assert cache.evictions >= 1
    assert cache.resident_bytes <= cache.max_bytes


def test_disabled_lineage_signature_cache_is_an_identical_miss_path() -> None:
    cache = LineageSignatureCache(max_bytes=1024, enabled=False)
    signatures = [("canonical:message", "f" * 64)]
    cache["session"] = signatures
    assert cache.get("session") is None
    assert len(cache) == 0
    assert cache.misses == 1


def test_disabling_cache_preserves_lineage_output(tmp_path: Path) -> None:
    """The cache is an optimization hint, never a source of archive truth."""
    parent = _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)])
    child = _session(
        "child",
        [_msg("c0", "hello", 0), _msg("c1", "reply", 1), _msg("c2", "tail", 2)],
        parent="parent",
    )
    cached_conn = _connect(tmp_path / "cached.db")
    uncached_conn = _connect(tmp_path / "uncached.db")
    try:
        cached = LineageSignatureCache(max_bytes=1024 * 1024)
        write_parsed_session_to_archive(cached_conn, parent, signature_cache=cached)
        write_parsed_session_to_archive(cached_conn, child, signature_cache=cached)

        disabled = LineageSignatureCache(max_bytes=1024 * 1024, enabled=False)
        write_parsed_session_to_archive(uncached_conn, parent, signature_cache=disabled)
        write_parsed_session_to_archive(uncached_conn, child, signature_cache=disabled)

        for table in ("sessions", "messages", "blocks", "session_links"):
            left = [tuple(row) for row in cached_conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            right = [tuple(row) for row in uncached_conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            assert left == right, table
    finally:
        cached_conn.close()
        uncached_conn.close()


def test_composed_cache_reuses_canonical_parent_identity_for_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sibling alignment reuses the composed parent, including its ids."""
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    db = tmp_path / "index.db"
    conn = _connect(db)
    cache = LineageSignatureCache(max_bytes=1024 * 1024)
    parent = _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)])
    parent_id = write_parsed_session_to_archive(conn, parent, signature_cache=cache)

    calls = 0
    original = write_module._own_db_signatures

    def counted(conn_arg: sqlite3.Connection, session_id_arg: str) -> list[tuple[str, str]]:
        nonlocal calls
        calls += 1
        return original(conn_arg, session_id_arg)

    monkeypatch.setattr(write_module, "_own_db_signatures", counted)
    child_a = _session(
        "child-a",
        [_msg("a0", "hello", 0), _msg("a1", "reply", 1), _msg("a2", "A tail", 2)],
        parent="parent",
    )
    child_b = _session(
        "child-b",
        [_msg("b0", "hello", 0), _msg("b1", "reply", 1), _msg("b2", "B tail", 2)],
        parent="parent",
    )
    child_a_id = write_parsed_session_to_archive(conn, child_a, signature_cache=cache)
    child_b_id = write_parsed_session_to_archive(conn, child_b, signature_cache=cache)

    # Parent own signatures are read once; the second sibling consumes the
    # composed cache hit. The branch point remains the parent's canonical row.
    assert calls == 1
    assert cache.hits >= 1
    branch_a = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_a_id,)
    ).fetchone()[0]
    branch_b = conn.execute(
        "SELECT branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_b_id,)
    ).fetchone()[0]
    canonical_branch = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1", (parent_id,)
    ).fetchone()[0]
    assert branch_a == canonical_branch == branch_b
    conn.close()


def test_one_byte_prefix_difference_is_a_miss_and_stays_spawned_fresh(tmp_path: Path) -> None:
    """A semantic prefix mismatch cannot inherit a nearby parent row."""
    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(
        conn,
        _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)]),
    )
    child_id = write_parsed_session_to_archive(
        conn,
        _session(
            "child",
            # One byte differs before the would-be branch point.
            [_msg("c0", "hellO", 0), _msg("c1", "reply", 1), _msg("c2", "tail", 2)],
            parent="parent",
        ),
    )
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    assert tuple(link) == ("spawned-fresh", None)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0] == 3
    conn.close()
