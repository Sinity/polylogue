"""Bounded semantic lineage-prefix cache coverage.

The cache is deliberately tested at the writer boundary: raw-byte identity is
not a reliable key for provider revisions, while canonical message signatures
and archive message ids are the identity/provenance evidence the writer
already trusts for branch extraction.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import LineageSignatureCache
from tests.infra.index_writer import (
    close_fixture_index_connection,
    fixture_index_mutation_scope,
    write_fixture_index_session,
)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_measured(path)
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
    cached_conn = _connect(tmp_path / "cached" / "index.db")
    uncached_conn = _connect(tmp_path / "uncached" / "index.db")
    try:
        cached = LineageSignatureCache(max_bytes=1024 * 1024)
        write_fixture_index_session(cached_conn, parent, signature_cache=cached)
        write_fixture_index_session(cached_conn, child, signature_cache=cached)

        disabled = LineageSignatureCache(max_bytes=1024 * 1024, enabled=False)
        write_fixture_index_session(uncached_conn, parent, signature_cache=disabled)
        write_fixture_index_session(uncached_conn, child, signature_cache=disabled)

        for table in ("sessions", "messages", "blocks", "session_links"):
            left = [tuple(row) for row in cached_conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            right = [tuple(row) for row in uncached_conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            assert left == right, table
    finally:
        close_fixture_index_connection(cached_conn)
        close_fixture_index_connection(uncached_conn)


def test_composed_cache_reuses_canonical_parent_identity_for_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sibling alignment reuses the composed parent, including its ids."""
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    db = tmp_path / "index.db"
    conn = _connect(db)
    cache = LineageSignatureCache(max_bytes=1024 * 1024)
    parent = _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)])
    parent_id = write_fixture_index_session(conn, parent, signature_cache=cache)

    preparation_calls = 0
    validation_calls = 0
    preparing = False
    original = write_module._own_db_signatures
    original_context = write_module._prepared_message_context

    def prepare_context(*args: Any, **kwargs: Any) -> write_module.PreparedMessageContext:
        nonlocal preparing
        assert not preparing
        preparing = True
        try:
            return original_context(*args, **kwargs)
        finally:
            preparing = False

    def counted(
        conn_arg: sqlite3.Connection,
        session_id_arg: str,
        before_input: write_module.BeforeIndexInput | None = None,
    ) -> list[tuple[str, str]]:
        nonlocal preparation_calls, validation_calls
        if session_id_arg == parent_id:
            if preparing:
                preparation_calls += 1
            else:
                validation_calls += 1
        return original(conn_arg, session_id_arg, before_input)

    monkeypatch.setattr(write_module, "_own_db_signatures", counted)
    monkeypatch.setattr(write_module, "_prepared_message_context", prepare_context)
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
    child_a_id = write_fixture_index_session(conn, child_a, signature_cache=cache)
    child_b_id = write_fixture_index_session(conn, child_b, signature_cache=cache)

    # Preparation reconstructs the parent once and the second sibling reuses
    # it. Each publication separately validates the current authoritative prefix.
    assert preparation_calls == 1
    assert validation_calls == 2
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
    close_fixture_index_connection(conn)


def test_changed_parent_refuses_prepared_child_even_with_stale_batch_cache(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers import write as write_module

    conn = _connect(tmp_path / "index.db")
    cache = LineageSignatureCache(max_bytes=1024 * 1024)
    prepared = None
    try:
        parent = _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)])
        parent_id = write_fixture_index_session(conn, parent)
        child = _session(
            "child",
            [_msg("c0", "hello", 0), _msg("c1", "reply", 1), _msg("c2", "tail", 2)],
            parent="parent",
        )
        prepared = write_module.prepare_session_write(conn, child, merge_append=False, signature_cache=cache)
        stale_prefix = cache.get_composed(parent_id)
        assert stale_prefix is not None
        replacement = _session("parent", [_msg("p0", "changed", 0), _msg("p1", "reply", 1)])
        write_fixture_index_session(conn, replacement)
        assert cache.get_composed(parent_id) == stale_prefix
        assert write_module._composed_db_signatures(conn, parent_id) != stale_prefix
        with pytest.raises(write_module.PreparedSessionWriteRefusedError), fixture_index_mutation_scope(conn):
            write_fixture_index_session(conn, child, signature_cache=cache, prepared_write=prepared)
        assert conn.execute("SELECT 1 FROM sessions WHERE native_id = 'child'").fetchone() is None
    finally:
        if prepared is not None:
            prepared.close()
        close_fixture_index_connection(conn)


def test_one_byte_prefix_difference_is_a_miss_and_stays_spawned_fresh(tmp_path: Path) -> None:
    """A semantic prefix mismatch cannot inherit a nearby parent row."""
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(
        conn,
        _session("parent", [_msg("p0", "hello", 0), _msg("p1", "reply", 1)]),
    )
    child_id = write_fixture_index_session(
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
    close_fixture_index_connection(conn)


def test_evicting_a_cached_ancestor_keeps_the_descendant_invalidatable(tmp_path: Path) -> None:
    """A composed entry stays tied to every ancestor after the one it stopped at is evicted.

    C composes by stopping at B's cached entry, so its own walk never visits
    A. Red twin: record only the walked sessions as C's dependencies. Once B's
    entry is evicted, rewriting A pops nothing that reaches C, and a later
    child of C is aligned against A's pre-rewrite signatures: it lands
    spawned-fresh and stores the whole replayed transcript.
    """
    conn = _connect(tmp_path / "index.db")
    cache = LineageSignatureCache(max_bytes=64 * 1024 * 1024)
    prefix = ["hello", "reply", "B tail", "C tail"]

    def lineage_messages(stem: str, texts: list[str]) -> list[ParsedMessage]:
        return [_msg(f"{stem}{position}", text, position) for position, text in enumerate(texts)]

    write_fixture_index_session(conn, _session("a", lineage_messages("a", prefix[:2])), signature_cache=cache)
    b_id = write_fixture_index_session(
        conn, _session("b", lineage_messages("b", prefix[:3]), parent="a"), signature_cache=cache
    )
    c_id = write_fixture_index_session(
        conn, _session("c", lineage_messages("c", prefix), parent="b"), signature_cache=cache
    )
    write_fixture_index_session(
        conn, _session("d1", lineage_messages("d1", [*prefix, "D1 tail"]), parent="c"), signature_cache=cache
    )
    composed_c = cache.get_composed(c_id)
    assert composed_c is not None, "C's composition is cached once a child of C was aligned"

    # Memory pressure: size the cache so the next entry evicts everything
    # older than C's composed entry, B's included.
    filler = [("filler:message", "f" * 64)]
    cache.max_bytes = LineageSignatureCache._weight(
        c_id, composed_c, dependencies=cache.composed_dependencies(c_id) or frozenset()
    ) + LineageSignatureCache._weight("filler", filler)
    cache["filler"] = filler
    cache.max_bytes = 64 * 1024 * 1024
    assert cache.get_composed(b_id) is None
    assert cache.get_composed(c_id) == composed_c

    # Rewrite A in place: same message ids, the first message's content edited.
    edited = ["hello, edited", *prefix[1:]]
    write_fixture_index_session(conn, _session("a", lineage_messages("a", edited[:2])), signature_cache=cache)

    d2_id = write_fixture_index_session(
        conn, _session("d2", lineage_messages("d2", [*edited, "D2 tail"]), parent="c"), signature_cache=cache
    )
    link = conn.execute(
        "SELECT inheritance, branch_point_message_id FROM session_links WHERE src_session_id = ?", (d2_id,)
    ).fetchone()
    c_last = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1", (c_id,)
    ).fetchone()[0]
    assert tuple(link) == ("prefix-sharing", c_last)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (d2_id,)).fetchone()[0] == 1
    close_fixture_index_connection(conn)


def test_pop_reaches_a_descendant_through_its_recorded_closure() -> None:
    """``pop`` finds every dependent from its own entry, not through intermediates."""
    cache = LineageSignatureCache(max_bytes=1024 * 1024)
    signatures = [("message", "s" * 64)]
    cache.set_composed("b", signatures, dependencies=frozenset({"b", "a"}))
    cache.set_composed("c", signatures, dependencies=frozenset({"c", "b", "a"}))
    cache.set_composed("unrelated", signatures, dependencies=frozenset({"unrelated"}))
    assert cache.composed_dependencies("c") == frozenset({"c", "b", "a"})

    cache.pop("a")

    assert cache.get_composed("b") is None
    assert cache.get_composed("c") is None
    assert cache.composed_dependencies("c") is None
    assert cache.get_composed("unrelated") == signatures


def test_an_oversized_own_entry_keeps_the_composed_entry_dependencies() -> None:
    """Refusing a whale own entry must not strip the same session's composed dependencies.

    Red twin: drop the dependency set on every oversized put. C's composed
    entry then stays resident with nothing tying it to A, and ``pop('a')``
    leaves it serving pre-rewrite signatures.
    """
    signatures = [("message", "s" * 64)]
    cache = LineageSignatureCache(
        max_bytes=LineageSignatureCache._weight("c", signatures, dependencies=frozenset({"a", "b", "c"})) * 2
    )
    cache.set_composed("c", signatures, dependencies=frozenset({"c", "b", "a"}))
    cache["c"] = [(f"message-{index}", "w" * 64) for index in range(64)]
    assert cache.get("c") is None
    assert cache.composed_dependencies("c") == frozenset({"c", "b", "a"})

    cache.pop("a")

    assert cache.get_composed("c") is None


def test_ancestor_dependency_closures_share_the_cache_byte_budget() -> None:
    signatures = [("message", "s" * 64)]
    dependencies = frozenset(f"ancestor-{number}" for number in range(1000))
    cache = LineageSignatureCache(max_bytes=4096)
    cache["unrelated"] = signatures
    resident = cache.resident_bytes
    cache.set_composed("deep-child", signatures, dependencies=dependencies)
    assert cache.get_composed("deep-child") is None
    assert cache.composed_dependencies("deep-child") is None
    assert cache.get("unrelated") == signatures
    assert cache.resident_bytes == resident <= cache.max_bytes
