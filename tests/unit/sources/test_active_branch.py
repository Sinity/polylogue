"""The resident and disk-backed lowerings publish one active-branch meaning."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import Role
from polylogue.sources.active_branch import normalize_active_branch
from polylogue.sources.parsers.base import ParsedMessage
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore


def _message(
    native_id: str,
    *,
    parent: str | None = None,
    path: bool | None = None,
    leaf: bool | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        parent_message_provider_id=parent,
        role=Role.USER,
        text=f"text of {native_id}",
        is_active_path=path,
        is_active_leaf=leaf,
    )


def _disk_lowering(tmp_path: Path, messages: list[ParsedMessage], *, passes: int = 1) -> list[ParsedMessage]:
    store = SqliteMessageStore(tmp_path / f"branch-{passes}.db")
    sink = store.new_sink()
    sink.extend(messages)
    for _ in range(passes):
        sink.normalize_active_path()
    store.conn.commit()
    store.close()
    return list(SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink)))


def _branch(messages: list[ParsedMessage]) -> list[tuple[bool | None, bool | None, bool]]:
    return [(message.is_active_path, message.is_active_leaf, message.active_leaf_fallback) for message in messages]


def _paths(messages: list[ParsedMessage]) -> list[bool | None]:
    return [message.is_active_path for message in messages]


def test_the_marked_leaf_occurrence_owns_its_parent_chain(tmp_path: Path) -> None:
    """Two occurrences share a provider id; only the marked one is the leaf.

    Anti-vacuity: resolving the leaf by provider id and taking its last
    occurrence walks to root B; marking by provider id also marks the
    unmarked ``dup`` occurrence.
    """
    messages = [
        _message("A", path=False),
        _message("B", path=False),
        _message("dup", parent="A", leaf=True),
        _message("dup", parent="B", path=False, leaf=False),
    ]

    resident = normalize_active_branch(list(messages))
    disk = _disk_lowering(tmp_path, messages)

    assert _paths(resident) == [True, False, True, False]
    assert _branch(disk) == _branch(resident)


def test_a_parent_id_names_its_nearest_earlier_occurrence(tmp_path: Path) -> None:
    """Anti-vacuity: the last occurrence of ``p`` (after the leaf) would pull
    ``x`` onto the path and leave the real parent ``p#0`` off it."""
    messages = [
        _message("p", path=False),
        _message("c", parent="p", leaf=True),
        _message("p", parent="x", path=False),
        _message("x", path=False),
    ]

    resident = normalize_active_branch(list(messages))

    assert _paths(resident) == [True, True, False, False]
    assert _branch(_disk_lowering(tmp_path, messages)) == _branch(resident)


def test_a_parent_cycle_ends_the_walk(tmp_path: Path) -> None:
    messages = [
        _message("a", parent="b", path=False),
        _message("b", parent="a", path=False, leaf=True),
    ]

    resident = normalize_active_branch(list(messages))

    assert _paths(resident) == [True, True]
    assert _branch(_disk_lowering(tmp_path, messages)) == _branch(resident)


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_a_fallback_leaf_never_becomes_path_evidence(tmp_path: Path, passes: int) -> None:
    """No producer leaf: the last message is the storage default and no path
    is inferred, however often the session is lowered.

    Anti-vacuity: without ``active_leaf_fallback`` the second pass sees one
    marked leaf, reads it as producer evidence and flips both explicit
    ``False`` paths to ``True``.
    """
    messages = [_message("root", path=False), _message("child", parent="root", path=False)]

    resident = list(messages)
    for _ in range(passes):
        resident = normalize_active_branch(resident)
    disk = _disk_lowering(tmp_path, messages, passes=passes)

    assert _branch(resident) == [(False, None, False), (False, True, True)]
    assert _branch(disk) == _branch(resident)
    # A sealed disk session lowered again by the resident form is unchanged.
    assert _branch(normalize_active_branch(disk)) == _branch(resident)


def test_a_producer_leaf_lowering_is_stable_on_repeat(tmp_path: Path) -> None:
    messages = [
        _message("root"),
        _message("side", parent="root", path=False),
        _message("tip", parent="root", leaf=True),
    ]

    once = normalize_active_branch(list(messages))
    twice = normalize_active_branch(once)

    assert _branch(once) == [(True, None, False), (False, None, False), (True, True, False)]
    assert _branch(twice) == _branch(once)
    assert _branch(_disk_lowering(tmp_path, messages, passes=2)) == _branch(once)
