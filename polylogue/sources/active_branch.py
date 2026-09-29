"""The active-branch meaning a prepared session is lowered with.

One meaning, two physical forms: :func:`normalize_active_branch` lowers a
resident message list, and ``SqliteMessageSink.normalize_active_path`` lowers
a disk-backed session with the same rules, so either preparation route
publishes the same leaf and path values.

- **Leaf.** A session whose producer marked exactly one occurrence as the
  active leaf keeps that occurrence. Otherwise the last message becomes the
  leaf as a storage default, and it carries ``active_leaf_fallback`` so a
  later pass never reads its own default as producer evidence.
- **Path.** Only a producer leaf implies an active path. The walk starts at
  the leaf occurrence itself and follows each occurrence's own parent id.
  A parent id names the nearest earlier occurrence with that provider id
  (a transcript records a parent before its child); with no earlier one it
  names the last occurrence. Only the occurrences on that chain are marked:
  another occurrence repeating a chain member's provider id is a different
  message and keeps its own value. A cycle ends the walk.
- **Idempotence.** Lowering an already lowered session changes nothing.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from polylogue.sources.parsers.base import ParsedMessage


def producer_leaf_position(messages: Sequence[ParsedMessage]) -> int | None:
    """The producer-marked leaf occurrence, or ``None`` when the storage default applies."""
    marked = [position for position, message in enumerate(messages) if message.is_active_leaf]
    if len(marked) != 1 or messages[marked[0]].active_leaf_fallback:
        return None
    return marked[0]


def parent_occurrence(
    occurrences: Sequence[int],
    child_position: int,
) -> int | None:
    """The occurrence a child's parent id names, from that id's ascending positions."""
    if not occurrences:
        return None
    earlier = bisect_left(occurrences, child_position)
    return occurrences[earlier - 1] if earlier else occurrences[-1]


def normalize_active_branch(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Settle leaf and active-path values for one resident session."""
    if not messages:
        return messages
    leaf = producer_leaf_position(messages)
    if leaf is None:
        last = len(messages) - 1
        return [
            message.model_copy(update={"is_active_leaf": position == last, "active_leaf_fallback": position == last})
            if bool(message.is_active_leaf) != (position == last) or message.active_leaf_fallback != (position == last)
            else message
            for position, message in enumerate(messages)
        ]
    if not messages[leaf].provider_message_id:
        return messages
    positions_by_id: dict[str, list[int]] = {}
    for position, message in enumerate(messages):
        if message.provider_message_id:
            positions_by_id.setdefault(message.provider_message_id, []).append(position)
    chain: set[int] = set()
    cursor: int | None = leaf
    while cursor is not None and cursor not in chain:
        chain.add(cursor)
        parent_id = messages[cursor].parent_message_provider_id
        cursor = parent_occurrence(positions_by_id.get(parent_id, ()), cursor) if parent_id else None
    return [
        message.model_copy(update={"is_active_path": True})
        if position in chain and message.is_active_path is not True
        else message
        for position, message in enumerate(messages)
    ]


__all__ = ["normalize_active_branch", "parent_occurrence", "producer_leaf_position"]
