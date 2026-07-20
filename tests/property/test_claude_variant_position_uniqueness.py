"""Property test: Claude tree-mode message normalization must never assign
two emitted messages the same (position, variant_index) pair.

``messages`` is unique on exactly ``(session_id, position, variant_index)``
(see ``storage/sqlite/archive_tiers/index.py``). Before the fix, a message
that was the sole ("rank-0") child of its parent always reset to
``variant_index=0`` regardless of which sibling-variant world its parent
belonged to. Two sibling variants at the same depth, each with their own
rank-0 continuation, then produced two distinct native messages at the exact
same ``(position, variant_index)`` coordinate -- silently dropping one via
``INSERT OR REPLACE`` on write (operation ab5bad1f / claude-ai-export:d24081f5).

This sweep generates arbitrary parent/branch/depth tree shapes (including
nested branch-under-branch chains) and asserts the parser's own output
satisfies the uniqueness invariant the messages table's primary key demands,
independent of which heuristic (rank-0 inheritance vs. the final
deduplication safety net) ends up producing it.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.sources.parsers.claude.common import normalize_chat_messages


@st.composite
def _message_trees(draw: st.DrawFn) -> list[object]:
    node_count = draw(st.integers(min_value=2, max_value=25))
    nodes: list[object] = []
    for index in range(node_count):
        parent_id: str | None = None
        if index > 0:
            # Any earlier node may be the parent -- this is what produces
            # arbitrary branching fan-out and depth, including nested
            # branch-under-branch shapes (a variant's variant).
            parent_index = draw(st.integers(min_value=0, max_value=index - 1))
            parent_id = f"n{parent_index}"
        role = draw(st.sampled_from(["human", "assistant"]))
        # Occasionally omit the timestamp to exercise the None-timestamp sort
        # path (_timestamp_sort_value treats it as +inf) alongside the normal
        # monotonic-by-creation-order path.
        has_timestamp = draw(st.booleans())
        node: dict[str, object] = {
            "uuid": f"n{index}",
            "sender": role,
            "text": f"body-{index}",
            "parent_message_uuid": parent_id,
        }
        if has_timestamp:
            node["created_at"] = f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z"
        nodes.append(node)
    return nodes


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_message_trees())
def test_variant_position_pairs_are_unique_for_any_tree_shape(chat_messages: list[object]) -> None:
    normalized = normalize_chat_messages(chat_messages)
    coordinates = [(message.position, message.variant_index) for message in normalized.messages]
    assert len(coordinates) == len(set(coordinates)), (
        f"duplicate (position, variant_index) pairs for tree shape: {coordinates}"
    )
    # Every emitted message must have a non-negative position and variant --
    # the invariant is meaningless if either coordinate went missing.
    assert all(message.position is not None and message.position >= 0 for message in normalized.messages)
    assert all(message.variant_index is not None and message.variant_index >= 0 for message in normalized.messages)
