"""Property-based tests using Hypothesis for polylogue core invariants.

These tests verify that critical system properties hold for arbitrary inputs,
providing broader coverage than example-based tests.
"""

from __future__ import annotations

import random
from datetime import datetime

import pytest
from hypothesis import given, strategies as st

from polylogue.core.hashing import hash_payload, hash_text, hash_text_short
from polylogue.core.timestamps import format_timestamp, parse_timestamp
from polylogue.importers.base import normalize_role
from polylogue.lib.models import Conversation, Message


# =============================================================================
# Hashing Properties
# =============================================================================


@given(st.text())
def test_hash_text_deterministic_property(text: str) -> None:
    """Same input always produces same hash."""
    assert hash_text(text) == hash_text(text)


@given(st.text())
def test_hash_text_length_invariant(text: str) -> None:
    """Hash is always 64 characters (SHA-256 hex digest)."""
    assert len(hash_text(text)) == 64


@given(st.text(), st.integers(min_value=1, max_value=64))
def test_hash_short_is_prefix(text: str, length: int) -> None:
    """Short hash is prefix of full hash."""
    full = hash_text(text)
    short = hash_text_short(text, length)
    assert full.startswith(short)
    assert len(short) == length


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_hash_payload_order_independent(d: dict) -> None:
    """Dict hash doesn't depend on key order."""
    if not d:  # Skip empty dicts
        return

    # Create shuffled copy
    keys = list(d.keys())
    random.shuffle(keys)
    shuffled = {k: d[k] for k in keys}

    assert hash_payload(d) == hash_payload(shuffled)


@given(st.text())
def test_hash_text_hex_characters(text: str) -> None:
    """Hash only contains valid hex characters."""
    result = hash_text(text)
    assert all(c in "0123456789abcdef" for c in result)


# =============================================================================
# Timestamp Properties
# =============================================================================


@given(st.floats(min_value=0, max_value=2**31 - 1, allow_nan=False, allow_infinity=False))
def test_parse_timestamp_valid_epoch(epoch: float) -> None:
    """Valid epochs are parsed successfully."""
    result = parse_timestamp(epoch)
    assert result is not None


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_timestamp_format_parse_roundtrip(epoch: int) -> None:
    """Format then parse preserves value (to second precision)."""
    formatted = format_timestamp(epoch)
    parsed = parse_timestamp(formatted)

    assert parsed is not None
    # Allow 1 second tolerance due to formatting precision
    assert abs(parsed.timestamp() - epoch) < 2


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_parse_timestamp_epoch_int(epoch: int) -> None:
    """Integer epochs parse correctly."""
    result = parse_timestamp(epoch)
    assert result is not None
    assert abs(result.timestamp() - epoch) < 1


@given(st.text())
def test_parse_timestamp_never_crashes(invalid: str) -> None:
    """parse_timestamp never crashes on arbitrary input."""
    # This tests that the function handles any input gracefully
    # without raising exceptions - it should return datetime or None
    result = parse_timestamp(invalid)
    assert result is None or isinstance(result, datetime)


# =============================================================================
# Role Normalization Properties
# =============================================================================


@given(st.sampled_from(["user", "assistant", "system", "human", "model"]))
def test_normalize_role_idempotent(role: str) -> None:
    """Normalizing twice gives same result."""
    once = normalize_role(role)
    twice = normalize_role(once)
    assert once == twice


@given(st.sampled_from(["USER", "Assistant", "SYSTEM", "Human", "MODEL"]))
def test_normalize_role_case_insensitive(role: str) -> None:
    """Role normalization is case-insensitive."""
    lowered = normalize_role(role.lower())
    mixed = normalize_role(role)
    assert lowered == mixed


@given(st.text(min_size=1))
def test_normalize_role_always_returns_string(role: str) -> None:
    """normalize_role always returns a non-empty string."""
    result = normalize_role(role)
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_role_none_handling() -> None:
    """normalize_role handles None gracefully."""
    result = normalize_role(None)
    assert isinstance(result, str)


# =============================================================================
# Message Properties
# =============================================================================


@given(st.sampled_from(["user", "assistant", "system", "human", "model"]))
def test_message_role_enum_cached(role: str) -> None:
    """role_enum property returns same object on repeated access."""
    msg = Message(id="1", role=role, text="test")
    first = msg.role_enum
    second = msg.role_enum
    assert first is second  # Same object (cached)


@given(st.text(min_size=1, max_size=1000), st.sampled_from(["user", "assistant"]))
def test_message_word_count_nonnegative(text: str, role: str) -> None:
    """Word count is always non-negative."""
    msg = Message(id="1", role=role, text=text)
    assert msg.word_count >= 0


@given(st.sampled_from(["user", "human"]))
def test_message_is_user_consistency(role: str) -> None:
    """Messages with user/human role are classified as user."""
    msg = Message(id="1", role=role, text="test")
    assert msg.is_user
    assert msg.is_dialogue


@given(st.sampled_from(["assistant", "model"]))
def test_message_is_assistant_consistency(role: str) -> None:
    """Messages with assistant/model role are classified as assistant."""
    msg = Message(id="1", role=role, text="test")
    assert msg.is_assistant
    assert msg.is_dialogue


# =============================================================================
# Conversation Projection Properties
# =============================================================================


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20))
def test_projection_count_matches_length(texts: list[str]) -> None:
    """Projection count() equals len(to_list())."""
    messages = [
        Message(id=str(i), role="user", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    proj = conv.project()
    assert proj.count() == len(proj.to_list())


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20), st.integers(min_value=0, max_value=30))
def test_projection_limit_respects_bound(texts: list[str], limit: int) -> None:
    """Projection with limit(n) returns at most n items."""
    messages = [
        Message(id=str(i), role="user", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    result = conv.project().limit(limit).to_list()
    assert len(result) <= limit


@given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
def test_projection_execute_returns_conversation(texts: list[str]) -> None:
    """Projection execute() returns a Conversation instance."""
    messages = [
        Message(id=str(i), role="user", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    result = conv.project().execute()
    assert isinstance(result, Conversation)
    assert result.id == conv.id


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20))
def test_projection_filter_preserves_order(texts: list[str]) -> None:
    """Filtering preserves message order."""
    messages = [
        Message(id=str(i), role="user" if i % 2 == 0 else "assistant", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    # Filter to user messages only
    user_messages = conv.project().user_messages().to_list()

    # Verify order is preserved (ids should be increasing)
    if len(user_messages) > 1:
        ids = [int(m.id) for m in user_messages]
        assert ids == sorted(ids)


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20))
def test_projection_offset_respects_bound(texts: list[str]) -> None:
    """Projection with offset(n) skips first n items."""
    messages = [
        Message(id=str(i), role="user", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    offset = min(5, len(texts))
    result = conv.project().offset(offset).to_list()
    expected_count = max(0, len(texts) - offset)

    assert len(result) == expected_count


@given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
def test_projection_reverse_inverts_order(texts: list[str]) -> None:
    """Reverse projection inverts message order."""
    messages = [
        Message(id=str(i), role="user", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    normal = conv.project().to_list()
    reversed_result = conv.project().reverse().to_list()

    assert len(normal) == len(reversed_result)
    if len(normal) > 0:
        assert normal[0].id == reversed_result[-1].id
        assert normal[-1].id == reversed_result[0].id


# =============================================================================
# Conversation Filtering Properties
# =============================================================================


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20))
def test_conversation_filter_never_increases_count(texts: list[str]) -> None:
    """Filtering never increases message count."""
    messages = [
        Message(id=str(i), role="user" if i % 2 == 0 else "assistant", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    original_count = len(conv.messages)
    filtered = conv.user_only()

    assert len(filtered.messages) <= original_count


@given(st.lists(st.text(min_size=1), min_size=0, max_size=20))
def test_conversation_substantive_subset_of_dialogue(texts: list[str]) -> None:
    """Substantive messages are a subset of dialogue messages."""
    messages = [
        Message(id=str(i), role="user" if i % 2 == 0 else "assistant", text=t)
        for i, t in enumerate(texts)
    ]
    conv = Conversation(
        id="test",
        provider="test",
        title="Test",
        messages=messages,
    )

    dialogue_count = len(conv.dialogue_only().messages)
    substantive_count = len(conv.substantive_only().messages)

    assert substantive_count <= dialogue_count
