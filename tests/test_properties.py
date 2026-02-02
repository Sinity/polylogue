"""Property-based tests using Hypothesis for polylogue core invariants.

These tests verify that critical system properties hold for arbitrary inputs,
providing broader coverage than example-based tests.
"""

from __future__ import annotations

import random
from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st

from polylogue.core.code_detection import detect_language, extract_code_block
from polylogue.core.hashing import hash_payload, hash_text, hash_text_short
from polylogue.core.json import dumps, loads
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


@given(st.text(min_size=1).filter(lambda s: s.strip()))
def test_normalize_role_returns_string_for_valid_input(role: str) -> None:
    """Valid non-whitespace input produces a string result."""
    result = normalize_role(role)
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_role_none_handling() -> None:
    """normalize_role rejects None - handle missing roles at parse time."""
    import pytest
    with pytest.raises(AttributeError):
        normalize_role(None)


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


# =============================================================================
# Code Detection Properties
# =============================================================================


@given(st.text())
def test_detect_language_never_crashes(code: str) -> None:
    """detect_language never crashes on arbitrary input."""
    result = detect_language(code)
    assert result is None or isinstance(result, str)


@given(st.text(), st.text(min_size=1, max_size=20))
def test_detect_language_with_hint_never_crashes(code: str, hint: str) -> None:
    """detect_language with hint never crashes."""
    result = detect_language(code, declared_lang=hint)
    assert isinstance(result, str)  # Always returns hint (normalized)


@given(st.sampled_from(["py", "js", "ts", "rs", "sh", "zsh"]))
def test_detect_language_normalizes_aliases(alias: str) -> None:
    """Language aliases are normalized correctly."""
    result = detect_language("", declared_lang=alias)
    expected = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "rs": "rust",
        "sh": "bash",
        "zsh": "bash",
    }
    assert result == expected[alias]


@given(st.text())
def test_extract_code_block_never_crashes(text: str) -> None:
    """extract_code_block never crashes on arbitrary input."""
    result = extract_code_block(text)
    assert isinstance(result, str)


@given(st.text(min_size=1, max_size=100))
def test_extract_code_block_from_fenced(code: str) -> None:
    """Fenced code blocks are extracted correctly."""
    # Create a fenced code block
    fenced = f"```python\n{code}\n```"
    result = extract_code_block(fenced)
    assert code in result or result == code


@given(st.text(min_size=1, max_size=100))
def test_extract_code_block_from_thinking_tags(code: str) -> None:
    """Thinking tags extract content correctly."""
    thinking = f"<thinking>{code}</thinking>"
    result = extract_code_block(thinking)
    assert code.strip() in result or result == code.strip()


# =============================================================================
# JSON Serialization Properties
# =============================================================================


@given(st.text())
def test_json_loads_handles_arbitrary_text(text: str) -> None:
    """loads() handles arbitrary text gracefully (may raise ValueError)."""
    try:
        result = loads(text)
        # If it parses, result should be some JSON value
        assert result is not None or result is None  # Always true, just checking no crash
    except ValueError:
        # Expected for invalid JSON
        pass


@given(st.binary())
def test_json_loads_handles_arbitrary_bytes(data: bytes) -> None:
    """loads() handles arbitrary bytes gracefully (may raise ValueError)."""
    try:
        result = loads(data)
        assert result is not None or result is None
    except ValueError:
        pass


# JSON uses IEEE 754 doubles, so integers beyond ±2^53 lose precision
# Constrain tests to safe integer range
JSON_SAFE_INT = st.integers(min_value=-(2**53), max_value=2**53)


@given(st.dictionaries(st.text(min_size=1, max_size=20), JSON_SAFE_INT))
def test_json_dumps_roundtrip_dicts(d: dict) -> None:
    """JSON dumps/loads roundtrip preserves dicts with safe integers."""
    serialized = dumps(d)
    deserialized = loads(serialized)
    assert deserialized == d


@given(st.lists(JSON_SAFE_INT, max_size=50))
def test_json_dumps_roundtrip_lists(items: list) -> None:
    """JSON dumps/loads roundtrip preserves lists with safe integers."""
    serialized = dumps(items)
    deserialized = loads(serialized)
    assert deserialized == items


@given(st.one_of(st.none(), st.booleans(), JSON_SAFE_INT, st.floats(allow_nan=False, allow_infinity=False), st.text()))
def test_json_dumps_handles_primitives(value) -> None:
    """JSON dumps handles all primitive types."""
    serialized = dumps(value)
    assert isinstance(serialized, str)
    deserialized = loads(serialized)
    if isinstance(value, float):
        # Float precision may vary
        assert abs(deserialized - value) < 1e-10 or deserialized == value
    else:
        assert deserialized == value


# =============================================================================
# Extended Fuzz Properties (Phase 3.6)
# =============================================================================


@given(st.text(min_size=0, max_size=1000))
def test_safe_path_component_never_crashes(text: str) -> None:
    """safe_path_component never crashes on arbitrary input."""
    from polylogue.paths import safe_path_component

    result = safe_path_component(text)
    # Should always return a string
    assert isinstance(result, str)
    # Should never be empty
    assert len(result) > 0
    # Should not contain path separators
    assert "/" not in result
    assert "\\" not in result


@given(st.binary(min_size=0, max_size=1000))
def test_content_hash_deterministic_bytes(data: bytes) -> None:
    """hash_text on decoded bytes is deterministic."""
    try:
        text = data.decode("utf-8", errors="replace")
        result1 = hash_text(text)
        result2 = hash_text(text)
        assert result1 == result2
    except UnicodeDecodeError:
        pass  # Skip invalid UTF-8


@given(st.text(min_size=1, max_size=500))
def test_hash_text_unique_for_different_inputs(text: str) -> None:
    """Different inputs produce different hashes (with high probability).

    Note: This tests collision resistance, not mathematical proof.
    """
    # Adding a suffix should produce different hash
    original = hash_text(text)
    modified = hash_text(text + "x")
    assert original != modified


@given(st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
def test_normalize_role_never_crashes(role: str) -> None:
    """normalize_role never crashes on valid non-empty input."""
    result = normalize_role(role)
    assert isinstance(result, str)
    assert len(result) > 0  # Always returns something


@given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=100))
def test_conversation_messages_preserved(texts: list[str]) -> None:
    """Conversation preserves all messages through transformations."""
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

    # All messages should be present
    assert len(conv.messages) == len(texts)

    # Round-trip through projection
    projected = conv.project().to_list()
    assert len(projected) == len(texts)


@given(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=0, max_size=100)))
def test_hash_payload_never_crashes(d: dict) -> None:
    """hash_payload never crashes on arbitrary dict."""
    result = hash_payload(d)
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hex digest
