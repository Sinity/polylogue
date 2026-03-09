"""Semantic invariant laws for Message and Conversation models.

These laws encode structural guarantees that hold for any Message, regardless
of input. They supersede specific example tests in test_models.py.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.infra.strategies.messages import conversation_model_strategy, message_model_strategy


# ---------------------------------------------------------------------------
# Law 1: is_noise and is_substantive are mutually exclusive
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_noise_and_substantive_mutually_exclusive(msg) -> None:
    """No message can be both noise and substantive."""
    assert not (msg.is_noise and msg.is_substantive)


# ---------------------------------------------------------------------------
# Law 2: is_thinking and is_substantive are mutually exclusive
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_thinking_and_substantive_mutually_exclusive(msg) -> None:
    """Thinking messages are never substantive."""
    assert not (msg.is_thinking and msg.is_substantive)


# ---------------------------------------------------------------------------
# Law 3: tool_use implies noise
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_tool_use_implies_noise(msg) -> None:
    """If a message is tool use, it is always noise."""
    if msg.is_tool_use:
        assert msg.is_noise


# ---------------------------------------------------------------------------
# Law 4: context_dump implies noise
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_context_dump_implies_noise(msg) -> None:
    """Context dump messages are always noise."""
    if msg.is_context_dump:
        assert msg.is_noise


# ---------------------------------------------------------------------------
# Law 5: substantive implies dialogue
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_substantive_implies_dialogue(msg) -> None:
    """Substantive messages are always dialogue (user or assistant)."""
    if msg.is_substantive:
        assert msg.is_dialogue


# ---------------------------------------------------------------------------
# Law 6: dialogue implies user OR assistant
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_dialogue_implies_user_or_assistant(msg) -> None:
    """Dialogue messages have role user or assistant."""
    if msg.is_dialogue:
        assert msg.is_user or msg.is_assistant


# ---------------------------------------------------------------------------
# Law 7: without_noise removes all noise messages
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_without_noise_removes_all_noise(conv) -> None:
    """After without_noise(), no message is_noise."""
    clean = conv.without_noise()
    assert all(not m.is_noise for m in clean.messages)


# ---------------------------------------------------------------------------
# Law 8: substantive_only returns only substantive messages
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_substantive_only_all_substantive(conv) -> None:
    """After substantive_only(), every message is_substantive."""
    filtered = conv.substantive_only()
    assert all(m.is_substantive for m in filtered.messages)


# ---------------------------------------------------------------------------
# Law 9: without_noise preserves all non-noise messages (count + identity)
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_without_noise_preserves_non_noise_count(conv) -> None:
    """without_noise() keeps exactly the non-noise messages and nothing else.

    Motivated by mutmut finding: all(... for m in []) is vacuously True, so
    the emptiness law alone doesn't catch a filter that discards everything.
    """
    expected_ids = [m.id for m in conv.messages if not m.is_noise]
    clean = conv.without_noise()
    assert [m.id for m in clean.messages] == expected_ids


# ---------------------------------------------------------------------------
# Law 10: substantive_only preserves all substantive messages (count + identity)
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_substantive_only_preserves_count(conv) -> None:
    """substantive_only() keeps exactly the substantive messages."""
    expected_ids = [m.id for m in conv.messages if m.is_substantive]
    filtered = conv.substantive_only()
    assert [m.id for m in filtered.messages] == expected_ids


# ---------------------------------------------------------------------------
# Law 11: user_only preserves all user messages (count + identity)
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_user_only_preserves_count(conv) -> None:
    """user_only() keeps exactly the user-role messages."""
    expected_ids = [m.id for m in conv.messages if m.is_user]
    filtered = conv.user_only()
    assert [m.id for m in filtered.messages] == expected_ids


# ---------------------------------------------------------------------------
# Law 13: without_noise is idempotent (now verified sound by Laws 9+11)
# ---------------------------------------------------------------------------

@given(conversation_model_strategy())
def test_without_noise_idempotent(conv) -> None:
    """without_noise(without_noise(c)) equals without_noise(c)."""
    once = conv.without_noise()
    twice = once.without_noise()
    assert [m.id for m in once.messages] == [m.id for m in twice.messages]


# ---------------------------------------------------------------------------
# Law 14: extract_thinking is None or non-empty (never empty string)
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_extract_thinking_never_empty_string(msg) -> None:
    """extract_thinking() returns None or a non-empty string."""
    result = msg.extract_thinking()
    assert result is None or (isinstance(result, str) and len(result.strip()) > 0)


# ---------------------------------------------------------------------------
# Law 15: cost_usd is None or positive
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_cost_usd_none_or_positive(msg) -> None:
    """cost_usd is None or strictly positive."""
    if msg.cost_usd is not None:
        assert msg.cost_usd > 0


# ---------------------------------------------------------------------------
# Law 16: duration_ms is None or positive
# ---------------------------------------------------------------------------

@given(message_model_strategy())
def test_duration_ms_none_or_positive(msg) -> None:
    """duration_ms is None or strictly positive."""
    if msg.duration_ms is not None:
        assert msg.duration_ms > 0
