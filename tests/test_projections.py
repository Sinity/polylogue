"""Comprehensive tests for polylogue.lib.projections module.

Tests the ConversationProjection fluent API for filtering, transforming,
and querying conversation messages with lazy evaluation.
"""
import re
from datetime import datetime

import pytest

from polylogue.lib.models import Attachment, Conversation, Message


# --- Fixtures ---


@pytest.fixture
def sample_messages():
    """Create a diverse set of messages for testing filters."""
    return [
        Message(
            id="m1",
            role="user",
            text="Hello, can you help me debug this error?",
            timestamp=datetime(2024, 1, 1, 10, 0),
        ),
        Message(
            id="m2",
            role="assistant",
            text="Of course! Please share the error message.",
            timestamp=datetime(2024, 1, 1, 10, 1),
        ),
        Message(
            id="m3",
            role="user",
            text="Here's the error log with details about the exception",
            timestamp=datetime(2024, 1, 1, 10, 2),
            attachments=[
                Attachment(
                    id="att1",
                    name="error.log",
                    mime_type="text/plain",
                    size_bytes=1024,
                )
            ],
        ),
        Message(
            id="m4",
            role="assistant",
            text="Let me analyze this. <thinking>Need to check the stack trace carefully</thinking> The error is caused by...",
            timestamp=datetime(2024, 1, 1, 10, 3),
        ),
        Message(
            id="m5",
            role="user",
            text="Thanks!",
            timestamp=datetime(2024, 1, 1, 10, 4),
        ),
        Message(
            id="m6",
            role="assistant",
            text="You're welcome! Let me know if you need anything else.",
            timestamp=datetime(2024, 1, 1, 10, 5),
        ),
        Message(
            id="m7",
            role="tool",
            text='{"tool_use": "read_file", "result": "success"}',
            timestamp=datetime(2024, 1, 1, 10, 6),
        ),
        Message(
            id="m8",
            role="system",
            text="System prompt with instructions",
            timestamp=datetime(2024, 1, 1, 10, 7),
        ),
    ]


@pytest.fixture
def sample_conversation(sample_messages):
    """Create a conversation with diverse messages."""
    return Conversation(
        id="conv1",
        provider="test",
        title="Test Conversation",
        messages=sample_messages,
        created_at=datetime(2024, 1, 1, 9, 0),
    )


@pytest.fixture
def empty_conversation():
    """Create an empty conversation."""
    return Conversation(
        id="empty",
        provider="test",
        messages=[],
    )


# --- Filter Tests ---


def test_projection_user_messages_filter(sample_conversation):
    """Filter to user messages only."""
    result = sample_conversation.project().user_messages().execute()

    assert len(result.messages) == 3
    for msg in result.messages:
        assert msg.is_user
    assert {m.id for m in result.messages} == {"m1", "m3", "m5"}


def test_projection_assistant_messages_filter(sample_conversation):
    """Filter to assistant messages only."""
    result = sample_conversation.project().assistant_messages().execute()

    assert len(result.messages) == 3
    for msg in result.messages:
        assert msg.is_assistant
    assert {m.id for m in result.messages} == {"m2", "m4", "m6"}


def test_projection_dialogue_filter(sample_conversation):
    """Filter to dialogue messages (user + assistant)."""
    result = sample_conversation.project().dialogue().execute()

    assert len(result.messages) == 6
    for msg in result.messages:
        assert msg.is_dialogue
    # Should exclude tool (m7) and system (m8)
    assert all(msg.id not in {"m7", "m8"} for msg in result.messages)


def test_projection_substantive_filter(sample_conversation):
    """Filter to substantive messages only."""
    result = sample_conversation.project().substantive().execute()

    # m1, m2, m3, m6 are substantive
    # m4 has thinking so not substantive
    # m5 is too short
    # m7, m8 are noise
    for msg in result.messages:
        assert msg.is_substantive


def test_projection_without_noise_filter(sample_conversation):
    """Exclude noise messages (tool use, context dumps, system)."""
    result = sample_conversation.project().without_noise().execute()

    for msg in result.messages:
        assert not msg.is_noise
    # Should exclude m7 (tool) and m8 (system)
    assert all(msg.id not in {"m7", "m8"} for msg in result.messages)


def test_projection_with_attachments_filter(sample_conversation):
    """Filter to messages with attachments."""
    result = sample_conversation.project().with_attachments().execute()

    assert len(result.messages) == 1
    assert result.messages[0].id == "m3"
    assert len(result.messages[0].attachments) == 1


def test_projection_min_words_filter(sample_conversation):
    """Filter to messages with minimum word count."""
    result = sample_conversation.project().min_words(8).execute()

    # Only messages with >= 8 words
    for msg in result.messages:
        assert msg.word_count >= 8
    # m1, m3, m6 should match
    assert len(result.messages) >= 2


def test_projection_max_words_filter(sample_conversation):
    """Filter to messages with maximum word count."""
    result = sample_conversation.project().max_words(5).execute()

    # Only messages with <= 5 words
    for msg in result.messages:
        assert msg.word_count <= 5
    # m5 "Thanks!" should match
    assert any(msg.id == "m5" for msg in result.messages)


def test_projection_contains_case_insensitive(sample_conversation):
    """Filter messages containing text (case insensitive by default)."""
    result = sample_conversation.project().contains("ERROR").execute()

    # Should match m1, m2, m3, m4 (all contain "error")
    assert len(result.messages) == 4
    for msg in result.messages:
        assert "error" in msg.text.lower()


def test_projection_contains_case_sensitive(sample_conversation):
    """Filter messages containing text (case sensitive)."""
    result = sample_conversation.project().contains("error", case_sensitive=True).execute()

    # Should match messages with lowercase "error"
    for msg in result.messages:
        assert "error" in msg.text


def test_projection_matches_regex(sample_conversation):
    """Filter messages matching regex pattern."""
    result = sample_conversation.project().matches(r"error|exception").execute()

    # Should match m1 and m3
    assert len(result.messages) >= 2
    for msg in result.messages:
        assert re.search(r"error|exception", msg.text) is not None


def test_projection_since_filter(sample_conversation):
    """Filter messages after a timestamp."""
    cutoff = datetime(2024, 1, 1, 10, 3)
    result = sample_conversation.project().since(cutoff).execute()

    for msg in result.messages:
        assert msg.timestamp >= cutoff
    # Should include m4, m5, m6, m7, m8
    assert len(result.messages) == 5


def test_projection_until_filter(sample_conversation):
    """Filter messages before a timestamp."""
    cutoff = datetime(2024, 1, 1, 10, 2)
    result = sample_conversation.project().until(cutoff).execute()

    for msg in result.messages:
        assert msg.timestamp <= cutoff
    # Should include m1, m2, m3
    assert len(result.messages) == 3


def test_projection_between_filter(sample_conversation):
    """Filter messages within a time range."""
    start = datetime(2024, 1, 1, 10, 1)
    end = datetime(2024, 1, 1, 10, 4)
    result = sample_conversation.project().between(start, end).execute()

    for msg in result.messages:
        assert start <= msg.timestamp <= end
    # Should include m2, m3, m4, m5
    assert len(result.messages) == 4


def test_projection_thinking_only_filter(sample_conversation):
    """Filter to messages containing thinking traces."""
    result = sample_conversation.project().thinking_only().execute()

    for msg in result.messages:
        assert msg.is_thinking
    # m4 has <thinking> tags
    assert any(msg.id == "m4" for msg in result.messages)


def test_projection_tool_use_only_filter(sample_conversation):
    """Filter to tool use messages."""
    result = sample_conversation.project().tool_use_only().execute()

    for msg in result.messages:
        assert msg.is_tool_use
    # m7 is a tool message
    assert len(result.messages) == 1
    assert result.messages[0].id == "m7"


def test_projection_where_custom_predicate(sample_conversation):
    """Filter with custom predicate."""
    result = sample_conversation.project().where(lambda m: len(m.id) == 2).execute()

    # All message IDs have length 2
    assert len(result.messages) == len(sample_conversation.messages)


# --- Chaining Tests ---


def test_projection_chaining_multiple_filters(sample_conversation):
    """Chain multiple filters together."""
    result = (
        sample_conversation.project()
        .user_messages()
        .min_words(5)
        .contains("error")
        .execute()
    )

    # Should match user messages with >= 5 words containing "error"
    for msg in result.messages:
        assert msg.is_user
        assert msg.word_count >= 5
        assert "error" in msg.text.lower()
    # m1 and m3 should match
    assert len(result.messages) == 2


def test_projection_chaining_filter_and_transform(sample_conversation):
    """Chain filters with transforms."""
    result = (
        sample_conversation.project()
        .assistant_messages()
        .strip_attachments()
        .execute()
    )

    for msg in result.messages:
        assert msg.is_assistant
        assert len(msg.attachments) == 0


# --- Transform Tests ---


def test_projection_truncate_text_transform(sample_conversation):
    """Transform to truncate message text."""
    result = sample_conversation.project().truncate_text(20).execute()

    for msg in result.messages:
        if msg.text and len(msg.text) > 23:  # 20 + "..."
            # Should be truncated
            assert len(msg.text) <= 23
            assert msg.text.endswith("...")


def test_projection_truncate_text_custom_suffix(sample_conversation):
    """Transform to truncate with custom suffix."""
    result = sample_conversation.project().truncate_text(20, suffix=" [...]").execute()

    for msg in result.messages:
        if msg.text and len(msg.text) > 26:  # 20 + " [...]"
            assert " [...]" in msg.text


def test_projection_strip_attachments_transform(sample_conversation):
    """Transform to remove attachments."""
    result = sample_conversation.project().strip_attachments().execute()

    for msg in result.messages:
        assert len(msg.attachments) == 0
    # Original message m3 had attachment, should be removed
    m3_result = next(m for m in result.messages if m.id == "m3")
    assert len(m3_result.attachments) == 0


def test_projection_transform_custom(sample_conversation):
    """Apply custom transform function."""
    def uppercase_text(msg: Message) -> Message:
        if msg.text:
            return msg.model_copy(update={"text": msg.text.upper()})
        return msg

    result = sample_conversation.project().transform(uppercase_text).execute()

    for msg in result.messages:
        if msg.text:
            assert msg.text == msg.text.upper()


# --- Pagination Tests ---


def test_projection_limit_offset(sample_conversation):
    """Test limit and offset pagination."""
    result = sample_conversation.project().offset(2).limit(3).execute()

    assert len(result.messages) == 3
    # Should be m3, m4, m5 (skip first 2)
    assert result.messages[0].id == "m3"
    assert result.messages[1].id == "m4"
    assert result.messages[2].id == "m5"


def test_projection_first_n(sample_conversation):
    """Get first N messages."""
    result = sample_conversation.project().first_n(3).execute()

    assert len(result.messages) == 3
    assert result.messages[0].id == "m1"
    assert result.messages[1].id == "m2"
    assert result.messages[2].id == "m3"


def test_projection_last_n(sample_conversation):
    """Get last N messages."""
    result = sample_conversation.project().last_n(3).execute()

    assert len(result.messages) == 3
    # Should be last 3 in reverse order due to reverse + limit
    # But the implementation uses reverse().limit(n), so they come out reversed
    assert result.messages[0].id == "m8"
    assert result.messages[1].id == "m7"
    assert result.messages[2].id == "m6"


def test_projection_reverse_order(sample_conversation):
    """Reverse message order."""
    result = sample_conversation.project().reverse().execute()

    assert len(result.messages) == len(sample_conversation.messages)
    # Should be in reverse order
    assert result.messages[0].id == "m8"
    assert result.messages[-1].id == "m1"


def test_projection_reverse_twice_restores_order(sample_conversation):
    """Calling reverse twice restores original order."""
    result = sample_conversation.project().reverse().reverse().execute()

    assert len(result.messages) == len(sample_conversation.messages)
    assert result.messages[0].id == "m1"
    assert result.messages[-1].id == "m8"


# --- Terminal Operation Tests ---


def test_projection_execute_returns_new_conversation(sample_conversation):
    """Execute returns a new Conversation instance."""
    result = sample_conversation.project().user_messages().execute()

    assert isinstance(result, Conversation)
    assert result.id == sample_conversation.id
    assert result.provider == sample_conversation.provider
    assert result is not sample_conversation  # Different instance


def test_projection_count_without_materialization(sample_conversation):
    """Count matching messages without creating full list."""
    count = sample_conversation.project().user_messages().count()

    assert count == 3
    assert isinstance(count, int)


def test_projection_first_returns_first_match(sample_conversation):
    """Get first matching message."""
    first = sample_conversation.project().assistant_messages().first()

    assert first is not None
    assert first.id == "m2"
    assert first.is_assistant


def test_projection_first_returns_none_when_empty(sample_conversation):
    """First returns None when no matches."""
    first = sample_conversation.project().contains("nonexistent_text_xyz").first()

    assert first is None


def test_projection_last_returns_last_match(sample_conversation):
    """Get last matching message."""
    last = sample_conversation.project().assistant_messages().last()

    assert last is not None
    assert last.id == "m6"
    assert last.is_assistant


def test_projection_last_returns_none_when_empty(sample_conversation):
    """Last returns None when no matches."""
    last = sample_conversation.project().contains("nonexistent_text_xyz").last()

    assert last is None


def test_projection_to_list(sample_conversation):
    """Materialize projection as a list."""
    result = sample_conversation.project().user_messages().to_list()

    assert isinstance(result, list)
    assert len(result) == 3
    for msg in result:
        assert isinstance(msg, Message)
        assert msg.is_user


def test_projection_exists_true_when_matches(sample_conversation):
    """Exists returns True when matches found."""
    exists = sample_conversation.project().contains("error").exists()

    assert exists is True


def test_projection_exists_false_when_no_matches(sample_conversation):
    """Exists returns False when no matches found."""
    exists = sample_conversation.project().contains("nonexistent_text_xyz").exists()

    assert exists is False


def test_projection_to_text_formatting(sample_conversation):
    """Render matching messages as text."""
    text = sample_conversation.project().user_messages().to_text()

    assert "user:" in text.lower()
    assert "Hello, can you help me debug this error?" in text
    assert "\n\n" in text  # Default separator


def test_projection_to_text_without_role(sample_conversation):
    """Render text without role prefix."""
    text = sample_conversation.project().user_messages().to_text(include_role=False)

    assert "user:" not in text.lower()
    assert "Hello, can you help me debug this error?" in text


def test_projection_to_text_custom_separator(sample_conversation):
    """Render text with custom separator."""
    text = sample_conversation.project().user_messages().to_text(separator=" | ")

    assert " | " in text


def test_projection_iter_lazy_evaluation(sample_conversation):
    """Iter provides lazy evaluation without materialization."""
    projection = sample_conversation.project().assistant_messages()
    iterator = projection.iter()

    # Iterator should be lazy
    assert hasattr(iterator, "__next__")

    # Consume manually
    messages = []
    for msg in iterator:
        messages.append(msg)
        if len(messages) == 2:
            break  # Stop early

    assert len(messages) == 2
    assert all(msg.is_assistant for msg in messages)


# --- Empty Conversation Tests ---


def test_projection_empty_conversation(empty_conversation):
    """Handle empty conversation gracefully."""
    result = empty_conversation.project().user_messages().execute()

    assert len(result.messages) == 0


def test_projection_empty_count(empty_conversation):
    """Count on empty conversation returns 0."""
    count = empty_conversation.project().count()

    assert count == 0


def test_projection_empty_first(empty_conversation):
    """First on empty conversation returns None."""
    first = empty_conversation.project().first()

    assert first is None


def test_projection_empty_last(empty_conversation):
    """Last on empty conversation returns None."""
    last = empty_conversation.project().last()

    assert last is None


def test_projection_empty_exists(empty_conversation):
    """Exists on empty conversation returns False."""
    exists = empty_conversation.project().exists()

    assert exists is False


def test_projection_empty_to_list(empty_conversation):
    """To_list on empty conversation returns empty list."""
    result = empty_conversation.project().to_list()

    assert result == []


def test_projection_empty_to_text(empty_conversation):
    """To_text on empty conversation returns empty string."""
    text = empty_conversation.project().to_text()

    assert text == ""


# --- Edge Cases ---


def test_projection_filter_no_matches(sample_conversation):
    """Filter that matches nothing returns empty conversation."""
    result = sample_conversation.project().where(lambda m: False).execute()

    assert len(result.messages) == 0


def test_projection_limit_larger_than_results(sample_conversation):
    """Limit larger than available results returns all matches."""
    result = sample_conversation.project().user_messages().limit(100).execute()

    assert len(result.messages) == 3  # Only 3 user messages


def test_projection_offset_larger_than_results(sample_conversation):
    """Offset larger than available results returns empty."""
    result = sample_conversation.project().offset(100).execute()

    assert len(result.messages) == 0


def test_projection_limit_zero(sample_conversation):
    """Limit of zero returns no messages."""
    result = sample_conversation.project().limit(0).execute()

    assert len(result.messages) == 0


def test_projection_offset_zero(sample_conversation):
    """Offset of zero starts from beginning."""
    result = sample_conversation.project().offset(0).limit(3).execute()

    assert len(result.messages) == 3
    assert result.messages[0].id == "m1"


def test_projection_messages_with_none_text():
    """Handle messages with None text gracefully."""
    messages = [
        Message(id="m1", role="user", text=None),
        Message(id="m2", role="assistant", text="Hello"),
    ]
    conv = Conversation(id="conv", provider="test", messages=messages)

    result = conv.project().contains("Hello").execute()

    # Should only match m2, not crash on None
    assert len(result.messages) == 1
    assert result.messages[0].id == "m2"


def test_projection_messages_with_none_timestamp():
    """Handle messages with None timestamp gracefully."""
    messages = [
        Message(id="m1", role="user", text="Hello", timestamp=None),
        Message(id="m2", role="assistant", text="Hi", timestamp=datetime(2024, 1, 1)),
    ]
    conv = Conversation(id="conv", provider="test", messages=messages)

    result = conv.project().since(datetime(2024, 1, 1)).execute()

    # Should only match m2 (m1 has None timestamp)
    assert len(result.messages) == 1
    assert result.messages[0].id == "m2"


# --- Complex Scenarios ---


def test_projection_complex_pipeline(sample_conversation):
    """Test complex filter + transform + pagination pipeline."""
    result = (
        sample_conversation.project()
        .dialogue()
        .without_noise()
        .min_words(5)
        .since(datetime(2024, 1, 1, 10, 0))
        .until(datetime(2024, 1, 1, 10, 5))
        .truncate_text(50)
        .limit(10)
        .execute()
    )

    for msg in result.messages:
        assert msg.is_dialogue
        assert not msg.is_noise
        assert msg.word_count >= 5
        assert msg.timestamp >= datetime(2024, 1, 1, 10, 0)
        assert msg.timestamp <= datetime(2024, 1, 1, 10, 5)
        if msg.text:
            assert len(msg.text) <= 53  # 50 + "..."


def test_projection_reusability(sample_conversation):
    """Projection builder can be reused with different terminals."""
    projection = sample_conversation.project().user_messages()

    # Use different terminal operations
    count = projection.count()
    first = projection.first()
    exists = projection.exists()
    result = projection.execute()

    assert count == 3
    assert first.id == "m1"
    assert exists is True
    assert len(result.messages) == 3


def test_projection_immutability(sample_conversation):
    """Original conversation is not modified by projections."""
    original_count = len(sample_conversation.messages)
    original_first_id = sample_conversation.messages[0].id

    _ = sample_conversation.project().user_messages().execute()

    # Original should be unchanged
    assert len(sample_conversation.messages) == original_count
    assert sample_conversation.messages[0].id == original_first_id
