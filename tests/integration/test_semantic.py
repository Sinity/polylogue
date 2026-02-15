"""Comprehensive tests for polylogue.lib.models semantic API.

Covers:
1. DialoguePair validation and properties
2. Message metadata extraction and properties
3. Conversation metadata and aggregation
4. Iteration helpers (pairs, thinking, substantive, dialogue)
5. Conversation rendering (to_text, to_clean_text)
"""

from datetime import datetime

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Attachment, Conversation, DialoguePair, Message
from tests.infra.helpers import assert_contains_all, assert_not_contains_any

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_messages():
    """Create basic user/assistant message pair."""
    return [
        Message(id="u1", role="user", text="What is machine learning?"),
        Message(id="a1", role="assistant", text="Machine learning is a subset of AI..."),
    ]


@pytest.fixture
def messages_with_metadata():
    """Messages with provider-specific cost/duration metadata."""
    return [
        Message(
            id="u1",
            role="user",
            text="Can you help with this?",
            provider_meta={"costUSD": 0.001},
        ),
        Message(
            id="a1",
            role="assistant",
            text="Yes, I can help.",
            provider_meta={"costUSD": 0.005, "durationMs": 2500},
        ),
        Message(
            id="u2",
            role="user",
            text="Great, now what?",
            provider_meta={"costUSD": 0.001},
        ),
        Message(
            id="a2",
            role="assistant",
            text="Let me explain further.",
            provider_meta={"costUSD": 0.008, "durationMs": 3000},
        ),
    ]


@pytest.fixture
def messages_with_thinking():
    """Messages with thinking/reasoning traces."""
    return [
        Message(
            id="u1",
            role="user",
            text="Solve: 2x + 5 = 13",
        ),
        Message(
            id="a1",
            role="assistant",
            text="<thinking>Let me work through this step by step.\n2x + 5 = 13\n2x = 8\nx = 4</thinking>\nThe answer is x = 4.",
            provider_meta={"content_blocks": [{"type": "thinking"}]},
        ),
    ]


@pytest.fixture
def messages_with_tool_use():
    """Messages including tool calls and results."""
    return [
        Message(id="u1", role="user", text="What time is it in Tokyo?"),
        Message(
            id="a1",
            role="assistant",
            text="Let me check the current time...",
            provider_meta={"content_blocks": [{"type": "tool_use", "name": "get_time"}]},
        ),
        Message(
            id="t1",
            role="tool",
            text='{"time": "2024-01-15T14:30:00+09:00"}',
        ),
        Message(
            id="a2",
            role="assistant",
            text="The current time in Tokyo is 2:30 PM (14:30).",
        ),
    ]


@pytest.fixture
def complex_conversation(messages_with_metadata, messages_with_thinking, messages_with_tool_use):
    """Complex conversation with various message types."""
    return Conversation(
        id="complex-conv",
        provider="claude",
        title="Complex Conversation",
        messages=MessageCollection(messages=[
            *messages_with_metadata,
            *messages_with_thinking,
            *messages_with_tool_use,
        ]),
        created_at=datetime(2024, 1, 15, 10, 0),
        metadata={"tags": ["test", "comprehensive"], "summary": "A test conversation"},
    )


@pytest.fixture
def empty_conversation():
    """Empty conversation for edge cases."""
    return Conversation(
        id="empty",
        provider="test",
        messages=MessageCollection(messages=[]),
    )


# =============================================================================
# 1. DIALOGUEPAIR VALIDATION TESTS (10 tests)
# =============================================================================


def test_dialogue_pair_validates_roles():
    """DialoguePair ensures user→assistant role order."""
    user_msg = Message(id="u1", role="user", text="Hello?")
    assistant_msg = Message(id="a1", role="assistant", text="Hi there!")
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    assert pair.user.is_user
    assert pair.assistant.is_assistant


def test_dialogue_pair_rejects_invalid_roles():
    """DialoguePair rejects non-user user message."""
    assistant_msg = Message(id="a1", role="assistant", text="Hi")
    wrong_assistant = Message(id="a2", role="assistant", text="Hello")
    with pytest.raises(ValueError, match="user message must have user role"):
        DialoguePair(user=assistant_msg, assistant=wrong_assistant)


def test_dialogue_pair_rejects_non_assistant_response():
    """DialoguePair rejects non-assistant assistant message."""
    user_msg = Message(id="u1", role="user", text="Hi")
    system_msg = Message(id="s1", role="system", text="System")
    with pytest.raises(ValueError, match="assistant message must have assistant role"):
        DialoguePair(user=user_msg, assistant=system_msg)


def test_dialogue_pair_exchange_property():
    """DialoguePair.exchange renders user/assistant text."""
    user_msg = Message(id="u1", role="user", text="What's 2+2?")
    assistant_msg = Message(id="a1", role="assistant", text="The answer is 4.")
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    exchange = pair.exchange
    assert "User: What's 2+2?" in exchange
    assert "Assistant: The answer is 4." in exchange


def test_dialogue_pair_from_consecutive_messages():
    """DialoguePair created from consecutive user/assistant messages."""
    messages = [
        Message(id="u1", role="user", text="Question"),
        Message(id="a1", role="assistant", text="Answer"),
    ]
    pair = DialoguePair(user=messages[0], assistant=messages[1])
    assert pair.user.text == "Question"
    assert pair.assistant.text == "Answer"


def test_dialogue_pair_handles_empty_text():
    """DialoguePair accepts messages with empty text."""
    user_msg = Message(id="u1", role="user", text="")
    assistant_msg = Message(id="a1", role="assistant", text="")
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    assert pair.user.text == ""
    assert pair.assistant.text == ""


def test_dialogue_pair_preserves_metadata():
    """DialoguePair preserves message metadata."""
    user_meta = {"source": "cli"}
    assistant_meta = {"tokens": 150}
    user_msg = Message(id="u1", role="user", text="Q", provider_meta=user_meta)
    assistant_msg = Message(id="a1", role="assistant", text="A", provider_meta=assistant_meta)
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    assert pair.user.provider_meta == user_meta
    assert pair.assistant.provider_meta == assistant_meta


def test_dialogue_pair_with_thinking_messages():
    """DialoguePair preserves thinking blocks in assistant message."""
    user_msg = Message(id="u1", role="user", text="Hard problem")
    assistant_msg = Message(
        id="a1",
        role="assistant",
        text="<thinking>Complex reasoning</thinking>\nAnswer",
        provider_meta={"content_blocks": [{"type": "thinking"}]},
    )
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    assert pair.assistant.is_thinking
    assert pair.assistant.extract_thinking() == "Complex reasoning"


def test_extract_thinking_from_content_blocks_only():
    """extract_thinking returns text from content_blocks when no XML tags in text."""
    msg = Message(
        id="a1",
        role="assistant",
        text="The answer is 42.",
        provider_meta={
            "content_blocks": [
                {"type": "thinking", "text": "Let me reason through this step by step."},
                {"type": "text", "text": "The answer is 42."},
            ]
        },
    )
    assert msg.is_thinking
    assert msg.extract_thinking() == "Let me reason through this step by step."


def test_extract_thinking_multiple_content_blocks():
    """extract_thinking joins multiple thinking blocks."""
    msg = Message(
        id="a1",
        role="assistant",
        text="Done.",
        provider_meta={
            "content_blocks": [
                {"type": "thinking", "text": "First thought."},
                {"type": "text", "text": "Done."},
                {"type": "thinking", "text": "Second thought."},
            ]
        },
    )
    assert msg.extract_thinking() == "First thought.\n\nSecond thought."


def test_extract_thinking_gemini_is_thought():
    """extract_thinking returns full text for Gemini isThought messages."""
    msg = Message(
        id="g1",
        role="model",
        text="Considering the implications of quantum entanglement...",
        provider_meta={"isThought": True},
    )
    assert msg.is_thinking
    assert msg.extract_thinking() == "Considering the implications of quantum entanglement..."


def test_extract_thinking_chatgpt_thoughts():
    """extract_thinking returns full text for ChatGPT thinking messages."""
    msg = Message(
        id="c1",
        role="tool",
        text="The user is asking about error handling patterns.",
        provider_meta={
            "raw": {
                "content": {"content_type": "thoughts"},
                "metadata": {},
            }
        },
    )
    assert msg.is_thinking
    assert msg.extract_thinking() == "The user is asking about error handling patterns."


def test_extract_thinking_none_when_not_thinking():
    """extract_thinking returns None for non-thinking messages."""
    msg = Message(id="m1", role="assistant", text="Hello!")
    assert not msg.is_thinking
    assert msg.extract_thinking() is None


def test_dialogue_pair_with_tool_use():
    """DialoguePair assistant message can contain tool use."""
    user_msg = Message(id="u1", role="user", text="Search for X")
    assistant_msg = Message(
        id="a1",
        role="assistant",
        text="Searching...",
        provider_meta={"content_blocks": [{"type": "tool_use"}]},
    )
    pair = DialoguePair(user=user_msg, assistant=assistant_msg)
    assert pair.assistant.is_tool_use


# =============================================================================
# 2. MESSAGE METADATA TESTS (8 tests)
# =============================================================================


def test_message_cost_usd_from_provider_meta():
    """Message.cost_usd extracts from provider_meta."""
    msg = Message(id="1", role="assistant", text="Response", provider_meta={"costUSD": 0.042})
    assert msg.cost_usd == 0.042


def test_message_cost_usd_fallback_none():
    """Message.cost_usd returns None when not present."""
    msg = Message(id="1", role="assistant", text="Response")
    assert msg.cost_usd is None


def test_message_duration_ms_from_provider_meta():
    """Message.duration_ms extracts from provider_meta."""
    msg = Message(id="1", role="assistant", text="Response", provider_meta={"durationMs": 2500})
    assert msg.duration_ms == 2500


def test_message_duration_ms_fallback_none():
    """Message.duration_ms returns None when not present."""
    msg = Message(id="1", role="assistant", text="Response")
    assert msg.duration_ms is None


def test_message_provider_meta_different_formats():
    """Message handles different provider metadata structures."""
    # Claude-code style
    claude_msg = Message(
        id="1",
        role="assistant",
        text="Response",
        provider_meta={"costUSD": 0.01, "durationMs": 1000},
    )
    assert claude_msg.cost_usd == 0.01
    assert claude_msg.duration_ms == 1000

    # ChatGPT style with nested structure
    chatgpt_msg = Message(
        id="2",
        role="assistant",
        text="Response",
        provider_meta={"raw": {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}},
    )
    assert chatgpt_msg.provider_meta["raw"]["usage"]["prompt_tokens"] == 10


def test_message_metadata_immutability():
    """Message provider_meta should not be modifiable after creation."""
    msg = Message(id="1", role="user", text="Hello", provider_meta={"key": "value"})
    # Accessing provider_meta should work
    assert msg.provider_meta["key"] == "value"
    # But mutation of the dict affects the message
    msg.provider_meta["new_key"] = "new_value"
    assert msg.provider_meta["new_key"] == "new_value"


def test_message_with_attachments_metadata():
    """Message preserves attachment metadata."""
    attach = Attachment(
        id="a1",
        name="doc.pdf",
        mime_type="application/pdf",
        size_bytes=5000,
        provider_meta={"uploaded_by": "user"},
    )
    msg = Message(id="1", role="user", text="Review this", attachments=[attach])
    assert len(msg.attachments) == 1
    assert msg.attachments[0].name == "doc.pdf"
    assert msg.attachments[0].provider_meta["uploaded_by"] == "user"


def test_message_classification_metadata():
    """Message classification works with various metadata formats."""
    # Thinking via content_blocks
    thinking_msg = Message(
        id="1",
        role="assistant",
        text="<thinking>...</thinking>",
        provider_meta={"content_blocks": [{"type": "thinking"}]},
    )
    assert thinking_msg.is_thinking

    # Tool use via content_blocks
    tool_msg = Message(
        id="2",
        role="assistant",
        text="Calling tool",
        provider_meta={"content_blocks": [{"type": "tool_use"}]},
    )
    assert tool_msg.is_tool_use


# =============================================================================
# 3. CONVERSATION METADATA TESTS (12 tests)
# =============================================================================


def test_conversation_user_title_override():
    """Conversation.user_title from metadata overrides stored title."""
    conv = Conversation(
        id="c1",
        provider="test",
        title="Stored Title",
        messages=MessageCollection(messages=[]),
        metadata={"title": "User Override"},
    )
    assert conv.user_title == "User Override"


def test_conversation_display_title_precedence():
    """Conversation.display_title precedence: user_title > title > id."""
    # User title has priority
    conv1 = Conversation(
        id="abc123def456",
        provider="test",
        title="Stored Title",
        messages=MessageCollection(messages=[]),
        metadata={"title": "User Title"},
    )
    assert conv1.display_title == "User Title"

    # Falls back to stored title
    conv2 = Conversation(
        id="abc123def456",
        provider="test",
        title="Stored Title",
        messages=MessageCollection(messages=[]),
    )
    assert conv2.display_title == "Stored Title"

    # Falls back to ID (truncated)
    conv3 = Conversation(
        id="abc123def456",
        provider="test",
        messages=MessageCollection(messages=[]),
    )
    assert conv3.display_title == "abc123de"


def test_conversation_summary_property():
    """Conversation.summary extracts from metadata."""
    conv = Conversation(
        id="c1",
        provider="test",
        messages=MessageCollection(messages=[]),
        metadata={"summary": "This conversation discusses AI basics."},
    )
    assert conv.summary == "This conversation discusses AI basics."


def test_conversation_tags_parsing_from_metadata():
    """Conversation.tags extracts list from metadata."""
    conv = Conversation(
        id="c1",
        provider="test",
        messages=MessageCollection(messages=[]),
        metadata={"tags": ["python", "debugging", "urgent"]},
    )
    assert conv.tags == ["python", "debugging", "urgent"]


def test_conversation_tags_empty_default():
    """Conversation.tags returns empty list when absent."""
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=[]))
    assert conv.tags == []


def test_conversation_total_cost_usd_aggregation():
    """Conversation.total_cost_usd sums all message costs."""
    messages = [
        Message(id="1", role="user", text="Q1", provider_meta={"costUSD": 0.001}),
        Message(id="2", role="assistant", text="A1", provider_meta={"costUSD": 0.005}),
        Message(id="3", role="user", text="Q2", provider_meta={"costUSD": 0.001}),
        Message(id="4", role="assistant", text="A2", provider_meta={"costUSD": 0.008}),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    assert conv.total_cost_usd == 0.015


def test_conversation_total_duration_ms_aggregation():
    """Conversation.total_duration_ms sums all message durations."""
    messages = [
        Message(id="1", role="user", text="Q1"),
        Message(id="2", role="assistant", text="A1", provider_meta={"durationMs": 2000}),
        Message(id="3", role="user", text="Q2"),
        Message(id="4", role="assistant", text="A2", provider_meta={"durationMs": 3000}),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    assert conv.total_duration_ms == 5000


def test_conversation_metadata_immutability():
    """Conversation metadata dict can be accessed and modified."""
    conv = Conversation(
        id="c1",
        provider="test",
        messages=MessageCollection(messages=[]),
        metadata={"key1": "value1"},
    )
    assert conv.metadata["key1"] == "value1"


def test_conversation_provider_specific_metadata():
    """Conversation preserves provider-specific metadata."""
    provider_meta = {"model": "gpt-4", "temperature": 0.8}
    conv = Conversation(
        id="c1",
        provider="chatgpt",
        messages=MessageCollection(messages=[]),
        provider_meta=provider_meta,
    )
    assert conv.provider_meta["model"] == "gpt-4"


def test_conversation_with_multiple_branches():
    """Conversation with branching dialogue sequences."""
    messages = [
        Message(id="u1", role="user", text="First question?"),
        Message(id="a1", role="assistant", text="First answer."),
        Message(id="u2", role="user", text="Second question?"),
        Message(id="a2", role="assistant", text="Second answer."),
        Message(id="u3", role="user", text="Follow-up?"),
        Message(id="a3", role="assistant", text="Follow-up answer."),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    assert conv.user_message_count == 3
    assert conv.assistant_message_count == 3


def test_conversation_equality():
    """Conversation instances with same data are equal."""
    messages = [Message(id="1", role="user", text="Hi")]
    conv1 = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    conv2 = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    assert conv1 == conv2


# =============================================================================
# 4. ITERATION HELPERS TESTS (15 tests)
# =============================================================================


def test_iter_pairs_with_valid_dialogues():
    """iter_pairs yields valid user/assistant pairs."""
    messages = [
        Message(id="u1", role="user", text="First question here"),
        Message(id="a1", role="assistant", text="First answer here"),
        Message(id="u2", role="user", text="Second question here"),
        Message(id="a2", role="assistant", text="Second answer here"),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    pairs = list(conv.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].user.id == "u1"
    assert pairs[0].assistant.id == "a1"
    assert pairs[1].user.id == "u2"
    assert pairs[1].assistant.id == "a2"


def test_iter_pairs_with_orphaned_user_message():
    """iter_pairs skips orphaned user message."""
    messages = [
        Message(id="u1", role="user", text="First question here"),
        Message(id="a1", role="assistant", text="First answer here"),
        Message(id="u2", role="user", text="Second question orphaned no reply"),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    pairs = list(conv.iter_pairs())
    assert len(pairs) == 1
    assert pairs[0].user.id == "u1"


def test_iter_pairs_with_orphaned_assistant_message():
    """iter_pairs skips orphaned assistant message."""
    messages = [
        Message(id="u1", role="user", text="First question here"),
        Message(id="a1", role="assistant", text="First answer here"),
        Message(id="a2", role="assistant", text="Second answer orphaned no user"),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    pairs = list(conv.iter_pairs())
    assert len(pairs) == 1


def test_iter_pairs_empty_conversation():
    """iter_pairs returns empty iterator for empty conversation."""
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=[]))
    pairs = list(conv.iter_pairs())
    assert pairs == []


def test_iter_thinking_extraction():
    """iter_thinking yields extracted thinking content."""
    messages = [
        Message(id="a1", role="assistant", text="<thinking>Step 1</thinking>\nAnswer"),
        Message(id="a2", role="assistant", text="<thinking>Step 2\nStep 3</thinking>\nMore"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    conv.messages[0].provider_meta = {"content_blocks": [{"type": "thinking"}]}
    conv.messages[1].provider_meta = {"content_blocks": [{"type": "thinking"}]}
    thinking = list(conv.iter_thinking())
    assert len(thinking) == 2
    assert "Step 1" in thinking[0]
    assert "Step 3" in thinking[1]


def test_iter_thinking_empty_when_none():
    """iter_thinking yields nothing for messages without thinking."""
    messages = [
        Message(id="a1", role="assistant", text="Just an answer"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    thinking = list(conv.iter_thinking())
    assert thinking == []


def test_iter_substantive_filters_noise():
    """iter_substantive filters out tool/system/context messages."""
    messages = [
        Message(id="u1", role="user", text="Actual question with substance"),
        Message(id="a1", role="assistant", text="Actual answer here"),
        Message(id="t1", role="tool", text="Tool result"),
        Message(id="s1", role="system", text="System prompt"),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    substantive = list(conv.iter_substantive())
    assert len(substantive) == 2
    assert substantive[0].id == "u1"
    assert substantive[1].id == "a1"


def test_iter_substantive_includes_user_assistant():
    """iter_substantive includes user and assistant messages."""
    messages = [
        Message(id="u1", role="user", text="Question with enough words here"),
        Message(id="a1", role="assistant", text="Answer with enough words here"),
    ]
    conv = Conversation(id="c1", provider="test", messages=messages)
    substantive = list(conv.iter_substantive())
    assert len(substantive) == 2


def test_iter_substantive_excludes_thinking_tool():
    """iter_substantive excludes thinking and tool messages."""
    messages = [
        Message(
            id="a1",
            role="assistant",
            text="<thinking>Reasoning</thinking>",
            provider_meta={"content_blocks": [{"type": "thinking"}]},
        ),
        Message(
            id="a2",
            role="assistant",
            text="Tool call",
            provider_meta={"content_blocks": [{"type": "tool_use"}]},
        ),
        Message(id="u1", role="user", text="Real question here with enough words"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    substantive = list(conv.iter_substantive())
    assert len(substantive) == 1
    assert substantive[0].id == "u1"


def test_iter_dialogue_filters_user_assistant_only():
    """iter_dialogue includes only user/assistant messages."""
    messages = [
        Message(id="u1", role="user", text="Q"),
        Message(id="a1", role="assistant", text="A"),
        Message(id="s1", role="system", text="System"),
        Message(id="t1", role="tool", text="Tool"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    dialogue = list(conv.iter_dialogue())
    assert len(dialogue) == 2
    assert dialogue[0].id == "u1"
    assert dialogue[1].id == "a1"


def test_iter_dialogue_preserves_order():
    """iter_dialogue preserves message order."""
    messages = [
        Message(id="u1", role="user", text="Q1"),
        Message(id="a1", role="assistant", text="A1"),
        Message(id="u2", role="user", text="Q2"),
        Message(id="a2", role="assistant", text="A2"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    dialogue = list(conv.iter_dialogue())
    ids = [m.id for m in dialogue]
    assert ids == ["u1", "a1", "u2", "a2"]


def test_iter_with_filtered_messages():
    """Iterators work on filtered conversations."""
    messages = [
        Message(id="u1", role="user", text="Q1"),
        Message(id="a1", role="assistant", text="A1"),
        Message(id="s1", role="system", text="Sys"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    filtered = conv.dialogue_only()
    dialogue = list(filtered.iter_dialogue())
    assert len(dialogue) == 2


def test_iter_with_projected_fields():
    """Iterators work with projected conversations."""
    messages = [
        Message(id="u1", role="user", text="Q" * 100),
        Message(id="a1", role="assistant", text="A"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    substantive = list(conv.iter_substantive())
    assert len(substantive) >= 1


def test_iter_performance_large_conversation():
    """Iterators handle large conversations efficiently."""
    messages = [
        Message(id=f"m{i}", role="user" if i % 2 == 0 else "assistant", text=f"Message {i}")
        for i in range(1000)
    ]
    conv = Conversation(id="large", provider="test", messages=MessageCollection(messages=messages))
    # Just ensure it doesn't crash
    count = sum(1 for _ in conv.iter_dialogue())
    assert count == 1000


def test_iter_generators_memory_efficient():
    """Iterators are generators (lazy evaluation)."""
    messages = [Message(id=f"m{i}", role="user", text=f"Q{i}") for i in range(100)]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    gen = conv.iter_dialogue()
    # Should have generator protocol
    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")


# =============================================================================
# 5. CONVERSATION RENDERING TESTS (8 tests)
# =============================================================================


def test_to_text_default_format():
    """to_text renders with role prefixes by default."""
    messages = [
        Message(id="u1", role="user", text="Hello"),
        Message(id="a1", role="assistant", text="Hi there"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    text = conv.to_text()
    assert "user: Hello" in text
    assert "assistant: Hi there" in text
    assert "\n\n" in text  # Default separator


def test_to_text_custom_role_prefixes():
    """to_text respects include_role parameter."""
    messages = [
        Message(id="u1", role="user", text="Q"),
        Message(id="a1", role="assistant", text="A"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    text = conv.to_text(include_role=False)
    assert_not_contains_any(text, "user:", "assistant:")
    assert_contains_all(text, "Q", "A")


def test_to_text_with_thinking_blocks():
    """to_text includes thinking blocks."""
    messages = [
        Message(
            id="a1",
            role="assistant",
            text="<thinking>Reasoning</thinking>\nAnswer",
            provider_meta={"content_blocks": [{"type": "thinking"}]},
        ),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    text = conv.to_text()
    assert "Reasoning" in text
    assert "Answer" in text


def test_to_clean_text_substantive_only():
    """to_clean_text includes only substantive messages."""
    messages = [
        Message(id="u1", role="user", text="Real question with enough substance here"),
        Message(id="a1", role="assistant", text="Real answer with enough substance here"),
        Message(id="t1", role="tool", text="Tool output"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    clean = conv.to_clean_text()
    assert "Real question" in clean
    assert "Real answer" in clean
    assert "Tool output" not in clean


def test_to_clean_text_filters_noise():
    """to_clean_text excludes tool/system/context messages."""
    messages = [
        Message(id="u1", role="user", text="Important question with detail"),
        Message(id="s1", role="system", text="System instructions"),
        Message(id="a1", role="assistant", text="Important answer with detail"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    clean = conv.to_clean_text()
    assert "Important question" in clean
    assert "Important answer" in clean
    assert "System instructions" not in clean


def test_empty_conversation_rendering():
    """Empty conversation renders as empty string."""
    conv = Conversation(id="empty", provider="test", messages=MessageCollection(messages=[]))
    text = conv.to_text()
    clean = conv.to_clean_text()
    assert text == ""
    assert clean == ""


def test_unicode_special_characters():
    """Rendering handles unicode and special characters."""
    messages = [
        Message(id="u1", role="user", text="What's the meaning of 🎯?"),
        Message(id="a1", role="assistant", text="It means目的 (mokutek) in Japanese."),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    text = conv.to_text()
    assert "🎯" in text
    assert "目的" in text


def test_rendering_with_attachments():
    """Rendering includes messages with attachments."""
    attach = Attachment(id="a1", name="doc.pdf")
    messages = [
        Message(id="u1", role="user", text="Here's the document", attachments=[attach]),
        Message(id="a1", role="assistant", text="I'll review it"),
    ]
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=messages))
    text = conv.to_text()
    assert "Here's the document" in text
    assert "I'll review it" in text
