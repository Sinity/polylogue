"""Example test file demonstrating test helper utilities.

This file shows before/after patterns for common test scenarios.
Created during test consolidation to demonstrate helper utility value.

Run with: pytest tests/test_example_using_helpers.py -v
"""

from __future__ import annotations

import pytest

from polylogue.rendering.core import ConversationFormatter
from tests.helpers import (
    ConversationBuilder,
    assert_contains_all,
    assert_messages_ordered,
    assert_not_contains_any,
    db_setup,
    make_chatgpt_node,
    make_claude_chat_message,
    make_message,
)

# =============================================================================
# EXAMPLE 1: Building Conversations for Database Tests
# =============================================================================


def test_conversation_builder_example(workspace_env):
    """Example: Using ConversationBuilder for database tests."""
    db_path = db_setup(workspace_env)

    # Create conversation with messages and attachments
    conv = (
        ConversationBuilder(db_path, "example-conv")
        .title("Example Conversation")
        .provider("chatgpt")
        .add_message("m1", role="user", text="What is Python?")
        .add_message("m2", role="assistant", text="Python is a programming language.")
        .add_attachment("att1", mime_type="image/png", provider_meta={"name": "diagram.png"})
        .save()
    )

    # Verify conversation was saved
    assert conv.conversation_id == "example-conv"
    assert conv.title == "Example Conversation"

    # Format and verify output
    formatter = ConversationFormatter(workspace_env["archive_root"])
    result = formatter.format("example-conv")

    assert_messages_ordered(result.markdown_text, "What is Python?", "Python is a programming language.")
    assert_contains_all(result.markdown_text, "## user", "## assistant", "diagram.png")


# =============================================================================
# EXAMPLE 2: Parametrized Tests with Helpers
# =============================================================================


MESSAGE_ORDERING_CASES = [
    (
        [("m1", "First", "2024-01-01T10:00:00Z"), ("m2", "Second", "2024-01-01T10:01:00Z")],
        ["First", "Second"],
        "chronological order",
    ),
    (
        [("m1", "Later", "2024-01-01T10:01:00Z"), ("m2", "Earlier", "2024-01-01T10:00:00Z")],
        ["Earlier", "Later"],
        "reverse insertion order",
    ),
    (
        [("m1", "Timestamped", "2024-01-01T10:00:00Z"), ("m2", "NoTimestamp", None)],
        ["Timestamped", "NoTimestamp"],
        "null timestamps sort last",
    ),
]


@pytest.mark.parametrize("message_data,expected_order,desc", MESSAGE_ORDERING_CASES)
def test_message_ordering_parametrized(workspace_env, message_data, expected_order, desc):
    """Example: Parametrized test using ConversationBuilder."""
    db_path = db_setup(workspace_env)

    builder = ConversationBuilder(db_path, f"order-{hash(desc) % 1000}")

    # Add messages from test data
    for msg_id, text, timestamp in message_data:
        builder.add_message(msg_id, role="user", text=text, timestamp=timestamp)

    builder.save()

    # Format and verify order
    formatter = ConversationFormatter(workspace_env["archive_root"])
    result = formatter.format(builder.conv.conversation_id)

    assert_messages_ordered(result.markdown_text, *expected_order)


# =============================================================================
# EXAMPLE 3: Importer Tests with Data Generators
# =============================================================================


CHATGPT_EXTRACTION_CASES = [
    (
        make_chatgpt_node("msg1", "user", ["Hello world"], timestamp=1704067200),
        1,
        "basic message",
    ),
    (
        make_chatgpt_node("msg1", "assistant", ["Response"], metadata={"attachments": [{"file_name": "doc.pdf"}]}),
        1,
        "with attachments metadata",
    ),
]


@pytest.mark.parametrize("node,expected_count,desc", CHATGPT_EXTRACTION_CASES)
def test_chatgpt_extraction_example(node, expected_count, desc):
    """Example: Using make_chatgpt_node for importer tests."""
    from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping

    mapping = {"node1": node}
    messages, attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == expected_count, f"Failed {desc}"


CLAUDE_EXTRACTION_CASES = [
    (
        make_claude_chat_message("u1", "human", "Question"),
        "user",
        "human role maps to user",
    ),
    (
        make_claude_chat_message("u2", "assistant", "Answer"),
        "assistant",
        "assistant role preserved",
    ),
]


@pytest.mark.parametrize("msg_data,expected_role,desc", CLAUDE_EXTRACTION_CASES)
def test_claude_extraction_example(msg_data, expected_role, desc):
    """Example: Using make_claude_chat_message for importer tests."""
    from polylogue.sources.parsers.claude import extract_messages_from_chat_messages

    messages, attachments = extract_messages_from_chat_messages([msg_data])

    assert len(messages) == 1
    assert messages[0].role == expected_role, f"Failed {desc}"


# =============================================================================
# EXAMPLE 4: Assertion Helpers
# =============================================================================


def test_assertion_helpers_example(workspace_env):
    """Example: Using assertion helpers for cleaner tests."""
    db_path = db_setup(workspace_env)

    (
        ConversationBuilder(db_path, "assert-example")
        .add_message("m1", role="user", text="Question")
        .add_message("m2", role="assistant", text="Answer")
        .save()
    )

    formatter = ConversationFormatter(workspace_env["archive_root"])
    result = formatter.format("assert-example")

    # Multiple assertions in one line
    assert_contains_all(result.markdown_text, "Question", "Answer", "## user", "## assistant")

    # Verify unwanted content absent
    assert_not_contains_any(result.markdown_text, "ERROR", "FAIL", "```json")

    # Verify ordering
    assert_messages_ordered(result.markdown_text, "Question", "Answer")


# =============================================================================
# EXAMPLE 5: Quick Builders vs Fluent Builders
# =============================================================================


def test_quick_vs_fluent_builders(workspace_env):
    """Example: When to use make_message() vs MessageBuilder."""
    # Quick builder for simple cases
    simple_msg = make_message("m1", role="user", text="Simple message")
    assert simple_msg.role == "user"
    assert simple_msg.text == "Simple message"

    # Fluent builder for complex cases with metadata
    from tests.helpers import MessageBuilder

    complex_msg = (
        MessageBuilder("m2", "conv1")
        .role("assistant")
        .text("Complex response")
        .timestamp("2024-01-01T10:00:00Z")
        .meta({"thinking": "Let me analyze...", "cost_usd": 0.005, "duration_ms": 1234})
        .build()
    )

    assert complex_msg.provider_meta == {
        "thinking": "Let me analyze...",
        "cost_usd": 0.005,
        "duration_ms": 1234,
    }
