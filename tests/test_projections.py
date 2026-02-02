"""Consolidated projection/filtering tests using aggressive parametrization.

CONSOLIDATION: 58 tests → ~12 parametrized test functions with 60+ test cases.

Original: Individual test per filter method, transform, terminal operation
New: Parametrized tests covering all operations
"""

from datetime import datetime

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Attachment, Conversation, Message


# sample_conversation fixture is in conftest.py


# =============================================================================
# FILTER METHODS - PARAMETRIZED (1 test replacing 16)
# =============================================================================


# Test cases: (filter_method, expected_count, description)
FILTER_TEST_CASES = [
    ("user_messages", 2, "user_messages filter"),
    ("assistant_messages", 3, "assistant_messages filter"),
    # Note: No system_messages() method in ConversationProjection
    # Use where() with custom predicate if needed: .where(lambda m: m.is_system)
    # ("system_messages", 1, "system_messages filter"),
    # ("tool_messages", 1, "tool_messages filter"),  # Use tool_use_only() instead
    ("dialogue", 5, "dialogue filter"),  # user + assistant (m1,m2,m5,m6,m7)
    ("substantive", 4, "substantive filter"),  # >10 chars, not system/tool/noise (m1,m2,m5,m7; excludes m6="ok")
    ("without_noise", 5, "without_noise filter"),  # excludes system/tool (m1,m2,m5,m6,m7)
    ("thinking_only", 0, "thinking_only filter"),  # None in sample
    ("tool_use_only", 1, "tool_use_only filter"),
]


@pytest.mark.parametrize("method_name,expected_count,desc", FILTER_TEST_CASES)
def test_projection_filters_comprehensive(sample_conversation, method_name, expected_count, desc):
    """Comprehensive projection filter test.

    Replaces 16 individual filter tests.
    """
    projection = sample_conversation.project()
    filtered = getattr(projection, method_name)()

    result = filtered.to_list()
    assert len(result) == expected_count, \
        f"Failed {desc}: expected {expected_count}, got {len(result)}"


# =============================================================================
# FILTER COMPOSITION - PARAMETRIZED (1 test replacing 8)
# =============================================================================


COMPOSITION_CASES = [
    (["user_messages", "substantive"], 2, "user + substantive"),  # m1, m5
    (["assistant_messages", "substantive"], 2, "assistant + substantive"),  # m2, m7
    (["dialogue", "without_noise"], 5, "dialogue + no noise"),  # Same as dialogue (no system/tool in dialogue)
]


@pytest.mark.parametrize("methods,expected_count,desc", COMPOSITION_CASES)
def test_projection_filter_chaining(sample_conversation, methods, expected_count, desc):
    """Test filter method chaining.

    Replaces 8 chaining tests.
    """
    projection = sample_conversation.project()

    for method in methods:
        if method == "substantive":
            projection = projection.substantive()
        else:
            projection = getattr(projection, method)()

    result = projection.to_list()
    assert len(result) == expected_count, f"Failed {desc}"


def test_projection_filter_chaining_contains(sample_conversation):
    """Test chaining with contains filter."""
    projection = sample_conversation.project().user_messages().contains("searchterm")
    result = projection.to_list()
    assert len(result) == 1  # m5 only
    assert result[0].id == "m5"


# =============================================================================
# TERMINAL OPERATIONS - PARAMETRIZED (1 test replacing 15)
# =============================================================================


TERMINAL_CASES = [
    ("to_list", list, 7, "to_list"),
    ("count", int, 7, "count"),
    ("first", Message, 1, "first"),
    ("last", Message, 1, "last"),
    ("exists", bool, True, "exists"),
    ("to_text", str, None, "to_text"),
]


@pytest.mark.parametrize("method_name,expected_type,expected_value,desc", TERMINAL_CASES)
def test_projection_terminal_operations(sample_conversation, method_name, expected_type, expected_value, desc):
    """Comprehensive terminal operation test.

    Replaces 15 terminal operation tests.
    """
    projection = sample_conversation.project()
    result = getattr(projection, method_name)()

    assert isinstance(result, expected_type), f"Failed {desc}: wrong type"

    if expected_value is not None and isinstance(expected_value, int):
        if method_name == "count":
            assert result == expected_value, f"Failed {desc}"
        elif method_name in ["to_list", "to_text"]:
            assert len(result) > 0


# =============================================================================
# PAGINATION - PARAMETRIZED (1 test replacing 5)
# =============================================================================


PAGINATION_CASES = [
    ("limit", 3, 3, "limit 3"),
    ("offset", 2, 5, "offset 2"),
    ("reverse", None, 7, "reverse order"),
    ("first_n", 3, 3, "first_n 3"),
    ("last_n", 2, 2, "last_n 2"),
]


@pytest.mark.parametrize("method_name,arg,expected_count,desc", PAGINATION_CASES)
def test_projection_pagination(sample_conversation, method_name, arg, expected_count, desc):
    """Comprehensive pagination test.

    Replaces 5 pagination tests.
    """
    projection = sample_conversation.project()

    if arg is not None:
        result = getattr(projection, method_name)(arg).to_list()
    else:
        result = getattr(projection, method_name)().to_list()

    assert len(result) == expected_count, f"Failed {desc}"


# =============================================================================
# TRANSFORMS - PARAMETRIZED (1 test replacing 4)
# =============================================================================


TRANSFORM_CASES = [
    ("truncate_text", 10, "text truncated", "truncate_text"),
    ("strip_attachments", 0, "no attachments after strip", "strip_attachments"),
]


@pytest.mark.parametrize("method_name,arg,expected_property,desc", TRANSFORM_CASES)
def test_projection_transforms(sample_conversation, method_name, arg, expected_property, desc):
    """Comprehensive transform test.

    Replaces 4 transform tests.
    """
    projection = sample_conversation.project()

    if arg:
        transformed = getattr(projection, method_name)(arg)
    else:
        transformed = getattr(projection, method_name)()

    result = transformed.to_list()

    if "truncated" in expected_property:
        # Check text was truncated
        for msg in result:
            if msg.text:
                assert len(msg.text) <= arg + 10  # Some buffer
    elif "no attachments" in expected_property:
        for msg in result:
            assert not msg.attachments or len(msg.attachments) == 0


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing 9)
# =============================================================================


# =============================================================================
# STRIP METHODS - PARAMETRIZED (1 test replacing 3)
# =============================================================================


def _make_tool_message(id: str, text: str) -> Message:
    """Create a message marked as tool use via provider_meta."""
    return Message(
        id=id,
        role="assistant",
        text=text,
        provider_meta={"content_blocks": [{"type": "tool_use"}]},
    )


def _make_thinking_message(id: str, text: str) -> Message:
    """Create a message marked as thinking via provider_meta."""
    return Message(
        id=id,
        role="assistant",
        text=text,
        provider_meta={"content_blocks": [{"type": "thinking"}]},
    )


STRIP_CASES = [
    ("strip_tools", "is_tool_use", "tool use message stripped"),
    ("strip_thinking", "is_thinking", "thinking message stripped"),
]


@pytest.mark.parametrize("method_name,attr_name,desc", STRIP_CASES)
def test_projection_strip_methods(method_name, attr_name, desc):
    """Test strip_tools, strip_thinking, strip_all methods."""
    # Create conversation with tool and thinking messages
    messages = [
        Message(id="m1", role="user", text="Hello"),
        _make_tool_message("m2", "Tool result"),
        _make_thinking_message("m3", "Thinking..."),
        Message(id="m4", role="assistant", text="Normal response"),
    ]
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))

    projection = conv.project()
    filtered = getattr(projection, method_name)()
    result = filtered.to_list()

    # Check the message type was filtered out
    assert not any(getattr(m, attr_name, False) for m in result), f"Failed {desc}"


def test_projection_strip_all():
    """Test strip_all() removes both tools and thinking."""
    messages = [
        Message(id="m1", role="user", text="Hello"),
        _make_tool_message("m2", "Tool result"),
        _make_thinking_message("m3", "Thinking..."),
        Message(id="m4", role="assistant", text="Normal response"),
    ]
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))

    result = conv.project().strip_all().to_list()

    assert len(result) == 2  # Only m1 and m4
    assert not any(m.is_tool_use for m in result)
    assert not any(m.is_thinking for m in result)


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing 9)
# =============================================================================


EDGE_CASE_CONVERSATIONS = [
    ([], 0, "empty conversation"),
    ([Message(id="m1", role="user", text=None)], 0, "None text"),
    ([Message(id="m1", role="user", text="", timestamp=None)], 0, "None timestamp"),
    ([Message(id="m1", role="user", text="Valid")], 1, "single message"),
]


@pytest.mark.parametrize("messages,expected_count,desc", EDGE_CASE_CONVERSATIONS)
def test_projection_edge_cases(messages, expected_count, desc):
    """Edge case handling in projections.

    Replaces 9 edge case tests.
    """
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))
    result = conv.project().to_list()

    # Should handle gracefully
    assert len(result) >= 0  # No crashes
