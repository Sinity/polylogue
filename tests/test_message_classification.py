"""Tests for message classification in Message model.

Tests cover role classification, thinking detection (via content_blocks/provider_meta),
tool use detection, context dump detection, and noise/substantive classification.
"""


from polylogue.lib.models import Attachment, Message, Role

# =============================================================================
# Role Classification Tests
# =============================================================================


def test_is_user_with_user_role():
    """User role correctly identified."""
    msg = Message(id="1", role="user", text="Hello")
    assert msg.is_user is True
    assert msg.is_assistant is False
    assert msg.is_dialogue is True


def test_is_user_with_human_role():
    """Human role (alias) correctly mapped to user."""
    msg = Message(id="1", role="human", text="Hello")
    assert msg.is_user is True
    assert msg.is_assistant is False


def test_is_assistant_with_assistant_role():
    """Assistant role correctly identified."""
    msg = Message(id="1", role="assistant", text="Hi there")
    assert msg.is_assistant is True
    assert msg.is_user is False
    assert msg.is_dialogue is True


def test_is_assistant_with_model_role():
    """Model role (Gemini alias) correctly mapped to assistant."""
    msg = Message(id="1", role="model", text="Response")
    assert msg.is_assistant is True
    assert msg.is_user is False


def test_is_system_role():
    """System role correctly identified."""
    msg = Message(id="1", role="system", text="System prompt")
    assert msg.is_system is True
    assert msg.is_dialogue is False


def test_is_dialogue_includes_user_and_assistant():
    """Dialogue includes user and assistant, excludes system."""
    user_msg = Message(id="1", role="user", text="Hi")
    assistant_msg = Message(id="2", role="assistant", text="Hello")
    system_msg = Message(id="3", role="system", text="Instructions")

    assert user_msg.is_dialogue is True
    assert assistant_msg.is_dialogue is True
    assert system_msg.is_dialogue is False


def test_role_normalization_case_insensitive():
    """Role strings are normalized case-insensitively."""
    msg1 = Message(id="1", role="USER", text="Test")
    msg2 = Message(id="2", role="User", text="Test")
    msg3 = Message(id="3", role="ASSISTANT", text="Test")

    assert msg1.is_user is True
    assert msg2.is_user is True
    assert msg3.is_assistant is True


# =============================================================================
# Thinking Detection Tests
# =============================================================================


def test_is_thinking_with_content_blocks():
    """Thinking detected via content_blocks."""
    msg = Message(
        id="1",
        role="assistant",
        text="<thinking>Let me analyze...</thinking>\nAnswer.",
        provider_meta={"content_blocks": [{"type": "thinking", "text": "Let me analyze..."}]}
    )
    assert msg.is_thinking is True


def test_is_thinking_with_gemini_thought_marker():
    """Gemini isThought in provider_meta detected."""
    msg = Message(
        id="1",
        role="assistant",
        text="Analyzing the problem...",
        provider_meta={"isThought": True}
    )
    assert msg.is_thinking is True


def test_is_thinking_with_gemini_raw_thought_marker():
    """Gemini isThought in raw provider_meta detected."""
    msg = Message(
        id="1",
        role="assistant",
        text="Analyzing the problem...",
        provider_meta={"raw": {"isThought": True}}
    )
    assert msg.is_thinking is True


def test_is_thinking_false_for_normal_message():
    """Normal assistant message not detected as thinking."""
    msg = Message(
        id="1",
        role="assistant",
        text="Here is the answer to your question."
    )
    assert msg.is_thinking is False


def test_is_thinking_false_without_content_blocks():
    """Message without content_blocks not detected as thinking (no heuristics)."""
    msg = Message(
        id="1",
        role="assistant",
        text="**Analyzing the request**\n\nSome content"  # Would match old heuristic
    )
    assert msg.is_thinking is False


def test_extract_thinking_content():
    """Thinking content extracted from XML tags."""
    msg = Message(
        id="1",
        role="assistant",
        text="<thinking>Step 1: Analyze\nStep 2: Implement</thinking>\nDone!"
    )
    extracted = msg.extract_thinking()
    assert extracted == "Step 1: Analyze\nStep 2: Implement"


def test_extract_thinking_returns_none_when_absent():
    """Extract thinking returns None when no thinking tags present."""
    msg = Message(id="1", role="assistant", text="Just a normal response")
    assert msg.extract_thinking() is None


# =============================================================================
# Tool Use Detection Tests
# =============================================================================


def test_is_tool_use_with_tool_role():
    """Tool role identified as tool use."""
    msg = Message(id="1", role="tool", text="Function result: success")
    assert msg.is_tool_use is True


def test_is_tool_use_with_content_blocks():
    """Tool use detected via content_blocks."""
    msg = Message(
        id="1",
        role="assistant",
        text="Using a tool...",
        provider_meta={"content_blocks": [{"type": "tool_use", "name": "Bash", "id": "t1"}]}
    )
    assert msg.is_tool_use is True


def test_is_tool_use_with_tool_result_content_blocks():
    """Tool result detected via content_blocks."""
    msg = Message(
        id="1",
        role="assistant",
        text="Tool result...",
        provider_meta={"content_blocks": [{"type": "tool_result", "tool_use_id": "t1"}]}
    )
    assert msg.is_tool_use is True


def test_is_tool_use_with_sidechain_marker():
    """Claude-code sidechain marker detected."""
    msg = Message(
        id="1",
        role="assistant",
        text="Some sidechain content",
        provider_meta={"raw": {"isSidechain": True}}
    )
    assert msg.is_tool_use is True


def test_is_tool_use_false_without_markers():
    """Text-only tool-like content not detected (no heuristics)."""
    xml_content = "<function_calls><invoke name='search'/></function_calls>"
    msg = Message(id="1", role="assistant", text=xml_content)
    assert msg.is_tool_use is False


def test_is_tool_use_false_for_normal_message():
    """Normal message not detected as tool use."""
    msg = Message(id="1", role="assistant", text="Let me help you with that.")
    assert msg.is_tool_use is False


# =============================================================================
# Context Dump Detection Tests
# =============================================================================


def test_is_context_dump_with_file_content():
    """Message with many code fences detected as context dump."""
    code_blocks = "\n".join([f"```python\nprint({i})\n```" for i in range(5)])
    msg = Message(id="1", role="user", text="Here is the code:\n" + code_blocks)
    assert msg.is_context_dump is True


def test_is_context_dump_with_system_content():
    """System prompt pasted as context detected."""
    system_text = "<system>You are a helpful assistant that follows rules carefully.</system>"
    msg = Message(id="1", role="user", text=system_text)
    assert msg.is_context_dump is True


def test_is_context_dump_false_for_normal_question():
    """Normal question not detected as context dump."""
    msg = Message(
        id="1",
        role="user",
        text="How do I implement a binary search tree in Python?"
    )
    assert msg.is_context_dump is False


# =============================================================================
# Substantive Classification Tests
# =============================================================================


def test_is_substantive_with_normal_response():
    """Normal response is substantive."""
    msg = Message(
        id="1",
        role="assistant",
        text="Here is how you can implement that feature:\n\n1. First, create a class..."
    )
    assert msg.is_substantive is True


def test_is_substantive_false_for_thinking_only():
    """Thinking-only message is not substantive."""
    msg = Message(
        id="1",
        role="assistant",
        text="<thinking>Let me think about this...</thinking>",
        provider_meta={"content_blocks": [{"type": "thinking", "text": "Let me think..."}]}
    )
    assert msg.is_substantive is False


def test_is_substantive_false_for_tool_use_only():
    """Tool use message is not substantive."""
    msg = Message(id="1", role="tool", text="Result: 42")
    assert msg.is_substantive is False


def test_is_substantive_false_for_empty_message():
    """Empty or whitespace message is not substantive."""
    msg = Message(id="1", role="assistant", text="   ")
    assert msg.is_substantive is False


def test_is_substantive_false_for_context_dump():
    """Context dump is not substantive."""
    system_text = "<system>Instructions...</system>" * 10
    msg = Message(id="1", role="user", text=system_text)
    assert msg.is_substantive is False


# =============================================================================
# Provider-Specific Tests
# =============================================================================


def test_role_enum_from_string_with_aliases():
    """Role.from_string handles all known aliases."""
    assert Role.from_string("user") == Role.USER
    assert Role.from_string("human") == Role.USER
    assert Role.from_string("assistant") == Role.ASSISTANT
    assert Role.from_string("model") == Role.ASSISTANT
    assert Role.from_string("system") == Role.SYSTEM


def test_role_enum_unknown():
    """Unknown roles return UNKNOWN."""
    assert Role.from_string("random_role") == Role.UNKNOWN
    assert Role.from_string("") == Role.UNKNOWN


def test_message_with_attachments():
    """Message with attachments handled correctly."""
    attach = Attachment(
        id="a1",
        name="test.py",
        mime_type="text/x-python",
    )
    msg = Message(
        id="1",
        role="user",
        text="Please review this code",
        attachments=[attach]
    )
    assert len(msg.attachments) == 1
    assert msg.attachments[0].name == "test.py"
