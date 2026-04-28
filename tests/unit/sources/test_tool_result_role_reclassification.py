"""Claude tool_result envelopes are reclassified from USER to TOOL (#428).

Anthropic's protocol requires ``tool_result`` blocks to be carried by
``role: user`` envelopes. Polylogue's outer-envelope role normalization
puts those under ``Role.USER``, polluting ``--message-role user`` filters.
The reclassification flips USER → TOOL when content is exclusively
``tool_result`` blocks; mixed envelopes (text + tool_result) keep USER.
"""

from __future__ import annotations

from polylogue.lib.roles import Role
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.sources.parsers.claude.common import (
    extract_messages_from_chat_messages,
    reclassify_tool_result_envelope,
)
from polylogue.types import ContentBlockType


def _tool_result(idx: int) -> ParsedContentBlock:
    return ParsedContentBlock(type=ContentBlockType.TOOL_RESULT, text=f"result {idx}")


def _text(text: str) -> ParsedContentBlock:
    return ParsedContentBlock(type=ContentBlockType.TEXT, text=text)


class TestReclassifyToolResultEnvelope:
    def test_user_with_only_tool_result_becomes_tool(self) -> None:
        assert reclassify_tool_result_envelope(Role.USER, [_tool_result(1)]) is Role.TOOL

    def test_user_with_multiple_tool_result_blocks_becomes_tool(self) -> None:
        assert reclassify_tool_result_envelope(Role.USER, [_tool_result(1), _tool_result(2)]) is Role.TOOL

    def test_user_with_text_only_stays_user(self) -> None:
        assert reclassify_tool_result_envelope(Role.USER, [_text("typed prose")]) is Role.USER

    def test_user_with_mixed_text_and_tool_result_stays_user(self) -> None:
        """Conservative: mixed envelopes preserve USER (split-on-parse is a separate change)."""
        assert reclassify_tool_result_envelope(Role.USER, [_text("hi"), _tool_result(1)]) is Role.USER

    def test_user_with_no_blocks_stays_user(self) -> None:
        assert reclassify_tool_result_envelope(Role.USER, []) is Role.USER

    def test_assistant_unaffected(self) -> None:
        """Assistant tool_use envelopes never get touched."""
        assert reclassify_tool_result_envelope(Role.ASSISTANT, [_tool_result(1)]) is Role.ASSISTANT

    def test_system_unaffected(self) -> None:
        assert reclassify_tool_result_envelope(Role.SYSTEM, [_tool_result(1)]) is Role.SYSTEM


class TestChatMessagesWithToolResults:
    def test_chat_message_with_tool_result_blocks_gets_tool_role(self) -> None:
        messages, _ = extract_messages_from_chat_messages(
            [
                {
                    "uuid": "m-typed",
                    "role": "user",
                    "content": [{"type": "text", "text": "typed prose"}],
                },
                {
                    "uuid": "m-tool",
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "tool output"}],
                    "text": "tool output",
                },
            ]
        )
        roles = {message.provider_message_id: message.role for message in messages}
        assert roles["m-typed"] is Role.USER
        assert roles["m-tool"] is Role.TOOL, "tool_result-only user envelope must be reclassified to TOOL (#428)"
