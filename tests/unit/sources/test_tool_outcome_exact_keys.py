from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSessionEvent
from polylogue.sources.tool_outcomes import derive_tool_outcomes

LONE = "\ud800"
_REASON = ToolResultUnknownReason.NOT_REPORTED.value


def _messages() -> list[ParsedMessage]:
    # The use and its result share a tool ID carrying a lone surrogate; a
    # second pair uses the replacement character in the same place, so a
    # lossy key would merge the two IDs.
    return [
        ParsedMessage(
            provider_message_id=f"use-{LONE}",
            role=Role.ASSISTANT,
            blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id=f"tool-{LONE}", tool_name="run")],
        ),
        ParsedMessage(
            provider_message_id=f"result-{LONE}",
            parent_message_provider_id=f"use-{LONE}",
            role=Role.TOOL,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT, tool_id=f"tool-{LONE}", text="synthetic", outcome_unknown_reason=_REASON
                )
            ],
        ),
        ParsedMessage(
            provider_message_id="use-\ufffd",
            role=Role.ASSISTANT,
            blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="tool-\ufffd", tool_name="run")],
        ),
        ParsedMessage(
            provider_message_id="result-\ufffd",
            parent_message_provider_id="use-\ufffd",
            role=Role.TOOL,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT, tool_id="tool-\ufffd", text="synthetic", outcome_unknown_reason=_REASON
                )
            ],
        ),
    ]


def _events() -> list[ParsedSessionEvent]:
    return [
        ParsedSessionEvent(
            event_type="claude_tool_execution_result",
            source_message_provider_id=f"result-{LONE}",
            payload={"tool_use_id": f"tool-{LONE}", "exit_code": 3},
        ),
        ParsedSessionEvent(
            event_type="claude_tool_execution_result",
            source_message_provider_id="result-\ufffd",
            payload={"tool_use_id": "tool-\ufffd", "exit_code": 0},
        ),
    ]


def _outcomes(messages: list[ParsedMessage]) -> list[ToolOutcome | None]:
    return [message.blocks[0].tool_outcome for message in messages]


def test_lone_surrogate_identifiers_resolve_exactly_in_memory() -> None:
    derived = derive_tool_outcomes(_messages(), _events(), origin=Origin.CLAUDE_CODE_SESSION)
    assert isinstance(derived, list)
    assert _outcomes(derived) == [ToolOutcome.ERROR, ToolOutcome.ERROR, ToolOutcome.OK, ToolOutcome.OK]
    assert derived[1].blocks[0].tool_id == f"tool-{LONE}"
