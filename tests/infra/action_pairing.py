"""Neutral retained tool streams for exercising the canonical archive writer."""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession


def action_stream(
    native_id: str,
    events: Sequence[tuple[str, str, str | None, bool | None]],
) -> ParsedSession:
    messages = []
    for name, kind, tool_id, is_error in events:
        is_use = kind == "use"
        block = ParsedContentBlock(
            type=BlockType.TOOL_USE if is_use else BlockType.TOOL_RESULT,
            tool_id=tool_id,
            tool_name="Bash" if is_use else None,
            tool_input={"command": name} if is_use else None,
            text=None if is_use else name,
            is_error=is_error,
        )
        messages.append(
            ParsedMessage(
                provider_message_id=name,
                role=Role.ASSISTANT if is_use else Role.TOOL,
                blocks=[block],
            )
        )
    return ParsedSession(source_name=Provider.CODEX, provider_session_id=native_id, messages=messages)
