"""Parsed session fixtures for durable reference publication laws."""

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession


def reference_session(
    native_id: str,
    *,
    provider: Provider = Provider.CODEX,
    parent: str | None = None,
    messages: tuple[tuple[str, str], ...] = (("message", "reference content"),),
) -> ParsedSession:
    return ParsedSession(
        source_name=provider,
        provider_session_id=native_id,
        title=native_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        messages=[
            ParsedMessage(
                provider_message_id=message_id,
                role=Role.USER,
                text=text,
                position=position,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=position == len(messages) - 1,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            )
            for position, (message_id, text) in enumerate(messages)
        ],
    )
