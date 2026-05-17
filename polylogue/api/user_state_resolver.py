"""Resolution helpers for non-conversation/message user-state targets (#1113).

These helpers validate that a target identified by
``(target_type, target_id, conversation_id, message_id)`` actually exists
in the archive before a mark or annotation is written, and produce the
canonical ``identity_key`` used by recall packs and workspaces.

The resolver returns the validated ``ResolvedTarget`` mapping or raises
``ValueError`` with a specific, surface-friendly message. Insight kinds
(``session``, ``work_event``, ``thread``) are validated against the
respective insight tables; ``content_block`` and ``attachment`` are
validated against the archive substrate; ``paste_span`` is treated as an
opaque content-block-derived identifier and only validated for non-empty
``target_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from polylogue.core.user_state_targets import (
    KINDS_BY_NAME,
    TARGET_KIND_NAMES,
    identity_key,
)

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


class ResolvedTarget(TypedDict, total=False):
    """Storage row payload + identity key for a resolved user-state target."""

    target_type: str
    target_id: str
    conversation_id: str
    message_id: str | None
    identity_key: str


_INSIGHT_QUERIES: dict[str, str] = {
    "session": "SELECT 1 FROM session_profiles WHERE conversation_id = ?",
    "work_event": "SELECT 1 FROM session_work_events WHERE event_id = ? AND conversation_id = ?",
    "thread": "SELECT 1 FROM work_threads WHERE thread_id = ? AND root_id = ?",
}


async def _row_exists(repository: ConversationRepository, sql: str, params: tuple[object, ...]) -> bool:
    """Run an existence probe via the repository's backend connection."""

    backend = repository._backend
    async with backend.connection() as conn:
        cursor = await conn.execute(sql, params)
        row = await cursor.fetchone()
    return row is not None


async def _content_block_exists(
    repository: ConversationRepository,
    *,
    message_id: str,
    block_index_token: str,
) -> bool:
    """Check that ``messages(message_id)`` exists and ``block_index`` is in range."""

    try:
        block_index = int(block_index_token)
    except ValueError:
        return False
    backend = repository._backend
    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS n FROM content_blocks WHERE message_id = ?",
            (message_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return False
    count = int(row["n"])
    return 0 <= block_index < count


def parse_content_block_target_id(target_id: str) -> tuple[str, str]:
    """Split ``"{message_id}:{block_index}"`` into its components.

    Raises ``ValueError`` if the token is malformed.
    """

    if ":" not in target_id:
        raise ValueError("content_block target_id must be 'message_id:block_index'")
    message_part, _, block_part = target_id.rpartition(":")
    if not message_part or not block_part:
        raise ValueError("content_block target_id must be 'message_id:block_index'")
    return message_part, block_part


async def resolve_insight_target(
    repository: ConversationRepository,
    *,
    target_type: str,
    target_id: str | None,
    conversation_id: str,
    message_id: str | None = None,
) -> ResolvedTarget:
    """Validate a non-conversation/non-message target and return its row payload.

    The caller is responsible for resolving ``conversation_id`` first via
    ``repository.resolve_id`` so this helper can assume the conversation
    exists. ``target_id`` is required for every kind except ``session``
    (where it defaults to the conversation_id).
    """

    if target_type not in KINDS_BY_NAME:
        raise ValueError(f"target_type must be one of: {', '.join(TARGET_KIND_NAMES)}")

    if target_type == "session":
        resolved_target_id = target_id or conversation_id
        if resolved_target_id != conversation_id:
            raise ValueError("session target_id must equal the conversation_id (session root)")
        if not await _row_exists(
            repository,
            _INSIGHT_QUERIES["session"],
            (conversation_id,),
        ):
            raise ValueError(f"session profile for conversation {conversation_id!r} is not materialized")
        return {
            "target_type": "session",
            "target_id": conversation_id,
            "conversation_id": conversation_id,
            "message_id": None,
            "identity_key": identity_key(
                "session",
                conversation_id=conversation_id,
                target_id=conversation_id,
            ),
        }

    if target_type == "work_event":
        if not target_id:
            raise ValueError("work_event target requires target_id (event_id)")
        if not await _row_exists(
            repository,
            _INSIGHT_QUERIES["work_event"],
            (target_id, conversation_id),
        ):
            raise ValueError(f"work_event {target_id!r} is not in conversation {conversation_id!r}")
        return {
            "target_type": "work_event",
            "target_id": target_id,
            "conversation_id": conversation_id,
            "message_id": None,
            "identity_key": identity_key(
                "work_event",
                conversation_id=conversation_id,
                target_id=target_id,
            ),
        }

    if target_type == "thread":
        if not target_id:
            raise ValueError("thread target requires target_id (thread_id)")
        if not await _row_exists(
            repository,
            _INSIGHT_QUERIES["thread"],
            (target_id, conversation_id),
        ):
            raise ValueError(f"thread {target_id!r} is not rooted at conversation {conversation_id!r}")
        return {
            "target_type": "thread",
            "target_id": target_id,
            "conversation_id": conversation_id,
            "message_id": None,
            "identity_key": identity_key(
                "thread",
                conversation_id=conversation_id,
                target_id=target_id,
            ),
        }

    if target_type == "content_block":
        if not target_id:
            raise ValueError("content_block target requires target_id 'message_id:block_index'")
        msg_id, block_part = parse_content_block_target_id(target_id)
        effective_message_id = message_id or msg_id
        if effective_message_id != msg_id:
            raise ValueError("content_block message_id must match the message_id in target_id")
        if not await _content_block_exists(
            repository,
            message_id=effective_message_id,
            block_index_token=block_part,
        ):
            raise ValueError(f"content_block {target_id!r} is not present in conversation {conversation_id!r}")
        return {
            "target_type": "content_block",
            "target_id": target_id,
            "conversation_id": conversation_id,
            "message_id": effective_message_id,
            "identity_key": identity_key(
                "content_block",
                conversation_id=conversation_id,
                target_id=target_id,
            ),
        }

    if target_type == "attachment":
        if not target_id:
            raise ValueError("attachment target requires target_id")
        # Attachments live as content blocks of kind 'tool_result' or as raw
        # blob refs; the durable identity contract is "non-empty token, scoped
        # to a conversation". A stronger FK lands when attachment identity is
        # promoted to a first-class table.
        return {
            "target_type": "attachment",
            "target_id": target_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "identity_key": identity_key(
                "attachment",
                conversation_id=conversation_id,
                target_id=target_id,
            ),
        }

    if target_type == "paste_span":
        if not target_id:
            raise ValueError("paste_span target requires target_id")
        return {
            "target_type": "paste_span",
            "target_id": target_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "identity_key": identity_key(
                "paste_span",
                conversation_id=conversation_id,
                target_id=target_id,
            ),
        }

    # conversation/message are resolved by the caller; this branch is
    # defensive and only fires if a future kind is added to the registry
    # without an explicit handler here.
    raise ValueError(f"no resolver handler for target_type {target_type!r}")


__all__ = [
    "ResolvedTarget",
    "parse_content_block_target_id",
    "resolve_insight_target",
]
