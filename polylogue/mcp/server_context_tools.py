"""Context pack MCP tool registration.

Registers ``build_context_pack`` — the agent-facing context assembly tool
that produces provenance-rich project/date/query context packs from canonical
archive tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from polylogue.mcp.context_pack import (
    ContextPackConversation,
    ContextPackDateRange,
    ContextPackMessage,
    ContextPackPayload,
    ContextPackProvenance,
    ContextPackQueryContext,
    ContextPackUnresolvedWork,
    _build_project_context,
    _summarize_action_events,
    redact_path,
    select_context_pack_conversations,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.storage.repository import ConversationRepository

_DEFAULT_MAX_CONVERSATIONS = 5
_DEFAULT_MAX_MESSAGES = 20
_DETAIL_DEFAULTS: dict[str, tuple[int, bool]] = {
    "summary": (0, True),
    "compact": (200, True),
    "full": (0, False),
}


def _get_detail_settings(detail_level: str) -> tuple[int, bool]:
    """Return (max_text_length, include_messages) for a detail level."""
    return _DETAIL_DEFAULTS.get(detail_level, _DETAIL_DEFAULTS["compact"])


def _message_has_tool_use(msg: object) -> bool:
    """Check whether a domain Message has tool-use content blocks."""
    content_blocks = getattr(msg, "content_blocks", None) or ()
    return any(isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result") for block in content_blocks)


def _message_has_thinking(msg: object) -> bool:
    """Check whether a domain Message has thinking content blocks."""
    content_blocks = getattr(msg, "content_blocks", None) or ()
    return any(isinstance(block, dict) and block.get("type") == "thinking" for block in content_blocks)


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending ellipsis if truncated."""
    if max_length > 0 and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def register_context_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def build_context_pack(
        project_path: str | None = None,
        project_repo: str | None = None,
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
        query: str | None = None,
        max_conversations: int = _DEFAULT_MAX_CONVERSATIONS,
        max_messages_per_conversation: int = _DEFAULT_MAX_MESSAGES,
        detail_level: str = "compact",
        redact_paths: bool = True,
    ) -> str:
        """Build a provenance-rich context pack for agent analysis.

        Assembles project context, date range, filtered conversations,
        action summaries, and unresolved work from the archive.

        Parameters:
            project_path: Filter conversations by cwd prefix pattern.
            project_repo: Filter conversations by git repo URL or name.
            since: Start date for conversation filter (ISO format).
            until: End date for conversation filter (ISO format).
            provider: Provider name filter.
            query: Free-text query for semantic narrowing.
            max_conversations: Maximum conversations to include (1-20).
            max_messages_per_conversation: Max messages per conversation (1-100).
            detail_level: 'summary' (metadata only), 'compact' (truncated), 'full'.
            redact_paths: Redact filesystem paths for privacy (default True).
        """

        async def run() -> str:
            conv_limit = max(1, min(max_conversations, 20))
            msg_limit = max(1, min(max_messages_per_conversation, 100))
            valid_detail = detail_level if detail_level in _DETAIL_DEFAULTS else "compact"
            max_text, include_messages = _get_detail_settings(valid_detail)

            ops = hooks.get_archive_ops()
            # cast: at runtime the query store IS the full ConversationRepository
            repo = cast("ConversationRepository", hooks.get_query_store())

            selection = await select_context_pack_conversations(
                ops.query_conversations,
                hooks.clamp_limit,
                project_path=project_path,
                project_repo=project_repo,
                since=since,
                until=until,
                provider=provider,
                query=query,
                limit=conv_limit,
            )
            conversations = selection.conversations
            total_matching = len(conversations)
            conv_ids = [str(conv.id) for conv in conversations]

            all_action_events: dict[str, tuple[Any, ...]] = {}
            if conv_ids:
                try:
                    all_action_events = await repo.get_action_events_batch(conv_ids)
                except Exception:
                    all_action_events = {}

            aggregated_events: list[Any] = []
            for events in all_action_events.values():
                aggregated_events.extend(events)

            project_ctx = _build_project_context(aggregated_events, redact=redact_paths)

            action_summary = _summarize_action_events(aggregated_events, redact=redact_paths)

            # Date range
            dates = [d for conv in conversations for d in (conv.created_at, conv.updated_at) if d is not None]
            earliest = min(dates).isoformat() if dates else None
            latest = max(dates).isoformat() if dates else None

            date_range = ContextPackDateRange(
                since=since,
                until=until,
                earliest=earliest,
                latest=latest,
                conversation_count_in_range=total_matching,
            )

            query_ctx = ContextPackQueryContext(
                total_matching_conversations=total_matching,
                conversations_included=min(total_matching, conv_limit),
                project_path=project_path,
                project_repo=project_repo,
                provider=provider,
                query=query,
                query_matched=total_matching,
                query_total=selection.query_total,
                match_strategy=selection.match_strategy,
                relaxed_filters=list(selection.relaxed_filters),
            )

            # Build conversation entries
            pack_conversations: list[ContextPackConversation] = []
            for conv in conversations[:conv_limit]:
                conv_id = str(conv.id)
                conv_events = all_action_events.get(conv_id, ())

                # Per-conversation project context
                cwd_set: list[str] = []
                branch_set: list[str] = []
                affected_set: list[str] = []
                for event in conv_events:
                    if event.cwd_path:
                        cwd_display = redact_path(event.cwd_path) if redact_paths else event.cwd_path
                        if cwd_display not in cwd_set:
                            cwd_set.append(cwd_display)
                    for branch in event.branch_names:
                        if branch and branch not in branch_set:
                            branch_set.append(branch)
                    for path in event.affected_paths:
                        affected = redact_path(path) if redact_paths else path
                        if affected not in affected_set:
                            affected_set.append(affected)

                # Messages
                messages: list[ContextPackMessage] = []
                if include_messages:
                    try:
                        poly = hooks.get_polylogue()
                        paginated, _total = await poly.get_messages_paginated(
                            conv_id,
                            limit=msg_limit,
                            offset=0,
                        )
                    except Exception:
                        paginated = []

                    for msg in paginated:
                        text = str(msg.text or "")
                        messages.append(
                            ContextPackMessage(
                                role=str(msg.role),
                                text=_truncate_text(text, max_text),
                                sort_key=getattr(msg, "sort_key", None),
                                has_tool_use=_message_has_tool_use(msg),
                                has_thinking=_message_has_thinking(msg),
                            )
                        )

                # Tool use count from action events
                tool_use_count = len(conv_events) if conv_events else None

                pack_conversations.append(
                    ContextPackConversation(
                        conversation_id=conv_id,
                        provider=str(conv.provider),
                        title=conv.display_title or None,
                        created_at=(conv.created_at.isoformat() if conv.created_at else None),
                        updated_at=(conv.updated_at.isoformat() if conv.updated_at else None),
                        message_count=len(conv.messages),
                        tool_use_count=tool_use_count,
                        messages=messages,
                        cwd_paths=cwd_set[:5],
                        branch_names=branch_set[:5],
                        affected_paths=affected_set[:10],
                    )
                )

            # Unresolved work
            unresolved: list[ContextPackUnresolvedWork] = []
            for conv in conversations[:conv_limit]:
                conv_id = str(conv.id)
                conv_events = all_action_events.get(conv_id, ())
                if not conv_events:
                    continue
                last_ts = None
                for event in conv_events:
                    if event.timestamp and (last_ts is None or event.timestamp > last_ts):
                        last_ts = event.timestamp
                unresolved.append(
                    ContextPackUnresolvedWork(
                        conversation_id=conv_id,
                        provider=str(conv.provider),
                        title=conv.display_title or None,
                        last_activity=last_ts.isoformat() if last_ts else None,
                        tool_use_count=len(conv_events),
                        reason="Has action events — may contain unresolved work",
                    )
                )

            provenance = ContextPackProvenance(
                generated_at=datetime.now(UTC).isoformat(),
                redacted=redact_paths,
            )

            payload = ContextPackPayload(
                project=project_ctx,
                date_range=date_range,
                query_context=query_ctx,
                conversations=pack_conversations,
                action_summaries=action_summary,
                unresolved_work=unresolved,
                provenance=provenance,
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("build_context_pack", run)


__all__ = ["register_context_tools"]
