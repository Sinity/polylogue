"""Context pack MCP tool registration.

Registers ``build_context_pack`` — the agent-facing context assembly tool
that produces provenance-rich project/date/query context packs from canonical
archive tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from polylogue.mcp.context_pack import (
    ContextPackActionSummary,
    ContextPackConversation,
    ContextPackDateRange,
    ContextPackMessage,
    ContextPackPayload,
    ContextPackProject,
    ContextPackProvenance,
    ContextPackQueryContext,
    ContextPackUnresolvedWork,
    _build_project_context,
    _summarize_action_events,
    redact_path,
)
from polylogue.mcp.query_contracts import MCPConversationQueryRequest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks

_DEFAULT_MAX_CONVERSATIONS = 5
_DEFAULT_MAX_MESSAGES = 20
_DETAIL_DEFAULTS: dict[str, tuple[int, bool]] = {
    "summary": (0, True),
    "compact": (200, True),
    "full": (0, False),
}


def _get_text_settings(detail_level: str, redact_paths: bool) -> tuple[int, bool]:
    """Return (max_text_length, should_redact) for a detail level."""
    return _DETAIL_DEFAULTS.get(detail_level, _DETAIL_DEFAULTS["compact"])


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
            max_text, redact = _get_text_settings(valid_detail, redact_paths)

            ops = hooks.get_archive_ops()
            repo = hooks.get_query_store()

            spec = MCPConversationQueryRequest(
                query=query,
                provider=provider,
                since=since,
                until=until,
                cwd_prefix=project_path,
                repo=project_repo,
                sort="received_desc",
                limit=conv_limit,
            ).build_spec(hooks.clamp_limit)

            conversations = await ops.query_conversations(spec)
            total_matching = len(conversations)
            conv_ids = [str(conv.id) for conv in conversations]

            all_action_events: dict[str, tuple] = {}
            if conv_ids:
                try:
                    all_action_events = await repo.get_action_events_batch(conv_ids)
                except Exception:
                    all_action_events = {}

            aggregated_events = []
            for events in all_action_events.values():
                aggregated_events.extend(events)

            project_ctx = _build_project_context(aggregated_events, redact=redact)
            action_summary = _summarize_action_events(aggregated_events, redact=redact)

            dates = []
            for conv in conversations:
                if conv.created_at:
                    dates.append(conv.created_at)
                elif conv.display_date:
                    dates.append(conv.display_date)
            earliest = min(dates).isoformat() if dates else None
            latest = max(dates).isoformat() if dates else None

            date_range = ContextPackDateRange(
                earliest=earliest,
                latest=latest,
                conversation_count_in_range=total_matching,
                has_gaps=False,
            )

            query_ctx = ContextPackQueryContext(
                total_matching_conversations=total_matching,
                conversations_included=min(total_matching, conv_limit),
                project_path=project_path,
                project_repo=project_repo,
                provider=provider,
                query=query,
            )

            pack_conversations: list[ContextPackConversation] = []
            poly = hooks.get_polylogue()
            for conv in conversations[:conv_limit]:
                conv_id = str(conv.id)
                conv_events = all_action_events.get(conv_id, ())

                cwd_set: list[str] = []
                branch_set: list[str] = []
                affected_set: list[str] = []
                for event in conv_events:
                    if event.cwd_path:
                        cwd_display = redact_path(event.cwd_path) if redact else event.cwd_path
                        if cwd_display not in cwd_set:
                            cwd_set.append(cwd_display)
                    for branch in event.branch_names:
                        if branch not in branch_set:
                            branch_set.append(branch)
                    for path in event.affected_paths:
                        affected = redact_path(path) if redact else path
                        if affected not in affected_set:
                            affected_set.append(affected)

                messages: list[ContextPackMessage] = []
                if valid_detail in ("compact", "full"):
                    try:
                        paginated, _ = await poly.get_messages_paginated(
                            conv_id, limit=msg_limit, offset=0,
                        )
                    except Exception:
                        paginated = []

                    event_by_msg_id: dict[str, object] = {}
                    for event in conv_events:
                        event_by_msg_id[event.message_id] = event

                    for msg in paginated:
                        event = event_by_msg_id.get(str(msg.id))
                        if event is not None and hasattr(event, "normalized_tool_name"):
                            messages.append(
                                ContextPackMessage.from_action_event(
                                    msg, event, redact=redact, max_text_length=max_text
                                )
                            )
                        else:
                            messages.append(
                                ContextPackMessage.from_message(
                                    msg, redact=redact, max_text_length=max_text
                                )
                            )

                tool_use_count = sum(
                    1 for msg in conv.messages
                    if hasattr(msg, "has_tool_use") and msg.has_tool_use
                )

                pack_conversations.append(
                    ContextPackConversation(
                        id=conv_id,
                        provider=str(conv.provider),
                        title=conv.display_title or "(untitled)",
                        created_at=conv.created_at.isoformat() if conv.created_at else None,
                        updated_at=conv.updated_at.isoformat() if conv.updated_at else None,
                        message_count=len(conv.messages),
                        tool_use_count=tool_use_count,
                        messages=tuple(messages),
                        cwd_paths=tuple(cwd_set[:5]),
                        branch_names=tuple(branch_set[:5]),
                        affected_paths=tuple(affected_set[:10]),
                    )
                )

            unresolved: list[ContextPackUnresolvedWork] = []
            for conv in conversations[:conv_limit]:
                conv_id = str(conv.id)
                conv_events = all_action_events.get(conv_id, ())
                if conv_events:
                    last_ts = None
                    for event in conv_events:
                        if event.timestamp:
                            if last_ts is None or event.timestamp > last_ts:
                                last_ts = event.timestamp
                    unresolved.append(
                        ContextPackUnresolvedWork(
                            conversation_id=conv_id,
                            provider=str(conv.provider),
                            title=conv.display_title or "(untitled)",
                            last_activity=last_ts.isoformat() if last_ts else None,
                            tool_use_count=len(conv_events),
                            reason="Has action events — may contain unresolved work",
                        )
                    )

            provenance = ContextPackProvenance(
                generated_at=datetime.now(UTC).isoformat(),
            )

            payload = ContextPackPayload(
                project=project_ctx,
                date_range=date_range,
                query_context=query_ctx,
                conversations=tuple(pack_conversations),
                action_summary=action_summary,
                unresolved_work=tuple(unresolved),
                provenance=provenance,
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("build_context_pack", run)


__all__ = ["register_context_tools"]
