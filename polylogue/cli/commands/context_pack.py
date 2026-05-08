"""CLI context-pack command — assemble provenance-rich context bundles for agents."""

from __future__ import annotations

from typing import Any

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.types import AppEnv
from polylogue.core.json import dumps as json_dumps
from polylogue.mcp.context_pack import (
    ContextPackConversation,
    ContextPackDateRange,
    ContextPackDecisions,
    ContextPackIntent,
    ContextPackMessage,
    ContextPackPayload,
    ContextPackProvenance,
    ContextPackQueryContext,
    _build_project_context,
    _summarize_action_events,
)
from polylogue.mcp.query_contracts import MCPConversationQueryRequest

_DEFAULT_MAX_CONVERSATIONS = 5
_DEFAULT_MAX_MESSAGES = 20


@click.command("context-pack")
@click.option("--project-path", "-P", default=None, help="Filter by cwd prefix pattern")
@click.option("--project-repo", "-R", default=None, help="Filter by git repo URL or name")
@click.option("--since", "-s", default=None, help="Start date (ISO 8601)")
@click.option("--until", "-u", default=None, help="End date (ISO 8601)")
@click.option("--provider", "-p", default=None, help="Provider name filter")
@click.option("--query", "-q", default=None, help="Free-text query")
@click.option(
    "--max-conversations", "-n", type=int, default=_DEFAULT_MAX_CONVERSATIONS, help="Max conversations (1-20)"
)
@click.option(
    "--max-messages", "-m", type=int, default=_DEFAULT_MAX_MESSAGES, help="Max messages per conversation (1-100)"
)
@click.option("--no-redact", "no_redact", is_flag=True, default=False, help="Do not redact filesystem paths")
@click.pass_obj
def context_pack_command(
    env: AppEnv,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    provider: str | None,
    query: str | None,
    max_conversations: int,
    max_messages: int,
    no_redact: bool,
) -> None:
    """Build a provenance-rich context pack for agent analysis.

    \b
    Examples:
        polylogue context-pack -P /realm/project/polylogue
        polylogue context-pack -R github.com/Sinity/polylogue -s 2026-01-01
        polylogue context-pack -q "cost tracking"
    """
    conv_limit = max(1, min(max_conversations, 20))
    msg_limit = max(1, min(max_messages, 100))

    ops = env.operations
    repo = env.repository

    spec = MCPConversationQueryRequest(
        query=query,
        provider=provider,
        since=since,
        until=until,
        cwd_prefix=project_path,
        repo=project_repo,
        sort="date",
        reverse=True,
        limit=conv_limit,
    ).build_spec(lambda x: max(1, min(int(x) + 0, 20)))  # type: ignore[arg-type]

    conversations = run_coroutine_sync(ops.query_conversations(spec))
    total_matching = len(conversations)
    conv_ids = [str(conv.id) for conv in conversations]

    all_action_events: dict[str, tuple[Any, ...]] = {}
    if conv_ids:
        try:
            all_action_events = run_coroutine_sync(repo.get_action_events_batch(conv_ids))
        except Exception:
            all_action_events = {}

    aggregated_events: list[Any] = []
    for events in all_action_events.values():
        aggregated_events.extend(events)

    action_summaries = _summarize_action_events(aggregated_events, redact=not no_redact)

    dates: list[str] = []
    for conv in conversations:
        if conv.created_at is not None:
            dates.append(str(conv.created_at))
        if conv.updated_at is not None:
            dates.append(str(conv.updated_at))
    earliest = min(dates) if dates else None
    latest = max(dates) if dates else None

    pack_conversations: list[ContextPackConversation] = []
    total_msg = 0
    total_tools = 0
    for conv in conversations[:conv_limit]:
        conv_id = str(conv.id)
        total_msg += conv.message_count
        tool_count = getattr(conv, "tool_use_count", 0) or 0
        total_tools += tool_count

        messages: list[ContextPackMessage] = []
        try:
            msg_list, _ = run_coroutine_sync(ops.get_messages_paginated(conv_id, limit=msg_limit, offset=0))
        except Exception:
            msg_list = []
        for m in msg_list:
            messages.append(
                ContextPackMessage(
                    role=m.role.value if m.role else "unknown",
                    text=m.text or "",
                )
            )

        pack_conversations.append(
            ContextPackConversation(
                conversation_id=conv_id,
                title=conv.title,
                provider=conv.provider.value if conv.provider else "unknown",
                created_at=str(conv.created_at) if conv.created_at is not None else None,
                updated_at=str(conv.updated_at) if conv.updated_at is not None else None,
                message_count=conv.message_count,
                tool_use_count=tool_count if tool_count else None,
                messages=messages,
            )
        )

    payload = ContextPackPayload(
        intent=ContextPackIntent(),
        decisions=ContextPackDecisions(),
        project=_build_project_context(aggregated_events, redact=not no_redact),
        date_range=ContextPackDateRange(
            since=since,
            until=until,
            earliest=earliest,
            latest=latest,
            conversation_count_in_range=total_matching,
        ),
        query_context=ContextPackQueryContext(
            total_matching_conversations=total_matching,
            conversations_included=min(total_matching, conv_limit),
            project_path=project_path,
            project_repo=project_repo,
            provider=provider,
            query=query,
        ),
        conversations=pack_conversations,
        action_summaries=action_summaries,
        provenance=ContextPackProvenance(redacted=not no_redact),
        total_conversations=total_matching,
        total_messages=total_msg,
        total_tool_calls=total_tools,
    )

    click.echo(json_dumps(payload))
