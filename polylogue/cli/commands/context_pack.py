"""CLI context-pack command — assemble provenance-rich context bundles for agents."""

from __future__ import annotations

from typing import Any, cast

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.types import AppEnv
from polylogue.core.json import dumps as json_dumps
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
)
from polylogue.mcp.query_contracts import MCPConversationQueryRequest
from polylogue.storage.repository import ConversationRepository

_DEFAULT_MAX_CONVERSATIONS = 5
_DEFAULT_MAX_MESSAGES = 20
_DETAIL_DEFAULTS: dict[str, tuple[int, bool]] = {
    "summary": (0, True),
    "compact": (200, True),
    "full": (0, False),
}


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
@click.option(
    "--detail", "-d", type=click.Choice(["summary", "compact", "full"]), default="compact", help="Detail level"
)
@click.option("--no-redact", "no_redact", is_flag=True, default=False, help="Do not redact filesystem paths")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default="json", help="Output format")
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
    detail: str,
    no_redact: bool,
    output_format: str,
) -> None:
    """Build a provenance-rich context pack for agent analysis.

    Assembles project context, date range, filtered conversations,
    action summaries, and unresolved work from the conversational archive.

    \b
    Examples:
        polylogue context-pack -P /realm/project/polylogue
        polylogue context-pack -R github.com/Sinity/polylogue -s 2026-01-01
        polylogue context-pack -q "cost tracking" -d summary
    """
    conv_limit = max(1, min(max_conversations, 20))
    msg_limit = max(1, min(max_messages, 100))

    ops = env.operations
    repo = cast("ConversationRepository", env.repository)

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
    ).build_spec(lambda x: max(1, min(int(x), 20)))

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

    project_ctx = _build_project_context(aggregated_events, redact=not no_redact)
    action_summary = _summarize_action_events(aggregated_events, redact=not no_redact)

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
    )

    pack_conversations: list[ContextPackConversation] = []
    for conv in conversations[:conv_limit]:
        conv_id = str(conv.id)
        conv_events = all_action_events.get(conv_id, ())
        conv_event_summary = _summarize_action_events(list(conv_events), redact=not no_redact)

        messages: list[ContextPackMessage] = []
        if detail != "summary":
            try:
                msg_list, _ = run_coroutine_sync(ops.get_messages_paginated(conv_id, limit=msg_limit, offset=0))
            except Exception:
                msg_list = []
            max_text = _DETAIL_DEFAULTS[detail][0]
            for m in msg_list:
                text = m.text or ""
                if max_text > 0 and len(text) > max_text:
                    text = text[:max_text] + "..."
                messages.append(
                    ContextPackMessage(
                        role=m.role.value if m.role else "unknown",
                        text=text,
                        tool_use_count=m.has_tool_use,
                        thinking_count=m.has_thinking,
                        paste_count=m.has_paste,
                        word_count=m.word_count,
                    )
                )

        pack_conversations.append(
            ContextPackConversation(
                id=conv_id,
                title=conv.title or "",
                provider=conv.provider.value if conv.provider else "unknown",
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=conv.message_count,
                word_count=conv.word_count,
                tool_use_count=conv.tool_use_count,
                thinking_count=conv.thinking_count,
                branch_type=conv.branch_type,
                action_summary=conv_event_summary,
                messages=tuple(messages),
            )
        )

    payload = ContextPackPayload(
        project=project_ctx,
        date_range=date_range,
        query_context=query_ctx,
        conversations=tuple(pack_conversations),
        unresolved_work=(
            ContextPackUnresolvedWork(
                items=action_summary.unresolved_items if action_summary else (),
            )
        ),
        provenance=ContextPackProvenance(),
    )

    click.echo(json_dumps(payload))
