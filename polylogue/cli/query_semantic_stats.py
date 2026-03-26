"""Semantic action/tool grouped stats output helpers."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from polylogue.cli.query_semantic_slice import (
    SemanticStatsSlice,
    action_matches_slice,
    normalized_tool_name,
)
from polylogue.cli.query_stats_structured import emit_structured_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import ConversationSummary
    from polylogue.lib.query_spec import ConversationQuerySpec
    from polylogue.storage.repository import ConversationRepository


async def _load_semantic_stats_conversations(
    repo: ConversationRepository,
    conversation_ids: list[str],
) -> list:
    get_many = getattr(repo, "get_many", None)
    if get_many is not None:
        conversations = get_many(conversation_ids)
        if inspect.isawaitable(conversations):
            return await conversations
        if isinstance(conversations, list):
            return conversations

    queries = getattr(repo, "queries", None)
    if queries is None:
        return []

    records_result = queries.get_conversations_batch(conversation_ids)
    messages_result = queries.get_messages_batch(conversation_ids)
    if not inspect.isawaitable(records_result) or not inspect.isawaitable(messages_result):
        return []

    from polylogue.storage.hydrators import conversation_from_records

    records = await records_result
    messages_by_conversation = await messages_result

    attachments_by_conversation: dict[str, list] = {}
    get_attachments_batch = getattr(queries, "get_attachments_batch", None)
    if get_attachments_batch is not None:
        attachments_result = get_attachments_batch(conversation_ids)
        if inspect.isawaitable(attachments_result):
            attachments_by_conversation = await attachments_result

    records_by_id = {str(record.conversation_id): record for record in records}
    hydrated = []
    for conversation_id in conversation_ids:
        record = records_by_id.get(conversation_id)
        if record is None:
            continue
        hydrated.append(
            conversation_from_records(
                record,
                messages_by_conversation.get(conversation_id, []),
                attachments_by_conversation.get(conversation_id, []),
            )
        )
    return hydrated


async def output_stats_by_semantic_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    repo: ConversationRepository,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    await output_stats_by_semantic_ids(
        env,
        [str(summary.id) for summary in summaries],
        repo,
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_semantic_query(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationRepository,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    await output_stats_by_semantic_ids(
        env,
        conversation_ids,
        repo,
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_semantic_ids(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationRepository,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    from collections import Counter, defaultdict

    from rich.table import Table

    if dimension not in {"action", "tool"}:
        raise ValueError(f"Unsupported semantic stats dimension: {dimension}")
    if not conversation_ids:
        env.ui.console.print("No conversations matched.")
        return

    semantic_slice = SemanticStatsSlice.from_selection(selection)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
    matched_facts = 0
    matched_messages = 0
    key_func = (lambda action: action.kind.value) if dimension == "action" else normalized_tool_name
    action_event_status = repo.get_action_event_read_model_status()
    if inspect.isawaitable(action_event_status):
        action_event_status = await action_event_status
    if not isinstance(action_event_status, dict):
        action_event_status = {}
    action_read_model_ready = bool(action_event_status.get("ready", False))

    for offset in range(0, len(conversation_ids), batch_size):
        batch_ids = conversation_ids[offset : offset + batch_size]
        if action_read_model_ready:
            action_events_by_conversation = await repo.get_action_events_batch(batch_ids)
            conversation_actions = {
                conversation_id: tuple(
                    action
                    for action in action_events_by_conversation.get(conversation_id, ())
                    if action_matches_slice(action, semantic_slice)
                )
                for conversation_id in batch_ids
            }
        else:
            from polylogue.lib.semantic_facts import build_conversation_semantic_facts

            conversations = await _load_semantic_stats_conversations(repo, batch_ids)
            conversation_actions = {
                str(conversation.id): tuple(
                    action
                    for action in build_conversation_semantic_facts(conversation).action_events
                    if action_matches_slice(action, semantic_slice)
                )
                for conversation in conversations
            }

        for conversation_id in batch_ids:
            filtered_actions = conversation_actions.get(conversation_id, ())
            group_counts = Counter(key_func(action) for action in filtered_actions)
            if not group_counts:
                groups["none"]["convs"] += 1
                continue

            matched_facts += sum(group_counts.values())
            message_groups: dict[str, set[str]] = defaultdict(set)
            for action in filtered_actions:
                message_groups[key_func(action)].add(action.message_id)

            matched_messages += len({action.message_id for action in filtered_actions})
            for key, fact_count in group_counts.items():
                groups[key]["convs"] += 1
                groups[key]["facts"] += fact_count
                groups[key]["msgs"] += len(message_groups[key])

    rows = [
        {
            "group": key,
            "conversations": stats["convs"],
            "facts": stats["facts"],
            "messages": stats["msgs"],
        }
        for key, stats in sorted(groups.items())
    ]
    summary = {
        "group": "MATCHED",
        "conversations": len(conversation_ids),
        "facts": matched_facts,
        "messages": matched_messages,
    }
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=True,
    ):
        return

    env.ui.console.print(f"\nMatched: {len(conversation_ids)} conversations (by {dimension})\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Facts", justify="right")
    table.add_column("Msgs", justify="right")
    for row in rows:
        table.add_row(
            str(row["group"]),
            f"{row['conversations']:,}",
            f"{row['facts']:,}",
            f"{row['messages']:,}",
        )
    table.add_section()
    table.add_row(
        "[bold]MATCHED[/]",
        f"[bold]{summary['conversations']:,}[/]",
        f"[bold]{summary['facts']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
    )
    env.ui.console.print(table)
    env.ui.console.print(f"Note: conversations may appear in multiple {dimension} groups.")


__all__ = [
    "output_stats_by_semantic_ids",
    "output_stats_by_semantic_query",
    "output_stats_by_semantic_summaries",
]
