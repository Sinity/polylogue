"""Grouped stats output: structured serialization, SQL stats, summary/date/provider grouping,
semantic action/tool grouping, and profile-backed grouping."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_feedback import emit_no_results

if TYPE_CHECKING:
    from polylogue.cli.shared.types import AppEnv
    from polylogue.lib.action_event.action_events import ActionEvent
    from polylogue.lib.filter.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query.spec import ConversationQuerySpec
    from polylogue.protocols import ConversationArchiveStatsStore


DATE_GROUP_DIMENSIONS = frozenset({"month", "year", "day"})
DATE_GROUP_FORMATS = {
    "day": "%Y-%m-%d",
    "month": "%Y-%m",
    "year": "%Y",
}
GROUP_COUNT_COLUMNS = (
    ("conversations", "Convs"),
    ("messages", "Messages"),
)
GROUP_WORD_COLUMNS = (
    ("conversations", "Convs"),
    ("messages", "Messages"),
    ("words", "Words"),
)
SEMANTIC_COLUMNS = (
    ("conversations", "Convs"),
    ("facts", "Facts"),
    ("messages", "Msgs"),
)
PROFILE_COLUMNS = (
    ("conversations", "Convs"),
    ("work_events", "Events"),
    ("messages", "Msgs"),
)


@dataclass(frozen=True, slots=True)
class GroupedStatsPayload:
    rows: list[dict[str, object]]
    summary: dict[str, object]


def _sort_group_keys(groups: Mapping[str, object], dimension: str) -> list[str]:
    return sorted(groups.keys(), reverse=dimension in DATE_GROUP_DIMENSIONS)


def _summary_group_key(summary: ConversationSummary, dimension: str) -> str:
    if dimension == "provider":
        return str(summary.provider) if summary.provider else "unknown"
    if dimension in DATE_GROUP_DIMENSIONS:
        dt = summary.updated_at or summary.created_at
        return dt.strftime(DATE_GROUP_FORMATS[dimension]) if dt else "unknown"
    return "all"


def _conversation_group_key(conversation: Conversation, dimension: str) -> str:
    if dimension == "provider":
        return conversation.provider or "unknown"
    if dimension in DATE_GROUP_DIMENSIONS:
        dt = conversation.display_date
        return dt.strftime(DATE_GROUP_FORMATS[dimension]) if dt else "unknown"
    return "all"


def _count_value(row: dict[str, object], key: str) -> int:
    value = row[key]
    if isinstance(value, int):
        return value
    raise TypeError(f"Stats value {key!r} must be int, got {type(value).__name__}")


def _formatted_group_label(group_label: str, *, color_provider: bool) -> str:
    if not color_provider:
        return group_label

    from polylogue.ui.theme import provider_color

    return f"[{provider_color(group_label).hex}]{group_label}[/]"


def _emit_grouped_stats_table(
    env: AppEnv,
    *,
    dimension: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    columns: tuple[tuple[str, str], ...],
    total_label: str,
    matched_conversations: int,
    output_format: str,
    color_provider: bool = False,
    multi_membership: bool = False,
    note: str | None = None,
) -> None:
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=multi_membership,
    ):
        return

    from rich.table import Table

    env.ui.console.print(f"\nMatched: {matched_conversations:,} conversations (by {dimension})\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    for _, title in columns:
        table.add_column(title, justify="right")

    for row in rows:
        group_label = str(row["group"])
        table.add_row(
            _formatted_group_label(group_label, color_provider=color_provider),
            *(f"{_count_value(row, key):,}" for key, _ in columns),
        )

    table.add_section()
    table.add_row(
        f"[bold]{total_label}[/]",
        *(f"[bold]{_count_value(summary, key):,}[/]" for key, _ in columns),
    )
    env.ui.console.print(table)
    if note is not None:
        env.ui.console.print(note)


# ---------------------------------------------------------------------------
# Structured stats serialization (from query_stats_structured.py)
# ---------------------------------------------------------------------------


def emit_structured_stats(
    *,
    output_format: str,
    dimension: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    multi_membership: bool = False,
) -> bool:
    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "dimension": dimension,
                    "multi_membership": multi_membership,
                    "rows": rows,
                    "summary": summary,
                },
                indent=2,
            )
        )
        return True

    if output_format == "yaml":
        import yaml

        click.echo(
            yaml.dump(
                {
                    "dimension": dimension,
                    "multi_membership": multi_membership,
                    "rows": rows,
                    "summary": summary,
                },
                default_flow_style=False,
                allow_unicode=True,
            )
        )
        return True

    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
        click.echo(buf.getvalue().rstrip("\r\n"))
        return True

    return False


# ---------------------------------------------------------------------------
# SQL-backed archive stats (from query_sql_stats.py)
# ---------------------------------------------------------------------------


async def output_stats_sql(
    env: AppEnv,
    filter_chain: ConversationFilter,
    repo: ConversationArchiveStatsStore,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "markdown",
) -> None:
    """Output statistics using SQL aggregation without full message loading."""
    described_filters = filter_chain.describe()
    has_filters = bool(described_filters)

    archive_stats = None
    if has_filters:
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                emit_no_results(
                    env,
                    selection=selection,
                    output_format=output_format,
                    exit_code=2 if output_format == "json" else None,
                )
                return
            conv_ids = [str(summary.id) for summary in summaries]
            conv_count = len(conv_ids)
        else:
            conv_count = await filter_chain.count()
            if conv_count == 0:
                emit_no_results(
                    env,
                    selection=selection,
                    output_format=output_format,
                    exit_code=2 if output_format == "json" else None,
                )
                return
            conv_ids = None
    else:
        conv_ids = None
        archive_stats = await repo.get_archive_stats()
        conv_count = archive_stats.total_conversations
        if conv_count == 0:
            emit_no_results(
                env,
                output_format=output_format,
                message="No conversations in archive.",
                exit_code=2 if output_format == "json" else None,
            )
            return
        stats = await repo.aggregate_message_stats()

    if has_filters:
        stats = await repo.aggregate_message_stats(conv_ids)

    date_range = ""
    if stats["min_sort_key"] and stats["max_sort_key"]:
        min_date = datetime.fromtimestamp(stats["min_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_date = datetime.fromtimestamp(stats["max_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    structured_summary: dict[str, object] = {
        "conversations": conv_count,
        "messages_total": stats["total"],
        "messages_user": stats["user"],
        "messages_assistant": stats["assistant"],
        "words_approx": stats["words_approx"],
        "attachment_refs": stats["attachment_refs"],
        "distinct_attachments": stats["distinct_attachments"],
        "providers": stats.get("providers") or {},
        "date_range": None,
        "filtered": has_filters,
    }
    if not has_filters:
        assert archive_stats is not None
        pending_embedding_conversations = getattr(archive_stats, "pending_embedding_conversations", 0)
        stale_embedding_messages = getattr(archive_stats, "stale_embedding_messages", 0)
        embeddings_payload = {
            "embedded_conversations": getattr(archive_stats, "embedded_conversations", 0),
            "embedded_messages": getattr(archive_stats, "embedded_messages", 0),
            "pending_embedding_conversations": pending_embedding_conversations,
            "stale_embedding_messages": stale_embedding_messages,
            "messages_missing_embedding_provenance": getattr(
                archive_stats,
                "messages_missing_embedding_provenance",
                0,
            ),
            "embedding_coverage_percent": round(getattr(archive_stats, "embedding_coverage", 0.0), 1),
            "embedding_readiness_status": getattr(archive_stats, "embedding_readiness_status", None),
            "retrieval_ready": getattr(archive_stats, "retrieval_ready", None),
        }
        structured_summary["embeddings"] = embeddings_payload
    if date_range:
        structured_summary["date_range"] = date_range

    if emit_structured_stats(
        output_format=output_format,
        dimension="archive",
        rows=[],
        summary=structured_summary,
    ):
        return

    out = env.ui.console.print
    out(f"\nConversations: {conv_count:,}\n")
    if stats["user"] or stats["assistant"]:
        out(f"Messages: {stats['total']:,} total ({stats['user']:,} user, {stats['assistant']:,} assistant)")
    else:
        out(f"Messages: {stats['total']:,}")

    if stats["words_approx"]:
        out(f"Words: ~{stats['words_approx']:,}")

    providers = stats.get("providers")
    if isinstance(providers, dict) and providers:
        provider_parts = [f"{name} ({count:,})" for name, count in providers.items()]
        out(f"Providers: {', '.join(provider_parts)}")

    out(f"Attachment refs: {stats['attachment_refs']:,}")
    out(f"Unique attachments: {stats['distinct_attachments']:,}")
    if not has_filters:
        assert archive_stats is not None
        embedding_line = (
            f"Embeddings: {archive_stats.embedded_conversations:,}/{archive_stats.total_conversations:,} convs, "
            f"{archive_stats.embedded_messages:,} msgs ({archive_stats.embedding_coverage:.1f}%)"
        )
        if pending_embedding_conversations:
            embedding_line += f", pending {pending_embedding_conversations:,}"
        if stale_embedding_messages:
            embedding_line += f", stale {stale_embedding_messages:,}"
        out(embedding_line)
    if date_range:
        out(f"Date range: {date_range}")


# ---------------------------------------------------------------------------
# Summary/date/provider grouped stats (from query_grouped_stats_summary.py)
# ---------------------------------------------------------------------------


def output_stats_by_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    msg_counts: dict[str, int],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> None:
    if not summaries:
        emit_no_results(env, selection=selection, output_format=output_format)

    groups: dict[str, list[ConversationSummary]] = defaultdict(list)
    for summary in summaries:
        groups[_summary_group_key(summary, dimension)].append(summary)

    rows: list[dict[str, object]] = []

    for key in _sort_group_keys(groups, dimension):
        group_summaries = groups[key]
        rows.append(
            {
                "group": key,
                "conversations": len(group_summaries),
                "messages": sum(msg_counts.get(str(summary.id), 0) for summary in group_summaries),
            }
        )

    summary_row = {
        "group": "TOTAL",
        "conversations": len(summaries),
        "messages": sum(msg_counts.get(str(summary.id), 0) for summary in summaries),
    }
    _emit_grouped_stats_table(
        env,
        dimension=dimension,
        rows=rows,
        summary=summary_row,
        columns=GROUP_COUNT_COLUMNS,
        total_label="TOTAL",
        matched_conversations=len(summaries),
        output_format=output_format,
        color_provider=dimension == "provider",
    )


def output_stats_by_grouped_conversations(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    output_format: str = "text",
) -> None:
    groups: dict[str, list[Conversation]] = defaultdict(list)
    for conv in results:
        groups[_conversation_group_key(conv, dimension)].append(conv)

    rows: list[dict[str, object]] = []
    for key in _sort_group_keys(groups, dimension):
        convs = groups[key]
        rows.append(
            {
                "group": key,
                "conversations": len(convs),
                "messages": sum(len(conv.messages) for conv in convs),
                "words": sum(sum(message.word_count for message in conv.messages) for conv in convs),
            }
        )

    summary = {
        "group": "TOTAL",
        "conversations": len(results),
        "messages": sum(len(conv.messages) for conv in results),
        "words": sum(sum(message.word_count for message in conv.messages) for conv in results),
    }
    _emit_grouped_stats_table(
        env,
        dimension=dimension,
        rows=rows,
        summary=summary,
        columns=GROUP_WORD_COLUMNS,
        total_label="TOTAL",
        matched_conversations=len(results),
        output_format=output_format,
        color_provider=dimension == "provider",
    )


# ---------------------------------------------------------------------------
# Semantic action/tool grouped stats (from query_grouped_stats_semantic.py)
# ---------------------------------------------------------------------------


def _action_kind_name(action: ActionEvent) -> str:
    return action.kind.value


def _semantic_grouped_payload(
    results: list[Conversation],
    *,
    selection: ConversationQuerySpec | None,
    key_for_action: Callable[[ActionEvent], str],
) -> GroupedStatsPayload:
    from polylogue.cli.query_semantic import (
        SemanticStatsSlice,
        action_matches_slice,
        filtered_action_events,
    )
    from polylogue.lib.semantic.facts import build_conversation_semantic_facts

    semantic_slice = SemanticStatsSlice.from_selection(selection)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
    matched_facts = 0
    matched_messages = 0

    for conv in results:
        facts = build_conversation_semantic_facts(conv)
        filtered_actions = filtered_action_events(facts, semantic_slice)
        fact_counts = Counter(key_for_action(action) for action in filtered_actions)
        if not fact_counts:
            groups["none"]["convs"] += 1
            continue

        matched_facts += sum(fact_counts.values())
        matched_messages += sum(
            1
            for message in facts.message_facts
            if any(action_matches_slice(action, semantic_slice) for action in message.action_events)
        )

        message_groups: dict[str, set[str]] = defaultdict(set)
        for message in facts.message_facts:
            for key in {
                key_for_action(action)
                for action in message.action_events
                if action_matches_slice(action, semantic_slice)
            }:
                message_groups[key].add(message.message_id)

        for key, fact_count in fact_counts.items():
            groups[key]["convs"] += 1
            groups[key]["facts"] += fact_count
            groups[key]["msgs"] += len(message_groups[key])

    return GroupedStatsPayload(
        rows=[
            {
                "group": key,
                "conversations": stats["convs"],
                "facts": stats["facts"],
                "messages": stats["msgs"],
            }
            for key, stats in sorted(groups.items())
        ],
        summary={
            "group": "MATCHED",
            "conversations": len(results),
            "facts": matched_facts,
            "messages": matched_messages,
        },
    )


def output_semantic_grouped_stats(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> bool:
    from polylogue.cli.query_semantic import normalized_tool_name

    if dimension == "action":
        payload = _semantic_grouped_payload(results, selection=selection, key_for_action=_action_kind_name)
        note = "Note: conversations may appear in multiple action groups."
    elif dimension == "tool":
        payload = _semantic_grouped_payload(results, selection=selection, key_for_action=normalized_tool_name)
        note = "Note: conversations may appear in multiple tool groups."
    else:
        return False

    _emit_grouped_stats_table(
        env,
        dimension=dimension,
        rows=payload.rows,
        summary=payload.summary,
        columns=SEMANTIC_COLUMNS,
        total_label="MATCHED",
        matched_conversations=len(results),
        output_format=output_format,
        multi_membership=True,
        note=note,
    )
    return True


# ---------------------------------------------------------------------------
# Grouped stats dispatcher (from query_grouped_stats.py)
# ---------------------------------------------------------------------------


def output_stats_by_conversations(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> None:
    if not results:
        emit_no_results(env, selection=selection, output_format=output_format)

    if output_semantic_grouped_stats(
        env,
        results,
        dimension,
        selection=selection,
        output_format=output_format,
    ):
        return

    output_stats_by_grouped_conversations(
        env,
        results,
        dimension,
        output_format=output_format,
    )


# ---------------------------------------------------------------------------
# Profile-backed grouped stats (from query_profile_stats.py)
# ---------------------------------------------------------------------------


async def output_stats_by_profile_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    repo: ConversationArchiveStatsStore,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    await output_stats_by_profile_ids(
        env,
        [str(summary.id) for summary in summaries],
        repo,
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_profile_query(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationArchiveStatsStore,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    await output_stats_by_profile_ids(
        env,
        conversation_ids,
        repo,
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_profile_ids(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationArchiveStatsStore,
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    if dimension not in {"repo", "work-kind"}:
        raise ValueError(f"Unsupported profile stats dimension: {dimension}")
    if not conversation_ids:
        emit_no_results(env, selection=selection, output_format=output_format)

    from polylogue.lib.session.session_profile import build_session_profile

    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"conversations": 0, "work_events": 0, "messages": 0})
    matched_conversations = 0
    matched_work_events = 0
    matched_messages = 0

    for offset in range(0, len(conversation_ids), batch_size):
        batch_ids = conversation_ids[offset : offset + batch_size]
        profiles_by_id = await repo.get_session_profiles_batch(batch_ids)
        missing_ids = [conversation_id for conversation_id in batch_ids if conversation_id not in profiles_by_id]
        if missing_ids:
            for conversation in await repo.get_many(missing_ids):
                profiles_by_id[str(conversation.id)] = build_session_profile(conversation)

        for conversation_id in batch_ids:
            profile = profiles_by_id.get(conversation_id)
            if profile is None:
                groups["none"]["conversations"] += 1
                continue

            if dimension == "repo":
                keys = profile.repo_names or ("none",)
            else:
                primary_kind = next(
                    (tag.split(":", 1)[1] for tag in profile.auto_tags if tag.startswith("kind:")),
                    None,
                )
                keys = (primary_kind or "none",)

            matched_conversations += 1
            matched_work_events += len(profile.work_events)
            matched_messages += profile.message_count

            for key in keys:
                groups[key]["conversations"] += 1
                groups[key]["work_events"] += len(profile.work_events)
                groups[key]["messages"] += profile.message_count

    rows = [
        {
            "group": key,
            "conversations": stats["conversations"],
            "work_events": stats["work_events"],
            "messages": stats["messages"],
        }
        for key, stats in sorted(groups.items(), key=lambda item: item[0])
    ]
    summary = {
        "group": "MATCHED",
        "conversations": matched_conversations,
        "work_events": matched_work_events,
        "messages": matched_messages,
    }
    multi_membership = dimension == "repo"
    _emit_grouped_stats_table(
        env,
        dimension=dimension,
        rows=rows,
        summary=summary,
        columns=PROFILE_COLUMNS,
        total_label="MATCHED",
        matched_conversations=matched_conversations,
        output_format=output_format,
        multi_membership=multi_membership,
        note="Note: conversations may appear in multiple repo groups." if multi_membership else None,
    )


__all__ = [
    "emit_structured_stats",
    "output_semantic_grouped_stats",
    "output_stats_by_conversations",
    "output_stats_by_grouped_conversations",
    "output_stats_by_profile_ids",
    "output_stats_by_profile_query",
    "output_stats_by_profile_summaries",
    "output_stats_by_summaries",
    "output_stats_sql",
]
