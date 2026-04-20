"""Grouped stats output: structured serialization, SQL stats, summary/date/provider grouping,
semantic action/tool grouping, and profile-backed grouping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_feedback import emit_no_results

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_spec import ConversationQuerySpec
    from polylogue.protocols import ConversationArchiveStatsStore


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
    from collections import defaultdict

    from rich.table import Table

    from polylogue.ui.theme import provider_color

    if not summaries:
        emit_no_results(env, selection=selection, output_format=output_format)

    groups: dict[str, list[ConversationSummary]] = defaultdict(list)
    for summary in summaries:
        if dimension == "provider":
            key = str(summary.provider) if summary.provider else "unknown"
        elif dimension == "month":
            dt = summary.updated_at or summary.created_at
            key = dt.strftime("%Y-%m") if dt else "unknown"
        elif dimension == "year":
            dt = summary.updated_at or summary.created_at
            key = dt.strftime("%Y") if dt else "unknown"
        elif dimension == "day":
            dt = summary.updated_at or summary.created_at
            key = dt.strftime("%Y-%m-%d") if dt else "unknown"
        else:
            key = "all"
        groups[key].append(summary)

    sorted_keys = (
        sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())
    )
    rows: list[dict[str, object]] = []

    for key in sorted_keys:
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
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary_row,
    ):
        return

    env.ui.console.print(f"\nMatched: {len(summaries)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")

    for row in rows:
        group_label = str(row["group"])
        label = f"[{provider_color(group_label).hex}]{group_label}[/]" if dimension == "provider" else group_label
        table.add_row(label, f"{row['conversations']:,}", f"{row['messages']:,}")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]", f"[bold]{summary_row['conversations']:,}[/]", f"[bold]{summary_row['messages']:,}[/]"
    )

    env.ui.console.print(table)


def output_stats_by_grouped_conversations(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    output_format: str = "text",
) -> None:
    from collections import defaultdict

    from rich.table import Table

    from polylogue.ui.theme import provider_color

    groups: dict[str, list[Conversation]] = defaultdict(list)
    for conv in results:
        if dimension == "provider":
            key = conv.provider or "unknown"
            groups[key].append(conv)
        elif dimension == "month":
            dt = conv.display_date
            key = dt.strftime("%Y-%m") if dt else "unknown"
            groups[key].append(conv)
        elif dimension == "year":
            dt = conv.display_date
            key = dt.strftime("%Y") if dt else "unknown"
            groups[key].append(conv)
        elif dimension == "day":
            dt = conv.display_date
            key = dt.strftime("%Y-%m-%d") if dt else "unknown"
            groups[key].append(conv)
        else:
            groups["all"].append(conv)

    sorted_keys = (
        sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())
    )

    rows: list[dict[str, object]] = []
    for key in sorted_keys:
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
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
    ):
        return

    env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")
    table.add_column("Words", justify="right")

    for row in rows:
        group_label = str(row["group"])
        label = f"[{provider_color(group_label).hex}]{group_label}[/]" if dimension == "provider" else group_label
        table.add_row(label, f"{row['conversations']:,}", f"{row['messages']:,}", f"{row['words']:,}")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{summary['conversations']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
        f"[bold]{summary['words']:,}[/]",
    )

    env.ui.console.print(table)


# ---------------------------------------------------------------------------
# Semantic action/tool grouped stats (from query_grouped_stats_semantic.py)
# ---------------------------------------------------------------------------


def output_semantic_grouped_stats(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> bool:
    from collections import Counter, defaultdict

    from rich.table import Table

    from polylogue.cli.query_semantic import (
        SemanticStatsSlice,
        action_matches_slice,
        filtered_action_events,
        normalized_tool_name,
    )
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    semantic_slice = SemanticStatsSlice.from_selection(selection)

    if dimension == "action":
        action_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
        matched_action_events = 0
        matched_action_msgs = 0

        for conv in results:
            facts = build_conversation_semantic_facts(conv)
            filtered_actions = filtered_action_events(facts, semantic_slice)
            action_counts = Counter(action.kind.value for action in filtered_actions)
            if not action_counts:
                action_groups["none"]["convs"] += 1
                continue

            matched_action_events += sum(action_counts.values())
            matched_action_msgs += sum(
                1
                for message in facts.message_facts
                if any(action_matches_slice(action, semantic_slice) for action in message.action_events)
            )

            action_message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    action.kind.value
                    for action in message.action_events
                    if action_matches_slice(action, semantic_slice)
                }:
                    action_message_groups[key].add(message.message_id)

            for key, fact_count in action_counts.items():
                action_groups[key]["convs"] += 1
                action_groups[key]["facts"] += fact_count
                action_groups[key]["msgs"] += len(action_message_groups[key])

        rows = [
            {
                "group": key,
                "conversations": stats["convs"],
                "facts": stats["facts"],
                "messages": stats["msgs"],
            }
            for key, stats in sorted(action_groups.items())
        ]
        summary = {
            "group": "MATCHED",
            "conversations": len(results),
            "facts": matched_action_events,
            "messages": matched_action_msgs,
        }
        note = "Note: conversations may appear in multiple action groups."
    elif dimension == "tool":
        tool_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
        matched_tool_facts = 0
        matched_tool_msgs = 0

        for conv in results:
            facts = build_conversation_semantic_facts(conv)
            filtered_actions = filtered_action_events(facts, semantic_slice)
            tool_counts = Counter(normalized_tool_name(action) for action in filtered_actions)
            if not tool_counts:
                tool_groups["none"]["convs"] += 1
                continue

            matched_tool_facts += sum(tool_counts.values())
            matched_tool_msgs += sum(
                1
                for message in facts.message_facts
                if any(action_matches_slice(action, semantic_slice) for action in message.action_events)
            )

            tool_message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    normalized_tool_name(action)
                    for action in message.action_events
                    if action_matches_slice(action, semantic_slice)
                }:
                    tool_message_groups[key].add(message.message_id)

            for key, fact_count in tool_counts.items():
                tool_groups[key]["convs"] += 1
                tool_groups[key]["facts"] += fact_count
                tool_groups[key]["msgs"] += len(tool_message_groups[key])

        rows = [
            {
                "group": key,
                "conversations": stats["convs"],
                "facts": stats["facts"],
                "messages": stats["msgs"],
            }
            for key, stats in sorted(tool_groups.items())
        ]
        summary = {
            "group": "MATCHED",
            "conversations": len(results),
            "facts": matched_tool_facts,
            "messages": matched_tool_msgs,
        }
        note = "Note: conversations may appear in multiple tool groups."
    else:
        return False

    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=True,
    ):
        return True

    env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")
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
    env.ui.console.print(note)
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
    from collections import defaultdict

    from rich.table import Table

    if dimension not in {"repo", "work-kind"}:
        raise ValueError(f"Unsupported profile stats dimension: {dimension}")
    if not conversation_ids:
        emit_no_results(env, selection=selection, output_format=output_format)

    from polylogue.lib.session_profile import build_session_profile

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
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=multi_membership,
    ):
        return

    env.ui.console.print(f"\nMatched: {matched_conversations} conversations (by {dimension})\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Msgs", justify="right")
    for row in rows:
        table.add_row(
            str(row["group"]),
            f"{row['conversations']:,}",
            f"{row['work_events']:,}",
            f"{row['messages']:,}",
        )
    table.add_section()
    table.add_row(
        "[bold]MATCHED[/]",
        f"[bold]{summary['conversations']:,}[/]",
        f"[bold]{summary['work_events']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
    )
    env.ui.console.print(table)
    if multi_membership:
        env.ui.console.print("Note: conversations may appear in multiple repo groups.")


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
