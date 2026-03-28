"""Grouped stats output helpers for summaries and hydrated conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.query_semantic_slice import (
    SemanticStatsSlice,
    action_matches_slice,
    filtered_action_events,
    normalized_tool_name,
)
from polylogue.cli.query_stats_structured import emit_structured_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_spec import ConversationQuerySpec


def output_stats_by_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    msg_counts: dict[str, int],
    dimension: str,
    *,
    output_format: str = "text",
) -> None:
    from collections import defaultdict

    from rich.table import Table

    from polylogue.ui.theme import provider_color

    if not summaries:
        env.ui.console.print("No conversations matched.")
        return

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

    sorted_keys = sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())
    rows: list[dict[str, object]] = []

    for key in sorted_keys:
        group_summaries = groups[key]
        n_convs = len(group_summaries)
        n_msgs = sum(msg_counts.get(str(summary.id), 0) for summary in group_summaries)
        rows.append({
            "group": key,
            "conversations": n_convs,
            "messages": n_msgs,
        })

    summary = {
        "group": "TOTAL",
        "conversations": len(summaries),
        "messages": sum(msg_counts.get(str(summary.id), 0) for summary in summaries),
    }
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
    ):
        return

    env.ui.console.print(f"\nMatched: {len(summaries)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")

    for row in rows:
        label = f"[{provider_color(row['group']).hex}]{row['group']}[/]" if dimension == "provider" else str(row["group"])
        table.add_row(label, f"{row['conversations']:,}", f"{row['messages']:,}")

    table.add_section()
    table.add_row("[bold]TOTAL[/]", f"[bold]{summary['conversations']:,}[/]", f"[bold]{summary['messages']:,}[/]")

    env.ui.console.print(table)


def output_stats_by_conversations(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> None:
    from collections import Counter, defaultdict

    from rich.table import Table

    from polylogue.lib.semantic_facts import build_conversation_semantic_facts
    from polylogue.ui.theme import provider_color

    semantic_slice = SemanticStatsSlice.from_selection(selection)

    if not results:
        env.ui.console.print("No conversations matched.")
        return

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

            message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    action.kind.value
                    for action in message.action_events
                    if action_matches_slice(action, semantic_slice)
                }:
                    message_groups[key].add(message.message_id)

            for key, fact_count in action_counts.items():
                action_groups[key]["convs"] += 1
                action_groups[key]["facts"] += fact_count
                action_groups[key]["msgs"] += len(message_groups[key])

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
        if emit_structured_stats(
            output_format=output_format,
            dimension=dimension,
            rows=rows,
            summary=summary,
            multi_membership=True,
        ):
            return

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
        env.ui.console.print("Note: conversations may appear in multiple action groups.")
        return

    if dimension == "tool":
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

            message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    normalized_tool_name(action)
                    for action in message.action_events
                    if action_matches_slice(action, semantic_slice)
                }:
                    message_groups[key].add(message.message_id)

            for key, fact_count in tool_counts.items():
                tool_groups[key]["convs"] += 1
                tool_groups[key]["facts"] += fact_count
                tool_groups[key]["msgs"] += len(message_groups[key])

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
        if emit_structured_stats(
            output_format=output_format,
            dimension=dimension,
            rows=rows,
            summary=summary,
            multi_membership=True,
        ):
            return

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
        env.ui.console.print("Note: conversations may appear in multiple tool groups.")
        return

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

    sorted_keys = sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())

    matched_convs = len(results)
    matched_msgs = sum(len(conv.messages) for conv in results)
    matched_words = sum(sum(message.word_count for message in conv.messages) for conv in results)
    rows: list[dict[str, object]] = []

    for key in sorted_keys:
        convs = groups[key]
        n_convs = len(convs)
        n_msgs = sum(len(conv.messages) for conv in convs)
        n_words = sum(sum(message.word_count for message in conv.messages) for conv in convs)
        rows.append({
            "group": key,
            "conversations": n_convs,
            "messages": n_msgs,
            "words": n_words,
        })

    summary = {
        "group": "TOTAL",
        "conversations": matched_convs,
        "messages": matched_msgs,
        "words": matched_words,
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
        label = f"[{provider_color(row['group']).hex}]{row['group']}[/]" if dimension == "provider" else str(row["group"])
        table.add_row(label, f"{row['conversations']:,}", f"{row['messages']:,}", f"{row['words']:,}")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{summary['conversations']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
        f"[bold]{summary['words']:,}[/]",
    )

    env.ui.console.print(table)


__all__ = [
    "output_stats_by_conversations",
    "output_stats_by_summaries",
]
