"""Semantic action/tool grouped query stats output helpers."""

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
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_spec import ConversationQuerySpec


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


__all__ = ["output_semantic_grouped_stats"]
