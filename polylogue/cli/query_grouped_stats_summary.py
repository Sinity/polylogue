"""Summary/date/provider grouped query stats output helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.query_stats_structured import emit_structured_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary


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
        rows.append({
            "group": key,
            "conversations": len(group_summaries),
            "messages": sum(msg_counts.get(str(summary.id), 0) for summary in group_summaries),
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

    sorted_keys = sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())

    rows: list[dict[str, object]] = []
    for key in sorted_keys:
        convs = groups[key]
        rows.append({
            "group": key,
            "conversations": len(convs),
            "messages": sum(len(conv.messages) for conv in convs),
            "words": sum(sum(message.word_count for message in conv.messages) for conv in convs),
        })

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
    "output_stats_by_grouped_conversations",
    "output_stats_by_summaries",
]
