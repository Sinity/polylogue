"""Summary/list/stats output helpers for query execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_helpers import summary_to_dict

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.action_facts import ActionFact
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_spec import ConversationQuerySpec
    from polylogue.lib.semantic_facts import ConversationSemanticFacts
    from polylogue.storage.repository import ConversationRepository


@dataclass(frozen=True, slots=True)
class SemanticStatsSlice:
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()

    @classmethod
    def from_selection(cls, selection: ConversationQuerySpec | None) -> SemanticStatsSlice:
        if selection is None:
            return cls()
        return cls(
            path_terms=selection.path_terms,
            action_terms=selection.action_terms,
            excluded_action_terms=selection.excluded_action_terms,
            tool_terms=selection.tool_terms,
            excluded_tool_terms=selection.excluded_tool_terms,
        )

    def has_filters(self) -> bool:
        return any(
            (
                self.path_terms,
                self.action_terms,
                self.excluded_action_terms,
                self.tool_terms,
                self.excluded_tool_terms,
            )
        )


def _normalized_tool_name(action: ActionFact) -> str:
    return (action.tool_name or "unknown").strip().lower()


def _path_matches_slice(action: ActionFact, path_terms: tuple[str, ...]) -> bool:
    if not path_terms:
        return True
    affected_paths = tuple(path.lower().replace("\\", "/") for path in action.affected_paths)
    if not affected_paths:
        return False
    return any(
        any(term.lower().replace("\\", "/") in path for path in affected_paths)
        for term in path_terms
    )


def _action_matches_slice(action: ActionFact, semantic_slice: SemanticStatsSlice) -> bool:
    if not _path_matches_slice(action, semantic_slice.path_terms):
        return False

    if "none" in semantic_slice.action_terms:
        return False
    required_action_terms = {term for term in semantic_slice.action_terms if term != "none"}
    if required_action_terms and action.kind.value not in required_action_terms:
        return False
    blocked_action_terms = {term for term in semantic_slice.excluded_action_terms if term != "none"}
    if action.kind.value in blocked_action_terms:
        return False

    tool_name = _normalized_tool_name(action)
    if "none" in semantic_slice.tool_terms:
        return False
    required_tool_terms = {term for term in semantic_slice.tool_terms if term != "none"}
    if required_tool_terms and tool_name not in required_tool_terms:
        return False
    blocked_tool_terms = {term for term in semantic_slice.excluded_tool_terms if term != "none"}
    return tool_name not in blocked_tool_terms


def _filtered_action_facts(
    facts: ConversationSemanticFacts,
    semantic_slice: SemanticStatsSlice,
) -> tuple[ActionFact, ...]:
    if not semantic_slice.has_filters():
        return facts.action_facts
    return tuple(action for action in facts.action_facts if _action_matches_slice(action, semantic_slice))


def _emit_structured_stats(
    *,
    output_format: str,
    dimension: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    multi_membership: bool = False,
) -> bool:
    """Emit machine-readable grouped stats when requested."""
    if output_format == "json":
        click.echo(json.dumps(
            {
                "dimension": dimension,
                "multi_membership": multi_membership,
                "rows": rows,
                "summary": summary,
            },
            indent=2,
        ))
        return True

    if output_format == "yaml":
        import yaml

        click.echo(yaml.dump(
            {
                "dimension": dimension,
                "multi_membership": multi_membership,
                "rows": rows,
                "summary": summary,
            },
            default_flow_style=False,
            allow_unicode=True,
        ))
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


def format_summary_list(
    summaries: list[ConversationSummary],
    output_format: str,
    fields: str | None,
    *,
    message_counts: dict[str, int] | None = None,
) -> str:
    """Format summary-list output for deterministic machine/plain surfaces."""
    message_counts = message_counts or {}

    selected: set[str] | None = None
    if fields:
        selected = {field.strip() for field in fields.split(",")}

    data = [summary_to_dict(summary, message_counts.get(str(summary.id), 0)) for summary in summaries]
    if selected:
        data = [{key: value for key, value in item.items() if key in selected} for item in data]

    if output_format == "json":
        return json.dumps(data, indent=2)

    if output_format == "yaml":
        import yaml

        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "date", "provider", "title", "messages", "tags", "summary"])
        for summary in summaries:
            date = summary.display_date.strftime("%Y-%m-%d") if summary.display_date else ""
            tags_str = ",".join(summary.tags) if summary.tags else ""
            writer.writerow([
                str(summary.id),
                date,
                summary.provider,
                summary.display_title or "",
                message_counts.get(str(summary.id), 0),
                tags_str,
                summary.summary or "",
            ])
        return buf.getvalue().rstrip("\r\n")

    lines = []
    for summary in summaries:
        date = summary.display_date.strftime("%Y-%m-%d") if summary.display_date else ""
        raw_title = summary.display_title or str(summary.id)[:20]
        title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
        count = message_counts.get(str(summary.id), 0)
        lines.append(f"{str(summary.id)[:24]:24s}  {date:10s}  {summary.provider:12s}  {title} ({count} msgs)")
    return "\n".join(lines)


async def output_stats_sql(
    env: AppEnv,
    filter_chain: ConversationFilter,
    repo: ConversationRepository,
) -> None:
    """Output statistics using SQL aggregation without full message loading."""
    from datetime import datetime, timezone

    has_filters = bool(filter_chain.describe())

    if has_filters:
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = [str(summary.id) for summary in summaries]
            conv_count = len(conv_ids)
        else:
            conv_count = await filter_chain.count()
            if conv_count == 0:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = None
    else:
        conv_ids = None
        conv_count = await filter_chain.count()
        if conv_count == 0:
            env.ui.console.print("No conversations in archive.")
            return

    stats = await repo.queries.aggregate_message_stats(conv_ids)

    date_range = ""
    if stats["min_sort_key"] and stats["max_sort_key"]:
        min_date = datetime.fromtimestamp(stats["min_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_date = datetime.fromtimestamp(stats["max_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    out = env.ui.console.print
    out(f"\nConversations: {conv_count:,}\n")
    if stats["user"] or stats["assistant"]:
        out(
            f"Messages: {stats['total']:,} total "
            f"({stats['user']:,} user, {stats['assistant']:,} assistant)"
        )
    else:
        out(f"Messages: {stats['total']:,}")

    if stats["words_approx"]:
        out(f"Words: ~{stats['words_approx']:,}")

    if stats.get("providers"):
        provider_parts = [f"{name} ({count:,})" for name, count in stats["providers"].items()]
        out(f"Providers: {', '.join(provider_parts)}")

    out(f"Attachments: {stats['attachments']:,}")
    if date_range:
        out(f"Date range: {date_range}")


def output_stats_by_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    msg_counts: dict[str, int],
    dimension: str,
    *,
    output_format: str = "text",
) -> None:
    """Fast stats-by using lightweight summaries and precomputed message counts."""
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
    if _emit_structured_stats(
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
    """Output grouped statistics from fully hydrated conversations."""
    from collections import defaultdict

    from rich.table import Table

    from polylogue.lib.semantic_facts import build_conversation_semantic_facts
    from polylogue.ui.theme import provider_color

    semantic_slice = SemanticStatsSlice.from_selection(selection)

    if not results:
        env.ui.console.print("No conversations matched.")
        return

    if dimension == "action":
        from collections import Counter

        action_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
        matched_action_facts = 0
        matched_action_msgs = 0

        for conv in results:
            facts = build_conversation_semantic_facts(conv)
            filtered_actions = _filtered_action_facts(facts, semantic_slice)
            action_counts = Counter(action.kind.value for action in filtered_actions)
            if not action_counts:
                action_groups["none"]["convs"] += 1
                continue

            matched_action_facts += sum(action_counts.values())
            matched_action_msgs += sum(
                1
                for message in facts.message_facts
                if any(_action_matches_slice(action, semantic_slice) for action in message.action_facts)
            )

            message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    action.kind.value
                    for action in message.action_facts
                    if _action_matches_slice(action, semantic_slice)
                }:
                    message_groups[key].add(message.message_id)

            for key, fact_count in action_counts.items():
                action_groups[key]["convs"] += 1
                action_groups[key]["facts"] += fact_count
                action_groups[key]["msgs"] += len(message_groups[key])

        sorted_keys = sorted(action_groups.keys())

        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Group", style="bold", min_width=12)
        table.add_column("Convs", justify="right")
        table.add_column("Facts", justify="right")
        table.add_column("Msgs", justify="right")

        rows: list[dict[str, object]] = []
        for key in sorted_keys:
            stats = action_groups[key]
            rows.append({
                "group": key,
                "conversations": stats["convs"],
                "facts": stats["facts"],
                "messages": stats["msgs"],
            })

        summary = {
            "group": "MATCHED",
            "conversations": len(results),
            "facts": matched_action_facts,
            "messages": matched_action_msgs,
        }
        if _emit_structured_stats(
            output_format=output_format,
            dimension=dimension,
            rows=rows,
            summary=summary,
            multi_membership=True,
        ):
            return

        env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")

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
        from collections import Counter

        tool_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
        matched_tool_facts = 0
        matched_tool_msgs = 0

        for conv in results:
            facts = build_conversation_semantic_facts(conv)
            filtered_actions = _filtered_action_facts(facts, semantic_slice)
            tool_counts = Counter(_normalized_tool_name(action) for action in filtered_actions)
            if not tool_counts:
                tool_groups["none"]["convs"] += 1
                continue

            matched_tool_facts += sum(tool_counts.values())
            matched_tool_msgs += sum(
                1
                for message in facts.message_facts
                if any(_action_matches_slice(action, semantic_slice) for action in message.action_facts)
            )

            message_groups: dict[str, set[str]] = defaultdict(set)
            for message in facts.message_facts:
                for key in {
                    _normalized_tool_name(action)
                    for action in message.action_facts
                    if _action_matches_slice(action, semantic_slice)
                }:
                    message_groups[key].add(message.message_id)

            for key, fact_count in tool_counts.items():
                tool_groups[key]["convs"] += 1
                tool_groups[key]["facts"] += fact_count
                tool_groups[key]["msgs"] += len(message_groups[key])

        sorted_keys = sorted(tool_groups.keys())

        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Group", style="bold", min_width=12)
        table.add_column("Convs", justify="right")
        table.add_column("Facts", justify="right")
        table.add_column("Msgs", justify="right")

        rows: list[dict[str, object]] = []
        for key in sorted_keys:
            stats = tool_groups[key]
            rows.append({
                "group": key,
                "conversations": stats["convs"],
                "facts": stats["facts"],
                "messages": stats["msgs"],
            })

        summary = {
            "group": "MATCHED",
            "conversations": len(results),
            "facts": matched_tool_facts,
            "messages": matched_tool_msgs,
        }
        if _emit_structured_stats(
            output_format=output_format,
            dimension=dimension,
            rows=rows,
            summary=summary,
            multi_membership=True,
        ):
            return

        env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")

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
    if _emit_structured_stats(
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


async def output_summary_list(
    env: AppEnv,
    summaries: list[ConversationSummary],
    params: dict[str, object],
    repo: ConversationRepository | None = None,
) -> None:
    """Output a list of conversation summaries with optional rich table rendering."""
    output_format = str(params.get("output_format", "text"))
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [str(summary.id) for summary in summaries]
        msg_counts = await repo.queries.get_message_counts_batch(ids)

    fields = params.get("fields")
    fields_value = str(fields) if isinstance(fields, str) else None
    if output_format in {"json", "yaml", "csv"} or env.ui.plain:
        click.echo(
            format_summary_list(
                summaries,
                "text" if env.ui.plain and output_format not in {"json", "yaml", "csv"} else output_format,
                fields_value,
                message_counts=msg_counts,
            )
        )
        return

    from rich.table import Table
    from rich.text import Text

    from polylogue.ui.theme import provider_color

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False, show_edge=False)
    table.add_column("ID", style="dim", max_width=24, no_wrap=True)
    table.add_column("Date", style="dim")
    table.add_column("Provider")
    table.add_column("Title", ratio=1)
    table.add_column("Msgs", justify="right")

    for summary in summaries:
        date = summary.display_date.strftime("%Y-%m-%d") if summary.display_date else ""
        raw_title = summary.display_title or str(summary.id)[:20]
        title = (raw_title[:60] + "...") if len(raw_title) > 63 else raw_title
        count = msg_counts.get(str(summary.id), 0)
        provider_text = Text(summary.provider, style=provider_color(summary.provider).hex)
        table.add_row(str(summary.id)[:24], date, provider_text, title, str(count))

    env.ui.console.print(table)


def conversations_to_csv(results: list[Conversation]) -> str:
    """Convert hydrated conversations to CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "provider", "title", "messages", "words", "tags", "summary"])

    for conv in results:
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else ""
        tags_str = ",".join(conv.tags) if conv.tags else ""
        writer.writerow([
            str(conv.id),
            date,
            conv.provider,
            conv.display_title or "",
            len(conv.messages),
            sum(message.word_count for message in conv.messages),
            tags_str,
            conv.summary or "",
        ])

    return output.getvalue()


__all__ = [
    "conversations_to_csv",
    "format_summary_list",
    "output_stats_by_conversations",
    "output_stats_by_summaries",
    "output_stats_sql",
    "output_summary_list",
]
