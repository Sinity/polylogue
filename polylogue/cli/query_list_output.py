"""Summary-list and CSV output helpers for query surfaces."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_helpers import summary_to_dict

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.storage.repository import ConversationRepository


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
    "output_summary_list",
]
