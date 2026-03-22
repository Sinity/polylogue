"""Output and streaming helpers for CLI query execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.cli.query_helpers import no_results, summary_to_dict
from polylogue.logging import get_logger
from polylogue.rendering.formatting import format_conversation

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary, Message
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
        selected = {f.strip() for f in fields.split(",")}

    if output_format == "json":
        data = [summary_to_dict(s, message_counts.get(str(s.id), 0)) for s in summaries]
        if selected:
            data = [{k: v for k, v in d.items() if k in selected} for d in data]
        return json.dumps(data, indent=2)

    if output_format == "yaml":
        import yaml

        data = [summary_to_dict(s, message_counts.get(str(s.id), 0)) for s in summaries]
        if selected:
            data = [{k: v for k, v in d.items() if k in selected} for d in data]
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "date", "provider", "title", "messages", "tags", "summary"])
        for s in summaries:
            date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
            tags_str = ",".join(s.tags) if s.tags else ""
            writer.writerow([
                str(s.id),
                date,
                s.provider,
                s.display_title or "",
                message_counts.get(str(s.id), 0),
                tags_str,
                s.summary or "",
            ])
        return buf.getvalue().rstrip("\r\n")

    lines = []
    for s in summaries:
        date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
        raw_title = s.display_title or str(s.id)[:20]
        title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
        count = message_counts.get(str(s.id), 0)
        lines.append(f"{str(s.id)[:24]:24s}  {date:10s}  {s.provider:12s}  {title} ({count} msgs)")
    return "\n".join(lines)


def render_stream_message(message: Message, output_format: str) -> str:
    """Render a single streamed message chunk."""
    if output_format == "plaintext":
        if not message.text:
            return ""
        role_label = (message.role or "unknown").upper().replace("[", "").replace("]", "")
        return f"[{role_label}]\n{message.text}\n\n"

    if output_format == "markdown":
        if not message.text:
            return ""
        role_label = (message.role or "unknown").capitalize()
        return f"## {role_label}\n\n{message.text}\n\n"

    if output_format == "json-lines":
        record = {
            "type": "message",
            "id": message.id,
            "role": message.role,
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "text": message.text,
            "word_count": message.word_count,
        }
        return json.dumps(record, ensure_ascii=False) + "\n"

    return ""


def render_stream_header(
    *,
    conversation_id: str,
    title: str | None,
    provider: str | None,
    display_date: object | None,
    output_format: str,
    dialogue_only: bool,
    message_limit: int | None,
    stats: dict[str, Any] | None,
) -> str:
    """Render any stream prelude/header for the selected output format."""
    if hasattr(display_date, "strftime"):
        display_date_text = display_date.strftime("%Y-%m-%d %H:%M")
        display_date_value = display_date.isoformat() if hasattr(display_date, "isoformat") else str(display_date)
    elif display_date:
        display_date_text = str(display_date)
        display_date_value = str(display_date)
    else:
        display_date_text = None
        display_date_value = None

    if output_format == "markdown":
        lines = [f"# {title or conversation_id[:24]}", ""]
        if display_date_text is not None:
            lines.append(f"**Date**: {display_date_text}")
        if provider:
            lines.append(f"**Provider**: {provider}")
        if display_date_text is not None or provider:
            lines.append("")
        if dialogue_only and stats:
            line = f"_Showing {stats['dialogue_messages']} dialogue messages"
            if message_limit:
                line += f" (limit: {message_limit})"
            line += f" of {stats['total_messages']} total_"
            lines.extend([line, ""])
        return "\n".join(lines)

    if output_format == "json-lines":
        header = {
            "type": "header",
            "conversation_id": conversation_id,
            "title": title,
            "provider": provider,
            "date": display_date_value,
            "dialogue_only": dialogue_only,
            "message_limit": message_limit,
            "stats": stats,
        }
        return json.dumps(header) + "\n"

    return ""


def render_stream_footer(*, output_format: str, emitted_messages: int) -> str:
    """Render any stream closing/footer fragment."""
    if output_format == "markdown":
        return f"\n---\n_Streamed {emitted_messages} messages_\n"
    if output_format == "json-lines":
        return json.dumps({"type": "footer", "message_count": emitted_messages}) + "\n"
    return ""


def render_stream_transcript(
    *,
    conversation_id: str,
    title: str | None,
    provider: str | None,
    display_date: object | None,
    messages: list[Message],
    output_format: str,
    dialogue_only: bool = False,
    message_limit: int | None = None,
    stats: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Render the full stream transcript deterministically for proof/tests."""
    parts = [
        render_stream_header(
            conversation_id=conversation_id,
            title=title,
            provider=provider,
            display_date=display_date,
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_limit=message_limit,
            stats=stats,
        )
    ]
    emitted = 0
    for message in messages[: message_limit if message_limit is not None else None]:
        chunk = render_stream_message(message, output_format)
        if chunk:
            parts.append(chunk)
            emitted += 1
    parts.append(render_stream_footer(output_format=output_format, emitted_messages=emitted))
    return "".join(parts), emitted


async def output_stats_sql(
    env: AppEnv,
    filter_chain: ConversationFilter,
    repo: ConversationRepository,
) -> None:
    """Output statistics using SQL aggregation (no full message loading)."""
    from datetime import datetime, timezone

    has_filters = bool(filter_chain.describe())

    if has_filters:
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = [str(s.id) for s in summaries]
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
        provider_parts = [f"{name} ({cnt:,})" for name, cnt in stats["providers"].items()]
        out(f"Providers: {', '.join(provider_parts)}")

    out(f"Attachments: {stats['attachments']:,}")
    if date_range:
        out(f"Date range: {date_range}")


def output_stats_by_summaries(
    env: AppEnv,
    summaries: list,
    msg_counts: dict[str, int],
    dimension: str,
) -> None:
    """Fast stats-by using lightweight summaries and precomputed message counts."""
    from collections import defaultdict

    from rich.table import Table

    from polylogue.ui.theme import provider_color

    if not summaries:
        env.ui.console.print("No conversations matched.")
        return

    groups: dict[str, list] = defaultdict(list)
    for s in summaries:
        if dimension == "provider":
            key = str(s.provider) if s.provider else "unknown"
        elif dimension == "month":
            dt = s.updated_at or s.created_at
            key = dt.strftime("%Y-%m") if dt else "unknown"
        elif dimension == "year":
            dt = s.updated_at or s.created_at
            key = dt.strftime("%Y") if dt else "unknown"
        elif dimension == "day":
            dt = s.updated_at or s.created_at
            key = dt.strftime("%Y-%m-%d") if dt else "unknown"
        else:
            key = "all"
        groups[key].append(s)

    sorted_keys = sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())

    env.ui.console.print(f"\nMatched: {len(summaries)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")

    total_convs = 0
    total_msgs = 0

    for key in sorted_keys:
        group_summaries = groups[key]
        n_convs = len(group_summaries)
        n_msgs = sum(msg_counts.get(str(s.id), 0) for s in group_summaries)
        label = f"[{provider_color(key).hex}]{key}[/]" if dimension == "provider" else key
        table.add_row(label, f"{n_convs:,}", f"{n_msgs:,}")
        total_convs += n_convs
        total_msgs += n_msgs

    table.add_section()
    table.add_row("[bold]TOTAL[/]", f"[bold]{total_convs:,}[/]", f"[bold]{total_msgs:,}[/]")

    env.ui.console.print(table)


def _output_stats_by(env: AppEnv, results: list[Conversation], dimension: str) -> None:
    """Output statistics grouped by a dimension."""
    from collections import defaultdict

    from rich.table import Table

    from polylogue.ui.theme import provider_color

    if not results:
        env.ui.console.print("No conversations matched.")
        return

    groups: dict[str, list[Conversation]] = defaultdict(list)
    for conv in results:
        if dimension == "provider":
            key = conv.provider or "unknown"
        elif dimension == "month":
            dt = conv.display_date
            key = dt.strftime("%Y-%m") if dt else "unknown"
        elif dimension == "year":
            dt = conv.display_date
            key = dt.strftime("%Y") if dt else "unknown"
        elif dimension == "day":
            dt = conv.display_date
            key = dt.strftime("%Y-%m-%d") if dt else "unknown"
        else:
            key = "all"
        groups[key].append(conv)

    sorted_keys = sorted(groups.keys(), reverse=True) if dimension in {"month", "year", "day"} else sorted(groups.keys())

    env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")
    table.add_column("Words", justify="right")

    total_convs = 0
    total_msgs = 0
    total_words = 0

    for key in sorted_keys:
        convs = groups[key]
        n_convs = len(convs)
        n_msgs = sum(len(c.messages) for c in convs)
        n_words = sum(sum(m.word_count for m in c.messages) for c in convs)
        label = f"[{provider_color(key).hex}]{key}[/]" if dimension == "provider" else key
        table.add_row(label, f"{n_convs:,}", f"{n_msgs:,}", f"{n_words:,}")
        total_convs += n_convs
        total_msgs += n_msgs
        total_words += n_words

    table.add_section()
    table.add_row("[bold]TOTAL[/]", f"[bold]{total_convs:,}[/]", f"[bold]{total_msgs:,}[/]", f"[bold]{total_words:,}[/]")

    env.ui.console.print(table)


def output_results(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Output query results."""
    if not results:
        no_results(env, params)

    output_format = params.get("output_format", "markdown")
    output_dest = params.get("output", "stdout")
    list_mode = params.get("list_mode", False)
    fields = params.get("fields")
    destinations = [d.strip() for d in output_dest.split(",")] if output_dest else ["stdout"]

    if len(results) == 1 and not list_mode:
        conv = results[0]
        if output_format == "markdown" and destinations == ["stdout"] and not env.ui.plain:
            _render_conversation_rich(env, conv)
            return
        content = format_conversation(conv, output_format, fields)
        _send_output(env, content, destinations, output_format, conv)
        return

    content = _format_list(results, output_format, fields)
    _send_output(env, content, destinations, output_format, None)


def _format_list(
    results: list[Conversation],
    output_format: str,
    fields: str | None,
) -> str:
    """Format a list of conversations for output."""
    from polylogue.rendering.formatting import _conv_to_dict

    if output_format == "json":
        return json.dumps([_conv_to_dict(c, fields) for c in results], indent=2)
    if output_format == "yaml":
        import yaml  # type: ignore[import-untyped]

        return str(yaml.dump([_conv_to_dict(c, fields) for c in results], default_flow_style=False, allow_unicode=True))
    if output_format == "csv":
        return _conv_to_csv(results)

    lines = []
    for conv in results:
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
        raw_title = conv.display_title or conv.id[:20]
        title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
        msg_count = len(conv.messages)
        lines.append(f"{conv.id[:24]:24s}  {date:10s}  {conv.provider:12s}  {title} ({msg_count} msgs)")
    return "\n".join(lines)


async def _output_summary_list(
    env: AppEnv,
    summaries: list[ConversationSummary],
    params: dict[str, Any],
    repo: ConversationRepository | None = None,
) -> None:
    """Output a list of conversation summaries (memory-efficient)."""
    output_format = params.get("output_format", "text")
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [str(s.id) for s in summaries]
        msg_counts = await repo.queries.get_message_counts_batch(ids)

    fields = params.get("fields")
    if output_format in {"json", "yaml", "csv"} or env.ui.plain:
        click.echo(
            format_summary_list(
                summaries,
                "text" if env.ui.plain and output_format not in {"json", "yaml", "csv"} else output_format,
                fields,
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

    for s in summaries:
        date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
        raw_title = s.display_title or str(s.id)[:20]
        title = (raw_title[:60] + "...") if len(raw_title) > 63 else raw_title
        count = msg_counts.get(str(s.id), 0)
        pc = provider_color(s.provider)
        prov_text = Text(s.provider, style=pc.hex)
        table.add_row(str(s.id)[:24], date, prov_text, title, str(count))

    env.ui.console.print(table)


def _conv_to_csv(results: list[Conversation]) -> str:
    """Convert conversations to CSV string."""
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
            sum(m.word_count for m in conv.messages),
            tags_str,
            conv.summary or "",
        ])

    return output.getvalue()


def _render_conversation_rich(env: AppEnv, conv: Conversation) -> None:
    """Render a conversation with Rich role colors and thinking block styling."""
    from rich import box
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    from polylogue.ui.theme import THINKING_STYLE, provider_color, role_color

    console = env.ui.console
    title = conv.display_title or conv.id
    pc = provider_color(conv.provider)
    header = Text()
    header.append(title, style="bold")
    if conv.display_date:
        header.append(f"  {conv.display_date.strftime('%Y-%m-%d %H:%M')}", style="dim")
    header.append(f"  [{pc.hex}]{conv.provider}[/{pc.hex}]")
    console.print(header)
    console.print()

    for msg in conv.messages:
        if not msg.text:
            continue
        role = msg.role or "unknown"
        rc = role_color(role)
        is_thinking = msg.is_thinking
        if is_thinking:
            content = Text(msg.text[:500], style=THINKING_STYLE["rich_style"])
            if len(msg.text) > 500:
                content.append(f"\n... ({len(msg.text):,} chars)", style="dim")
            panel = Panel(
                content,
                title=f"{THINKING_STYLE['icon']} Thinking",
                title_align="left",
                border_style=THINKING_STYLE["border_color"],
                box=box.SIMPLE,
                padding=(0, 1),
            )
            console.print(panel)
            continue
        try:
            md = Markdown(msg.text)
            panel = Panel(
                md,
                title=f"[{rc.label}]{role.capitalize()}[/{rc.label}]",
                title_align="left",
                border_style=rc.hex,
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
        except Exception:
            console.print(f"[{rc.label}]{role.capitalize()}:[/{rc.label}] {msg.text[:200]}")
        console.print()


async def stream_conversation(
    env: AppEnv,
    repo: ConversationRepository,
    conversation_id: str,
    *,
    output_format: str = "plaintext",
    dialogue_only: bool = False,
    message_limit: int | None = None,
) -> int:
    """Stream conversation messages to stdout without buffering."""
    conv_record = await repo.queries.get_conversation(conversation_id)
    if not conv_record:
        click.echo(f"Conversation not found: {conversation_id}", err=True)
        raise SystemExit(1)

    stats = await repo.queries.get_conversation_stats(conversation_id)
    sys.stdout.write(
        render_stream_header(
            conversation_id=conversation_id,
            title=conv_record.title,
            provider=getattr(conv_record, "provider_name", None),
            display_date=(getattr(conv_record, "updated_at", None) or getattr(conv_record, "created_at", None)),
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_limit=message_limit,
            stats=stats,
        )
    )
    sys.stdout.flush()

    count = 0
    async for msg in repo.iter_messages(
        conversation_id,
        dialogue_only=dialogue_only,
        limit=message_limit,
    ):
        chunk = render_stream_message(msg, output_format)
        if chunk:
            sys.stdout.write(chunk)
            count += 1
        sys.stdout.flush()

    sys.stdout.write(render_stream_footer(output_format=output_format, emitted_messages=count))
    sys.stdout.flush()

    return count


def _write_message_streaming(msg: Message, output_format: str) -> None:
    """Write a single message to stdout in streaming mode."""
    chunk = render_stream_message(msg, output_format)
    if chunk:
        sys.stdout.write(chunk)
        sys.stdout.flush()


def _send_output(
    env: AppEnv,
    content: str,
    destinations: list[str],
    output_format: str,
    conv: Conversation | None,
) -> None:
    """Send output to specified destinations."""
    for dest in destinations:
        if dest == "stdout":
            click.echo(content)
        elif dest == "browser":
            _open_in_browser(env, content, output_format, conv)
        elif dest == "clipboard":
            _copy_to_clipboard(env, content)
        else:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            env.ui.console.print(f"Wrote to {path}")


def _open_in_browser(
    env: AppEnv,
    content: str,
    output_format: str,
    conv: Conversation | None,
) -> None:
    """Open content in browser."""
    import tempfile
    import webbrowser

    if output_format != "html":
        if conv:
            from polylogue.rendering.formatting import _conv_to_html

            content = _conv_to_html(conv)
        else:
            from html import escape as _html_escape

            content = f"<html><body><pre>{_html_escape(content)}</pre></body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temp_path = handle.name

    webbrowser.open(f"file://{temp_path}")
    env.ui.console.print(f"Opened in browser: {temp_path}")


def _copy_to_clipboard(env: AppEnv, content: str) -> None:
    """Copy content to clipboard."""
    import subprocess

    clipboard_cmds = [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["clip"],
    ]

    for cmd in clipboard_cmds:
        try:
            subprocess.run(
                cmd,
                input=content.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            env.ui.console.print("Copied to clipboard.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    click.echo("Could not copy to clipboard (no clipboard tool found).", err=True)


def _open_result(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Open result in browser or editor."""
    if not results:
        env.ui.console.print("No conversations matched.")
        raise SystemExit(2)

    conv = results[0]

    from polylogue.cli.helpers import latest_render_path, load_effective_config

    try:
        config = load_effective_config(env)
    except Exception as exc:
        logger.warning("Config load failed, falling back to defaults: %s", exc)
        config = None

    render_root = None
    if config and hasattr(config, "render_root") and config.render_root:
        render_root = Path(config.render_root)
    else:
        import os

        render_root_env = os.environ.get("POLYLOGUE_RENDER_ROOT")
        if render_root_env:
            render_root = Path(render_root_env)

    if not render_root or not render_root.exists():
        click.echo("No rendered outputs found.", err=True)
        click.echo("Run 'polylogue run' first to render conversations.", err=True)
        raise SystemExit(1)

    conv_id_short = str(conv.id)[:8] if conv.id else ""
    html_file = next(render_root.rglob(f"*{conv_id_short}*/conversation.html"), None)
    md_file = next(render_root.rglob(f"*{conv_id_short}*/conversation.md"), None)

    render_file = html_file or md_file
    if not render_file:
        render_file = latest_render_path(render_root)

    if not render_file:
        click.echo("No rendered output found for this conversation.", err=True)
        click.echo("Run 'polylogue run' to render conversations.", err=True)
        raise SystemExit(1)

    import webbrowser

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")
