"""Output formatting, delivery, streaming, and summary-list helpers for CLI query execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import webbrowser
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.cli.query_semantic import (
    SemanticStatsSlice,
    action_matches_slice,
    filtered_action_events,
    normalized_tool_name,
    output_stats_by_semantic_ids,
    output_stats_by_semantic_query,
    output_stats_by_semantic_summaries,
)
from polylogue.cli.query_stats import (
    emit_structured_stats,
    output_stats_by_conversations,
    output_stats_by_profile_ids,
    output_stats_by_profile_query,
    output_stats_by_profile_summaries,
    output_stats_by_summaries,
    output_stats_sql,
)
from polylogue.logging import get_logger
from polylogue.rendering.formatting import format_conversation

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import MessageRecord


# ---------------------------------------------------------------------------
# Output formatting (from query_output_formatting.py)
# ---------------------------------------------------------------------------


def format_list(
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
        return conversations_to_csv(results)

    lines = []
    for conv in results:
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
        raw_title = conv.display_title or conv.id[:20]
        title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
        msg_count = len(conv.messages)
        lines.append(f"{conv.id[:24]:24s}  {date:10s}  {conv.provider:12s}  {title} ({msg_count} msgs)")
    return "\n".join(lines)


def render_conversation_rich(env: AppEnv, conv: Conversation) -> None:
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


# ---------------------------------------------------------------------------
# Delivery and external-output helpers (from query_output_delivery.py)
# ---------------------------------------------------------------------------


def send_output(
    env: AppEnv,
    content: str,
    destinations: list[str],
    output_format: str,
    conv: Conversation | None,
) -> None:
    for dest in destinations:
        if dest == "stdout":
            click.echo(content)
        elif dest == "browser":
            open_in_browser(env, content, output_format, conv)
        elif dest == "clipboard":
            copy_to_clipboard(env, content)
        else:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            env.ui.console.print(f"Wrote to {path}")


def open_in_browser(
    env: AppEnv,
    content: str,
    output_format: str,
    conv: Conversation | None,
) -> None:
    if output_format != "html":
        if conv:
            from polylogue.rendering.formatting import _conv_to_html

            content = _conv_to_html(conv)
        else:
            content = f"<html><body><pre>{html_escape(content)}</pre></body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temp_path = handle.name

    webbrowser.open(f"file://{temp_path}")
    env.ui.console.print(f"Opened in browser: {temp_path}")


def copy_to_clipboard(env: AppEnv, content: str) -> None:
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


def open_result(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, object],
) -> None:
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

    from polylogue.paths import render_root as default_render_root

    render_root = None
    if config and hasattr(config, "render_root") and config.render_root:
        render_root = Path(config.render_root)
    else:
        render_root = default_render_root()

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

    if bool(params.get("print_path")):
        output_format = str(params.get("output_format") or "markdown")
        if output_format == "json":
            click.echo(json.dumps({"path": str(render_file)}, indent=2))
        else:
            click.echo(str(render_file))
        return

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")


# ---------------------------------------------------------------------------
# List output and CSV (from query_list_output.py)
# ---------------------------------------------------------------------------


def summary_to_dict(summary: ConversationSummary, message_count: int) -> dict[str, object]:
    return {
        "id": str(summary.id),
        "provider": str(summary.provider),
        "title": summary.display_title,
        "date": summary.display_date.isoformat() if summary.display_date else None,
        "tags": summary.tags,
        "summary": summary.summary,
        "messages": message_count,
    }


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
            writer.writerow(
                [
                    str(summary.id),
                    date,
                    summary.provider,
                    summary.display_title or "",
                    message_counts.get(str(summary.id), 0),
                    tags_str,
                    summary.summary or "",
                ]
            )
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
        writer.writerow(
            [
                str(conv.id),
                date,
                conv.provider,
                conv.display_title or "",
                len(conv.messages),
                sum(message.word_count for message in conv.messages),
                tags_str,
                conv.summary or "",
            ]
        )

    return output.getvalue()


# ---------------------------------------------------------------------------
# Streaming output (from query_stream_output.py)
# ---------------------------------------------------------------------------


def render_stream_message(message: Message | MessageRecord, output_format: str) -> str:
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
            "id": getattr(message, "id", getattr(message, "message_id", None)),
            "role": message.role,
            "timestamp": message.timestamp.isoformat() if getattr(message, "timestamp", None) else None,
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
    async for message in repo.iter_messages(
        conversation_id,
        dialogue_only=dialogue_only,
        limit=message_limit,
    ):
        chunk = render_stream_message(message, output_format)
        if chunk:
            sys.stdout.write(chunk)
            count += 1
        sys.stdout.flush()

    sys.stdout.write(render_stream_footer(output_format=output_format, emitted_messages=count))
    sys.stdout.flush()

    return count


def write_message_streaming(message: Message | MessageRecord, output_format: str) -> None:
    """Write a single streamed message to stdout."""
    chunk = render_stream_message(message, output_format)
    if chunk:
        sys.stdout.write(chunk)
        sys.stdout.flush()


def no_results(env: AppEnv, params: dict[str, Any], *, exit_code: int = 2) -> None:
    """Delegate the no-results contract to the canonical query helper."""
    from polylogue.cli.query import no_results as query_no_results

    query_no_results(env, params, exit_code=exit_code)


# ---------------------------------------------------------------------------
# Main output dispatch (from original query_output.py)
# ---------------------------------------------------------------------------


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


# Internal aliases used by query.py and tests
_output_summary_list = output_summary_list
_output_stats_by = output_stats_by_conversations
_write_message_streaming = write_message_streaming
_copy_to_clipboard = copy_to_clipboard
_open_in_browser = open_in_browser
_open_result = open_result
_format_list = format_list
_render_conversation_rich = render_conversation_rich


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


__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "conversations_to_csv",
    "copy_to_clipboard",
    "emit_structured_stats",
    "filtered_action_events",
    "format_list",
    "format_summary_list",
    "normalized_tool_name",
    "open_in_browser",
    "open_result",
    "output_results",
    "output_stats_by_conversations",
    "output_stats_by_profile_ids",
    "output_stats_by_profile_query",
    "output_stats_by_profile_summaries",
    "output_stats_by_semantic_ids",
    "output_stats_by_semantic_query",
    "output_stats_by_semantic_summaries",
    "output_stats_by_summaries",
    "output_stats_sql",
    "output_summary_list",
    "render_conversation_rich",
    "render_stream_footer",
    "render_stream_header",
    "render_stream_message",
    "render_stream_transcript",
    "send_output",
    "stream_conversation",
    "summary_to_dict",
    "write_message_streaming",
]
