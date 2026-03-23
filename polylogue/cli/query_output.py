"""Output and streaming helpers for CLI query execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.cli.query_helpers import no_results
from polylogue.cli.query_stream_output import (
    render_stream_footer as _render_stream_footer,
)
from polylogue.cli.query_stream_output import (
    render_stream_header as _render_stream_header,
)
from polylogue.cli.query_stream_output import (
    render_stream_message as _render_stream_message,
)
from polylogue.cli.query_stream_output import (
    render_stream_transcript as _render_stream_transcript,
)
from polylogue.cli.query_stream_output import (
    stream_conversation as _stream_conversation,
)
from polylogue.cli.query_stream_output import (
    write_message_streaming as _write_message_streaming_impl,
)
from polylogue.cli.query_summary_output import (
    conversations_to_csv as _conversations_to_csv,
)
from polylogue.cli.query_summary_output import (
    format_summary_list as _format_summary_list,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_conversations as _output_stats_by_conversations,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_profile_query as _output_stats_by_profile_query,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_profile_summaries as _output_stats_by_profile_summaries,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_semantic_query as _output_stats_by_semantic_query,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_semantic_summaries as _output_stats_by_semantic_summaries,
)
from polylogue.cli.query_summary_output import (
    output_stats_by_summaries as _output_stats_by_summaries,
)
from polylogue.cli.query_summary_output import (
    output_stats_sql as _output_stats_sql,
)
from polylogue.cli.query_summary_output import (
    output_summary_list as _output_summary_list_impl,
)
from polylogue.logging import get_logger
from polylogue.rendering.formatting import format_conversation

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation


render_stream_footer = _render_stream_footer
render_stream_header = _render_stream_header
render_stream_message = _render_stream_message
render_stream_transcript = _render_stream_transcript
stream_conversation = _stream_conversation
format_summary_list = _format_summary_list
output_stats_by_summaries = _output_stats_by_summaries
output_stats_by_profile_query = _output_stats_by_profile_query
output_stats_by_profile_summaries = _output_stats_by_profile_summaries
output_stats_by_semantic_query = _output_stats_by_semantic_query
output_stats_by_semantic_summaries = _output_stats_by_semantic_summaries
output_stats_sql = _output_stats_sql


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
        return _conversations_to_csv(results)

    lines = []
    for conv in results:
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
        raw_title = conv.display_title or conv.id[:20]
        title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
        msg_count = len(conv.messages)
        lines.append(f"{conv.id[:24]:24s}  {date:10s}  {conv.provider:12s}  {title} ({msg_count} msgs)")
    return "\n".join(lines)


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


_output_summary_list = _output_summary_list_impl
_output_stats_by = _output_stats_by_conversations
_write_message_streaming = _write_message_streaming_impl


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
