"""Output and streaming helpers for CLI query execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.cli.query_helpers import no_results
from polylogue.cli.query_output_delivery import (
    copy_to_clipboard as _copy_to_clipboard_impl,
)
from polylogue.cli.query_output_delivery import (
    open_in_browser as _open_in_browser_impl,
)
from polylogue.cli.query_output_delivery import (
    open_result as _open_result_impl,
)
from polylogue.cli.query_output_formatting import (
    format_list as _format_list,
)
from polylogue.cli.query_output_formatting import (
    render_conversation_rich as _render_conversation_rich,
)
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
from polylogue.rendering.formatting import format_conversation

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


_output_summary_list = _output_summary_list_impl
_output_stats_by = _output_stats_by_conversations
_write_message_streaming = _write_message_streaming_impl
_copy_to_clipboard = _copy_to_clipboard_impl
_open_in_browser = _open_in_browser_impl
_open_result = _open_result_impl


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
