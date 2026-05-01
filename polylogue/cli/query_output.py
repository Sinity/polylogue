"""Output formatting, delivery, streaming, and summary-list helpers for CLI query execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Sequence
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import click

from polylogue.archive.message.roles import MessageRoleFilter, message_role_count_key, message_role_labels
from polylogue.archive.semantic.content_projection import ContentProjectionSpec, coerce_content_projection_spec
from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryOutputSpec
from polylogue.cli.query_feedback import emit_no_results
from polylogue.cli.query_output_contracts import QueryOutputDocument, StructuredRowsDocument
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
from polylogue.lib.json import JSONDocument
from polylogue.lib.roles import Role
from polylogue.logging import get_logger
from polylogue.rendering.formatting import format_conversation
from polylogue.surfaces.payloads import ConversationListRowPayload, ConversationSearchHitPayload, model_json_document

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.spec import ConversationQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.lib.search_hits import ConversationSearchHit
    from polylogue.protocols import ConversationOutputStore
    from polylogue.storage.runtime import MessageRecord

ConversationStats: TypeAlias = dict[str, int]
MACHINE_OUTPUT_FORMATS = frozenset({"json", "yaml", "csv"})
DIALOGUE_MESSAGE_ROLES: MessageRoleFilter = (Role.USER, Role.ASSISTANT)


def _display_date(value: datetime | None, date_format: str = "%Y-%m-%d") -> str:
    return value.strftime(date_format) if value else ""


def _ellipsize(value: str, max_width: int) -> str:
    return (value[: max_width - 3] + "...") if len(value) > max_width else value


def _conversation_list_line(conv: Conversation) -> str:
    date = _display_date(conv.display_date) or "unknown"
    title = _ellipsize(conv.display_title or conv.id[:20], 50)
    return f"{conv.id[:24]:24s}  {date:10s}  {conv.provider:12s}  {title} ({len(conv.messages)} msgs)"


def _summary_list_line(summary: ConversationSummary, message_count: int) -> str:
    date = _display_date(summary.display_date)
    title = _ellipsize(summary.display_title or str(summary.id)[:20], 50)
    return f"{str(summary.id)[:24]:24s}  {date:10s}  {summary.provider:12s}  {title} ({message_count} msgs)"


def _search_hit_list_line(hit: ConversationSearchHit, message_count: int) -> str:
    base = _summary_list_line(hit.summary, message_count)
    evidence_parts = [hit.match_surface, hit.retrieval_lane]
    if hit.message_id:
        evidence_parts.append(f"message {hit.message_id}")
    evidence = "/".join(evidence_parts)
    snippet = f": {hit.snippet}" if hit.snippet else ""
    return f"{base}\n  match[{hit.rank}]: {evidence}{snippet}"


def _stream_date_parts(display_date: object | None) -> tuple[str | None, str | None]:
    if isinstance(display_date, datetime):
        return display_date.strftime("%Y-%m-%d %H:%M"), display_date.isoformat()
    if hasattr(display_date, "strftime"):
        text = display_date.strftime("%Y-%m-%d %H:%M")
        value = display_date.isoformat() if hasattr(display_date, "isoformat") else str(display_date)
        return text, value
    if display_date:
        value = str(display_date)
        return value, value
    return None, None


def _role_filter_label(message_roles: MessageRoleFilter) -> str:
    if message_roles == DIALOGUE_MESSAGE_ROLES:
        return "dialogue"
    labels = message_role_labels(message_roles)
    if len(labels) == 1:
        return labels[0]
    return "selected-role"


def _role_filter_count(stats: ConversationStats, message_roles: MessageRoleFilter) -> int:
    if message_roles == DIALOGUE_MESSAGE_ROLES:
        return stats.get("dialogue_messages", 0)
    return sum(stats.get(message_role_count_key(role), 0) for role in message_roles)


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
        import yaml

        return str(yaml.dump([_conv_to_dict(c, fields) for c in results], default_flow_style=False, allow_unicode=True))
    if output_format == "csv":
        return conversations_to_csv(results)

    return "\n".join(_conversation_list_line(conv) for conv in results)


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
            logger.exception("render_conversation_rich: Panel rendering failed for role %s", role)
            console.print(f"[{rc.label}]{role.capitalize()}:[/{rc.label}] {msg.text[:200]}")
        console.print()


# ---------------------------------------------------------------------------
# Delivery and external-output helpers (from query_output_delivery.py)
# ---------------------------------------------------------------------------


def deliver_query_output(
    env: AppEnv,
    document: QueryOutputDocument,
) -> None:
    """Deliver a rendered query document to every requested destination."""
    for destination in document.destinations:
        if destination.kind == "stdout":
            click.echo(document.content)
        elif destination.kind == "browser":
            _open_in_browser(env, document.content, document.output_format, document.conversation)
        elif destination.kind == "clipboard":
            _copy_to_clipboard(env, document.content)
        else:
            assert destination.path is not None
            path = destination.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document.content, encoding="utf-8")
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
    results: Sequence[Conversation | ConversationSummary],
    output: QueryOutputSpec,
    *,
    selection: ConversationQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
) -> None:
    if not results:
        emit_no_results(env, selection=selection, diagnostics=diagnostics, output_format=output.output_format)

    conv = results[0]

    from polylogue.cli.shared.helpers import latest_render_path, load_effective_config

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

    from polylogue.paths.sanitize import conversation_render_root

    render_dir = conversation_render_root(render_root, str(conv.provider), str(conv.id))
    html_file = render_dir / "conversation.html"
    md_file = render_dir / "conversation.md"

    render_file = html_file if html_file.exists() else md_file if md_file.exists() else None
    if render_file is None:
        render_file = latest_render_path(render_root)

    if not render_file:
        click.echo("No rendered output found for this conversation.", err=True)
        click.echo("Run 'polylogue run' to render conversations.", err=True)
        raise SystemExit(1)

    if output.print_path:
        if output.output_format == "json":
            click.echo(json.dumps({"path": str(render_file)}, indent=2))
        else:
            click.echo(str(render_file))
        return

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")


# ---------------------------------------------------------------------------
# List output and CSV (from query_list_output.py)
# ---------------------------------------------------------------------------


def summary_to_dict(summary: ConversationSummary, message_count: int) -> JSONDocument:
    return ConversationListRowPayload.from_summary(
        summary,
        message_count=message_count,
    ).selected()


def format_summary_list(
    summaries: list[ConversationSummary],
    output_format: str,
    fields: str | None,
    *,
    message_counts: dict[str, int] | None = None,
) -> str:
    """Format summary-list output for deterministic machine/plain surfaces."""
    message_counts = message_counts or {}
    document = StructuredRowsDocument(
        rows=tuple(summary_to_dict(summary, message_counts.get(str(summary.id), 0)) for summary in summaries),
        csv_headers=("id", "date", "provider", "title", "messages", "tags", "summary"),
        csv_rows=tuple(
            (
                str(summary.id),
                _display_date(summary.display_date),
                summary.provider,
                summary.display_title or "",
                message_counts.get(str(summary.id), 0),
                ",".join(summary.tags) if summary.tags else "",
                summary.summary or "",
            )
            for summary in summaries
        ),
        text_lines=tuple(_summary_list_line(summary, message_counts.get(str(summary.id), 0)) for summary in summaries),
    )
    return document.with_selected_fields(fields).render(output_format)


def _search_hit_to_payload(
    hit: ConversationSearchHit,
    *,
    message_count: int,
) -> JSONDocument:
    return model_json_document(
        ConversationSearchHitPayload.from_search_hit(hit, message_count=message_count),
        exclude_none=True,
    )


def format_search_hit_list(
    hits: list[ConversationSearchHit],
    output_format: str,
    fields: str | None,
    *,
    message_counts: dict[str, int] | None = None,
) -> str:
    """Format evidence-bearing search hits for deterministic surfaces."""
    message_counts = message_counts or {}
    document = StructuredRowsDocument(
        rows=tuple(
            _search_hit_to_payload(
                hit, message_count=message_counts.get(hit.conversation_id, hit.summary.message_count or 0)
            )
            for hit in hits
        ),
        csv_headers=(
            "id",
            "date",
            "provider",
            "title",
            "messages",
            "rank",
            "retrieval_lane",
            "match_surface",
            "message_id",
            "snippet",
        ),
        csv_rows=tuple(
            (
                str(hit.summary.id),
                _display_date(hit.summary.display_date),
                hit.summary.provider,
                hit.summary.display_title or "",
                message_counts.get(hit.conversation_id, hit.summary.message_count or 0),
                hit.rank,
                hit.retrieval_lane,
                hit.match_surface,
                hit.message_id or "",
                hit.snippet or "",
            )
            for hit in hits
        ),
        text_lines=tuple(
            _search_hit_list_line(hit, message_counts.get(hit.conversation_id, hit.summary.message_count or 0))
            for hit in hits
        ),
    )
    return document.with_selected_fields(fields).render(output_format)


async def output_search_hits(
    env: AppEnv,
    hits: list[ConversationSearchHit],
    output: QueryOutputSpec,
    repo: ConversationOutputStore | None = None,
) -> None:
    """Output evidence-bearing search hits with optional rich table rendering."""
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [hit.conversation_id for hit in hits]
        msg_counts = await repo.get_message_counts_batch(ids)

    if output.output_format in MACHINE_OUTPUT_FORMATS or env.ui.plain:
        click.echo(
            format_search_hit_list(
                hits,
                "text" if env.ui.plain and output.output_format not in MACHINE_OUTPUT_FORMATS else output.output_format,
                output.fields,
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
    table.add_column("Match", ratio=1)

    for hit in hits:
        summary = hit.summary
        date = _display_date(summary.display_date)
        title = _ellipsize(summary.display_title or str(summary.id)[:20], 54)
        count = msg_counts.get(hit.conversation_id, summary.message_count or 0)
        provider_text = Text(summary.provider, style=provider_color(summary.provider).hex)
        snippet = _ellipsize(hit.snippet or "", 80)
        match = f"{hit.match_surface}/{hit.retrieval_lane}"
        if hit.message_id:
            match = f"{match} {hit.message_id}"
        if snippet:
            match = f"{match}: {snippet}"
        table.add_row(str(summary.id)[:24], date, provider_text, title, str(count), match)

    env.ui.console.print(table)


async def output_summary_list(
    env: AppEnv,
    summaries: list[ConversationSummary],
    output: QueryOutputSpec,
    repo: ConversationOutputStore | None = None,
) -> None:
    """Output a list of conversation summaries with optional rich table rendering."""
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [str(summary.id) for summary in summaries]
        msg_counts = await repo.get_message_counts_batch(ids)

    if output.output_format in MACHINE_OUTPUT_FORMATS or env.ui.plain:
        click.echo(
            format_summary_list(
                summaries,
                "text" if env.ui.plain and output.output_format not in MACHINE_OUTPUT_FORMATS else output.output_format,
                output.fields,
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
        date = _display_date(summary.display_date)
        title = _ellipsize(summary.display_title or str(summary.id)[:20], 63)
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
        tags_str = ",".join(conv.tags) if conv.tags else ""
        writer.writerow(
            [
                str(conv.id),
                _display_date(conv.display_date),
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
        raw_timestamp = getattr(message, "timestamp", None)
        timestamp = raw_timestamp.isoformat() if isinstance(raw_timestamp, datetime) else None
        record = {
            "type": "message",
            "id": getattr(message, "id", getattr(message, "message_id", None)),
            "role": message.role,
            "timestamp": timestamp,
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
    message_roles: MessageRoleFilter,
    message_limit: int | None,
    stats: ConversationStats | None,
) -> str:
    """Render any stream prelude/header for the selected output format."""
    display_date_text, display_date_value = _stream_date_parts(display_date)

    if output_format == "markdown":
        lines = [f"# {title or conversation_id[:24]}", ""]
        if display_date_text is not None:
            lines.append(f"**Date**: {display_date_text}")
        if provider:
            lines.append(f"**Provider**: {provider}")
        if display_date_text is not None or provider:
            lines.append("")
        if message_roles and stats:
            shown = _role_filter_count(stats, message_roles)
            label = _role_filter_label(message_roles)
            line = f"_Showing {shown} {label} messages"
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
            "message_roles": list(message_role_labels(message_roles)),
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
    message_roles: MessageRoleFilter = (),
    message_limit: int | None = None,
    stats: ConversationStats | None = None,
) -> tuple[str, int]:
    """Render the full stream transcript deterministically for proof/tests."""
    effective_roles = message_roles or (DIALOGUE_MESSAGE_ROLES if dialogue_only else ())
    filtered_messages = [message for message in messages if not effective_roles or message.role in effective_roles]
    parts = [
        render_stream_header(
            conversation_id=conversation_id,
            title=title,
            provider=provider,
            display_date=display_date,
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_roles=effective_roles,
            message_limit=message_limit,
            stats=stats,
        )
    ]
    emitted = 0
    for message in filtered_messages[: message_limit if message_limit is not None else None]:
        chunk = render_stream_message(message, output_format)
        if chunk:
            parts.append(chunk)
            emitted += 1
    parts.append(render_stream_footer(output_format=output_format, emitted_messages=emitted))
    return "".join(parts), emitted


async def stream_conversation(
    env: AppEnv,
    repo: ConversationOutputStore,
    conversation_id: str,
    *,
    output_format: str = "plaintext",
    dialogue_only: bool = False,
    message_roles: MessageRoleFilter = (),
    content_projection: ContentProjectionSpec | None = None,
    message_limit: int | None = None,
) -> int:
    """Stream conversation messages to stdout without buffering."""
    projection = await repo.get_render_projection(conversation_id)
    if projection is None:
        click.echo(f"Conversation not found: {conversation_id}", err=True)
        raise SystemExit(1)
    conv_record = projection.conversation

    stats = await repo.get_conversation_stats(conversation_id)
    effective_roles = message_roles or (DIALOGUE_MESSAGE_ROLES if dialogue_only else ())
    projection_spec = coerce_content_projection_spec(content_projection)
    sys.stdout.write(
        render_stream_header(
            conversation_id=conversation_id,
            title=conv_record.title,
            provider=getattr(conv_record, "provider_name", None),
            display_date=(getattr(conv_record, "updated_at", None) or getattr(conv_record, "created_at", None)),
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_roles=effective_roles,
            message_limit=message_limit,
            stats=stats,
        )
    )
    sys.stdout.flush()

    count = 0
    if projection_spec.filters_content():
        conversation = await repo.get(conversation_id)
        if conversation is None:
            click.echo(f"Conversation not found: {conversation_id}", err=True)
            raise SystemExit(1)
        if effective_roles:
            conversation = conversation.with_roles(effective_roles)
        conversation = conversation.with_content_projection(projection_spec)
        for message in list(conversation.messages)[: message_limit if message_limit is not None else None]:
            chunk = render_stream_message(message, output_format)
            if chunk:
                sys.stdout.write(chunk)
                count += 1
            sys.stdout.flush()
    else:
        async for message in repo.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            message_roles=effective_roles,
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


def no_results(
    env: AppEnv,
    output: QueryOutputSpec,
    *,
    selection: ConversationQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
    exit_code: int | None = 2,
) -> None:
    """Emit the canonical no-results contract for output surfaces."""
    emit_no_results(
        env,
        selection=selection,
        diagnostics=diagnostics,
        output_format=output.output_format,
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Main output dispatch (from original query_output.py)
# ---------------------------------------------------------------------------


def output_results(
    env: AppEnv,
    results: list[Conversation],
    output: QueryOutputSpec,
    *,
    selection: ConversationQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
) -> None:
    """Output query results."""
    if not results:
        no_results(env, output, selection=selection, diagnostics=diagnostics)

    if len(results) == 1 and not output.list_mode:
        conv = results[0]
        if output.output_format == "markdown" and output.destination_labels() == ("stdout",) and not env.ui.plain:
            _render_conversation_rich(env, conv)
            return
        content = format_conversation(conv, output.output_format, output.fields)
        _send_output(env, content, output.destinations, output.output_format, conv)
        return

    content = _format_list(results, output.output_format, output.fields)
    _send_output(env, content, output.destinations, output.output_format, None)


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
    destinations: Sequence[QueryDeliveryTarget],
    output_format: str,
    conv: Conversation | None,
) -> None:
    """Send output to specified destinations."""
    deliver_query_output(
        env,
        QueryOutputDocument(
            content=content,
            output_format=output_format,
            destinations=tuple(destinations),
            conversation=conv,
        ),
    )


__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "conversations_to_csv",
    "copy_to_clipboard",
    "deliver_query_output",
    "emit_structured_stats",
    "filtered_action_events",
    "format_list",
    "format_search_hit_list",
    "format_summary_list",
    "normalized_tool_name",
    "open_in_browser",
    "open_result",
    "output_results",
    "output_search_hits",
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
    "stream_conversation",
    "summary_to_dict",
    "write_message_streaming",
]
