"""Output formatting, delivery, streaming, and summary-list helpers for CLI query execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from html import escape as html_escape
from typing import TYPE_CHECKING, TypeAlias
from urllib.parse import quote

import click

from polylogue.archive.query.search_hits import bound_search_snippet
from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryOutputSpec
from polylogue.cli.query_feedback import emit_no_results
from polylogue.cli.query_output_contracts import QueryOutputDocument, StructuredRowsDocument
from polylogue.cli.query_semantic import (
    SemanticStatsSlice,
    action_matches_slice,
    filtered_actions,
    normalized_tool_name,
    output_stats_by_semantic_ids,
    output_stats_by_semantic_query,
    output_stats_by_semantic_summaries,
)
from polylogue.cli.query_stats import (
    emit_structured_stats,
    output_stats_by_profile_ids,
    output_stats_by_profile_query,
    output_stats_by_profile_summaries,
    output_stats_by_sessions,
    output_stats_by_summaries,
    output_stats_sql,
)
from polylogue.core.json import JSONDocument
from polylogue.logging import get_logger
from polylogue.rendering.formatting import format_session
from polylogue.surfaces.payloads import (
    SearchCursor,
    SessionListRowPayload,
    SessionSearchHitPayload,
    build_search_envelope,
    model_json_document,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.archive.models import Message, Session, SessionSummary
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.core.protocols import SessionOutputStore
    from polylogue.storage.runtime import MessageRecord

SessionStats: TypeAlias = dict[str, int]
MACHINE_OUTPUT_FORMATS = frozenset({"json", "ndjson", "yaml", "csv"})


def _display_date(value: datetime | None, date_format: str = "%Y-%m-%d") -> str:
    return value.strftime(date_format) if value else ""


def _ellipsize(value: str, max_width: int) -> str:
    if max_width <= 3:
        return value[:max_width]
    return (value[: max_width - 3] + "...") if len(value) > max_width else value


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _display_title(value: str | None, fallback: str, *, max_width: int) -> str:
    title = _single_line(value or fallback)
    return _ellipsize(title, max_width)


class _LayoutBreakpoints:
    """Terminal-width breakpoints for graceful narrow rendering.

    Listed widest-first. The first breakpoint whose ``min_width`` fits the
    current terminal wins. Below the narrowest breakpoint we drop to a single
    "id/title" line per row so output stays legible at ~40 columns.

    Pinned at 80 / 60 / 40 so the contract is observable in tests without
    depending on host terminal size.
    """

    WIDE_MIN = 80
    MID_MIN = 60
    NARROW_MIN = 40


def _terminal_width(env: AppEnv) -> int:
    """Return the rendering terminal width, falling back to 80 columns."""
    width = getattr(env.ui.console, "width", None)
    if isinstance(width, int) and width > 0:
        return width
    return 80


def _summary_list_layout(width: int) -> tuple[str, ...]:
    """Pick a column set for the session-list table at ``width`` columns.

    ``width`` is the rendered terminal width. The selection is deterministic
    and pinned by ``test_narrow_terminal_layout``; do not reorder columns
    without updating that contract.
    """
    if width >= _LayoutBreakpoints.WIDE_MIN:
        return ("id", "date", "origin", "title", "msgs")
    if width >= _LayoutBreakpoints.MID_MIN:
        # Drop the wide ID column; titles get more room.
        return ("date", "origin", "title", "msgs")
    if width >= _LayoutBreakpoints.NARROW_MIN:
        # Drop provider too; date + title + msgs survives at 40 cols.
        return ("date", "title", "msgs")
    # Below the narrowest breakpoint we go single-column.
    return ("title",)


def _search_hit_layout(width: int) -> tuple[str, ...]:
    """Pick a column set for the search-hit table at ``width`` columns."""
    if width >= _LayoutBreakpoints.WIDE_MIN:
        return ("id", "date", "origin", "title", "msgs", "match")
    if width >= _LayoutBreakpoints.MID_MIN:
        # Drop ID column; keep match snippet for context.
        return ("date", "origin", "title", "match")
    if width >= _LayoutBreakpoints.NARROW_MIN:
        # Drop provider and date; keep title + match snippet.
        return ("title", "match")
    return ("title",)


def _title_budget(width: int) -> int:
    """Title truncation budget for the rendered terminal width."""
    if width >= _LayoutBreakpoints.WIDE_MIN:
        return 63
    if width >= _LayoutBreakpoints.MID_MIN:
        return max(30, width - 24)
    if width >= _LayoutBreakpoints.NARROW_MIN:
        return max(18, width - 18)
    return max(12, width - 2)


def _session_list_line(conv: Session) -> str:
    date = _display_date(conv.display_date) or "unknown"
    title = _display_title(conv.display_title, conv.id[:20], max_width=50)
    return f"{conv.id[:24]:24s}  {date:10s}  {str(conv.origin):20s}  {title} ({len(conv.messages)} msgs)"


def _summary_list_line(summary: SessionSummary, message_count: int) -> str:
    date = _display_date(summary.display_date)
    title = _display_title(summary.display_title, str(summary.id)[:20], max_width=50)
    return f"{str(summary.id)[:24]:24s}  {date:10s}  {str(summary.origin):20s}  {title} ({message_count} msgs)"


def _search_hit_list_line(hit: SessionSearchHit, message_count: int) -> str:
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


# ---------------------------------------------------------------------------
# Output formatting (from query_output_formatting.py)
# ---------------------------------------------------------------------------


def format_list(
    results: list[Session],
    output_format: str,
    fields: str | None,
) -> str:
    """Format a list of sessions for output.

    #1618: JSON and YAML emit a paginated envelope
    (``{"items": [...], "total": N, "limit": N, "offset": 0}``) instead
    of a bare array so the shape matches the MCP
    ``list_sessions`` tool. ``next_offset`` is omitted because
    the CLI doesn't paginate today (it returns the full match set);
    when CLI pagination lands it will populate the same field MCP
    already does. Bare-array consumers must read ``.items``.
    """
    from polylogue.rendering.formatting import _conv_to_dict

    if output_format == "json":
        items = [_conv_to_dict(c, fields) for c in results]
        envelope = {"items": items, "total": len(items), "limit": len(items), "offset": 0}
        return json.dumps(envelope, indent=2)
    if output_format == "yaml":
        import yaml

        items = [_conv_to_dict(c, fields) for c in results]
        envelope = {"items": items, "total": len(items), "limit": len(items), "offset": 0}
        return str(yaml.dump(envelope, default_flow_style=False, allow_unicode=True, sort_keys=False))
    if output_format == "csv":
        return sessions_to_csv(results)

    return "\n".join(_session_list_line(conv) for conv in results)


def render_session_rich(env: AppEnv, conv: Session) -> None:
    """Render a session with Rich role colors and thinking block styling."""
    from rich import box
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    from polylogue.ui.theme import THINKING_STYLE, provider_color, role_color

    console = env.ui.console
    title = conv.display_title or conv.id
    pc = provider_color(str(conv.origin))
    header = Text()
    header.append(title, style="bold")
    if conv.display_date:
        header.append(f"  {conv.display_date.strftime('%Y-%m-%d %H:%M')}", style="dim")
    header.append(f"  [{pc.hex}]{str(conv.origin)}[/{pc.hex}]")
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
            logger.exception("render_session_rich: Panel rendering failed for role %s", role)
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
            _open_in_browser(env, document.content, document.output_format, document.session)
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
    conv: Session | None,
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
    results: Sequence[Session | SessionSummary],
    output: QueryOutputSpec,
    *,
    selection: SessionQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
) -> None:
    if not results:
        emit_no_results(env, selection=selection, diagnostics=diagnostics, output_format=output.output_format)

    conv = results[0]

    daemon_url = str(getattr(env, "daemon_url", None) or "http://127.0.0.1:8766").rstrip("/")
    web_url = f"{daemon_url}/?session={quote(str(conv.id), safe='')}"
    if output.print_url:
        if output.output_format == "json":
            click.echo(json.dumps({"url": web_url}, indent=2))
        else:
            click.echo(web_url)
        return

    webbrowser.open(web_url)
    env.ui.console.print(f"Opened: {web_url}")


# ---------------------------------------------------------------------------
# List output and CSV (from query_list_output.py)
# ---------------------------------------------------------------------------


def summary_to_dict(summary: SessionSummary, message_count: int) -> JSONDocument:
    return SessionListRowPayload.from_summary(
        summary,
        message_count=message_count,
    ).selected()


def format_summary_list(
    summaries: list[SessionSummary],
    output_format: str,
    fields: str | None,
    *,
    message_counts: dict[str, int] | None = None,
) -> str:
    """Format summary-list output for deterministic machine/plain surfaces."""
    message_counts = message_counts or {}
    document = StructuredRowsDocument(
        rows=tuple(summary_to_dict(summary, message_counts.get(str(summary.id), 0)) for summary in summaries),
        csv_headers=("id", "date", "origin", "title", "messages", "tags", "summary"),
        csv_rows=tuple(
            (
                str(summary.id),
                _display_date(summary.display_date),
                str(summary.origin),
                _single_line(summary.display_title or ""),
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
    hit: SessionSearchHit,
    *,
    message_count: int,
) -> JSONDocument:
    hit = _bounded_search_hit(hit)
    return model_json_document(
        SessionSearchHitPayload.from_search_hit(hit, message_count=message_count),
        exclude_none=True,
    )


def _bounded_search_hit(hit: SessionSearchHit) -> SessionSearchHit:
    bounded = bound_search_snippet(hit.snippet)
    if bounded == hit.snippet:
        return hit
    return replace(hit, snippet=bounded)


def format_search_hit_list(
    hits: list[SessionSearchHit],
    output_format: str,
    fields: str | None,
    *,
    message_counts: dict[str, int] | None = None,
) -> str:
    """Format evidence-bearing search hits for deterministic surfaces."""
    message_counts = message_counts or {}
    bounded_hits = [_bounded_search_hit(hit) for hit in hits]
    document = StructuredRowsDocument(
        rows=tuple(
            _search_hit_to_payload(
                hit, message_count=message_counts.get(hit.session_id, hit.summary.message_count or 0)
            )
            for hit in bounded_hits
        ),
        csv_headers=(
            "id",
            "date",
            "origin",
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
                str(hit.summary.origin),
                _single_line(hit.summary.display_title or ""),
                message_counts.get(hit.session_id, hit.summary.message_count or 0),
                hit.rank,
                hit.retrieval_lane,
                hit.match_surface,
                hit.message_id or "",
                hit.snippet or "",
            )
            for hit in bounded_hits
        ),
        text_lines=tuple(
            _search_hit_list_line(hit, message_counts.get(hit.session_id, hit.summary.message_count or 0))
            for hit in bounded_hits
        ),
    )
    return document.with_selected_fields(fields).render(output_format)


def format_search_envelope(
    hits: list[SessionSearchHit],
    *,
    query: str,
    retrieval_lane: str,
    limit: int,
    offset: int,
    sort: str | None,
    total: int | None = None,
    message_counts: dict[str, int] | None = None,
    cursor: SearchCursor | None = None,
) -> str:
    """Render the canonical :class:`SearchEnvelope` JSON for ranked search.

    Used for ``--format json`` so CLI JSON output matches MCP, daemon HTTP,
    and the Python API's ``Polylogue.search_envelope()`` shape (#1266, #1749).
    ``total`` is the shared "count when known" field: callers that hold the
    query spec thread the ``spec.count()`` result so the CLI reports a
    concrete count like every other surface. ``None`` is retained only for
    the genuine no-spec case where no count is available.
    """
    counts = message_counts or {}
    bounded_hits = [_bounded_search_hit(hit) for hit in hits]
    hit_payloads = [
        SessionSearchHitPayload.from_search_hit(
            hit,
            message_count=counts.get(hit.session_id, hit.summary.message_count or 0),
        )
        for hit in bounded_hits
    ]
    resolved_lane = bounded_hits[0].retrieval_lane if bounded_hits else (retrieval_lane or "auto")
    envelope = build_search_envelope(
        hit_payloads,
        total=total,
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=resolved_lane,
        sort=sort,
        cursor=cursor,
    )
    return envelope.model_dump_json(indent=2, exclude_none=True)


async def output_search_hits(
    env: AppEnv,
    hits: list[SessionSearchHit],
    output: QueryOutputSpec,
    repo: SessionOutputStore | None = None,
    *,
    total: int | None = None,
    cursor: SearchCursor | None = None,
) -> None:
    """Output evidence-bearing search hits with optional rich table rendering.

    ``total`` is the count of matching sessions (from ``spec.count()``)
    threaded by the caller so the JSON envelope reports a concrete total like
    every other read surface (#1749). ``cursor`` carries a previously-decoded
    :class:`SearchCursor` when the request is a paginated follow-up (#1268);
    JSON envelope output uses it to drop hits up to and including the anchor
    and to mint a fresh ``next_cursor`` from the page tail.
    """
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [hit.session_id for hit in hits]
        msg_counts = await repo.get_message_counts_batch(ids)

    if output.output_format == "json":
        # JSON format uses the typed SearchEnvelope shared across surfaces (#1266).
        # ndjson/yaml/csv keep emitting one-per-row for streaming-friendly use.
        query_text = getattr(output, "search_query", "") or ""
        retrieval_lane = getattr(output, "retrieval_lane", "") or "auto"
        limit_value = getattr(output, "limit", None) or len(hits)
        offset_value = getattr(output, "offset", 0) or 0
        sort_value = getattr(output, "sort", None)
        click.echo(
            format_search_envelope(
                hits,
                query=query_text,
                retrieval_lane=retrieval_lane,
                limit=limit_value,
                offset=offset_value,
                sort=sort_value,
                total=total,
                message_counts=msg_counts,
                cursor=cursor,
            )
        )
        return

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

    width = _terminal_width(env)
    columns = _search_hit_layout(width)
    title_budget = _title_budget(width)
    snippet_budget = max(20, min(80, width - 24))

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False, show_edge=False)
    for column in columns:
        if column == "id":
            table.add_column("ID", style="dim", max_width=24, no_wrap=True)
        elif column == "date":
            table.add_column("Date", style="dim")
        elif column == "origin":
            table.add_column("Origin")
        elif column == "title":
            table.add_column("Title", ratio=1)
        elif column == "msgs":
            table.add_column("Msgs", justify="right")
        elif column == "match":
            table.add_column("Match", ratio=1)

    for hit in (_bounded_search_hit(hit) for hit in hits):
        summary = hit.summary
        date = _display_date(summary.display_date)
        title = _display_title(summary.display_title, str(summary.id)[:20], max_width=title_budget)
        count = msg_counts.get(hit.session_id, summary.message_count or 0)
        origin_text = Text(
            str(summary.origin),
            style=provider_color(str(summary.origin)).hex,
        )
        snippet = _ellipsize(hit.snippet or "", snippet_budget)
        match = f"{hit.match_surface}/{hit.retrieval_lane}"
        if hit.message_id:
            match = f"{match} {hit.message_id}"
        if snippet:
            match = f"{match}: {snippet}"
        row: list[str | Text] = []
        for column in columns:
            if column == "id":
                row.append(str(summary.id)[:24])
            elif column == "date":
                row.append(date)
            elif column == "origin":
                row.append(origin_text)
            elif column == "title":
                row.append(title)
            elif column == "msgs":
                row.append(str(count))
            elif column == "match":
                row.append(match)
        table.add_row(*row)

    env.ui.console.print(table)


async def output_summary_list(
    env: AppEnv,
    summaries: list[SessionSummary],
    output: QueryOutputSpec,
    repo: SessionOutputStore | None = None,
) -> None:
    """Output a list of session summaries with optional rich table rendering."""
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

    width = _terminal_width(env)
    columns = _summary_list_layout(width)
    title_budget = _title_budget(width)

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False, show_edge=False)
    for column in columns:
        if column == "id":
            table.add_column("ID", style="dim", max_width=24, no_wrap=True)
        elif column == "date":
            table.add_column("Date", style="dim")
        elif column == "origin":
            table.add_column("Origin")
        elif column == "title":
            table.add_column("Title", ratio=1)
        elif column == "msgs":
            table.add_column("Msgs", justify="right")

    for summary in summaries:
        date = _display_date(summary.display_date)
        title = _display_title(summary.display_title, str(summary.id)[:20], max_width=title_budget)
        count = msg_counts.get(str(summary.id), 0)
        origin_text = Text(
            str(summary.origin),
            style=provider_color(str(summary.origin)).hex,
        )
        row: list[str | Text] = []
        for column in columns:
            if column == "id":
                row.append(str(summary.id)[:24])
            elif column == "date":
                row.append(date)
            elif column == "origin":
                row.append(origin_text)
            elif column == "title":
                row.append(title)
            elif column == "msgs":
                row.append(str(count))
        table.add_row(*row)

    env.ui.console.print(table)


def sessions_to_csv(results: list[Session]) -> str:
    """Convert hydrated sessions to CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "origin", "title", "messages", "words", "tags", "summary"])

    for conv in results:
        tags_str = ",".join(conv.tags) if conv.tags else ""
        writer.writerow(
            [
                str(conv.id),
                _display_date(conv.display_date),
                str(conv.origin),
                _single_line(conv.display_title or ""),
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
    session_id: str,
    title: str | None,
    origin: str | None,
    display_date: object | None,
    output_format: str,
    message_limit: int | None,
) -> str:
    """Render any stream prelude/header for the selected output format."""
    display_date_text, display_date_value = _stream_date_parts(display_date)

    if output_format == "markdown":
        lines = [f"# {title or session_id[:24]}", ""]
        if display_date_text is not None:
            lines.append(f"**Date**: {display_date_text}")
        if origin:
            lines.append(f"**Origin**: {origin}")
        if display_date_text is not None or origin:
            lines.append("")
        if message_limit:
            line = f"_Showing up to {message_limit} messages_"
            lines.extend([line, ""])
        return "\n".join(lines)

    if output_format == "json-lines":
        header = {
            "type": "header",
            "session_id": session_id,
            "title": title,
            "origin": origin,
            "date": display_date_value,
            "message_limit": message_limit,
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
    session_id: str,
    title: str | None,
    origin: str | None,
    display_date: object | None,
    messages: list[Message],
    output_format: str,
    message_limit: int | None = None,
) -> tuple[str, int]:
    """Render the full stream transcript deterministically for evidence/tests."""
    parts = [
        render_stream_header(
            session_id=session_id,
            title=title,
            origin=origin,
            display_date=display_date,
            output_format=output_format,
            message_limit=message_limit,
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
    selection: SessionQuerySpec | None = None,
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
    results: list[Session],
    output: QueryOutputSpec,
    *,
    selection: SessionQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
) -> None:
    """Output query results."""
    if not results:
        no_results(env, output, selection=selection, diagnostics=diagnostics)

    if len(results) == 1 and not output.list_mode:
        conv = results[0]
        if output.output_format == "markdown" and output.destination_labels() == ("stdout",) and not env.ui.plain:
            _render_session_rich(env, conv)
            return
        content = format_session(conv, output.output_format, output.fields)
        _send_output(env, content, output.destinations, output.output_format, conv)
        return

    content = _format_list(results, output.output_format, output.fields)
    _send_output(env, content, output.destinations, output.output_format, None)


# Internal aliases used by query.py and tests
_output_summary_list = output_summary_list
_output_stats_by = output_stats_by_sessions
_write_message_streaming = write_message_streaming
_copy_to_clipboard = copy_to_clipboard
_open_in_browser = open_in_browser
_open_result = open_result
_format_list = format_list
_render_session_rich = render_session_rich


def _send_output(
    env: AppEnv,
    content: str,
    destinations: Sequence[QueryDeliveryTarget],
    output_format: str,
    conv: Session | None,
) -> None:
    """Send output to specified destinations."""
    deliver_query_output(
        env,
        QueryOutputDocument(
            content=content,
            output_format=output_format,
            destinations=tuple(destinations),
            session=conv,
        ),
    )


__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "sessions_to_csv",
    "copy_to_clipboard",
    "deliver_query_output",
    "emit_structured_stats",
    "filtered_actions",
    "format_list",
    "format_search_hit_list",
    "format_summary_list",
    "normalized_tool_name",
    "open_in_browser",
    "open_result",
    "output_results",
    "format_search_envelope",
    "output_search_hits",
    "output_stats_by_sessions",
    "output_stats_by_profile_ids",
    "output_stats_by_profile_query",
    "output_stats_by_profile_summaries",
    "output_stats_by_semantic_ids",
    "output_stats_by_semantic_query",
    "output_stats_by_semantic_summaries",
    "output_stats_by_summaries",
    "output_stats_sql",
    "output_summary_list",
    "render_session_rich",
    "render_stream_footer",
    "render_stream_header",
    "render_stream_message",
    "render_stream_transcript",
    "summary_to_dict",
    "write_message_streaming",
]
