"""Seam B — the one row renderer over declared operation results.

Before this module the CLI had two row shapes for one query. The operation
returned a ``SessionListRowPayload`` document; the CLI had its own idea of what
a session row looked like; and three functions in ``archive_query.py``
(``_normalize_daemon_list_item``, ``_emit_daemon_list_payload``,
``_emit_daemon_search_payload``, plus the degraded variant) existed purely to
carry a row from the first shape to the second and to re-derive envelope fields
the operation had already decided.

The operation row is now the CLI row: ``operations.daemon_reads._session_list_row``
projects the declared row vocabulary once, at the boundary that builds it, and
this module renders exactly what it is handed. Nothing here reshapes a row,
re-counts a page, or re-decides a terminal outcome.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

import click

from polylogue.cli.render.outcome import emit_empty_page, exit_for
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome, render_outcome_line

if TYPE_CHECKING:
    from polylogue.cli.shared.types import AppEnv
    from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload

TextLine = Callable[[dict[str, object]], str]

#: The environment whose execute/render phase timings this renderer reports.
#: Bound once per invocation by the root query adapter so every emitter below
#: -- list page, ranked page, aggregate, unit rows -- attributes its own bytes
#: without each of them threading an ``AppEnv`` through.
TIMING_ENV: ContextVar[AppEnv | None] = ContextVar("cli_render_timing_env", default=None)


def _object_int(value: object) -> int:
    try:
        return int(cast(int, value))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Field projection and machine formats
# ---------------------------------------------------------------------------


def selected_fields(fields: str | None) -> frozenset[str] | None:
    if not fields:
        return None
    selected = frozenset(field.strip() for field in fields.split(",") if field.strip())
    return selected or None


def project_payload(payload: dict[str, object], fields: str | None) -> dict[str, object]:
    selected = selected_fields(fields)
    if selected is None:
        return dict(payload)
    return {key: value for key, value in payload.items() if key in selected}


def csv_text(items: list[dict[str, object]]) -> str:
    if not items:
        return ""
    fields = list(items[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(items)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Text lines
# ---------------------------------------------------------------------------


def attached_units_suffix(item: dict[str, object]) -> str:
    """Render a compact ``[+unit:N]`` summary of attached projection units."""

    attached = item.get("attached_units")
    if not isinstance(attached, dict) or not attached:
        return ""
    parts = [f"{unit}:{len(rows)}" for unit, rows in attached.items() if isinstance(rows, list)]
    return f"  [+{' '.join(parts)}]" if parts else ""


def summary_line_renderer(items: list[dict[str, object]]) -> TextLine:
    """Bind one identity frame to the whole rendered result set."""
    from polylogue.rendering.identity import identity_frame

    frame = identity_frame(str(item.get("id", "")) for item in items)

    def render(item: dict[str, object]) -> str:
        from polylogue.surfaces.query_rows import session_row

        row = session_row(item)
        identity = frame.display(row.id)
        line = (
            f"{identity:{frame.column_width}s}  {(row.date or 'unknown'):10s}  "
            f"{row.origin:24s}  {row.title} ({row.message_count} msgs)"
        )
        return line + attached_units_suffix(item)

    return render


def hit_line(item: dict[str, object]) -> str:
    session = item.get("session")
    match = item.get("match")
    if not isinstance(session, dict) or not isinstance(match, dict):
        return str(item)
    from polylogue.archive.query.search_hits import bound_search_snippet
    from polylogue.surfaces.query_rows import session_row

    row = session_row(session)
    snippet = bound_search_snippet(match.get("snippet"))
    line = f"{match['rank']}. {row.origin}  {row.title}  {snippet or ''}"
    return line + attached_units_suffix(item)


def stats_by_line(item: dict[str, object]) -> str:
    return f"{item['group']}: {item['count']}"


# ---------------------------------------------------------------------------
# The row emitter
# ---------------------------------------------------------------------------


def emit_rows(
    envelope: dict[str, object],
    items: list[dict[str, object]],
    *,
    output_format: str,
    text_line: TextLine,
    fields: str | None,
    item_key: str = "items",
) -> None:
    """Write one terminal envelope's rows in the requested format.

    The missing-outcome ``ValueError`` is this renderer's precondition, not a
    fallback: a terminal envelope that reached Seam B without the outcome its
    operation decided is a bug upstream, and inventing one here would be the
    second outcome authority the design exists to remove.
    """
    if "outcome" not in envelope:
        raise ValueError("terminal row envelope missing canonical outcome")
    outcome = OutcomeEnvelope.model_validate(envelope["outcome"])
    env = TIMING_ENV.get()
    if env is not None:
        env.finish_timing("execute")
        env.begin_timing("render")
    try:
        rendered_items = [project_payload(item, fields) for item in items]
        if output_format == "json":
            projected_envelope = {**envelope, item_key: rendered_items}
            click.echo(json.dumps(projected_envelope, indent=2, sort_keys=True))
            return
        if output_format == "ndjson":
            for item in rendered_items:
                click.echo(json.dumps(item, sort_keys=True))
            return
        if output_format == "csv":
            click.echo(csv_text(rendered_items), nl=False)
            return
        if output_format == "yaml":
            import yaml

            projected_envelope = {**envelope, item_key: rendered_items}
            click.echo(yaml.safe_dump(projected_envelope, sort_keys=False, allow_unicode=True), nl=False)
            return
        if output_format not in {"markdown", "plaintext"}:
            raise click.UsageError(f"Root query does not support --format {output_format}.")
        outcome_line = render_outcome_line(outcome)
        if outcome_line is not None:
            click.echo(outcome_line)
        click.echo("\n".join(text_line(item) for item in items))
    finally:
        if env is not None:
            env.finish_timing("render")


# ---------------------------------------------------------------------------
# The session page
# ---------------------------------------------------------------------------


def attach_projected_units(items: list[dict[str, object]], attached: object) -> None:
    """Fold the page-level ``with <unit>`` projection back onto its rows.

    The operation returns the projection once per page, keyed by unit and then
    by session, because that is how it was fetched -- one bounded query per
    unit rather than one per row.  The renderer reads it per row, so the join
    happens here, and every requested unit names every row (an empty list is a
    real answer, a missing key would look like "not projected").
    """

    if not isinstance(attached, Mapping) or not attached:
        return
    for item in items:
        session_id = str(item.get("id") or "")
        projected = {
            str(unit): list(rows.get(session_id, ()) if isinstance(rows, Mapping) else ())
            for unit, rows in attached.items()
        }
        if projected:
            item["attached_units"] = projected


def _page_rows(payload: Mapping[str, object], key: str) -> list[dict[str, object]]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _page_envelope(
    payload: Mapping[str, object],
    *,
    mode: str,
    rows: list[dict[str, object]],
    offset: int,
    limit: int,
    origin: str | None,
    source: str,
) -> dict[str, object]:
    """Build the terminal envelope from what the operation reported.

    ``total``/``limit``/``total_unit``/``next_cursor``/``outcome`` are the
    operation's facts. In particular the effective limit is the one the
    operation applied -- a daemon may clamp a request for 5000 to its own cap,
    and continuing from the requested number rendered a truncated first page as
    complete.
    """

    from polylogue.archive.query.spec import session_count_unit_label

    # A ``None`` total is the operation's honest answer, not a missing field: a
    # vector lane exposes a bounded nearest-neighbour page and deliberately
    # reports no archive-wide cardinality (``daemon_reads._search_payload``).
    # Substituting ``len(rows)`` turned that into a confident "3 of 3"
    # (polylogue-jfabc), so the unknown is carried through instead.
    raw_total = payload.get("total")
    total: int | None = _object_int(raw_total) if raw_total is not None else None
    effective_limit = _object_int(payload.get("limit") or len(rows) or limit)
    total_unit = payload.get("total_unit")
    next_cursor = payload.get("next_cursor")
    # The operation decides continuation (``daemon_reads.page_next_offset``);
    # the renderer only reports what it decided.
    raw_next_offset = payload.get("next_offset")
    envelope: dict[str, object] = {
        "mode": mode,
        "origin": origin,
        "items": rows,
        "total": total,
        "total_unit": total_unit if isinstance(total_unit, str) else session_count_unit_label(True),
        "limit": effective_limit,
        "offset": offset,
        "next_offset": raw_next_offset if isinstance(raw_next_offset, int) else None,
        # The operation mints the ranked continuation cursor itself
        # (``build_search_envelope`` -> ``build_search_cursor``); dropping it
        # here made ranked pages non-continuable by transport.
        "next_cursor": next_cursor if isinstance(next_cursor, str) else None,
        "source": source,
    }
    outcome = payload.get("outcome")
    envelope["outcome"] = (
        dict(outcome)
        if isinstance(outcome, Mapping)
        else decide_outcome(matched=total if total is not None else len(rows)).to_dict()
    )
    return envelope


def emit_session_list_page(
    payload: Mapping[str, object],
    *,
    limit: int,
    offset: int,
    output_format: str,
    origin: str | None,
    fields: str | None,
    source: str = "daemon",
) -> None:
    """Render an unranked session page exactly as the operation returned it."""

    rows = _page_rows(payload, "items")
    attach_projected_units(rows, payload.get("attached_units"))
    envelope = _page_envelope(payload, mode="list", rows=rows, offset=offset, limit=limit, origin=origin, source=source)
    emit_rows(envelope, rows, output_format=output_format, text_line=summary_line_renderer(rows), fields=fields)


def emit_session_search_page(
    payload: Mapping[str, object],
    *,
    query: str,
    limit: int,
    offset: int,
    output_format: str,
    origin: str | None,
    fields: str | None,
    typo_hint: str | None = None,
    diagnose_miss: Callable[[], QueryMissDiagnosticsPayload | Mapping[str, object] | None] | None = None,
    source: str = "daemon",
) -> None:
    """Render a ranked session page, including its degraded and empty forms.

    An operation-supplied ``diagnostics`` document always wins. ``diagnose_miss``
    is the caller's fallback, called only when the page is actually empty and
    the operation named no diagnosis: a zero-hit page still owes the operator an
    explanation, and computing one costs a second query nobody wants on a page
    that did match.
    """

    _emit_degraded_search_page(payload, query=query, output_format=output_format, fields=fields, source=source)
    hits = _page_rows(payload, "hits")
    envelope = _page_envelope(
        payload, mode="search", rows=hits, offset=offset, limit=limit, origin=origin, source=source
    )
    envelope["query"] = query
    envelope["retrieval_lane"] = str(payload.get("retrieval_lane") or "dialogue")
    if not hits:
        operation_diagnostics = payload.get("diagnostics")
        diagnostics: QueryMissDiagnosticsPayload | Mapping[str, object] | None
        if isinstance(operation_diagnostics, Mapping):
            diagnostics = operation_diagnostics
        else:
            diagnostics = diagnose_miss() if diagnose_miss is not None else None
        emit_empty_page(envelope, output_format=output_format, typo_hint=typo_hint, diagnostics=diagnostics)
    emit_rows(envelope, hits, output_format=output_format, text_line=hit_line, fields=fields)


def _emit_degraded_search_page(
    payload: Mapping[str, object],
    *,
    query: str,
    output_format: str,
    fields: str | None,
    source: str,
) -> None:
    """Report a search route the operation declared degraded, then exit on it.

    A degraded page is not an empty one: the reason is named, and the exit
    status comes from the degraded outcome rather than from the zero rows.
    """

    route_state = payload.get("route_state")
    if not isinstance(route_state, Mapping) or route_state.get("state") != "degraded":
        return
    reason = str(route_state.get("reason") or "Search index unavailable.")
    outcome = decide_outcome(matched=0, degraded=(reason,))
    envelope: dict[str, object] = {
        "mode": "search",
        "query": query,
        "retrieval_lane": str(payload.get("retrieval_lane") or "dialogue"),
        "items": [],
        "total": None,
        "source": source,
        "route_state": dict(route_state),
        "outcome": outcome.to_dict(),
    }
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        envelope["diagnostics"] = dict(diagnostics)
    if output_format in {"json", "yaml"}:
        emit_rows(envelope, [], output_format=output_format, text_line=hit_line, fields=fields)
    elif output_format in {"ndjson", "csv"}:
        pass
    else:
        outcome_line = render_outcome_line(outcome)
        if outcome_line is not None:
            click.echo(outcome_line, err=True)
        click.echo(reason, err=True)
    exit_for(outcome)


__all__ = [
    "TextLine",
    "attach_projected_units",
    "attached_units_suffix",
    "csv_text",
    "emit_rows",
    "emit_session_list_page",
    "emit_session_search_page",
    "hit_line",
    "project_payload",
    "selected_fields",
    "stats_by_line",
    "summary_line_renderer",
]
