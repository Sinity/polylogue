"""Query-set read-view adapter."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.query import project_query_results
from polylogue.cli.query_contracts import QueryExecutionPlan
from polylogue.cli.read_views.base import ReadViewInvocation, ReadViewOptions
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.surfaces.projection_spec import QueryProjectionSpec, RenderDestination

#: Views whose bulk renderer owns the whole selection, so a query-set read
#: hands them the matched set directly instead of dispatching a handler.
_BULK_RENDERED_VIEWS = frozenset({"summary", "transcript"})


def run_query_set_read_view(
    env: AppEnv,
    request: RootModeRequest,
    *,
    view: str = "",
    output_format: str | None,
    fields: str | None,
    destination: str,
    out_path: str | None,
    projection_spec: QueryProjectionSpec | None = None,
    option_values: Mapping[str, object] | None = None,
) -> None:
    """Render all matched sessions through the query-set read path."""

    if view == "dialogue":
        _run_dialogue_query_set(
            env,
            request,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            projection_spec=projection_spec,
        )
        return

    if view not in _BULK_RENDERED_VIEWS:
        if _view_accepts_query_set(view):
            _run_query_set_native_view(
                env,
                request,
                view=view,
                output_format=output_format,
                destination=destination,
                out_path=out_path,
                projection_spec=projection_spec,
                option_values=option_values,
            )
            return
        _run_registered_view_query_set(
            env,
            request,
            view=view,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            projection_spec=projection_spec,
            option_values=option_values,
        )
        return

    from polylogue.cli.query_set_read import run_query_set_read

    fmt = output_format or "ndjson"
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt

    if destination == RenderDestination.FILE:
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")

        buf = io.StringIO()

        def _captured_echo_read_set(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_read_set  # type: ignore[assignment]
        try:
            run_query_set_read(
                env,
                request,
                output_format=bulk_fmt,
                fields=fields,
            )
        finally:
            click.echo = _orig_echo
        rendered = buf.getvalue()
        from polylogue.cli.read_views.base import _warn_on_secret_candidates

        _warn_on_secret_candidates(env, rendered, label=out_path)
        Path(out_path).write_text(rendered, encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    if destination in (RenderDestination.CLIPBOARD, RenderDestination.BROWSER):
        # Query-set rendering writes one document per match. Capture that
        # stream before delivery so clipboard/browser destinations receive the
        # complete set instead of silently falling back to stdout.
        buf = io.StringIO()

        def _captured_echo_read_set(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_read_set  # type: ignore[assignment]
        try:
            run_query_set_read(
                env,
                request,
                output_format=bulk_fmt,
                fields=fields,
            )
        finally:
            click.echo = _orig_echo
        from polylogue.cli.read_views.base import deliver_content

        deliver_content(
            env,
            buf.getvalue(),
            destination=destination,
            out_path=out_path,
            output_format=fmt,
        )
        return

    run_query_set_read(
        env,
        request,
        output_format=bulk_fmt,
        fields=fields,
    )


def _run_dialogue_query_set(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str | None,
    destination: str,
    out_path: str | None,
    projection_spec: QueryProjectionSpec | None,
) -> None:
    """Select through cli.query and project each match through read.dialogue."""

    from polylogue.cli.read_dispatch import daemon_route_disabled
    from polylogue.cli.read_views.base import deliver_content
    from polylogue.cli.read_views.standard import _format_dialogue_session, _read_dialogue_session
    from polylogue.cli.session_rows import query_complete_session_ids, query_session_rows

    spec = request.query_spec()
    disabled = daemon_route_disabled(flag=bool(request.params.get("no_daemon")))
    if spec.limit is None:
        session_ids = query_complete_session_ids(env.config, request, daemon_disabled=disabled)[spec.offset :]
    else:
        rows = query_session_rows(
            env.config,
            request,
            limit=spec.limit,
            offset=spec.offset,
            daemon_disabled=disabled,
        )
        session_ids = [row.session_id for row in rows]
    fmt = output_format or "markdown"
    projection = projection_spec.projection if projection_spec is not None else None
    rendered: list[str] = []
    for session_id in session_ids:
        session = _read_dialogue_session(env, request, session_id, projection)
        if session is not None:
            rendered.append(
                _format_dialogue_session(session, "json" if fmt in {"ndjson", "jsonl"} else fmt, projection=projection)
            )
    if fmt in {"ndjson", "jsonl"}:
        content = "".join(json.dumps(json.loads(part), separators=(",", ":")) + "\n" for part in rendered)
    elif fmt == "json":
        content = json.dumps([json.loads(part) for part in rendered], indent=2) + "\n"
    else:
        content = "\n---\n".join(rendered)
        if content:
            content += "\n"
    deliver_content(env, content, destination=destination, out_path=out_path, output_format=fmt)


def _view_accepts_query_set(view: str) -> bool:
    """Does this view's declaration accept the whole matched selection?"""
    from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA

    metadata = READ_VIEW_HANDLER_METADATA.get(view)
    return metadata is not None and metadata.accepts_query_set


def _ndjson_requested(output_format: str | None) -> bool:
    return (output_format or "").lower() in {"ndjson", "jsonl"}


#: Views whose registered handler renders its own per-record NDJSON stream.
#: For every other view a query-set NDJSON read emits one self-contained
#: document per selected session, which is the same framing contract at the
#: grain that view owns.
_NDJSON_STREAM_VIEWS = frozenset({"messages"})


def _handler_option_values(
    view: str,
    projection_spec: QueryProjectionSpec | None,
    option_values: Mapping[str, object] | None,
) -> ReadViewOptions | None:
    """Build one view's typed options from the projection *and* its Click values.

    The projection carries the shared body window. The view-specific values a
    caller supplied on the command line (``--full``, ``--continuation``,
    ``--around``, ...) are overlaid on top: building options from the
    projection alone silently dropped ``read --all --view messages --full``,
    leaving ``ReadViewMessageOptions.full`` false so every selected session was
    truncated at the handler's page fallback despite the explicit request.
    """
    from polylogue.cli.read_view_handlers import read_view_options_for_view

    projection = projection_spec.projection if projection_spec is not None else None
    values: dict[str, object] = {}
    if projection is not None:
        values.update(
            {
                "limit": projection.body_limit,
                "offset": projection.body_offset or 0,
                "window_hours": projection.neighbor_window_hours or 24,
            }
        )
    for name, value in (option_values or {}).items():
        if value is None and name in values:
            continue
        values[name] = value
    return read_view_options_for_view(view, values)


def _run_query_set_native_view(
    env: AppEnv,
    request: RootModeRequest,
    *,
    view: str,
    output_format: str | None,
    destination: str,
    out_path: str | None,
    projection_spec: QueryProjectionSpec | None,
    option_values: Mapping[str, object] | None,
) -> None:
    """Dispatch a query-set-capable handler once over the original selection.

    ``temporal`` and ``chronicle`` declare ``accepts_query_set`` and build one
    cross-session projection from the whole request. Narrowing the request to
    one ``conv_id`` per invocation turned a single chronology into an array of
    unrelated per-session projections and repeated the query work once per
    match.
    """
    from polylogue.cli.read_view_handlers import run_read_view

    run_read_view(
        env,
        request,
        ReadViewInvocation(
            view=view,
            session_id=None,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            options=_handler_option_values(view, projection_spec, option_values),
            projection_spec=projection_spec,
        ),
    )


def _run_registered_view_query_set(
    env: AppEnv,
    request: RootModeRequest,
    *,
    view: str,
    output_format: str | None,
    destination: str,
    out_path: str | None,
    projection_spec: QueryProjectionSpec | None,
    option_values: Mapping[str, object] | None = None,
) -> None:
    """Run a non-session-list view once per selected session.

    The generic session formatter is suitable for summary/transcript/dialogue
    exports only. Other read views own pagination and payload semantics in
    their registered handlers, so query-set reads narrow the request to each
    selected session and reuse those handlers instead of silently exporting a
    whole ``Session`` object.
    """
    from polylogue.cli.read_view_handlers import run_read_view

    spec = request.query_spec()
    plan = QueryExecutionPlan.from_params(request.query_params())
    sessions = run_coroutine_sync(env.polylogue.list_sessions_for_spec(spec))
    sessions = project_query_results(sessions, plan)
    options = _handler_option_values(view, projection_spec, option_values)
    wants_ndjson = _ndjson_requested(output_format)
    handler_streams_ndjson = wants_ndjson and view in _NDJSON_STREAM_VIEWS
    if handler_streams_ndjson:
        render_format = "ndjson"
    elif output_format in (None, "ndjson", "jsonl"):
        render_format = "json"
    else:
        render_format = output_format
    rendered_parts: list[str] = []
    for session in sessions:
        session_id = str(session.id)
        narrowed = request.with_param_updates(conv_id=session_id).with_query_terms(())
        buf = io.StringIO()

        def _captured_echo(message: object = None, *, captured_buf: io.StringIO = buf, **_kwargs: object) -> None:
            captured_buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_read_view(
                env,
                narrowed,
                ReadViewInvocation(
                    view=view,
                    session_id=session_id,
                    output_format=render_format,
                    destination="stdout",
                    out_path=None,
                    options=options,
                    projection_spec=projection_spec,
                ),
            )
        finally:
            click.echo = _orig_echo
        rendered_parts.append(buf.getvalue().rstrip("\n"))

    # ``--format ndjson`` is a framing contract: one self-contained document
    # per line. Serializing the per-session envelopes as one pretty-printed
    # JSON array gave line-oriented consumers both the wrong framing and the
    # wrong record grain.
    if handler_streams_ndjson:
        content = "\n".join(part for part in rendered_parts if part)
        if content:
            content += "\n"
    elif wants_ndjson:
        lines: list[str] = []
        for rendered in rendered_parts:
            if not rendered:
                continue
            try:
                lines.append(json.dumps(json.loads(rendered), sort_keys=True))
            except json.JSONDecodeError:
                lines.append(json.dumps(rendered))
        content = "\n".join(lines)
        if content:
            content += "\n"
    elif render_format == "json":
        documents: list[object] = []
        for rendered in rendered_parts:
            if not rendered:
                continue
            try:
                documents.append(json.loads(rendered))
            except json.JSONDecodeError:
                documents.append(rendered)
        content = json.dumps(documents, indent=2) + "\n"
    else:
        separator = "\n---\n" if render_format in {"markdown", "plaintext", "yaml"} else "\n"
        content = separator.join(rendered_parts)
        if content:
            content += "\n"

    from polylogue.cli.read_views.base import deliver_content

    deliver_content(
        env,
        content,
        destination=destination,
        out_path=out_path,
        output_format=("ndjson" if wants_ndjson else render_format) or "markdown",
    )


__all__ = ["run_query_set_read_view"]
