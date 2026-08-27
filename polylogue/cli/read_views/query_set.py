"""Query-set read-view adapter."""

from __future__ import annotations

import io
import json
from pathlib import Path

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.query import project_query_results
from polylogue.cli.query_contracts import QueryExecutionPlan
from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.surfaces.projection_spec import QueryProjectionSpec


def _dialogue_query_set_renderer(
    session: object,
    output_format: str,
    fields: str | None,
    projection_spec: QueryProjectionSpec | None,
) -> str:
    del fields
    from polylogue.archive.session.domain_models import Session
    from polylogue.cli.read_views.standard import _format_dialogue_session

    assert isinstance(session, Session)
    projection = projection_spec.projection if projection_spec is not None else None
    return _format_dialogue_session(session, output_format, projection=projection)


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
) -> None:
    """Render all matched sessions through the query-set read path."""

    if view not in {"summary", "transcript", "dialogue"}:
        _run_registered_view_query_set(
            env,
            request,
            view=view,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            projection_spec=projection_spec,
        )
        return

    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.cli.query_set_read import run_query_set_read

    is_dialogue = view == "dialogue"
    fmt = output_format or ("markdown" if is_dialogue else "ndjson")
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt
    content_projection = ContentProjectionSpec.prose_only() if is_dialogue else None
    renderer = (
        (
            lambda session, output_format, fields: _dialogue_query_set_renderer(
                session, output_format, fields, projection_spec
            )
        )
        if is_dialogue
        else None
    )

    if destination == "file":
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
                content_projection=content_projection,
                renderer=renderer,
            )
        finally:
            click.echo = _orig_echo
        rendered = buf.getvalue()
        from polylogue.cli.read_views.base import _warn_on_secret_candidates

        _warn_on_secret_candidates(env, rendered, label=out_path)
        Path(out_path).write_text(rendered, encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    if destination in ("clipboard", "browser"):
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
                content_projection=content_projection,
                renderer=renderer,
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
        content_projection=content_projection,
        renderer=renderer,
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
) -> None:
    """Run a non-session-list view once per selected session.

    The generic session formatter is suitable for summary/transcript/dialogue
    exports only. Other read views own pagination and payload semantics in
    their registered handlers, so query-set reads narrow the request to each
    selected session and reuse those handlers instead of silently exporting a
    whole ``Session`` object.
    """
    from polylogue.cli.read_view_handlers import read_view_options_for_view, run_read_view

    spec = request.query_spec()
    plan = QueryExecutionPlan.from_params(request.query_params())
    sessions = run_coroutine_sync(env.polylogue.list_sessions_for_spec(spec))
    sessions = project_query_results(sessions, plan)
    projection = projection_spec.projection if projection_spec is not None else None
    option_values: dict[str, object] = {}
    if projection is not None:
        option_values.update(
            {
                "limit": projection.body_limit,
                "offset": projection.body_offset or 0,
                "window_hours": projection.neighbor_window_hours or 24,
            }
        )
    options = read_view_options_for_view(view, option_values)
    render_format = "json" if output_format in (None, "ndjson", "jsonl") else output_format
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

    if render_format == "json":
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
        output_format=render_format or "markdown",
    )


__all__ = ["run_query_set_read_view"]
