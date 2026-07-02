"""Query-set read-view adapter."""

from __future__ import annotations

import io
from pathlib import Path

import click

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
        Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    run_query_set_read(
        env,
        request,
        output_format=bulk_fmt,
        fields=fields,
        content_projection=content_projection,
        renderer=renderer,
    )


__all__ = ["run_query_set_read_view"]
