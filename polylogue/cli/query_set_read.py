"""Query-set reading for every session matching the parent filter chain."""

from __future__ import annotations

import json
from collections.abc import Callable

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.archive.session.domain_models import Session
from polylogue.cli.query import project_query_results
from polylogue.cli.query_contracts import QueryExecutionPlan
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.rendering.formatting import format_session

_PER_LINE_FORMATS = frozenset({"jsonl"})
_SEPARATED_FORMATS = frozenset({"markdown", "obsidian", "org", "plaintext", "html", "yaml"})
_ARRAY_FORMATS = frozenset({"json"})


def _emit_jsonl(rendered_json: str) -> None:
    """Collapse a multi-line session JSON to a single line."""
    click.echo(json.dumps(json.loads(rendered_json), separators=(",", ":")))


def run_query_set_read(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str,
    fields: str | None,
    content_projection: ContentProjectionSpec | None = None,
    renderer: Callable[[Session, str, str | None], str] | None = None,
) -> None:
    """Render every matched session in one process.

    The parent filter chain (origin, since, path, message-role, etc.) is
    consumed via the ``RootModeRequest``. Output is streamed to stdout.
    """
    spec = request.query_spec()
    plan = QueryExecutionPlan.from_params(request.query_params())
    sessions = run_coroutine_sync(env.polylogue.list_sessions_for_spec(spec, content_projection=content_projection))
    sessions = project_query_results(sessions, plan)

    render_format = "json" if output_format == "jsonl" else output_format

    if output_format in _ARRAY_FORMATS:
        click.echo("[")
    for index, session in enumerate(sessions):
        rendered = (
            renderer(session, render_format, fields)
            if renderer is not None
            else format_session(session, render_format, fields)
        )
        if output_format in _PER_LINE_FORMATS:
            _emit_jsonl(rendered)
            continue
        if output_format in _ARRAY_FORMATS:
            suffix = "," if index < len(sessions) - 1 else ""
            click.echo(f"{rendered}{suffix}")
            continue
        if index > 0 and output_format in _SEPARATED_FORMATS:
            click.echo("\n---\n")
        click.echo(rendered)
    if output_format in _ARRAY_FORMATS:
        click.echo("]")


__all__ = ["run_query_set_read"]
