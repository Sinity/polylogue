"""Bulk export of every conversation matching the parent filter chain."""

from __future__ import annotations

import json

import click

from polylogue.cli.query import project_query_results
from polylogue.cli.query_contracts import QueryExecutionPlan
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.types import AppEnv
from polylogue.rendering.formatting import format_conversation
from polylogue.sync_bridge import run_coroutine_sync

_PER_LINE_FORMATS = frozenset({"jsonl"})
_SEPARATED_FORMATS = frozenset({"markdown", "obsidian", "org", "plaintext", "html", "yaml"})
_ARRAY_FORMATS = frozenset({"json"})


def _emit_jsonl(rendered_json: str) -> None:
    """Collapse a multi-line conversation JSON to a single line."""
    click.echo(json.dumps(json.loads(rendered_json), separators=(",", ":")))


def run_bulk_export(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str,
    fields: str | None,
) -> None:
    """Render every matched conversation in one process.

    The parent filter chain (provider, since, path, message-role, etc.) is
    consumed via the ``RootModeRequest``. Output is streamed to stdout.
    """
    spec = request.query_spec()
    plan = QueryExecutionPlan.from_params(request.query_params())
    conversations = run_coroutine_sync(env.operations.query_conversations(spec))
    conversations = project_query_results(conversations, plan)

    render_format = "json" if output_format == "jsonl" else output_format

    if output_format in _ARRAY_FORMATS:
        click.echo("[")
    for index, conversation in enumerate(conversations):
        rendered = format_conversation(conversation, render_format, fields)
        if output_format in _PER_LINE_FORMATS:
            _emit_jsonl(rendered)
            continue
        if output_format in _ARRAY_FORMATS:
            suffix = "," if index < len(conversations) - 1 else ""
            click.echo(f"{rendered}{suffix}")
            continue
        if index > 0 and output_format in _SEPARATED_FORMATS:
            click.echo("\n---\n")
        click.echo(rendered)
    if output_format in _ARRAY_FORMATS:
        click.echo("]")


__all__ = ["run_bulk_export"]
