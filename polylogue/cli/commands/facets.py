"""Top-level facets command."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.query_verbs import emit_facets_response
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


@click.command("facets")
@click.option(
    "-q",
    "--query",
    "query_text",
    metavar="TEXT",
    help="Scope facets to a full-text query. Omit for global archive facets.",
)
@click.option(
    "-o",
    "--origin",
    metavar="ORIGIN",
    help="Scope facets to one archive origin, for example chatgpt-export.",
)
@click.option(
    "--include-deferred",
    is_flag=True,
    help="Materialize expensive detail families such as repos, roles, material origins, and message types.",
)
@click.option("--no-idf", is_flag=True, help="Omit inverse-document-frequency weights from JSON output.")
@click.option("-f", "--format", "output_format", type=click.Choice(["text", "json"]), default="text", show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Alias for --format json.")
@click.pass_context
def facets_command(
    ctx: click.Context,
    *,
    query_text: str | None,
    origin: str | None,
    include_deferred: bool,
    no_idf: bool,
    output_format: str,
    json_output: bool,
) -> None:
    """Show global or scoped archive facet families.

    ``polylogue facets`` is the direct command for the same typed facet
    envelope used by ``find QUERY then analyze --facets``. By default it keeps
    expensive/noisy detail families deferred and reports their state explicitly
    instead of rendering empty buckets as authoritative facts.
    """

    env = ctx.obj if isinstance(ctx.obj, AppEnv) else AppEnv()

    spec = RootModeRequest.from_params(
        {
            "query": (query_text,) if query_text else (),
            "origin": origin,
        }
    ).query_spec()
    response = run_coroutine_sync(env.polylogue.facets(spec, include_idf=not no_idf, include_deferred=include_deferred))
    emit_facets_response(response, output_format="json" if json_output else output_format)
