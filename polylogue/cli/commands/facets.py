"""Top-level facets command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from polylogue.cli.query_verbs import emit_facets_response
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.surfaces.payloads import FacetsResponse


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

    daemon_response = _fetch_daemon_facets(
        env,
        query_text=query_text,
        origin=origin,
        include_deferred=include_deferred,
        no_idf=no_idf,
        disabled=bool(ctx.parent and ctx.parent.params.get("no_daemon")),
    )
    emit_facets_response(daemon_response, output_format="json" if json_output else output_format)


def _fetch_daemon_facets(
    env: AppEnv,
    *,
    query_text: str | None,
    origin: str | None,
    include_deferred: bool,
    no_idf: bool,
    disabled: bool,
) -> FacetsResponse:
    """Use the same supplied-reader facet operation for both serving modes."""
    from polylogue.cli.operation_kernel import configured_read_operation
    from polylogue.cli.shared.helpers import load_effective_config
    from polylogue.config import load_polylogue_config
    from polylogue.surfaces.payloads import FacetsResponse

    settings = load_polylogue_config()
    config = load_effective_config(env)
    params: dict[str, object] = {
        "query": query_text or "",
        "include_deferred": include_deferred,
        "no_idf": no_idf,
    }
    if origin:
        params["origin"] = origin
    result = configured_read_operation(
        config,
        "facets",
        {"params": params},
        daemon_disabled=disabled or settings.no_daemon or settings.daemon_client_mode == "off",
    )
    return FacetsResponse.model_validate(result.value)
