"""Top-level facets command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.query_verbs import emit_facets_response
from polylogue.cli.root_request import RootModeRequest
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
    )
    if daemon_response is not None:
        emit_facets_response(daemon_response, output_format="json" if json_output else output_format)
        return

    spec = RootModeRequest.from_params(
        {
            "query": (query_text,) if query_text else (),
            "origin": origin,
        }
    ).query_spec()
    response = run_coroutine_sync(env.polylogue.facets(spec, include_idf=not no_idf, include_deferred=include_deferred))
    emit_facets_response(response, output_format="json" if json_output else output_format)


def _fetch_daemon_facets(
    env: AppEnv,
    *,
    query_text: str | None,
    origin: str | None,
    include_deferred: bool,
    no_idf: bool,
) -> FacetsResponse | None:
    """Use the config-matched UDS daemon for read-only facets when lossless."""

    if no_idf or os.environ.get("POLYLOGUE_NO_DAEMON", "").lower() in {"1", "true", "yes", "on"}:
        return None
    if os.environ.get("POLYLOGUE_DAEMON", "").lower() == "off":
        return None
    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.cli.shared.helpers import load_effective_config
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.surfaces.payloads import FacetsResponse
    from polylogue.version import POLYLOGUE_VERSION

    config = load_effective_config(env)
    client = DaemonClient(Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "polylogue" / "daemon.sock")
    if (
        client.probe(
            archive_root=str(config.archive_root),
            index_schema_version=INDEX_SCHEMA_VERSION,
            daemon_version=POLYLOGUE_VERSION,
        )
        is None
    ):
        return None
    params: dict[str, str] = {}
    if query_text:
        params["query"] = query_text
    if origin:
        params["origin"] = origin
    if include_deferred:
        params["include_expensive"] = "1"
    payload = client.request_json("GET", "/api/facets?" + urlencode(params))
    if payload is None:
        return None
    try:
        return FacetsResponse.model_validate(payload)
    except ValueError:
        return None
