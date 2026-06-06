"""Facets command: scoped vs global aggregate counts (#1269)."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv


@click.command("facets")
@click.option("--origin", "-o", "origin", default=None, help="Filter to this origin")
@click.option("--tag", "-t", "tag", default=None, help="Filter to this tag")
@click.option(
    "--query",
    "-q",
    "query",
    default=None,
    help="Free-text FTS query for the scoped view",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--no-idf",
    "no_idf",
    is_flag=True,
    default=False,
    help="Skip inverse-document-frequency weighting",
)
@click.pass_obj
def facets_command(
    env: AppEnv,
    origin: str | None,
    tag: str | None,
    query: str | None,
    output_format: str,
    no_idf: bool,
) -> None:
    """Show scoped and global facet aggregates.

    With no filters the scoped and global counts are identical. When
    ``--origin``, ``--tag``, or ``--query`` narrows the view, the
    scoped buckets describe the filtered slice while the global
    buckets describe the whole archive (#1269 / slice D of #873).

    \b
    Examples:
        polylogue facets                           # Global counts only
        polylogue facets -o chatgpt-export         # Scoped to ChatGPT exports
        polylogue facets -t draft -f json          # JSON, scoped to "draft"
        polylogue facets -q "vector store" --no-idf
    """
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.archive.query.spec import SessionQuerySpec

    params: dict[str, object] = {}
    if origin:
        params["origin"] = origin
    if tag:
        params["tag"] = tag
    if query:
        params["query"] = query

    spec = SessionQuerySpec.from_params(params) if params else None

    response = run_coroutine_sync(env.polylogue.facets(spec, include_idf=not no_idf))

    if output_format == "json":
        click.echo(json.dumps(response.model_dump(mode="json", by_alias=True), indent=2))
        return

    del env  # printed via click.echo for backend-agnostic output
    scope_label = "scoped" if response.scoped_to_query else "global"
    click.echo(f"Facets ({scope_label}):")
    click.echo(f"  scoped:   {response.scoped.total_sessions} sessions, {response.scoped.total_messages} messages")
    click.echo(f"  global:   {response.global_.total_sessions} sessions, {response.global_.total_messages} messages")
    click.echo("")
    click.echo("Origins (scoped / global):")
    keys = sorted(set(response.scoped.origins) | set(response.global_.origins))
    for key in keys:
        s = response.scoped.origins.get(key, 0)
        g = response.global_.origins.get(key, 0)
        click.echo(f"  {key:24s} {s:>6d} / {g:>6d}")
    if response.scoped.tags or response.global_.tags:
        click.echo("")
        click.echo("Tags (scoped / global):")
        tag_keys = sorted(set(response.scoped.tags) | set(response.global_.tags))
        for key in tag_keys:
            s = response.scoped.tags.get(key, 0)
            g = response.global_.tags.get(key, 0)
            click.echo(f"  {key:24s} {s:>6d} / {g:>6d}")
    if response.idf:
        click.echo("")
        click.echo("IDF (higher = rarer, partitions more strongly):")
        for family, values in response.idf.items():
            click.echo(f"  [{family}]")
            for value, weight in sorted(values.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {value:24s} {weight:7.3f}")
