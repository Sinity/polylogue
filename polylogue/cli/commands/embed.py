"""Embedding generation command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from polylogue.cli.embed_runtime import embed_batch, embed_single
from polylogue.cli.embed_stats import embedding_status_payload, render_embedding_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv

_embed_single = embed_single
_embed_batch = embed_batch


def _embedding_status_payload(env: AppEnv) -> dict[str, object]:
    return embedding_status_payload(env)


def _show_embedding_stats(env: AppEnv, *, json_output: bool = False) -> None:
    render_embedding_stats(_embedding_status_payload(env), json_output=json_output)


@click.command("embed")
@click.option(
    "--conversation", "-c",
    type=str,
    default=None,
    help="Embed a specific conversation by ID",
)
@click.option(
    "--model",
    type=click.Choice(["voyage-4", "voyage-4-large", "voyage-4-lite"]),
    default="voyage-4",
    help="Voyage AI model: voyage-4 (default), voyage-4-large, voyage-4-lite",
)
@click.option(
    "--rebuild", "-r",
    is_flag=True,
    help="Re-embed all conversations (ignore existing embeddings)",
)
@click.option(
    "--stats", "-s",
    is_flag=True,
    help="Show embedding statistics only",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit embedding statistics as JSON (requires --stats)",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=None,
    help="Maximum number of conversations to embed",
)
@click.pass_obj
def embed_command(
    env: AppEnv,
    conversation: str | None,
    model: str,
    rebuild: bool,
    stats: bool,
    json_output: bool,
    limit: int | None,
) -> None:
    """Generate semantic embeddings for conversations."""
    import os

    from polylogue.storage.search_providers import create_vector_provider

    if json_output and not stats:
        click.echo("Error: --json requires --stats", err=True)
        raise click.Abort()

    voyage_key = os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not stats:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key  (or POLYLOGUE_VOYAGE_API_KEY)", err=True)
        raise click.Abort()

    if stats:
        _show_embedding_stats(env, json_output=json_output)
        return

    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        click.echo("sqlite-vec is not available (ensure it is in your Nix flake or virtualenv)", err=True)
        raise click.Abort()

    if model != "voyage-4":
        vec_provider.model = model

    repo = env.repository

    if conversation:
        _embed_single(env, repo, vec_provider, conversation)
        return

    _embed_batch(env, repo, vec_provider, rebuild=rebuild, limit=limit)


__all__ = [
    "_embed_batch",
    "_embed_single",
    "_embedding_status_payload",
    "_show_embedding_stats",
    "embed_command",
]
