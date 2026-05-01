"""CLI presentation for embedding statistics."""

from __future__ import annotations

import json
from collections.abc import Mapping

import click

from polylogue.storage.embeddings.status_payload import (
    EmbeddingStatusPayload,
    RetrievalBandPayload,
    embedding_status_payload,
)


def _payload_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _render_embedding_window(payload: EmbeddingStatusPayload) -> None:
    if payload["oldest_embedded_at"] or payload["newest_embedded_at"]:
        click.echo(
            f"  Embedded at:           {payload['oldest_embedded_at'] or '-'} -> {payload['newest_embedded_at'] or '-'}"
        )


def _render_named_counts(label: str, values: Mapping[str, int] | Mapping[int, int]) -> None:
    if values:
        click.echo(f"  {label}:                {', '.join(f'{name} ({count})' for name, count in values.items())}")


def _render_retrieval_bands(payload: EmbeddingStatusPayload) -> None:
    if not payload["retrieval_bands"]:
        return
    click.echo("  Retrieval bands:")
    for band_name, band in payload["retrieval_bands"].items():
        status_text = "ready" if band.get("ready") else str(band.get("status", "pending"))
        click.echo(
            f"    {band_name}: {status_text}; "
            f"rows={_payload_int(band.get('materialized_rows', 0)):,}/{_payload_int(band.get('source_rows', 0)):,}; "
            f"docs={_payload_int(band.get('materialized_documents', 0)):,}/{_payload_int(band.get('source_documents', 0)):,}"
        )


def render_embedding_stats(payload: EmbeddingStatusPayload, *, json_output: bool = False) -> None:
    """Render an embedding statistics payload."""
    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("\nEmbedding Statistics")
    click.echo(f"  Status:                {payload['status']}")
    click.echo(f"  Total conversations:   {payload['total_conversations']}")
    click.echo(f"  Embedded conversations:{payload['embedded_conversations']:>4}")
    click.echo(f"  Embedded messages:     {payload['embedded_messages']}")
    click.echo(f"  Coverage:              {payload['embedding_coverage_percent']:.1f}%")
    click.echo(f"  Pending:               {payload['pending_conversations']}")
    click.echo(f"  Retrieval ready:       {'yes' if payload['retrieval_ready'] else 'no'}")
    click.echo(f"  Freshness:             {payload['freshness_status']}")
    click.echo(f"  Stale messages:        {payload['stale_messages']}")
    click.echo(f"  Missing provenance:    {payload['messages_missing_provenance']}")
    _render_embedding_window(payload)
    _render_named_counts("Models", payload["embedding_models"])
    _render_named_counts("Dimensions", payload["embedding_dimensions"])
    _render_retrieval_bands(payload)


def show_embedding_stats(env: object, *, json_output: bool = False) -> None:
    """Display embedding statistics."""
    render_embedding_stats(embedding_status_payload(env), json_output=json_output)  # type: ignore[arg-type]


__all__ = [
    "EmbeddingStatusPayload",
    "RetrievalBandPayload",
    "embedding_status_payload",
    "render_embedding_stats",
    "show_embedding_stats",
]
