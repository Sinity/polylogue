"""Embedding status payload and rendering helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

import click
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from polylogue.config import Config


class _HasConfig(Protocol):
    @property
    def config(self) -> Config: ...


class RetrievalBandPayload(TypedDict, total=False):
    ready: bool
    status: str
    materialized_rows: int
    source_rows: int
    materialized_documents: int
    source_documents: int


class EmbeddingStatusPayload(TypedDict):
    status: str
    total_conversations: int
    embedded_conversations: int
    embedded_messages: int
    pending_conversations: int
    embedding_coverage_percent: float
    retrieval_ready: bool
    freshness_status: str
    stale_messages: int
    messages_missing_provenance: int
    oldest_embedded_at: str | None
    newest_embedded_at: str | None
    embedding_models: dict[str, int]
    embedding_dimensions: dict[int, int]
    retrieval_bands: dict[str, dict[str, object]]


def _payload_int(value: object) -> int:
    """Coerce loosely typed payload counters to ints for display."""
    return int(value) if isinstance(value, bool | int | float | str) else 0


def embedding_status_payload(env: _HasConfig) -> EmbeddingStatusPayload:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.storage.backends.connection import open_read_connection
    from polylogue.storage.embedding_stats import read_embedding_stats_sync

    with open_read_connection(env.config.db_path) as conn:
        total_convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        embedding_stats = read_embedding_stats_sync(conn)

    embedded_convs = embedding_stats.embedded_conversations
    embedded_msgs = embedding_stats.embedded_messages
    pending = embedding_stats.pending_conversations or max(total_convs - embedded_convs, 0)
    coverage = (embedded_convs / total_convs * 100) if total_convs > 0 else 0
    if total_convs <= 0:
        status = "empty"
    elif embedded_convs <= 0:
        status = "none"
    elif pending > 0:
        status = "partial"
    else:
        status = "complete"
    freshness_status = status
    if embedding_stats.embedded_messages > 0 and (
        embedding_stats.stale_messages > 0 or embedding_stats.messages_missing_provenance > 0
    ):
        freshness_status = "stale"

    return {
        "status": status,
        "total_conversations": int(total_convs),
        "embedded_conversations": int(embedded_convs),
        "embedded_messages": int(embedded_msgs),
        "pending_conversations": int(pending),
        "embedding_coverage_percent": round(float(coverage), 1),
        "retrieval_ready": bool(embedded_msgs > embedding_stats.stale_messages),
        "freshness_status": freshness_status,
        "stale_messages": int(embedding_stats.stale_messages),
        "messages_missing_provenance": int(embedding_stats.messages_missing_provenance),
        "oldest_embedded_at": embedding_stats.oldest_embedded_at,
        "newest_embedded_at": embedding_stats.newest_embedded_at,
        "embedding_models": embedding_stats.model_counts,
        "embedding_dimensions": embedding_stats.dimension_counts,
        "retrieval_bands": embedding_stats.retrieval_bands,
    }


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
    if payload["oldest_embedded_at"] or payload["newest_embedded_at"]:
        click.echo(
            f"  Embedded at:           {payload['oldest_embedded_at'] or '-'} -> {payload['newest_embedded_at'] or '-'}"
        )
    if payload["embedding_models"]:
        click.echo(
            f"  Models:                {', '.join(f'{name} ({count})' for name, count in payload['embedding_models'].items())}"
        )
    if payload["embedding_dimensions"]:
        click.echo(
            f"  Dimensions:            {', '.join(f'{dimension} ({count})' for dimension, count in payload['embedding_dimensions'].items())}"
        )
    if payload["retrieval_bands"]:
        click.echo("  Retrieval bands:")
        for band_name, band in payload["retrieval_bands"].items():
            status_text = "ready" if band.get("ready") else str(band.get("status", "pending"))
            click.echo(
                f"    {band_name}: {status_text}; "
                f"rows={_payload_int(band.get('materialized_rows', 0)):,}/{_payload_int(band.get('source_rows', 0)):,}; "
                f"docs={_payload_int(band.get('materialized_documents', 0)):,}/{_payload_int(band.get('source_documents', 0)):,}"
            )


def show_embedding_stats(env: _HasConfig, *, json_output: bool = False) -> None:
    """Display embedding statistics."""
    render_embedding_stats(embedding_status_payload(env), json_output=json_output)
