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


def _render_latest_catchup_run(payload: EmbeddingStatusPayload) -> None:
    latest = payload["latest_catchup_run"]
    if latest is None:
        return
    click.echo("  Latest catch-up:")
    click.echo(
        f"    {latest['status']}; processed={latest['processed_sessions']:,}/{latest['planned_sessions']:,} convs; "
        f"embedded={latest['embedded_messages']:,} msgs; errors={latest['error_count']:,}; "
        f"est. cost ~${latest['estimated_cost_usd']:.4f}"
    )
    if latest["stop_reason"]:
        click.echo(f"    reason: {latest['stop_reason']}")


def _render_next_actions(payload: EmbeddingStatusPayload) -> None:
    action = payload["next_action"]
    command = action["command"]
    if command is None:
        return
    click.echo(f"  Next action:           {action['code']}")
    click.echo(f"  Reason:                {action['reason']}")
    click.echo(f"  Command:               {command}")


def render_embedding_stats(payload: EmbeddingStatusPayload, *, json_output: bool = False) -> None:
    """Render an embedding statistics payload."""
    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("\nEmbedding Statistics")
    click.echo(f"  Config enabled:        {'yes' if payload['config_enabled'] else 'no'}")
    click.echo(f"  Voyage key:            {'present' if payload['has_voyage_api_key'] else 'missing'}")
    click.echo(f"  Daemon stage:          {'enabled' if payload['daemon_stage_enabled'] else 'disabled'}")
    click.echo(f"  Configured model:      {payload['configured_model']} ({payload['configured_dimension']}d)")
    if payload["monthly_cost_cap_usd"] > 0:
        click.echo(f"  Monthly cost cap:      ${payload['monthly_cost_cap_usd']:.2f}")
    else:
        click.echo("  Monthly cost cap:      unbounded")
    click.echo(f"  Status:                {payload['status']}")
    click.echo(f"  Total sessions:   {payload['total_sessions']}")
    click.echo(f"  Embedded sessions:{payload['embedded_sessions']:>4}")
    click.echo(f"  Embedded messages:     {payload['embedded_messages']}")
    click.echo(f"  Coverage:              {payload['embedding_coverage_percent']:.1f}%")
    pending_messages = (
        f"{payload['pending_messages']} msgs"
        if payload["pending_messages_exact"]
        else "msgs not calculated (use --detail)"
    )
    click.echo(f"  Pending:               {payload['pending_sessions']} convs, {pending_messages}")
    click.echo(f"  Retrieval ready:       {'yes' if payload['retrieval_ready'] else 'no'}")
    click.echo(f"  Freshness:             {payload['freshness_status']}")
    click.echo(f"  Stale messages:        {payload['stale_messages']}")
    click.echo(f"  Missing provenance:    {payload['messages_missing_provenance']}")
    click.echo(f"  Estimated total cost:  ~${payload['total_estimated_cost_usd']:.2f}")
    _render_next_actions(payload)
    _render_embedding_window(payload)
    _render_named_counts("Models", payload["embedding_models"])
    _render_named_counts("Dimensions", payload["embedding_dimensions"])
    _render_latest_catchup_run(payload)
    _render_retrieval_bands(payload)


def show_embedding_stats(env: object, *, json_output: bool = False, detail: bool = False) -> None:
    """Display embedding statistics."""
    render_embedding_stats(
        embedding_status_payload(env, include_retrieval_bands=detail, include_detail=detail),  # type: ignore[arg-type]
        json_output=json_output,
    )


__all__ = [
    "EmbeddingStatusPayload",
    "RetrievalBandPayload",
    "embedding_status_payload",
    "render_embedding_stats",
    "show_embedding_stats",
]
