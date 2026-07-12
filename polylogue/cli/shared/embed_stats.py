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

_FIELD_WIDTH = 22


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


def _render_field(label: str, value: object) -> None:
    click.echo(f"  {label + ':':<{_FIELD_WIDTH}}{value}")


def _render_embedding_window(payload: EmbeddingStatusPayload) -> None:
    if payload["oldest_embedded_at"] or payload["newest_embedded_at"]:
        _render_field(
            "Embedded at", f"{payload['oldest_embedded_at'] or '-'} -> {payload['newest_embedded_at'] or '-'}"
        )


def _render_named_counts(label: str, values: Mapping[str, int] | Mapping[int, int]) -> None:
    if values:
        _render_field(label, ", ".join(f"{name} ({count})" for name, count in values.items()))


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
    material = payload["latest_material_catchup_run"]
    if material is not None and material["run_id"] != latest["run_id"]:
        click.echo(
            "  Latest material catch-up: "
            f"{material['status']}; processed={material['processed_sessions']:,}/{material['planned_sessions']:,} convs; "
            f"embedded={material['embedded_messages']:,} msgs; errors={material['error_count']:,}; "
            f"est. cost ~${material['estimated_cost_usd']:.4f}"
        )


def _render_next_actions(payload: EmbeddingStatusPayload) -> None:
    action = payload["next_action"]
    command = action["command"]
    if command is None:
        return
    _render_field("Next action", action["code"])
    _render_field("Reason", action["reason"])
    _render_field("Command", command)


def _render_failure_details(payload: EmbeddingStatusPayload) -> None:
    """Render bounded actionable identities for the human status surface."""

    details = payload["failure_details"]
    if not details:
        return
    click.echo("  Active embedding failures:")
    for detail in details:
        refs = ", ".join(detail["message_refs"]) or "session-only"
        click.echo(
            f"    {detail['failure_id']}: {detail['lifecycle_state']}; "
            f"{detail['origin']} / {detail['session_id']}; {detail['provider']} {detail['model']}; "
            f"{detail['error_class']}"
        )
        click.echo(f"      refs: {refs}")
        click.echo(f"      error: {detail['error_message']}")
        click.echo(f"      resolve: {detail['resolution_command']}")


def render_embedding_stats(payload: EmbeddingStatusPayload, *, json_output: bool = False) -> None:
    """Render an embedding statistics payload."""
    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("\nEmbedding Statistics")
    _render_field("Config enabled", "yes" if payload["config_enabled"] else "no")
    _render_field("Voyage key", "present" if payload["has_voyage_api_key"] else "missing")
    _render_field("Daemon stage", "enabled" if payload["daemon_stage_enabled"] else "disabled")
    _render_field("Configured model", f"{payload['configured_model']} ({payload['configured_dimension']}d)")
    if payload["monthly_cost_cap_usd"] > 0:
        _render_field("Monthly cost cap", f"${payload['monthly_cost_cap_usd']:.2f}")
    else:
        _render_field("Monthly cost cap", "unbounded")
    _render_field("Status", payload["status"])
    _render_field("Total sessions", payload["total_sessions"])
    _render_field("Embedded sessions", payload["embedded_sessions"])
    _render_field("Embedded messages", payload["embedded_messages"])
    _render_field("Session coverage", f"{payload['embedding_coverage_percent']:.1f}%")
    candidate_prose_messages = payload.get("candidate_prose_messages")
    message_coverage_percent = payload.get("message_coverage_percent")
    if candidate_prose_messages is not None and message_coverage_percent is not None:
        prefix = "" if payload.get("candidate_prose_messages_exact", False) else "~"
        _render_field(
            "Message coverage",
            f"{prefix}{message_coverage_percent:.1f}% of {candidate_prose_messages:,} candidate prose msgs",
        )
    pending_messages = (
        f"{payload['pending_messages']} msgs" if payload["pending_messages_exact"] else "msgs not calculated"
    )
    _render_field("Pending", f"{payload['pending_sessions']} convs, {pending_messages}")
    _render_field("Retrieval ready", "yes" if payload["retrieval_ready"] else "no")
    _render_field("Freshness", payload["freshness_status"])
    _render_field("Stale messages", payload["stale_messages"])
    _render_field("Missing provenance", payload["messages_missing_provenance"])
    if payload["total_estimated_cost_usd"] is None:
        _render_field("Estimated total cost", "unknown")
    else:
        _render_field("Estimated total cost", f"~${payload['total_estimated_cost_usd']:.2f}")
    _render_next_actions(payload)
    _render_failure_details(payload)
    _render_embedding_window(payload)
    _render_named_counts("Models", payload["embedding_models"])
    _render_named_counts("Dimensions", payload["embedding_dimensions"])
    _render_latest_catchup_run(payload)
    _render_retrieval_bands(payload)


def show_embedding_stats(env: object, *, json_output: bool = False, detail: bool = False) -> None:
    """Display embedding statistics."""
    render_embedding_stats(
        embedding_status_payload(env, include_retrieval_bands=False, include_detail=detail),  # type: ignore[arg-type]
        json_output=json_output,
    )


__all__ = [
    "EmbeddingStatusPayload",
    "RetrievalBandPayload",
    "embedding_status_payload",
    "render_embedding_stats",
    "show_embedding_stats",
]
