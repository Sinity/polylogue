"""Executable read-view registry for the query-first CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from polylogue.archive.viewport import read_view_choices
from polylogue.cli.read_views.base import (
    ReadViewContextOptions,
    ReadViewContextPackOptions,
    ReadViewCorrelationOptions,
    ReadViewHandler,
    ReadViewInvocation,
    ReadViewMessageOptions,
    ReadViewNeighborOptions,
    ReadViewRecoveryOptions,
)
from polylogue.cli.read_views.bulk import run_bulk_export_view
from polylogue.cli.read_views.context import run_read_context, run_read_context_pack
from polylogue.cli.read_views.correlation import run_read_correlation
from polylogue.cli.read_views.messages import run_read_messages, run_read_raw
from polylogue.cli.read_views.neighbors import run_read_neighbors
from polylogue.cli.read_views.recovery import run_read_recovery
from polylogue.cli.read_views.standard import run_read_summary_or_transcript
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest


READ_VIEW_HANDLERS: dict[str, ReadViewHandler] = {
    "summary": ReadViewHandler("summary", "optional", run_read_summary_or_transcript, default_format="markdown"),
    "transcript": ReadViewHandler("transcript", "optional", run_read_summary_or_transcript, default_format="markdown"),
    "messages": ReadViewHandler("messages", "required", run_read_messages, default_format="text"),
    "raw": ReadViewHandler("raw", "required", run_read_raw, default_format="json"),
    "context": ReadViewHandler("context", "required", run_read_context, default_format="json"),
    "context-pack": ReadViewHandler("context-pack", "none", run_read_context_pack, default_format="markdown"),
    "recovery": ReadViewHandler("recovery", "required", run_read_recovery, default_format="markdown"),
    "neighbors": ReadViewHandler("neighbors", "query_or_session", run_read_neighbors, default_format="text"),
    "correlation": ReadViewHandler("correlation", "required", run_read_correlation, default_format="text"),
}


def run_read_view(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Execute a registered read view."""

    try:
        handler = READ_VIEW_HANDLERS[invocation.view]
    except KeyError as exc:  # pragma: no cover - Click choice prevents this.
        raise click.UsageError(f"Unknown read view: {invocation.view}") from exc
    handler.validate(invocation, request)
    handler.handler(env, request, invocation)


def read_view_handler_ids() -> tuple[str, ...]:
    """Return executable read-view handler ids."""

    return tuple(READ_VIEW_HANDLERS)


def validate_read_view_handler_registry() -> None:
    """Fail fast if profile metadata and executable handlers drift."""

    profile_ids = set(read_view_choices())
    handler_ids = set(READ_VIEW_HANDLERS)
    missing = sorted(profile_ids - handler_ids)
    extra = sorted(handler_ids - profile_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing handlers: {', '.join(missing)}")
        if extra:
            details.append(f"handlers without profiles: {', '.join(extra)}")
        raise RuntimeError("read-view handler registry drift: " + "; ".join(details))


validate_read_view_handler_registry()

__all__ = [
    "READ_VIEW_HANDLERS",
    "ReadViewContextOptions",
    "ReadViewContextPackOptions",
    "ReadViewCorrelationOptions",
    "ReadViewHandler",
    "ReadViewInvocation",
    "ReadViewMessageOptions",
    "ReadViewNeighborOptions",
    "ReadViewRecoveryOptions",
    "read_view_handler_ids",
    "run_bulk_export_view",
    "run_read_view",
    "validate_read_view_handler_registry",
]
