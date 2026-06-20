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
    ReadViewOptions,
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


_MESSAGE_OPTIONS = frozenset(
    {
        "limit",
        "message_role",
        "message_type",
        "no_code_blocks",
        "no_file_reads",
        "no_tool_calls",
        "no_tool_outputs",
        "offset",
        "prose_only",
    }
)
_CONTEXT_PACK_OPTIONS = frozenset(
    {
        "max_messages",
        "max_sessions",
        "no_redact",
        "pack_origin",
        "pack_query",
        "project_path",
        "project_repo",
        "since",
        "until",
    }
)
_CORRELATION_OPTIONS = frozenset({"confidence_threshold", "github_api", "otlp", "repo_path", "since_hours"})


READ_VIEW_HANDLERS: dict[str, ReadViewHandler] = {
    "summary": ReadViewHandler("summary", "optional", run_read_summary_or_transcript, default_format="markdown"),
    "transcript": ReadViewHandler("transcript", "optional", run_read_summary_or_transcript, default_format="markdown"),
    "messages": ReadViewHandler(
        "messages", "required", run_read_messages, default_format="text", accepted_options=_MESSAGE_OPTIONS
    ),
    "raw": ReadViewHandler("raw", "required", run_read_raw, default_format="json", accepted_options=_MESSAGE_OPTIONS),
    "context": ReadViewHandler(
        "context", "required", run_read_context, default_format="json", accepted_options=frozenset({"related_limit"})
    ),
    "context-pack": ReadViewHandler(
        "context-pack",
        "none",
        run_read_context_pack,
        default_format="markdown",
        accepted_options=_CONTEXT_PACK_OPTIONS,
    ),
    "recovery": ReadViewHandler(
        "recovery",
        "required",
        run_read_recovery,
        default_format="markdown",
        accepted_options=frozenset({"recovery_report"}),
    ),
    "neighbors": ReadViewHandler(
        "neighbors",
        "query_or_session",
        run_read_neighbors,
        default_format="text",
        accepted_options=frozenset({"limit", "window_hours"}),
    ),
    "correlation": ReadViewHandler(
        "correlation", "required", run_read_correlation, default_format="text", accepted_options=_CORRELATION_OPTIONS
    ),
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


def read_view_option_names() -> frozenset[str]:
    """Return every view-specific option name owned by read-view handlers."""

    return frozenset(option_name for handler in READ_VIEW_HANDLERS.values() for option_name in handler.accepted_options)


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
    "ReadViewOptions",
    "ReadViewRecoveryOptions",
    "read_view_handler_ids",
    "read_view_option_names",
    "run_bulk_export_view",
    "run_read_view",
    "validate_read_view_handler_registry",
]
