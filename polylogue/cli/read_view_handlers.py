"""Executable read-view registry for the query-first CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from polylogue.archive.viewport import read_view_choices
from polylogue.cli.read_view_registry import (
    CHRONICLE_READ_VIEW_OPTION_NAMES,
    CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES,
    CONTEXT_READ_VIEW_OPTION_NAMES,
    CORRELATION_READ_VIEW_OPTION_NAMES,
    MESSAGE_READ_VIEW_OPTION_NAMES,
    NEIGHBOR_READ_VIEW_OPTION_NAMES,
    READ_VIEW_HANDLER_METADATA,
)
from polylogue.cli.read_views.base import (
    ReadViewChronicleOptions,
    ReadViewContextImageOptions,
    ReadViewContextOptions,
    ReadViewCorrelationOptions,
    ReadViewHandler,
    ReadViewInvocation,
    ReadViewMessageOptions,
    ReadViewNeighborOptions,
    ReadViewOptions,
)
from polylogue.cli.read_views.chronicle import build_chronicle_options, run_read_chronicle
from polylogue.cli.read_views.context import (
    build_context_image_options,
    build_context_options,
    run_read_context,
    run_read_context_image,
)
from polylogue.cli.read_views.correlation import build_correlation_options, run_read_correlation
from polylogue.cli.read_views.messages import (
    build_message_options,
    run_read_messages,
    run_read_raw,
)
from polylogue.cli.read_views.neighbors import build_neighbor_options, run_read_neighbors
from polylogue.cli.read_views.query_set import run_query_set_read_view
from polylogue.cli.read_views.standard import run_read_dialogue, run_read_summary_or_transcript, run_read_temporal
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest


READ_VIEW_HANDLERS: dict[str, ReadViewHandler] = {
    "summary": ReadViewHandler(
        "summary",
        "optional",
        run_read_summary_or_transcript,
        default_format="markdown",
        accepts_query_set=True,
    ),
    "transcript": ReadViewHandler(
        "transcript",
        "optional",
        run_read_summary_or_transcript,
        default_format="markdown",
        accepts_query_set=True,
    ),
    "dialogue": ReadViewHandler(
        "dialogue",
        "required",
        run_read_dialogue,
        default_format="markdown",
        accepts_query_set=True,
    ),
    "messages": ReadViewHandler(
        "messages",
        "required",
        run_read_messages,
        default_format="text",
        accepted_options=MESSAGE_READ_VIEW_OPTION_NAMES,
        option_builder=build_message_options,
    ),
    "raw": ReadViewHandler(
        "raw",
        "required",
        run_read_raw,
        default_format="json",
        accepted_options=MESSAGE_READ_VIEW_OPTION_NAMES,
        option_builder=build_message_options,
    ),
    "context": ReadViewHandler(
        "context",
        "required",
        run_read_context,
        default_format="json",
        accepted_options=CONTEXT_READ_VIEW_OPTION_NAMES,
        option_builder=build_context_options,
    ),
    "context-image": ReadViewHandler(
        "context-image",
        "none",
        run_read_context_image,
        default_format="markdown",
        accepted_options=CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES,
        option_builder=build_context_image_options,
    ),
    "neighbors": ReadViewHandler(
        "neighbors",
        "query_or_session",
        run_read_neighbors,
        default_format="text",
        accepted_options=NEIGHBOR_READ_VIEW_OPTION_NAMES,
        option_builder=build_neighbor_options,
    ),
    "correlation": ReadViewHandler(
        "correlation",
        "required",
        run_read_correlation,
        default_format="text",
        accepted_options=CORRELATION_READ_VIEW_OPTION_NAMES,
        option_builder=build_correlation_options,
    ),
    "temporal": ReadViewHandler(
        "temporal",
        "optional",
        run_read_temporal,
        default_format="markdown",
        accepts_query_set=True,
    ),
    "chronicle": ReadViewHandler(
        "chronicle",
        "optional",
        run_read_chronicle,
        default_format="markdown",
        accepted_options=CHRONICLE_READ_VIEW_OPTION_NAMES,
        option_builder=build_chronicle_options,
        accepts_query_set=True,
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


def read_view_options_for_view(view: str, values: dict[str, object]) -> ReadViewOptions | None:
    """Build typed options for one read view."""

    try:
        handler = READ_VIEW_HANDLERS[view]
    except KeyError as exc:  # pragma: no cover - Click choice prevents this.
        raise click.UsageError(f"Unknown read view: {view}") from exc
    return handler.build_options(values)


def validate_read_view_handler_registry() -> None:
    """Fail fast if profile metadata and executable handlers drift."""

    profile_ids = set(read_view_choices())
    handler_ids = set(READ_VIEW_HANDLERS)
    metadata_ids = set(READ_VIEW_HANDLER_METADATA)
    missing = sorted(profile_ids - handler_ids)
    extra = sorted(handler_ids - profile_ids)
    metadata_missing = sorted(handler_ids - metadata_ids)
    metadata_extra = sorted(metadata_ids - handler_ids)
    metadata_mismatch = [
        view_id
        for view_id, handler in READ_VIEW_HANDLERS.items()
        if view_id in READ_VIEW_HANDLER_METADATA
        and (
            handler.accepted_options != READ_VIEW_HANDLER_METADATA[view_id].accepted_options
            or handler.session_policy != READ_VIEW_HANDLER_METADATA[view_id].session_policy
            or handler.accepts_query_set != READ_VIEW_HANDLER_METADATA[view_id].accepts_query_set
        )
    ]
    if missing or extra or metadata_missing or metadata_extra or metadata_mismatch:
        details: list[str] = []
        if missing:
            details.append(f"missing handlers: {', '.join(missing)}")
        if extra:
            details.append(f"handlers without profiles: {', '.join(extra)}")
        if metadata_missing:
            details.append(f"handlers without metadata: {', '.join(metadata_missing)}")
        if metadata_extra:
            details.append(f"metadata without handlers: {', '.join(metadata_extra)}")
        if metadata_mismatch:
            details.append(f"handler metadata mismatch: {', '.join(sorted(metadata_mismatch))}")
        raise RuntimeError("read-view handler registry drift: " + "; ".join(details))


validate_read_view_handler_registry()

__all__ = [
    "READ_VIEW_HANDLERS",
    "ReadViewContextOptions",
    "ReadViewContextImageOptions",
    "ReadViewCorrelationOptions",
    "ReadViewChronicleOptions",
    "ReadViewHandler",
    "ReadViewInvocation",
    "ReadViewMessageOptions",
    "ReadViewNeighborOptions",
    "ReadViewOptions",
    "read_view_handler_ids",
    "read_view_option_names",
    "read_view_options_for_view",
    "run_query_set_read_view",
    "run_read_view",
    "validate_read_view_handler_registry",
]
