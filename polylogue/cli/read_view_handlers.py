"""Executable read-view registry for the query-first CLI.

This module owns one fact per read view: *which callable runs it*, and — for a
view with its own options — which builder types them.  Everything else a read
view declares (session policy, accepted option names, query-set admission) is
declared once in :mod:`polylogue.cli.read_view_registry` and read from there.

Those static facts used to be spelled a second time in this module's handler
table, and a ``metadata_mismatch`` check in
``validate_read_view_handler_registry`` compared the two copies row by row.
Two tables plus a drift check is the shape the read-algebra spec forbids
(polylogue-vbsc0): the check can only report a disagreement after someone has
already had to make the same edit twice, and it is dead weight once there is
nothing to disagree.  Deriving the handler from the declaration removes the
second edit site, so the mismatch it guarded is unrepresentable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import click

from polylogue.archive.viewport import read_view_choices
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
from polylogue.cli.read_views.base import (
    ReadViewChronicleOptions,
    ReadViewContextImageOptions,
    ReadViewContextOptions,
    ReadViewCorrelationOptions,
    ReadViewEventsOptions,
    ReadViewHandler,
    ReadViewHandlerFunc,
    ReadViewInvocation,
    ReadViewMessageOptions,
    ReadViewNeighborOptions,
    ReadViewOptionBuilder,
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
from polylogue.cli.read_views.effective_context import build_effective_context_options, run_read_effective_context
from polylogue.cli.read_views.events import build_events_options, run_read_events
from polylogue.cli.read_views.file_edits import run_read_agent_policies, run_read_file_edits
from polylogue.cli.read_views.lineage import (
    build_lineage_options,
    build_topology_options,
    run_read_lineage,
    run_read_topology,
)
from polylogue.cli.read_views.messages import (
    build_message_options,
    run_read_hooks,
    run_read_messages,
    run_read_raw,
)
from polylogue.cli.read_views.neighbors import build_neighbor_options, run_read_neighbors
from polylogue.cli.read_views.query_set import run_query_set_read_view
from polylogue.cli.read_views.standard import run_read_dialogue, run_read_summary_or_transcript, run_read_temporal
from polylogue.cli.read_views.web_content_constructs import run_read_web_content_constructs
from polylogue.cli.shared.types import AppEnv
from polylogue.operations.session_projections import (
    SESSION_LIST_PROJECTIONS,
    validate_session_list_projection_cli_contract,
)

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest


@dataclass(frozen=True, slots=True)
class ReadViewExecution:
    """The executable half of a read view: what runs it, and how it types options."""

    run: ReadViewHandlerFunc
    option_builder: ReadViewOptionBuilder | None = None


def build_read_view_handler(
    view_id: str,
    execution: ReadViewExecution,
    *,
    declared_as: str | None = None,
) -> ReadViewHandler:
    """Bind one declared read view to the callable that executes it.

    The declaration is the only source of the view's static contract, so a
    handler cannot claim a session policy or an option set the declaration does
    not carry.

    ``declared_as`` names the declaration a view *borrows*.  A session-list
    projection states which declared CLI handler serves it, so it is dispatched
    under its own name while carrying that handler's contract; the contract is
    still read from a declaration, never invented for the borrowing name.
    """

    declaration_id = declared_as or view_id
    try:
        metadata = READ_VIEW_HANDLER_METADATA[declaration_id]
    except KeyError as exc:
        raise RuntimeError(f"read view {declaration_id!r} has an executable handler but no declaration") from exc
    return ReadViewHandler(
        view_id=view_id,
        session_policy=metadata.session_policy,
        handler=execution.run,
        accepted_options=metadata.accepted_options,
        option_builder=execution.option_builder,
        accepts_query_set=metadata.accepts_query_set,
    )


#: Executable bindings for the views declared directly by the registry.
READ_VIEW_EXECUTION: dict[str, ReadViewExecution] = {
    "summary": ReadViewExecution(run_read_summary_or_transcript),
    "transcript": ReadViewExecution(run_read_summary_or_transcript),
    "dialogue": ReadViewExecution(run_read_dialogue),
    "messages": ReadViewExecution(run_read_messages, build_message_options),
    "raw": ReadViewExecution(run_read_raw, build_message_options),
    "hooks": ReadViewExecution(run_read_hooks),
    "effective_context": ReadViewExecution(run_read_effective_context, build_effective_context_options),
    "lineage": ReadViewExecution(run_read_lineage, build_lineage_options),
    "topology": ReadViewExecution(run_read_topology, build_topology_options),
    "context": ReadViewExecution(run_read_context, build_context_options),
    "context-image": ReadViewExecution(run_read_context_image, build_context_image_options),
    "neighbors": ReadViewExecution(run_read_neighbors, build_neighbor_options),
    "correlation": ReadViewExecution(run_read_correlation, build_correlation_options),
    "temporal": ReadViewExecution(run_read_temporal),
    "chronicle": ReadViewExecution(run_read_chronicle, build_chronicle_options),
}


#: Executable bindings the shared session-projection table names rather than
#: this module: a projection states which of these runs it, so one binding can
#: serve several declared views.
SESSION_LIST_READ_VIEW_EXECUTION: dict[str, ReadViewExecution] = {
    "events": ReadViewExecution(run_read_events, build_events_options),
    "file-edits": ReadViewExecution(run_read_file_edits),
    "agent-policies": ReadViewExecution(run_read_agent_policies),
    "web-content": ReadViewExecution(run_read_web_content_constructs),
}


def session_list_read_view_handlers() -> dict[str, ReadViewHandler]:
    """Build the CLI entries declared by the shared session-projection table."""

    handlers: dict[str, ReadViewHandler] = {}
    for projection in SESSION_LIST_PROJECTIONS.values():
        try:
            execution = SESSION_LIST_READ_VIEW_EXECUTION[projection.cli_handler]
        except KeyError as exc:
            raise RuntimeError(
                f"session projection {projection.name!r} names unknown CLI handler {projection.cli_handler!r}"
            ) from exc
        handlers[projection.name] = build_read_view_handler(
            projection.name, execution, declared_as=projection.cli_handler
        )
    return handlers


def _build_read_view_handlers() -> dict[str, ReadViewHandler]:
    """Compose every executable read view, in the profile registry's public order."""

    handlers = {
        view_id: build_read_view_handler(view_id, execution) for view_id, execution in READ_VIEW_EXECUTION.items()
    }
    handlers.update(session_list_read_view_handlers())
    # Public order first; anything the profile registry does not declare stays
    # in the table so ``validate_read_view_handler_registry`` can name it
    # instead of it disappearing into an ordering comprehension.
    order = [view_id for view_id in read_view_choices() if view_id in handlers]
    order.extend(view_id for view_id in handlers if view_id not in order)
    return {view_id: handlers[view_id] for view_id in order}


READ_VIEW_HANDLERS: dict[str, ReadViewHandler] = _build_read_view_handlers()


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


def read_view_options_for_view(view: str, values: dict[str, object]) -> ReadViewOptions | None:
    """Build typed options for one read view."""

    try:
        handler = READ_VIEW_HANDLERS[view]
    except KeyError as exc:  # pragma: no cover - Click choice prevents this.
        raise click.UsageError(f"Unknown read view: {view}") from exc
    return handler.build_options(values)


def validate_read_view_handler_registry() -> None:
    """Fail fast if profile metadata and executable handlers drift.

    Only *coverage* is checkable here.  A handler's static contract is read
    from its declaration, so it cannot disagree with one.
    """

    profile_ids = set(read_view_choices())
    handler_ids = set(READ_VIEW_HANDLERS)
    validate_session_list_projection_cli_contract(handler_ids)
    missing = sorted(profile_ids - handler_ids)
    extra = sorted(handler_ids - profile_ids)
    unbound = sorted(set(READ_VIEW_HANDLER_METADATA) - handler_ids)
    if missing or extra or unbound:
        details: list[str] = []
        if missing:
            details.append(f"missing handlers: {', '.join(missing)}")
        if extra:
            details.append(f"handlers without profiles: {', '.join(extra)}")
        if unbound:
            details.append(f"declarations without an executable binding: {', '.join(unbound)}")
        raise RuntimeError("read-view handler registry drift: " + "; ".join(details))


validate_read_view_handler_registry()

__all__ = [
    "READ_VIEW_EXECUTION",
    "READ_VIEW_HANDLERS",
    "SESSION_LIST_READ_VIEW_EXECUTION",
    "ReadViewContextOptions",
    "ReadViewContextImageOptions",
    "ReadViewCorrelationOptions",
    "ReadViewChronicleOptions",
    "ReadViewEventsOptions",
    "ReadViewExecution",
    "ReadViewHandler",
    "ReadViewInvocation",
    "ReadViewMessageOptions",
    "ReadViewNeighborOptions",
    "ReadViewOptions",
    "build_read_view_handler",
    "read_view_handler_ids",
    "read_view_options_for_view",
    "session_list_read_view_handlers",
    "run_query_set_read_view",
    "run_read_view",
    "validate_read_view_handler_registry",
]
