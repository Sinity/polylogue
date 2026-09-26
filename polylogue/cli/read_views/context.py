"""Context-oriented read-view handlers."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from datetime import datetime, timezone
from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest, configured_mutation_operation
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
from polylogue.cli.read_view_registry import CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES, CONTEXT_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewContextImageOptions,
    ReadViewContextOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.surfaces.payloads import serialize_surface_payload


def build_context_options(values: ReadViewOptionValues) -> ReadViewContextOptions:
    """Build options owned by the context preamble read view."""

    return ReadViewContextOptions(related_limit=cast(int, values.get("related_limit", 5)))


def build_context_image_options(values: ReadViewOptionValues) -> ReadViewContextImageOptions:
    """Build options owned by the context-image read view."""

    return ReadViewContextImageOptions(
        project_path=cast(str | None, values.get("project_path")),
        project_repo=cast(str | None, values.get("project_repo")),
        since=cast(str | None, values.get("since")),
        until=cast(str | None, values.get("until")),
        origin=cast(str | None, values.get("context_origin")),
        query=cast(str | None, values.get("context_query")),
        max_sessions=cast(int, values.get("max_sessions", 5)),
        no_redact=cast(bool, values.get("no_redact", False)),
    )


def run_read_context(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Compose the context preamble for the seed session."""

    from polylogue.context.preamble import _git_project_state

    assert invocation.session_id is not None
    options = cast(ReadViewContextOptions, invocation.options or ReadViewContextOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    related_limit = (
        projection.context_related_limit
        if projection and projection.context_related_limit is not None
        else options.related_limit
    )
    cwd = os.getcwd()
    git_state, git_failure = _git_project_state(cwd)
    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest(
                "read.context",
                {
                    "session_id": invocation.session_id,
                    "related_limit": max(1, related_limit),
                    "cwd": cwd,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "observed_project_state": git_state.model_dump(mode="json") if git_state is not None else None,
                    "project_failure": git_failure,
                    "source_tool_calls": {"compose_context_preamble": "polylogue-cli"},
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    payload = result.get("payload")
    if not isinstance(payload, dict):
        env.ui.error(f"Session not found: {invocation.session_id}")
        raise SystemExit(1)
    ledger = result.get("ledger")
    if isinstance(ledger, dict):
        with suppress(OperationKernelError):
            configured_mutation_operation(env.config, "mutation.facade.context_ledger", ledger)
    deliver_content(
        env,
        json.dumps(payload, indent=2, default=str) + "\n",
        destination=invocation.destination,
        out_path=invocation.out_path,
    )


def run_read_context_image(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the project/query-scoped context image as a compiled context image.

    The context image is a thin lens over ``compile_context``: the seed session
    (when the find selection resolved one) or the context-image selection filters
    pick the sessions, and the shared engine compiles the message transcript with
    omission accounting. The CLI ``read`` verb routes multi-session selections
    through ``run_read_context_image``; this handler covers the single resolved
    seed and direct handler invocation.
    """

    options = cast(ReadViewContextImageOptions, invocation.options or ReadViewContextImageOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    max_sessions = (
        projection.context_max_sessions
        if projection and projection.context_max_sessions is not None
        else options.max_sessions
    )
    redact_paths = projection.redact_paths if projection is not None else not options.no_redact
    image = run_coroutine_sync(
        env.polylogue.context_image_payload(
            seed_session_id=invocation.session_id,
            project_path=options.project_path,
            project_repo=options.project_repo,
            since=options.since,
            until=options.until,
            origin=options.origin,
            query=options.query,
            max_sessions=max_sessions,
            redact_paths=redact_paths,
        )
    )
    deliver_content(
        env,
        serialize_surface_payload(image, exclude_none=True) + "\n",
        destination=invocation.destination,
        out_path=invocation.out_path,
    )


__all__ = [
    "CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES",
    "CONTEXT_READ_VIEW_OPTION_NAMES",
    "build_context_options",
    "build_context_image_options",
    "run_read_context",
    "run_read_context_image",
]
