"""Context-oriented read-view handlers."""

from __future__ import annotations

from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync
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

CONTEXT_READ_VIEW_OPTION_NAMES = frozenset({"related_limit"})
CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES = frozenset(
    {
        "max_sessions",
        "no_redact",
        "context_origin",
        "context_query",
        "project_path",
        "project_repo",
        "since",
        "until",
    }
)


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

    from polylogue.context.preamble import compose_context_preamble

    del request
    assert invocation.session_id is not None
    options = cast(ReadViewContextOptions, invocation.options or ReadViewContextOptions())
    preamble = compose_context_preamble(
        env,
        session_id=invocation.session_id,
        related_limit=max(1, options.related_limit),
    )
    deliver_content(env, preamble + "\n", destination=invocation.destination, out_path=invocation.out_path)


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
    image = run_coroutine_sync(
        env.polylogue.context_image_payload(
            seed_session_id=invocation.session_id,
            project_path=options.project_path,
            project_repo=options.project_repo,
            since=options.since,
            until=options.until,
            origin=options.origin,
            query=options.query,
            max_sessions=options.max_sessions,
            redact_paths=not options.no_redact,
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
