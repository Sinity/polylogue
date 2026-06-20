"""Context-oriented read-view handlers."""

from __future__ import annotations

from typing import cast

from polylogue.cli.read_views.base import (
    ReadViewContextOptions,
    ReadViewContextPackOptions,
    ReadViewInvocation,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


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


def run_read_context_pack(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the project/query-scoped context pack."""

    from polylogue.context.pack import run_context_pack_view

    del request
    options = cast(ReadViewContextPackOptions, invocation.options or ReadViewContextPackOptions())
    run_context_pack_view(
        env,
        project_path=options.project_path,
        project_repo=options.project_repo,
        since=options.since,
        until=options.until,
        origin=options.origin,
        query=options.query,
        max_sessions=options.max_sessions,
        max_messages=options.max_messages,
        no_redact=options.no_redact,
    )


__all__ = ["run_read_context", "run_read_context_pack"]
