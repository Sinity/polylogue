"""Context-oriented read-view handlers."""

from __future__ import annotations

from typing import cast

from polylogue.cli.read_views.base import (
    ReadViewContextOptions,
    ReadViewContextPackOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

CONTEXT_READ_VIEW_OPTION_NAMES = frozenset({"related_limit"})
CONTEXT_PACK_READ_VIEW_OPTION_NAMES = frozenset(
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


def build_context_options(values: ReadViewOptionValues) -> ReadViewContextOptions:
    """Build options owned by the context preamble read view."""

    return ReadViewContextOptions(related_limit=cast(int, values.get("related_limit", 5)))


def build_context_pack_options(values: ReadViewOptionValues) -> ReadViewContextPackOptions:
    """Build options owned by the context-pack read view."""

    return ReadViewContextPackOptions(
        project_path=cast(str | None, values.get("project_path")),
        project_repo=cast(str | None, values.get("project_repo")),
        since=cast(str | None, values.get("since")),
        until=cast(str | None, values.get("until")),
        origin=cast(str | None, values.get("pack_origin")),
        query=cast(str | None, values.get("pack_query")),
        max_sessions=cast(int, values.get("max_sessions", 5)),
        max_messages=cast(int, values.get("max_messages", 20)),
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


__all__ = [
    "CONTEXT_PACK_READ_VIEW_OPTION_NAMES",
    "CONTEXT_READ_VIEW_OPTION_NAMES",
    "build_context_options",
    "build_context_pack_options",
    "run_read_context",
    "run_read_context_pack",
]
