"""Read view for the provider-effective post-compaction context."""

from __future__ import annotations

import json
from typing import cast

from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
from polylogue.cli.read_view_registry import EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewEffectiveContextOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def build_effective_context_options(values: ReadViewOptionValues) -> ReadViewEffectiveContextOptions:
    return ReadViewEffectiveContextOptions(at_position=cast(int | None, values.get("at_position")))


def run_read_effective_context(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    session_id = invocation.session_id
    assert session_id is not None
    options = cast(ReadViewEffectiveContextOptions, invocation.options or ReadViewEffectiveContextOptions())

    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest("read.effective_context", {"session_id": session_id, "at_position": options.at_position}),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("read.effective_context returned no payload")
    content = json.dumps(payload, indent=2) + "\n"
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )


__all__ = ["EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES", "build_effective_context_options", "run_read_effective_context"]
