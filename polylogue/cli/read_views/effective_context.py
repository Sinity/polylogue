"""Read view for the provider-effective post-compaction context."""

from __future__ import annotations

import json
from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.read_view_registry import EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewEffectiveContextOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


def build_effective_context_options(values: ReadViewOptionValues) -> ReadViewEffectiveContextOptions:
    return ReadViewEffectiveContextOptions(at_position=cast(int | None, values.get("at_position")))


def run_read_effective_context(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    assert invocation.session_id is not None
    options = cast(ReadViewEffectiveContextOptions, invocation.options or ReadViewEffectiveContextOptions())

    async def _run() -> list[dict[str, object]] | None:
        from polylogue.api import Polylogue

        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            return await api.get_effective_context(invocation.session_id, at_position=options.at_position)

    payload = run_coroutine_sync(_run())
    if payload is None:
        env.ui.error(f"Session not found: {invocation.session_id}")
        return
    content = (
        json.dumps(
            {"session_id": invocation.session_id, "at_position": options.at_position, "messages": payload}, indent=2
        )
        + "\n"
    )
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )


__all__ = ["EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES", "build_effective_context_options", "run_read_effective_context"]
