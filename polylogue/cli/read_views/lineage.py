"""Read view for the compact seed-relative lineage graph (polylogue-4ts.9)."""

from __future__ import annotations

import json
from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.read_view_registry import LINEAGE_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewInvocation,
    ReadViewLineageOptions,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


def build_lineage_options(values: ReadViewOptionValues) -> ReadViewLineageOptions:
    return ReadViewLineageOptions(
        node_offset=cast(int, values.get("node_offset", 0) or 0),
        node_limit=cast(int | None, values.get("node_limit")),
        edge_offset=cast(int, values.get("edge_offset", 0) or 0),
        edge_limit=cast(int | None, values.get("edge_limit")),
    )


def run_read_lineage(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the compact lineage graph for the seed session."""

    session_id = invocation.session_id
    assert session_id is not None
    options = cast(ReadViewLineageOptions, invocation.options or ReadViewLineageOptions())

    async def _run() -> object | None:
        from polylogue.api import Polylogue

        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            return await api.compact_lineage(
                session_id,
                node_offset=options.node_offset,
                node_limit=options.node_limit,
                edge_offset=options.edge_offset,
                edge_limit=options.edge_limit,
            )

    graph = run_coroutine_sync(_run())
    if graph is None:
        env.ui.error(f"Session not found: {session_id}")
        return
    content = json.dumps(graph.model_dump(mode="json"), indent=2) + "\n"  # type: ignore[attr-defined]
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )


__all__ = ["LINEAGE_READ_VIEW_OPTION_NAMES", "build_lineage_options", "run_read_lineage"]
