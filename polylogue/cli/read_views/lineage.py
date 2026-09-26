"""Read view for the compact seed-relative lineage graph (polylogue-4ts.9)."""

from __future__ import annotations

import json
from typing import cast

from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
from polylogue.cli.read_view_registry import LINEAGE_READ_VIEW_OPTION_NAMES, TOPOLOGY_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewInvocation,
    ReadViewLineageOptions,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


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

    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest(
                "read.lineage",
                {
                    "session_id": session_id,
                    "node_offset": options.node_offset,
                    "node_limit": options.node_limit,
                    "edge_offset": options.edge_offset,
                    "edge_limit": options.edge_limit,
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    content = json.dumps(result["payload"], indent=2) + "\n"
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )


def build_topology_options(values: ReadViewOptionValues) -> ReadViewLineageOptions:
    """Bind the topology paging window from already-declared query params."""

    return ReadViewLineageOptions(
        node_offset=cast(int, values.get("node_offset", 0) or 0),
        node_limit=cast(int | None, values.get("node_limit")),
        edge_limit=cast(int | None, values.get("edge_limit")),
    )


def run_read_topology(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the canonical session-links topology envelope for one session."""

    session_id = invocation.session_id
    assert session_id is not None
    options = cast(ReadViewLineageOptions, invocation.options or ReadViewLineageOptions())
    node_limit = options.node_limit or cast(int | None, request.params.get("limit")) or 200
    edge_limit = options.edge_limit or 500

    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest(
                "read.topology",
                {
                    "session_id": session_id,
                    "node_offset": options.node_offset,
                    "node_limit": node_limit,
                    "edge_limit": edge_limit,
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    content = json.dumps(result["payload"], indent=2) + "\n"
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )


__all__ = [
    "LINEAGE_READ_VIEW_OPTION_NAMES",
    "TOPOLOGY_READ_VIEW_OPTION_NAMES",
    "build_lineage_options",
    "build_topology_options",
    "run_read_lineage",
    "run_read_topology",
]
