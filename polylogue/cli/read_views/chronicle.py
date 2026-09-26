"""Thin CLI adapter for the typed chronicle read operation."""

from __future__ import annotations

import json
from typing import cast

from polylogue.cli.read_view_registry import CHRONICLE_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewChronicleOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.chronicle import (
    ChronicleProjectionPayload,
    chronicle_json_document,
    render_chronicle_markdown,
)

DEFAULT_CHRONICLE_EDGE_LIMIT = 8


def build_chronicle_options(values: ReadViewOptionValues) -> ReadViewChronicleOptions:
    """Build chronicle options from read command values."""

    raw_limit = values.get("limit")
    edge_limit = int(raw_limit) if isinstance(raw_limit, int | str) else DEFAULT_CHRONICLE_EDGE_LIMIT
    return ReadViewChronicleOptions(edge_limit=max(edge_limit, 1))


def run_read_chronicle(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Dispatch and render one chronicle read."""

    from polylogue.cli.lowering import _selection_params
    from polylogue.cli.operation_kernel import (
        OperationKernelError,
        OperationRequest,
        dispatch,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled

    options = invocation.options if isinstance(invocation.options, ReadViewChronicleOptions) else None
    projection = (
        invocation.projection_spec.projection.model_dump(mode="json") if invocation.projection_spec is not None else {}
    )
    if not projection:
        projection = {"edge_limit": options.edge_limit if options is not None else DEFAULT_CHRONICLE_EDGE_LIMIT}
    elif projection.get("edge_limit") is None:
        projection["edge_limit"] = options.edge_limit if options is not None else DEFAULT_CHRONICLE_EDGE_LIMIT
    params = _selection_params(request)
    params["query"] = list(request.query_terms)
    params.setdefault("limit", 5)
    operation = OperationRequest(
        "read.chronicle",
        {"session_id": invocation.session_id, "params": params, "projection": projection},
    )
    try:
        result = dispatch(
            cast(Config, request.config()),
            operation,
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    wire = result.value
    if not isinstance(wire, dict) or wire.get("view") != "chronicle" or not isinstance(wire.get("payload"), dict):
        from polylogue.cli.operation_kernel import OperationEnvelopeError

        raise OperationEnvelopeError("read.chronicle result has an invalid view payload")
    payload = ChronicleProjectionPayload.model_validate(wire["payload"])
    fmt = invocation.output_format or "markdown"
    content = (
        json.dumps(chronicle_json_document(payload), indent=2) + "\n"
        if fmt == "json"
        else render_chronicle_markdown(payload)
    )
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)


__all__ = [
    "CHRONICLE_READ_VIEW_OPTION_NAMES",
    "DEFAULT_CHRONICLE_EDGE_LIMIT",
    "build_chronicle_options",
    "run_read_chronicle",
]
