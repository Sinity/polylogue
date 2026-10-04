"""Thin CLI adapter for the typed corpus-compaction read operation."""

from __future__ import annotations

import json
from typing import cast

from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.compaction import CorpusCompactionPack, render_compaction_markdown


def run_read_compact(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Dispatch and render one compaction pack over the selected sessions."""

    from polylogue.cli.lowering import _selection_params
    from polylogue.cli.operation_kernel import (
        OperationKernelError,
        OperationRequest,
        dispatch,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled

    max_tokens = invocation.projection_spec.projection.max_tokens if invocation.projection_spec is not None else None
    params = _selection_params(request)
    params["query"] = list(request.query_terms)
    params.setdefault("limit", 5)
    operation = OperationRequest(
        "read.compact",
        {"session_id": invocation.session_id, "params": params, "projection": {"max_tokens": max_tokens}},
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
    if not isinstance(wire, dict) or wire.get("view") != "compact" or not isinstance(wire.get("payload"), dict):
        from polylogue.cli.operation_kernel import OperationEnvelopeError

        raise OperationEnvelopeError("read.compact result has an invalid view payload")
    pack = CorpusCompactionPack.model_validate(wire["payload"])
    fmt = invocation.output_format or "markdown"
    content = (
        json.dumps(pack.model_dump(mode="json"), indent=2) + "\n" if fmt == "json" else render_compaction_markdown(pack)
    )
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)
    from polylogue.cli.render.outcome import finish_supplied_outcome

    finish_supplied_outcome(pack.outcome)


__all__ = ["run_read_compact"]
