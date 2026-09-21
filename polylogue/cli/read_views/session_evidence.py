"""Per-session evidence read views, served by one declared ``session.read``.

``hooks``, ``file-edits``, ``agent-policies`` and ``web-content`` render four
index-tier relations that ride one exact session reference and have no
query-grammar unit of their own (design D3).  They are the same request with a
different relation named, so they are one adapter rather than four: each lowers
``session.read`` with its own kind, dispatches through the operation kernel and
renders the evidence body the operation answered with.

None of them opens an archive, and none branches on whether a daemon is
running: which executor answered is the kernel's decision, reported by the
result's own authority and named on stderr under ``--verbose`` exactly as
``read --view messages`` names it (polylogue-r3cuz).
"""

from __future__ import annotations

import json
from typing import cast

import click

from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.projection_spec import RenderDestination

__all__ = [
    "run_read_agent_policies",
    "run_read_file_edits",
    "run_read_hooks",
    "run_read_web_content",
    "run_session_evidence_view",
]


def run_session_evidence_view(
    env: AppEnv,
    request: RootModeRequest,
    invocation: ReadViewInvocation,
    *,
    kind: str,
) -> None:
    """Render one declared per-session evidence relation.

    A refusal names itself and exits non-zero; it must never render as an
    empty body that reads "this session recorded nothing".  Refusals are
    classed exactly as the transcript path classes them, through the one CLI
    read-failure terminal (polylogue-jtrtj).
    """

    from polylogue.cli.lowering import lower_session_read
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationFailedError,
        OperationUnavailableError,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"
    config = cast(Config, request.config())

    try:
        payload, served_by = dispatch_read(
            config,
            lower_session_read(invocation.session_id, kind=kind),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except (OperationFailedError, OperationUnavailableError) as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise OperationEnvelopeError(f"session.read {kind} result carries no evidence body")
    if bool(request.params.get("verbose")):
        click.echo(f"served-by: {served_by.line()}", err=True)

    if output_format == "json":
        # Machine output is rendered as raw bytes so Rich markup never rewrites
        # JSON and read-view delivery can capture file/clipboard targets.
        content = json.dumps(evidence, indent=2) + "\n"
    else:
        import yaml

        content = yaml.dump(evidence) + "\n"

    if invocation.destination in (RenderDestination.FILE, RenderDestination.CLIPBOARD, RenderDestination.STDOUT):
        deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)
        return
    click.echo(content, nl=False)


def run_read_hooks(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the per-session hook-event summary from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="hooks")


def run_read_file_edits(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render captured Edit/Write/MultiEdit tool-call evidence from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="file-edits")


def run_read_agent_policies(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render sandbox/approval/network policy facts from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="agent-policies")


def run_read_web_content(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render typed web-export constructs from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="web-content")
