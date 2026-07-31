"""File-edit and agent-policy evidence read-view handlers.

Renders two index-tier relations that had a complete, tested read chain
terminating at the repository layer with no surface consumer above it
(polylogue-nua7): ``file_edits`` (Claude Code Edit/Write/MultiEdit
structured diffs / pre-edit file content / old-new string pairs) and
``session_agent_policies`` (Codex sandbox/approval/network policy facts).
"""

from __future__ import annotations

import io

import click

from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

__all__ = ["run_read_agent_policies", "run_read_file_edits"]


def run_read_file_edits(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route the file-edits view to the file-edit evidence renderer."""

    from polylogue.cli.messages import run_session_file_edits

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_session_file_edits(
                env,
                request,
                session_id=invocation.session_id,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_session_file_edits(
        env,
        request,
        session_id=invocation.session_id,
        output_format=output_format,
    )


def run_read_agent_policies(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route the agent-policies view to the agent-policy evidence renderer."""

    from polylogue.cli.messages import run_session_agent_policies

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_session_agent_policies(
                env,
                request,
                session_id=invocation.session_id,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_session_agent_policies(
        env,
        request,
        session_id=invocation.session_id,
        output_format=output_format,
    )
