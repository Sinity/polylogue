"""Message and raw-session read-view handlers."""

from __future__ import annotations

import io

import click

from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def run_read_messages(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route messages view to messages renderer with destination handling."""

    from polylogue.cli.messages import run_messages

    assert invocation.session_id is not None
    options = invocation.messages
    limit = options.limit if options.limit is not None else 50

    if invocation.destination in ("file", "clipboard"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_messages(
                env,
                request,
                session_id=invocation.session_id,
                message_role=options.role,
                message_type=options.message_type,
                limit=limit,
                offset=options.offset,
                no_code_blocks=options.no_code_blocks,
                no_tool_calls=options.no_tool_calls,
                no_tool_outputs=options.no_tool_outputs,
                no_file_reads=options.no_file_reads,
                prose_only=options.prose_only,
                output_format=invocation.output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_messages(
        env,
        request,
        session_id=invocation.session_id,
        message_role=options.role,
        message_type=options.message_type,
        limit=limit,
        offset=options.offset,
        no_code_blocks=options.no_code_blocks,
        no_tool_calls=options.no_tool_calls,
        no_tool_outputs=options.no_tool_outputs,
        no_file_reads=options.no_file_reads,
        prose_only=options.prose_only,
        output_format=invocation.output_format,
    )


def run_read_raw(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route raw view to raw renderer with destination handling."""

    from polylogue.cli.messages import run_raw

    assert invocation.session_id is not None
    options = invocation.messages
    limit = options.limit if options.limit is not None else 50
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo_raw(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_raw  # type: ignore[assignment]
        try:
            run_raw(
                env,
                request,
                session_id=invocation.session_id,
                limit=limit,
                offset=options.offset,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_raw(
        env,
        request,
        session_id=invocation.session_id,
        limit=limit,
        offset=options.offset,
        output_format=output_format,
    )


__all__ = ["run_read_messages", "run_read_raw"]
