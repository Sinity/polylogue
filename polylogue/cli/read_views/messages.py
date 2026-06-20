"""Message and raw-session read-view handlers."""

from __future__ import annotations

import io
from typing import cast

import click

from polylogue.cli.read_views.base import (
    ReadViewInvocation,
    ReadViewMessageOptions,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

MESSAGE_READ_VIEW_OPTION_NAMES = frozenset(
    {
        "limit",
        "message_role",
        "message_type",
        "no_code_blocks",
        "no_file_reads",
        "no_tool_calls",
        "no_tool_outputs",
        "offset",
        "prose_only",
    }
)


def build_message_options(values: ReadViewOptionValues) -> ReadViewMessageOptions:
    """Build options shared by the messages and raw read views."""

    return ReadViewMessageOptions(
        limit=cast(int | None, values.get("limit")),
        offset=cast(int, values.get("offset", 0)),
        role=cast(tuple[str, ...], values.get("message_role", ())),
        message_type=cast(str | None, values.get("message_type")),
        no_code_blocks=cast(bool, values.get("no_code_blocks", False)),
        no_tool_calls=cast(bool, values.get("no_tool_calls", False)),
        no_tool_outputs=cast(bool, values.get("no_tool_outputs", False)),
        no_file_reads=cast(bool, values.get("no_file_reads", False)),
        prose_only=cast(bool, values.get("prose_only", False)),
    )


def run_read_messages(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route messages view to messages renderer with destination handling."""

    from polylogue.cli.messages import run_messages

    assert invocation.session_id is not None
    options = cast(ReadViewMessageOptions, invocation.options or ReadViewMessageOptions())
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
    options = cast(ReadViewMessageOptions, invocation.options or ReadViewMessageOptions())
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


__all__ = ["MESSAGE_READ_VIEW_OPTION_NAMES", "build_message_options", "run_read_messages", "run_read_raw"]
