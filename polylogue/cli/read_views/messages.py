"""Message and raw-session read-view handlers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import click

from polylogue.archive.query.spec import DEFAULT_MESSAGE_PAGE_LIMIT
from polylogue.cli.read_view_registry import MESSAGE_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewInvocation,
    ReadViewMessageOptions,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.projection_spec import RenderDestination


def build_message_options(values: ReadViewOptionValues) -> ReadViewMessageOptions:
    """Build options shared by the messages and raw read views."""

    return ReadViewMessageOptions(
        limit=cast(int | None, values.get("limit")),
        offset=cast(int, values.get("offset", 0)),
        full=cast(bool, values.get("full", False)),
        continuation=cast(str | None, values.get("continuation")),
        around=cast(str | None, values.get("around")),
    )


def run_read_messages(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route messages view to messages renderer with destination handling."""

    from polylogue.cli.messages import run_messages

    assert invocation.session_id is not None
    options = cast(ReadViewMessageOptions, invocation.options or ReadViewMessageOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    limit = projection.body_limit if projection is not None and projection.body_limit is not None else options.limit
    if options.full:
        limit = None
    limit = limit if limit is not None else DEFAULT_MESSAGE_PAGE_LIMIT
    offset = projection.body_offset if projection is not None and projection.body_offset is not None else options.offset

    if invocation.destination == RenderDestination.FILE and invocation.output_format in {"json", "ndjson"}:
        assert invocation.out_path is not None
        _write_messages_file(
            env,
            request,
            session_id=invocation.session_id,
            limit=limit,
            offset=offset,
            full=options.full,
            output_format=invocation.output_format,
            out_path=Path(invocation.out_path),
            around=options.around,
        )
        return

    if invocation.destination in (RenderDestination.FILE, RenderDestination.CLIPBOARD):
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
                limit=limit,
                offset=offset,
                full=options.full,
                output_format=invocation.output_format,
                around=options.around,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_messages(
        env,
        request,
        session_id=invocation.session_id,
        limit=limit,
        offset=offset,
        full=options.full,
        output_format=invocation.output_format,
        continuation=options.continuation,
        around=options.around,
    )


def _write_messages_file(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int,
    offset: int,
    full: bool,
    output_format: str,
    out_path: Path,
    around: str | None = None,
) -> None:
    """Stream one message window sequence straight to a file.

    Written window by window rather than composed in memory first: ``--full``
    on a long session is exactly the case this destination exists for, and the
    declared read answers it as a bounded sequence, so the rows are serialized
    as they arrive.
    """

    from polylogue.cli.messages import read_message_windows
    from polylogue.cli.operation_kernel import OperationKernelError
    from polylogue.cli.read_dispatch import daemon_route_disabled
    from polylogue.security.secret_scan import describe_path_scan_result, scan_path_for_secret_candidates

    config = cast(Config, request.config())
    windows = read_message_windows(
        config,
        session_id,
        limit=limit,
        offset=offset,
        full=full,
        continuation=None,
        daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        around=around,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    total = 0
    first_offset = offset
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            if output_format != "ndjson":
                fh.write("{\n")
                fh.write(f'  "session_id": {json.dumps(session_id)},\n')
                fh.write('  "messages": [')
            for window in windows:
                if emitted == 0:
                    first_offset = window.offset
                total = window.total
                for row in window.rows:
                    document = dict(row)
                    if output_format == "ndjson":
                        fh.write(json.dumps({"session_id": session_id, **document}))
                        fh.write("\n")
                    else:
                        if emitted:
                            fh.write(",")
                        fh.write("\n    ")
                        fh.write(json.dumps(document, indent=2).replace("\n", "\n    "))
                    emitted += 1
            if output_format != "ndjson":
                fh.write("\n  ],\n")
                fh.write(f'  "total": {total},\n')
                fh.write(f'  "limit": {emitted if full else limit},\n')
                fh.write(f'  "offset": {first_offset}\n')
                fh.write("}\n")
    except OperationKernelError as exc:
        from polylogue.cli.messages import message_read_failure

        message_read_failure(env, exc, session_id=session_id)
        return

    notice = describe_path_scan_result(scan_path_for_secret_candidates(out_path))
    if notice is not None:
        click.echo(notice)
    click.echo(f"Wrote to {out_path}")


def run_read_raw(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route raw view to raw renderer with destination handling."""

    from polylogue.cli.messages import run_raw

    assert invocation.session_id is not None
    options = cast(ReadViewMessageOptions, invocation.options or ReadViewMessageOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    limit = projection.body_limit if projection is not None and projection.body_limit is not None else options.limit
    limit = limit if limit is not None else DEFAULT_MESSAGE_PAGE_LIMIT
    offset = projection.body_offset if projection is not None and projection.body_offset is not None else options.offset
    output_format = invocation.output_format or "json"

    if invocation.destination in (RenderDestination.FILE, RenderDestination.CLIPBOARD, RenderDestination.STDOUT):
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
                offset=offset,
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
        offset=offset,
        output_format=output_format,
    )


def run_read_hooks(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the per-session hook-event summary from ``session.read``.

    The hook read model is a per-session evidence relation with no query-grammar
    unit of its own, so design D3 classifies this view as a ``session.read``
    projection.  The view lowers and dispatches; it never opens an archive and
    never branches on whether a daemon is present -- which executor answers is
    the kernel's decision, recorded in the result's own authority.
    """

    from polylogue.cli.lowering import lower_session_read
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationFailedError,
        OperationUnavailableError,
        dispatch,
    )

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"
    config = cast(Config, request.config())

    try:
        result = dispatch(config, lower_session_read(invocation.session_id, kind="hooks"))
    except (OperationFailedError, OperationUnavailableError) as exc:
        # A refusal names itself and exits non-zero; it must never render as an
        # empty summary that reads "this session recorded no hook events".
        # Classed exactly as the transcript path classes its refusals, through
        # the one CLI read-failure terminal (polylogue-jtrtj).
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    if not isinstance(result.value, dict):
        raise OperationEnvelopeError("session.read returned a non-object result")
    evidence = result.value.get("evidence")
    if not isinstance(evidence, dict):
        raise OperationEnvelopeError("session.read hooks result carries no evidence body")

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


__all__ = [
    "MESSAGE_READ_VIEW_OPTION_NAMES",
    "build_message_options",
    "run_read_hooks",
    "run_read_messages",
    "run_read_raw",
]
