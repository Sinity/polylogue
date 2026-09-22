"""Message-window read-view handlers."""

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
    """Build options owned by the messages read view."""

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


def _fsync_directory(directory: Path) -> None:
    """Make a rename into ``directory`` durable, not merely visible.

    ``os.replace`` publishes the new name atomically with respect to readers,
    but the directory entry itself is not on stable storage until its parent is
    synced; a crash between the two can leave the operator with neither the old
    export nor the new one.
    """
    import os

    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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

    The destination is replaced, never truncated in place. ``read_message_windows``
    is a generator, so the first read does not happen until the loop below --
    and opening ``out_path`` with ``"w"`` had already destroyed the operator's
    previous file and written a partial JSON header by then. A failing read
    (a missing session reference is the ordinary case) then reported the
    failure while leaving malformed output where a good file used to be. The
    rows still stream: they stream into a sibling temporary file, which is
    renamed over the destination only once the sequence has completed.
    """

    import os
    import stat
    import tempfile

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
    # A sibling of the destination, so the rename below is within one
    # filesystem and therefore atomic; a temporary directory elsewhere would
    # degrade the replacement into a copy that can fail half-written.
    descriptor, staged_name = tempfile.mkstemp(dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".partial")
    staged = Path(staged_name)
    try:
        # ``mkstemp`` creates at 0600. Replacing an existing destination keeps
        # that destination's mode, so a rewrite does not silently change the
        # permissions of a file the operator already placed; a new destination
        # keeps the private default rather than widening it to the umask.
        existing_mode = out_path.stat().st_mode if out_path.exists() else None
        if existing_mode is not None:
            os.chmod(staged, stat.S_IMODE(existing_mode))
    except OSError:
        pass
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
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
        staged.unlink(missing_ok=True)
        from polylogue.cli.messages import message_read_failure

        message_read_failure(env, exc, session_id=session_id)
        return
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, out_path)
    _fsync_directory(out_path.parent)

    notice = describe_path_scan_result(scan_path_for_secret_candidates(out_path))
    if notice is not None:
        click.echo(notice)
    click.echo(f"Wrote to {out_path}")


__all__ = [
    "MESSAGE_READ_VIEW_OPTION_NAMES",
    "build_message_options",
    "run_read_messages",
]
