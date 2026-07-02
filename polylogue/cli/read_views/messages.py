"""Message and raw-session read-view handlers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import click

from polylogue.api.archive import SessionNotFoundError
from polylogue.api.sync.bridge import run_coroutine_sync
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
from polylogue.surfaces.payloads import SessionMessageRowPayload, model_json_document


def build_message_options(values: ReadViewOptionValues) -> ReadViewMessageOptions:
    """Build options shared by the messages and raw read views."""

    return ReadViewMessageOptions(
        limit=cast(int | None, values.get("limit")),
        offset=cast(int, values.get("offset", 0)),
        full=cast(bool, values.get("full", False)),
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
    limit = limit if limit is not None else 50
    offset = projection.body_offset if projection is not None and projection.body_offset is not None else options.offset

    if invocation.destination == "file" and invocation.output_format in {"json", "ndjson"}:
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
        )
        return

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
                limit=limit,
                offset=offset,
                full=options.full,
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
        limit=limit,
        offset=offset,
        full=options.full,
        output_format=invocation.output_format,
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
) -> None:
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            try:
                _, total = await api.get_messages_paginated(session_id, limit=1, offset=0)
            except SessionNotFoundError:
                env.ui.error(f"Session not found: {session_id}")
                return

            effective_limit = max(total - offset, 0) if full else limit
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                if output_format == "ndjson":
                    emitted = 0
                    async for message in api.iter_messages(session_id, limit=offset + effective_limit):
                        if emitted < offset:
                            emitted += 1
                            continue
                        payload = {
                            "session_id": session_id,
                            **model_json_document(
                                SessionMessageRowPayload.from_message(message, session_id=session_id),
                                exclude_none=True,
                            ),
                        }
                        fh.write(json.dumps(payload))
                        fh.write("\n")
                        emitted += 1
                    return

                fh.write("{\n")
                fh.write(f'  "session_id": {json.dumps(session_id)},\n')
                fh.write('  "messages": [')
                first = True
                emitted = 0
                async for message in api.iter_messages(session_id, limit=offset + effective_limit):
                    if emitted < offset:
                        emitted += 1
                        continue
                    if not first:
                        fh.write(",")
                    fh.write("\n    ")
                    fh.write(
                        json.dumps(
                            model_json_document(
                                SessionMessageRowPayload.from_message(message, session_id=session_id),
                                exclude_none=True,
                            ),
                            indent=2,
                        ).replace("\n", "\n    ")
                    )
                    first = False
                    emitted += 1
                fh.write("\n  ],\n")
                fh.write(f'  "total": {total},\n')
                fh.write(f'  "limit": {effective_limit},\n')
                fh.write(f'  "offset": {offset}\n')
                fh.write("}\n")

        click.echo(f"Wrote to {out_path}")

    run_coroutine_sync(_run())


def run_read_raw(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route raw view to raw renderer with destination handling."""

    from polylogue.cli.messages import run_raw

    assert invocation.session_id is not None
    options = cast(ReadViewMessageOptions, invocation.options or ReadViewMessageOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    limit = projection.body_limit if projection is not None and projection.body_limit is not None else options.limit
    limit = limit if limit is not None else 50
    offset = projection.body_offset if projection is not None and projection.body_offset is not None else options.offset
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


__all__ = ["MESSAGE_READ_VIEW_OPTION_NAMES", "build_message_options", "run_read_messages", "run_read_raw"]
