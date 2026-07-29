"""Raw session-events read-view handler.

Renders ``Session.session_events`` -- provider evidence that rides the
session timeline instead of a dialogue message (Codex ``world_state``/
``agent_policy``/``turn_context`` policy facts, Claude Code sidecar events,
Hermes tool-availability/step spans, and similar). Before this view, the data
was populated on every full session read (for content hashing and a couple of
insight materializers) but never rendered or queried on any surface.
"""

from __future__ import annotations

import io
from typing import cast

import click

from polylogue.cli.read_view_registry import EVENTS_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewEventsOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

__all__ = ["EVENTS_READ_VIEW_OPTION_NAMES", "build_events_options", "run_read_events"]


def build_events_options(values: ReadViewOptionValues) -> ReadViewEventsOptions:
    """Build options owned by the events read view."""

    return ReadViewEventsOptions(limit=cast(int | None, values.get("limit")))


def run_read_events(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route the events view to the session-events renderer."""

    from polylogue.cli.messages import run_session_events

    assert invocation.session_id is not None
    options = cast(ReadViewEventsOptions, invocation.options or ReadViewEventsOptions())
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo_events(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_events  # type: ignore[assignment]
        try:
            run_session_events(
                env,
                request,
                session_id=invocation.session_id,
                limit=options.limit,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_session_events(
        env,
        request,
        session_id=invocation.session_id,
        limit=options.limit,
        output_format=output_format,
    )
