"""Web-content-construct evidence read-view handler.

Renders ``web_content_constructs`` (polylogue-kktg): a 155k+-row index
relation that had a complete, tested write path from the ChatGPT/Claude
parsers but no reader above the storage layer -- every production SELECT
against it existed only to DELETE orphans (an integrity sweep) or as a demo
smoke-probe COUNT(*).
"""

from __future__ import annotations

import io

import click

from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

__all__ = ["run_read_web_content_constructs"]


def run_read_web_content_constructs(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route the web-content view to the web-content-construct evidence renderer."""

    from polylogue.cli.messages import run_session_web_content_constructs

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_session_web_content_constructs(
                env,
                request,
                session_id=invocation.session_id,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_session_web_content_constructs(
        env,
        request,
        session_id=invocation.session_id,
        output_format=output_format,
    )
