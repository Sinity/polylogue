"""Delivery and external-output helpers for CLI query output."""

from __future__ import annotations

import os
import subprocess
import tempfile
import webbrowser
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation


def send_output(
    env: AppEnv,
    content: str,
    destinations: list[str],
    output_format: str,
    conv: Conversation | None,
) -> None:
    for dest in destinations:
        if dest == "stdout":
            click.echo(content)
        elif dest == "browser":
            open_in_browser(env, content, output_format, conv)
        elif dest == "clipboard":
            copy_to_clipboard(env, content)
        else:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            env.ui.console.print(f"Wrote to {path}")


def open_in_browser(
    env: AppEnv,
    content: str,
    output_format: str,
    conv: Conversation | None,
) -> None:
    if output_format != "html":
        if conv:
            from polylogue.rendering.formatting import _conv_to_html

            content = _conv_to_html(conv)
        else:
            content = f"<html><body><pre>{html_escape(content)}</pre></body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temp_path = handle.name

    webbrowser.open(f"file://{temp_path}")
    env.ui.console.print(f"Opened in browser: {temp_path}")


def copy_to_clipboard(env: AppEnv, content: str) -> None:
    clipboard_cmds = [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["clip"],
    ]

    for cmd in clipboard_cmds:
        try:
            subprocess.run(
                cmd,
                input=content.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            env.ui.console.print("Copied to clipboard.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    click.echo("Could not copy to clipboard (no clipboard tool found).", err=True)


def open_result(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, object],
) -> None:
    if not results:
        env.ui.console.print("No conversations matched.")
        raise SystemExit(2)

    conv = results[0]

    from polylogue.cli.helpers import latest_render_path, load_effective_config

    try:
        config = load_effective_config(env)
    except Exception as exc:
        logger.warning("Config load failed, falling back to defaults: %s", exc)
        config = None

    render_root = None
    if config and hasattr(config, "render_root") and config.render_root:
        render_root = Path(config.render_root)
    else:
        render_root_env = os.environ.get("POLYLOGUE_RENDER_ROOT")
        if render_root_env:
            render_root = Path(render_root_env)

    if not render_root or not render_root.exists():
        click.echo("No rendered outputs found.", err=True)
        click.echo("Run 'polylogue run' first to render conversations.", err=True)
        raise SystemExit(1)

    conv_id_short = str(conv.id)[:8] if conv.id else ""
    html_file = next(render_root.rglob(f"*{conv_id_short}*/conversation.html"), None)
    md_file = next(render_root.rglob(f"*{conv_id_short}*/conversation.md"), None)

    render_file = html_file or md_file
    if not render_file:
        render_file = latest_render_path(render_root)

    if not render_file:
        click.echo("No rendered output found for this conversation.", err=True)
        click.echo("Run 'polylogue run' to render conversations.", err=True)
        raise SystemExit(1)

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")
