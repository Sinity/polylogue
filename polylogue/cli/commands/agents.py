"""Agent coordination CLI projections."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click

if TYPE_CHECKING:
    from polylogue.coordination import AgentCoordinationPayload, CoordinationView

F = TypeVar("F", bound=Callable[..., object])


def _build_coordination_envelope(
    *,
    view: CoordinationView,
    cwd: Path | None,
    limit: int,
    detail: bool,
) -> AgentCoordinationPayload:
    """Load the coordination substrate only when a projection is requested."""
    from polylogue.coordination import build_coordination_envelope

    return build_coordination_envelope(view=view, cwd=cwd, limit=limit, detail=detail)


def _format_options(func: F) -> F:
    func = click.option(
        "--cwd", type=click.Path(file_okay=False, path_type=Path), default=None, help="Repo/cwd to inspect."
    )(func)
    func = click.option("--limit", "-l", type=int, default=10, show_default=True, help="Maximum peer/resource rows.")(
        func
    )
    func = click.option("--json", "json_output", is_flag=True, help="Emit JSON.")(func)
    func = click.option(
        "--detail",
        is_flag=True,
        help="Include bounded evidence details omitted from the compact default.",
    )(func)
    func = click.option(
        "--format",
        "-f",
        "output_format",
        type=click.Choice(["text", "json", "markdown", "tree"]),
        default="json",
        show_default=True,
        help="Output format.",
    )(func)
    return func


def _emit(
    view: CoordinationView,
    cwd: Path | None,
    limit: int,
    json_output: bool,
    output_format: str,
    detail: bool,
) -> None:
    payload = _build_coordination_envelope(view=view, cwd=cwd, limit=limit, detail=detail)
    if json_output or output_format == "json":
        click.echo(payload.to_json(exclude_none=True))
        return
    if output_format == "markdown":
        from polylogue.coordination.rendering import render_coordination_markdown

        click.echo(render_coordination_markdown(payload), nl=False)
        return
    if output_format == "tree":
        from polylogue.coordination.rendering import render_coordination_tree

        click.echo(render_coordination_tree(payload))
        return
    from polylogue.coordination.rendering import render_coordination_text

    click.echo(render_coordination_text(payload))


@click.group("agents", invoke_without_command=True)
@click.pass_context
def agents_command(ctx: click.Context) -> None:
    """Inspect agent coordination state from shared local evidence."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(status_command)


@agents_command.command("status")
@_format_options
def status_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Show the compact bounded coordination envelope."""
    _emit("status", cwd, limit, json_output, output_format, detail)


@agents_command.command("self")
@_format_options
def self_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Show this agent's repo, process, and current work-item projection."""
    _emit("self", cwd, limit, json_output, output_format, detail)


@agents_command.command("work-item")
@_format_options
def work_item_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Show the current work item projection."""
    _emit("work-item", cwd, limit, json_output, output_format, detail)


@agents_command.command("current")
@_format_options
def current_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Alias for ``work-item``."""
    _emit("work-item", cwd, limit, json_output, output_format, detail)


@agents_command.command("conflicts")
@_format_options
def conflicts_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Show overlap/resource awareness; same-file activity is not a blocker."""
    _emit("conflicts", cwd, limit, json_output, output_format, detail)


@agents_command.command("overlap")
@_format_options
def overlap_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Alias for ``conflicts``."""
    _emit("conflicts", cwd, limit, json_output, output_format, detail)


@agents_command.command("handoff")
@_format_options
def handoff_command(cwd: Path | None, limit: int, json_output: bool, output_format: str, detail: bool) -> None:
    """Show supported live handoff references for the current repo."""
    _emit("handoff", cwd, limit, json_output, output_format, detail)


__all__ = ["agents_command"]
