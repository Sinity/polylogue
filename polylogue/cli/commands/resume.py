"""Resume command for reconstructing agent handoff context."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.insight_command_contracts import find_root_params
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.resume_rendering import render_resume_brief
from polylogue.cli.shared.types import AppEnv


def _wants_json(ctx: click.Context, *, json_mode: bool, output_format: str | None) -> bool:
    if json_mode or output_format == "json":
        return True
    root_output = find_root_params(ctx).get("output_format")
    return root_output == "json"


@click.command("resume")
@click.argument("session_id")
@click.option("--related-limit", type=int, default=6, show_default=True, help="Maximum related sessions to include.")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def resume_command(
    ctx: click.Context,
    session_id: str,
    related_limit: int,
    json_mode: bool,
    output_format: str | None,
) -> None:
    """Reconstruct work-state context for a fresh agent session."""
    env: AppEnv = ctx.obj
    brief = run_coroutine_sync(
        env.operations.build_resume_brief(
            session_id,
            related_limit=related_limit,
        )
    )
    if brief is None:
        fail("resume", f"Conversation not found: {session_id}")
    if _wants_json(ctx, json_mode=json_mode, output_format=output_format):
        emit_success(brief.model_dump(mode="json"))
        return
    render_resume_brief(brief)


__all__ = ["resume_command"]
