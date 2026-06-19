"""Resume commands for reconstructing agent handoff context."""

from __future__ import annotations

import click

from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.insight_command_contracts import find_root_params
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.resume_rendering import render_resume_brief
from polylogue.cli.shared.types import AppEnv


class ResumeGroup(click.Group):
    """Click group that keeps ``resume SESSION_ID`` as the default action."""

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            return "session", self.commands["session"], args
        return super().resolve_command(ctx, args)


def _wants_json(ctx: click.Context, *, output_format: str | None) -> bool:
    if output_format == "json":
        return True
    root_output = find_root_params(ctx).get("output_format")
    return root_output == "json"


@click.group(
    "resume",
    cls=ResumeGroup,
)
def resume_command() -> None:
    """Reconstruct work-state context for a fresh agent session."""


@resume_command.command("session", hidden=True)
@click.argument("session_id")
@click.option("--related-limit", type=int, default=6, show_default=True, help="Maximum related sessions to include.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def resume_session_command(
    ctx: click.Context,
    session_id: str,
    related_limit: int,
    output_format: str | None,
) -> None:
    """Reconstruct work-state context for ``session_id``."""
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.insights.resume import build_resume_brief

    env: AppEnv = ctx.obj
    brief = run_coroutine_sync(
        build_resume_brief(
            env.polylogue,
            session_id,
            related_limit=related_limit,
        )
    )
    if brief is None:
        fail("resume", f"Session not found: {session_id}")
    if _wants_json(ctx, output_format=output_format):
        emit_success(brief.model_dump(mode="json"))
        return
    render_resume_brief(brief)


@resume_command.command("candidates")
@click.option("--repo", "repo_path", required=True, help="Repository path to rank sessions against.")
@click.option("--cwd", default=None, help="Current working directory for prefix matching.")
@click.option("--recent", "recent_files", multiple=True, help="Recently touched file path. Repeatable.")
@click.option("--limit", type=int, default=10, show_default=True, help="Maximum candidates to return.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def resume_candidates_command(
    ctx: click.Context,
    repo_path: str,
    cwd: str | None,
    recent_files: tuple[str, ...],
    limit: int,
    output_format: str | None,
) -> None:
    """Rank archived logical sessions for the current working context."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    env: AppEnv = ctx.obj
    candidates = run_coroutine_sync(
        env.polylogue.find_resume_candidates(
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=limit,
        )
    )
    payload = {
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "total": len(candidates),
    }
    if _wants_json(ctx, output_format=output_format):
        emit_success(payload)
        return
    for candidate in candidates:
        click.echo(f"{candidate.score:.3f} {candidate.logical_session_id} {candidate.title}")


__all__ = ["resume_command"]
