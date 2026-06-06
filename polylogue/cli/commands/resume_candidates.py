"""Resume-candidate command for read-pull context discovery."""

from __future__ import annotations

import click

from polylogue.cli.shared.insight_command_contracts import find_root_params
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv


def _wants_json(ctx: click.Context, *, output_format: str | None) -> bool:
    if output_format == "json":
        return True
    root_output = find_root_params(ctx).get("output_format")
    return root_output == "json"


@click.command("resume-candidates")
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


__all__ = ["resume_candidates_command"]
