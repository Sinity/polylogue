"""Context seed commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("context")
@click.option("--from-verify", is_flag=True, help="Compile a debugging seed from the latest failed verify run.")
@click.option(
    "--failure-context", type=click.Path(path_type=Path), default=None, help="Workspace failure-context JSON envelope."
)
@click.option("--format", "output_format", type=click.Choice(["json", "plain"]), default="json", show_default=True)
def context_command(from_verify: bool, failure_context: Path | None, output_format: str) -> None:
    """Compile bounded context seeds for the next agent session."""
    if not from_verify:
        raise click.UsageError("choose --from-verify")
    from polylogue.context.failure_seed import compile_failure_seed

    try:
        payload = compile_failure_seed(envelope_path=failure_context)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    seed = payload["seed"]
    click.echo(f"Debug seed: {', '.join(seed['failure_tests']) or 'unknown failing test'}")
    click.echo(f"Implicated files: {', '.join(seed['implicated_files']) or 'none recorded'}")
    click.echo(f"Next command: {seed['next_command']}")


__all__ = ["context_command"]
