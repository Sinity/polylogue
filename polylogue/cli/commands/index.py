"""Index command."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.container import create_config
from polylogue.cli.formatting import format_index_status
from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.ingestion import DriveError
from polylogue.pipeline.runner import run_sources


@click.command("index")
@click.option("--config", type=click.Path(path_type=Path), help="Path to config file")
@click.pass_obj
def index_command(env: AppEnv, config: Path | None) -> None:
    try:
        cfg = create_config(config or env.config_path)
    except ConfigError as exc:
        fail("index", str(exc))
    try:
        result = run_sources(
            config=cfg,
            stage="index",
            plan=None,
            ui=env.ui,
            source_names=None,
        )
    except DriveError as exc:
        fail("index", str(exc))
    env.ui.summary(
        "Index",
        [
            format_index_status("index", result.indexed, result.index_error),
            f"Duration: {result.duration_ms}ms",
        ],
    )
    if result.index_error:
        error_line = f"Index error: {result.index_error}"
        hint_line = "Hint: run `polylogue index` to rebuild the index."
        if env.ui.plain:
            env.ui.console.print(error_line)
            env.ui.console.print(hint_line)
        else:
            env.ui.console.print(f"[yellow]{error_line}[/yellow]")
            env.ui.console.print(f"[yellow]{hint_line}[/yellow]")
