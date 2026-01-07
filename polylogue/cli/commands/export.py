"""Export command."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.export import export_jsonl


@click.command("export")
@click.option("--out", type=click.Path(path_type=Path), help="Write JSONL export to path")
@click.pass_obj
def export_command(env: AppEnv, out: Path | None) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("export", str(exc))
    target = export_jsonl(archive_root=config.archive_root, output_path=out)
    env.ui.console.print(f"Exported {target}")
