"""polylogue commands — discoverable command surface audit (#1681)."""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv

_COMMAND_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Query & Search": ("list", "count", "stats", "read", "recent"),
    "Archive Management": ("import", "check", "reset", "backup", "maintenance"),
    "Insights & Analytics": ("insights", "analyze", "resume"),
    "User-State Objects": ("user-state", "blackboard", "tags", "feedback"),
    "Configuration": ("config", "init", "ops auth", "ops completions", "dashboard", "tutorial"),
    "Embeddings": ("embed",),
    "Schema": ("ops schema",),
    "Diagnostics": ("diagnostics", "status"),
}


@click.command("commands")
@click.pass_obj
def commands_command(env: AppEnv) -> None:
    """List available polylogue commands grouped by category (#1681)."""
    env.ui.console.print("\n[bold]Polylogue Commands[/bold]\n")
    for category, cmds in _COMMAND_CATEGORIES.items():
        env.ui.console.print(f"[bold]{category}[/bold]")
        env.ui.console.print(f"  {', '.join(cmds)}")
        env.ui.console.print()
