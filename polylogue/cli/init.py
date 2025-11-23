"""Interactive configuration initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..settings import Settings, persist_settings, SETTINGS_PATH
from ..commands import CommandEnv


def run_init_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    """Run interactive configuration wizard.

    Guides user through:
    - Output directory setup
    - HTML preview preferences
    - Theme selection
    - Directory creation

    Args:
        args: Command arguments (includes --force to overwrite existing config)
        env: Command environment with UI
    """
    ui = env.ui
    console = ui.console

    # Check if config already exists
    if SETTINGS_PATH.exists() and not getattr(args, "force", False):
        if not ui.plain:
            if not ui.confirm(
                f"Configuration already exists at {SETTINGS_PATH}. Overwrite?",
                default=False
            ):
                console.print("[yellow]Init cancelled.")
                return
        else:
            console.print(f"[yellow]Configuration exists at {SETTINGS_PATH}")
            console.print("[yellow]Use --force to overwrite, or run 'polylogue settings' to modify.")
            return

    console.print("\n[bold cyan]Welcome to Polylogue![/bold cyan]")
    console.print("Let's set up your configuration.\n")

    # Step 1: Output directory
    default_dir = Path.home() / "polylogue-data"
    console.print(f"[bold]Output Directory[/bold]")
    console.print(f"Where should rendered conversations be saved?")

    if not ui.plain:
        output_dir_input = ui.input(
            f"  [{default_dir}]: ",
            default=str(default_dir)
        )
        output_dir = Path(output_dir_input).expanduser()
    else:
        output_dir = default_dir
        console.print(f"  Using default: {default_dir}")

    # Step 2: HTML previews
    console.print(f"\n[bold]HTML Previews[/bold]")
    console.print("Generate interactive HTML files alongside Markdown?")

    if not ui.plain:
        html_enabled = ui.confirm("  Enable HTML previews?", default=True)
    else:
        html_enabled = True
        console.print("  Using default: Yes")

    # Step 3: Theme
    console.print(f"\n[bold]HTML Theme[/bold]")
    if not ui.plain:
        theme_options = ["light", "dark"]
        theme = ui.choose("  Choose theme:", theme_options)
        if not theme:
            theme = "dark"
    else:
        theme = "dark"
        console.print("  Using default: dark")

    # Save settings
    settings = Settings(html_previews=html_enabled, html_theme=theme)
    persist_settings(settings)

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[green]✓ Created output directory: {output_dir}")
    except OSError as e:
        console.print(f"\n[yellow]Warning: Could not create {output_dir}: {e}")
        console.print(f"[yellow]You may need to create it manually.")

    # Summary
    console.print(f"\n[bold green]✓ Configuration saved to {SETTINGS_PATH}[/bold green]\n")
    console.print("[bold]Your settings:[/bold]")
    console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
    console.print(f"  HTML previews: [cyan]{'enabled' if html_enabled else 'disabled'}[/cyan]")
    console.print(f"  Theme: [cyan]{theme}[/cyan]")

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print("  [green]polylogue help --examples[/green]     # See usage examples")
    console.print("  [green]polylogue import chatgpt FILE[/green] # Import ChatGPT export")
    console.print("  [green]polylogue sync codex[/green]          # Sync Codex sessions")
    console.print()
