"""Interactive configuration initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..settings import Settings, persist_settings, SETTINGS_PATH
from ..commands import CommandEnv
from ..drive_client import DriveClient


def run_init_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    """Run interactive configuration wizard.

    Guides user through:
    - Output directory setup
    - HTML preview preferences
    - Theme selection
    - Collapse threshold configuration
    - Google Drive credentials setup (optional)
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
            console.print("[yellow]Use --force to overwrite, or run 'polylogue config set' to modify.")
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

    # Step 4: Collapse threshold
    console.print(f"\n[bold]Collapse Threshold[/bold]")
    console.print("How many lines before long outputs are collapsed?")

    if not ui.plain:
        collapse_input = ui.input("  [25]: ", default="25")
        try:
            collapse_threshold = int(collapse_input) if collapse_input else 25
        except ValueError:
            collapse_threshold = 25
            console.print("  [yellow]Invalid number, using default: 25")
    else:
        collapse_threshold = 25
        console.print("  Using default: 25")

    # Step 5: Drive credentials (optional)
    drive_setup = False
    console.print(f"\n[bold]Google Drive Integration[/bold]")
    console.print("Set up Drive credentials now for syncing conversations from Google Drive?")
    console.print("[dim](You can skip this and set up later if needed)[/dim]")

    if not ui.plain:
        drive_setup = ui.confirm("  Set up Drive credentials?", default=False)
    else:
        console.print("  Skipping in plain mode (run 'polylogue sync drive' later to set up)")

    if drive_setup:
        try:
            drive_client = DriveClient(ui)
            drive_client.ensure_credentials()
            console.print("[green]✓ Drive credentials configured")
        except (SystemExit, KeyboardInterrupt):
            console.print("[yellow]Drive setup skipped")
        except Exception as e:
            console.print(f"[yellow]Drive setup failed: {e}")

    # Save settings
    settings = Settings(
        html_previews=html_enabled,
        html_theme=theme,
        collapse_threshold=collapse_threshold
    )
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
    console.print(f"  Collapse threshold: [cyan]{collapse_threshold}[/cyan]")
    if drive_setup:
        console.print(f"  Drive credentials: [cyan]configured[/cyan]")

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print("  [green]polylogue help[/green]                 # See commands and examples")
    console.print("  [green]polylogue search 'query'[/green]       # Search transcripts")
    console.print("  [green]polylogue import chatgpt FILE[/green]  # Import ChatGPT export")
    console.print("  [green]polylogue sync codex[/green]           # Sync Codex sessions")
    if drive_setup:
        console.print("  [green]polylogue sync drive[/green]           # Sync from Google Drive")
    console.print()
