"""Interactive configuration initialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from ..settings import Settings, persist_settings, SETTINGS_PATH
from ..commands import CommandEnv
from ..drive_client import DriveClient
from ..config import persist_config, IndexConfig, is_config_declarative


def run_init_cli(args: SimpleNamespace, env: CommandEnv) -> None:
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

    immutable, reason, cfg_path = is_config_declarative()
    if immutable:
        console.print(
            f"[red]Configuration is managed declaratively ({cfg_path}): {reason}. "
            "Edit your Nix/flake module instead of using 'config init'."
        )
        raise SystemExit(1)

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

    # Step 1: Roots
    default_output_root = env.config.defaults.output_dirs.render.parent
    default_input_root = env.config.exports.chatgpt
    console.print(f"[bold]Output Directory[/bold]")
    console.print(f"Where should rendered conversations be saved (root for all providers)?")

    if not ui.plain:
        output_dir_input = ui.input(
            f"  [{default_output_root}]: ",
            default=str(default_output_root)
        )
        output_dir = Path(output_dir_input).expanduser()
    else:
        output_dir = default_output_root
        console.print(f"  Using default: {default_output_root}")

    console.print(f"\n[bold]Inbox Directory[/bold]")
    console.print("Where should incoming exports/sessions be read from (ChatGPT/Claude ZIPs, etc.)?")
    if not ui.plain:
        input_dir_input = ui.input(f"  [{default_input_root}]: ", default=str(default_input_root))
        input_root = Path(input_dir_input).expanduser()
    else:
        input_root = default_input_root
        console.print(f"  Using default inbox: {input_root}")

    console.print(f"\n[bold]Index/Qdrant Settings[/bold]")
    console.print("Configure Qdrant URL/API key/collection (leave blank to keep defaults).")

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

    # Step 5: Index backend
    console.print(f"\n[bold]Index Backend[/bold]")
    console.print("Choose index backend (sqlite is default; qdrant enables vector search).")
    backend = "sqlite"
    index_cfg = env.config.index
    qdrant_url: Optional[str] = index_cfg.qdrant_url if index_cfg else None
    qdrant_key: Optional[str] = index_cfg.qdrant_api_key if index_cfg else None
    qdrant_collection: Optional[str] = index_cfg.qdrant_collection if index_cfg else None
    qdrant_vector: Optional[int] = index_cfg.qdrant_vector_size if index_cfg else None
    if not ui.plain:
        backend_choice = ui.choose("  Backend (sqlite/qdrant/none)", ["sqlite", "qdrant", "none"])
        backend = backend_choice or "sqlite"
        if backend == "qdrant":
            qdrant_url = ui.input("  Qdrant URL", default=qdrant_url or "http://localhost:6333")
            qdrant_key = ui.input("  Qdrant API key (blank for none)", default=qdrant_key or "")
            qdrant_collection = ui.input("  Qdrant collection", default=qdrant_collection or "polylogue")
            vector_raw = ui.input("  Vector size", default=str(qdrant_vector or 1536))
            try:
                qdrant_vector = int(vector_raw)
            except Exception:
                qdrant_vector = 1536
    else:
        console.print("  Using default backend: sqlite")

    index_cfg = IndexConfig(
        backend=backend,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        qdrant_collection=qdrant_collection,
        qdrant_vector_size=qdrant_vector,
    )

    # Optional labeled roots
    labeled_roots: dict[str, Path] = {}
    if not ui.plain:
        while ui.confirm("Add a labeled archive root (e.g., work/personal)?", default=False):
            label = ui.input("  Label (e.g., work)", default=f"root{len(labeled_roots)+1}") or f"root{len(labeled_roots)+1}"
            root_input = ui.input("  Root path", default=str(output_dir))
            if root_input:
                labeled_roots[label] = Path(root_input).expanduser()
            if not ui.confirm("Add another root?", default=False):
                break

    # Step 6: Drive credentials (optional)
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

    # Persist config.json so subsequent runs honor the chosen roots and UI defaults.
    config_path = persist_config(
        input_root=input_root,
        output_root=output_dir,
        collapse_threshold=collapse_threshold,
        html_previews=html_enabled,
        html_theme=theme,
        index=index_cfg,
        roots=labeled_roots or None,
    )

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[green]✓ Created output directory: {output_dir}")
    except OSError as e:
        console.print(f"\n[yellow]Warning: Could not create {output_dir}: {e}")
        console.print(f"[yellow]You may need to create it manually.")

    # Summary
    console.print(f"\n[bold green]✓ Settings saved to {SETTINGS_PATH}[/bold green]")
    console.print(f"[bold green]✓ Config saved to {config_path}[/bold green]\n")
    console.print("[bold]Your settings:[/bold]")
    console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
    console.print(f"  Inbox directory: [cyan]{input_root}[/cyan]")
    console.print(f"  HTML previews: [cyan]{'enabled' if html_enabled else 'disabled'}[/cyan]")
    console.print(f"  Theme: [cyan]{theme}[/cyan]")
    console.print(f"  Collapse threshold: [cyan]{collapse_threshold}[/cyan]")
    console.print(f"  Index backend: [cyan]{backend}[/cyan]")
    if drive_setup:
        console.print(f"  Drive credentials: [cyan]configured[/cyan]")

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print("  [green]polylogue help[/green]                 # See commands and examples")
    console.print("  [green]polylogue search 'query'[/green]       # Search transcripts")
    console.print("  [green]polylogue import run chatgpt FILE[/green]  # Import ChatGPT export")
    console.print("  [green]polylogue sync codex[/green]           # Sync Codex sessions")
    if drive_setup:
        console.print("  [green]polylogue sync drive[/green]           # Sync from Google Drive")
    console.print()
