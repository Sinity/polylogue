"""Browse command for interactive TUI conversation browser."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--provider",
    type=str,
    default=None,
    help="Filter conversations by provider (chatgpt, claude, etc.)",
)
@click.option(
    "--db-path",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Path to database file (default: ~/.local/state/polylogue/polylogue.db)",
)
def browse(provider: str | None, db_path: Path | None) -> None:
    """Launch interactive TUI browser for conversations.

    The browser provides a full-screen terminal interface for exploring
    your AI conversation archive:

    - Left panel: List of conversations
    - Right panel: Selected conversation content
    - Keyboard navigation: j/k (down/up), Enter (view), q (quit)

    Examples:

        # Browse all conversations
        polylogue browse

        # Browse only Claude conversations
        polylogue browse --provider claude

        # Use custom database
        polylogue browse --db-path /path/to/db.db
    """
    try:
        from polylogue.tui.app import run_browser
    except ImportError as e:
        click.echo(f"Error: TUI dependencies not available: {e}", err=True)
        click.echo("Install with: pip install 'polylogue[tui]'", err=True)
        raise click.Abort()

    try:
        run_browser(db_path=db_path, provider=provider)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        pass
    except Exception as e:
        click.echo(f"Error running browser: {e}", err=True)
        raise click.Abort()
