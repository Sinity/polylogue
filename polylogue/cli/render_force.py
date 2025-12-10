"""Render force command for regenerating markdown from database."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..db import open_connection
from ..persistence.database import ConversationDatabase
from ..renderers.db_renderer import DatabaseRenderer
from ..commands import CommandEnv


def run_render_force(
    env: CommandEnv,
    *,
    provider: Optional[str] = None,
    conversation_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> int:
    """Regenerate markdown files from database.

    This command reads conversations from the SQLite database and regenerates
    their markdown representations. This is useful when:
    - You've changed your markdown templates
    - You've updated file naming conventions
    - You want to add/remove sections from existing conversations
    - You've updated collapse thresholds

    Args:
        env: Command environment
        provider: Optional provider filter
        conversation_id: Optional specific conversation ID
        output_dir: Optional output directory override

    Returns:
        Exit code
    """
    console = env.ui.console
    db = ConversationDatabase()

    env.ui.banner(
        "Regenerate Markdown from Database",
        "This will recreate all markdown files from stored data"
    )

    # Get conversations from database
    with open_connection(db.resolve_path()) as conn:
        if conversation_id and provider:
            # Specific conversation
            query = """
                SELECT provider, conversation_id, slug, title
                FROM conversations
                WHERE provider = ? AND conversation_id = ?
            """
            rows = conn.execute(query, (provider, conversation_id)).fetchall()
        elif provider:
            # All conversations for provider
            query = """
                SELECT provider, conversation_id, slug, title
                FROM conversations
                WHERE provider = ?
            """
            rows = conn.execute(query, (provider,)).fetchall()
        else:
            # All conversations
            query = """
                SELECT provider, conversation_id, slug, title
                FROM conversations
            """
            rows = conn.execute(query).fetchall()

    if not rows:
        console.print("No conversations found in database.")
        return 0

    console.print(f"Found {len(rows)} conversation(s) to regenerate")

    # Determine output directory
    if output_dir is None:
        from ..config import CONFIG
        if provider:
            # Use provider-specific output directory
            provider_dirs = {
                "chatgpt": CONFIG.defaults.output_dirs.get("import_chatgpt"),
                "claude": CONFIG.defaults.output_dirs.get("import_claude"),
                "codex": CONFIG.defaults.output_dirs.get("sync_codex"),
                "claude_code": CONFIG.defaults.output_dirs.get("sync_claude_code"),
            }
            output_dir = Path(provider_dirs.get(provider, CONFIG.defaults.output_dirs.get("render")))
        else:
            output_dir = Path(CONFIG.defaults.output_dirs.get("render"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create renderer
    renderer = DatabaseRenderer(db_path=db.resolve_path())

    regenerated = 0
    failed = 0

    for row in rows:
        conv_provider = row["provider"]
        conv_id = row["conversation_id"]
        slug = row["slug"]
        title = row["title"] or "Untitled"

        console.print(f"\n[cyan]Regenerating:[/cyan] {conv_provider}/{slug}")
        console.print(f"  Title: {title}")

        try:
            # Render conversation from database
            markdown_path = renderer.render_conversation(
                provider=conv_provider,
                conversation_id=conv_id,
                output_dir=output_dir,
            )

            console.print(f"  [green]✓[/green] Regenerated: {markdown_path}")
            regenerated += 1

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed: {e}")
            failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Regenerated: {regenerated}")
    console.print(f"  Failed: {failed}")

    if regenerated > 0:
        console.print(f"\n[green]Success![/green] Markdown files regenerated from database to: {output_dir}")

    return 0 if failed == 0 else 1
