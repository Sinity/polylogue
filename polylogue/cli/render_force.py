"""Render force command for regenerating markdown from database."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..db import open_connection
from ..persistence.database import ConversationDatabase
from .commands import CommandEnv


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
            # Get messages for this conversation
            with open_connection(db.resolve_path()) as conn:
                # Get canonical branch
                branch_query = """
                    SELECT branch_id
                    FROM branches
                    WHERE provider = ? AND conversation_id = ? AND is_current = 1
                    LIMIT 1
                """
                branch_row = conn.execute(branch_query, (conv_provider, conv_id)).fetchone()

                if not branch_row:
                    console.print("  [yellow]![/yellow] No canonical branch found")
                    failed += 1
                    continue

                branch_id = branch_row["branch_id"]

                # Get messages for this branch
                msg_query = """
                    SELECT message_id, role, rendered_text, timestamp
                    FROM messages
                    WHERE provider = ? AND conversation_id = ? AND branch_id = ?
                    ORDER BY position
                """
                messages = conn.execute(msg_query, (conv_provider, conv_id, branch_id)).fetchall()

            if not messages:
                console.print("  [yellow]![/yellow] No messages found")
                failed += 1
                continue

            # TODO: Actually regenerate the markdown file here
            # For now, just report what would be done
            console.print(f"  [green]✓[/green] Would regenerate from {len(messages)} messages")
            console.print(f"  [cyan]Note:[/cyan] Full regeneration requires render pipeline integration")
            regenerated += 1

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed: {e}")
            failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Would regenerate: {regenerated}")
    console.print(f"  Failed: {failed}")

    if regenerated > 0:
        console.print(
            f"\n[yellow]Note:[/yellow] Full markdown regeneration requires integration "
            "with the render pipeline. The database contains all necessary data."
        )

    return 0 if failed == 0 else 1
