"""Reprocess command for retrying failed imports."""
from __future__ import annotations

from typing import Optional

from ..db import open_connection
from ..importers.raw_storage import get_failed_imports, retrieve_raw_import, mark_parse_success, mark_parse_failed
from ..importers.fallback_parser import extract_messages_heuristic, create_degraded_markdown
from ..commands import CommandEnv


def run_reprocess(
    env: CommandEnv,
    *,
    provider: Optional[str] = None,
    use_fallback: bool = False,
) -> int:
    """Reprocess failed imports.

    Args:
        env: Command environment
        provider: Optional provider filter
        use_fallback: If True, use fallback parser for failed imports

    Returns:
        Exit code
    """
    console = env.ui.console

    # Get failed imports
    failed = get_failed_imports(provider=provider)

    if not failed:
        console.print("No failed imports found.")
        return 0

    env.ui.banner(
        "Reprocess Failed Imports",
        f"Found {len(failed)} failed import(s)"
    )

    success_count = 0
    still_failed = 0

    for row in failed:
        data_hash = row["hash"]
        import_provider = row["provider"]
        source_path = row["source_path"] or "unknown"

        console.print(f"\n[cyan]Processing:[/cyan] {import_provider} - {source_path}")
        console.print(f"  Hash: {data_hash[:16]}...")

        # Retrieve raw data
        raw_data = retrieve_raw_import(data_hash)
        if not raw_data:
            console.print("  [red]✗[/red] Raw data not found")
            continue

        if use_fallback:
            # Use fallback parser
            try:
                import json
                data = json.loads(raw_data.decode('utf-8'))
                messages = extract_messages_heuristic(data)

                if messages:
                    console.print(f"  [green]✓[/green] Extracted {len(messages)} messages using fallback parser")
                    mark_parse_success(data_hash)
                    success_count += 1
                else:
                    console.print("  [yellow]![/yellow] No messages extracted")
                    still_failed += 1
            except Exception as e:
                console.print(f"  [red]✗[/red] Fallback failed: {e}")
                still_failed += 1
        else:
            # Try re-parsing with updated parser
            # This would require importing the actual parser for each provider
            # For now, just report that it's available
            console.print("  [yellow]![/yellow] Strict reprocessing not yet implemented")
            console.print("  [cyan]Hint:[/cyan] Use --fallback to extract text heuristically")
            still_failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Success: {success_count}")
    console.print(f"  Still failed: {still_failed}")

    return 0 if still_failed == 0 else 1
