"""Reprocess command for retrying failed imports."""
from __future__ import annotations

from typing import Optional

from ..importers.raw_storage import get_failed_imports, retrieve_raw_import, mark_parse_success, mark_parse_failed
from ..importers.fallback_parser import extract_messages_heuristic
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
    settings = env.settings
    outputs = env.config.defaults.output_dirs

    def _output_dir(import_provider: str):
        mapping = {
            "chatgpt": outputs.import_chatgpt,
            "claude": outputs.import_claude,
            "claude.ai": outputs.import_claude,
            "codex": outputs.sync_codex,
            "claude-code": outputs.sync_claude_code,
            "claude_code": outputs.sync_claude_code,
        }
        return mapping.get(import_provider, outputs.render)

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
            # Try re-parsing with updated parser (strict mode)
            try:
                import tempfile
                import json
                from pathlib import Path

                # Write raw data to temp file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as tmp:
                    tmp.write(raw_data)
                    tmp_path = Path(tmp.name)

                try:
                    # Import using provider-specific parser
                    if import_provider == 'chatgpt':
                        from ..importers.chatgpt import import_chatgpt_export
                        out_dir = _output_dir(import_provider)
                        results = import_chatgpt_export(
                            tmp_path,
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                        )
                    elif import_provider in ('claude', 'claude.ai'):
                        from ..importers.claude_ai import import_claude_export
                        out_dir = _output_dir(import_provider)
                        results = import_claude_export(
                            tmp_path,
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                        )
                    elif import_provider == 'codex':
                        from ..importers.codex import import_codex_session
                        out_dir = _output_dir(import_provider)
                        results = [import_codex_session(
                            session_id=str(tmp_path),
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                        )]
                    elif import_provider in ('claude-code', 'claude_code'):
                        from ..importers.claude_code import import_claude_code_session
                        out_dir = _output_dir(import_provider)
                        results = [import_claude_code_session(
                            session_id=str(tmp_path),
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                        )]
                    else:
                        console.print(f"  [red]✗[/red] Unknown provider: {import_provider}")
                        still_failed += 1
                        continue

                    # Check results
                    if not results:
                        console.print("  [yellow]![/yellow] No conversations returned; leaving import marked failed")
                        still_failed += 1
                    else:
                        skipped = sum(1 for r in results if getattr(r, "skipped", False))
                        written = len(results) - skipped
                        console.print(
                            f"  [green]✓[/green] Re-imported {written} conversation(s) "
                            f"(skipped {skipped})"
                        )
                        mark_parse_success(data_hash)
                        success_count += 1

                finally:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

            except Exception as e:
                console.print(f"  [red]✗[/red] Reprocessing failed: {e}")
                mark_parse_failed(data_hash, str(e))
                still_failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Success: {success_count}")
    console.print(f"  Still failed: {still_failed}")

    return 0 if still_failed == 0 else 1
