"""Reprocess command for retrying failed imports."""
from __future__ import annotations

import json
import tempfile
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
        conversation_id = row["conversation_id"] if "conversation_id" in row.keys() else None
        metadata: dict = {}
        raw_metadata = row["metadata_json"] if "metadata_json" in row.keys() else None
        if isinstance(raw_metadata, str) and raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                metadata = {}

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
                from pathlib import Path

                export_root = metadata.get("export_root") or metadata.get("exportRoot")
                if isinstance(export_root, str) and export_root.strip():
                    export_root = Path(export_root)
                else:
                    export_root = None
                temp_dir = None
                tmp_path: Optional[Path] = None
                resolved_source: Optional[Path] = None
                if isinstance(source_path, str):
                    candidate = Path(source_path)
                    if candidate.exists() and candidate.is_file():
                        resolved_source = candidate

                try:
                    # Import using provider-specific parser
                    if import_provider == 'chatgpt':
                        from ..importers.chatgpt import _render_chatgpt_conversation
                        conv = json.loads(raw_data.decode("utf-8"))
                        if not isinstance(conv, dict):
                            raise ValueError("ChatGPT raw import was not a conversation object.")
                        out_dir = _output_dir(import_provider)
                        root = export_root if isinstance(export_root, Path) else Path(source_path).parent
                        results = [
                            _render_chatgpt_conversation(
                                conv,
                                root,
                                out_dir,
                                collapse_threshold=settings.collapse_threshold,
                                html=settings.html_previews,
                                html_theme=settings.html_theme,
                                force=True,
                                allow_dirty=True,
                                registrar=env.registrar,
                                attachment_ocr=True,
                            )
                        ]
                    elif import_provider in ('claude', 'claude.ai'):
                        from ..importers.claude_ai import _render_claude_conversation
                        conv = json.loads(raw_data.decode("utf-8"))
                        if not isinstance(conv, dict):
                            raise ValueError("Claude raw import was not a conversation object.")
                        out_dir = _output_dir(import_provider)
                        root = export_root if isinstance(export_root, Path) else Path(source_path).parent
                        results = [
                            _render_claude_conversation(
                                conv,
                                root,
                                out_dir,
                                collapse_threshold=settings.collapse_threshold,
                                html=settings.html_previews,
                                html_theme=settings.html_theme,
                                force=True,
                                allow_dirty=True,
                                registrar=env.registrar,
                                attachment_ocr=True,
                            )
                        ]
                    elif import_provider == 'codex':
                        from ..importers.codex import import_codex_session
                        out_dir = _output_dir(import_provider)
                        if resolved_source is None:
                            temp_dir = tempfile.TemporaryDirectory(prefix="polylogue-reprocess-")
                            tmp_path = Path(temp_dir.name) / "session.jsonl"
                            tmp_path.write_bytes(raw_data)
                            resolved_source = tmp_path
                        results = [import_codex_session(
                            session_id=str(resolved_source),
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                            conversation_id_override=str(conversation_id) if conversation_id else None,
                        )]
                    elif import_provider in ('claude-code', 'claude_code'):
                        from ..importers.claude_code import import_claude_code_session
                        out_dir = _output_dir(import_provider)
                        if resolved_source is None:
                            temp_dir = tempfile.TemporaryDirectory(prefix="polylogue-reprocess-")
                            tmp_path = Path(temp_dir.name) / "session.jsonl"
                            tmp_path.write_bytes(raw_data)
                            resolved_source = tmp_path
                        results = [import_claude_code_session(
                            session_id=str(resolved_source),
                            output_dir=out_dir,
                            collapse_threshold=settings.collapse_threshold,
                            html=settings.html_previews,
                            html_theme=settings.html_theme,
                            force=True,
                            allow_dirty=True,
                            attachment_ocr=True,
                            conversation_id_override=str(conversation_id) if conversation_id else None,
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
                    if temp_dir is not None:
                        temp_dir.cleanup()

            except Exception as e:
                console.print(f"  [red]✗[/red] Reprocessing failed: {e}")
                mark_parse_failed(data_hash, str(e))
                still_failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Success: {success_count}")
    console.print(f"  Still failed: {still_failed}")

    return 0 if still_failed == 0 else 1
