from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from ..cli_common import sk_select
from ..commands import CommandEnv, render_command
from ..drive_client import DriveClient
from ..importers import ImportResult
from ..options import RenderOptions
from ..ui import UI
from ..util import write_clipboard_text
from .context import DEFAULT_COLLAPSE, DEFAULT_RENDER_OUT, resolve_html_enabled


def resolve_inputs(path: Path, plain: bool) -> Optional[List[Path]]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    candidates = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if not candidates:
        return []
    if plain:
        return candidates
    lines = [str(p) for p in candidates]
    selection = sk_select(
        lines,
        preview="bat --style=plain {}",
        bindings=["ctrl-g:execute(glow --style=dark {+})"],
    )
    if selection is None:
        return None
    if not selection:
        return []
    return [Path(s) for s in selection]


def run_render_cli(args: argparse.Namespace, env: CommandEnv, json_output: bool) -> None:
    ui = env.ui
    console = ui.console
    inputs = resolve_inputs(args.input, ui.plain)
    if inputs is None:
        console.print("[yellow]Render cancelled.")
        return
    if not inputs:
        console.print("No JSON files to render")
        return
    output = Path(args.out) if args.out else DEFAULT_RENDER_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    download_attachments = not args.links_only
    if not ui.plain and not args.links_only:
        download_attachments = ui.confirm("Download attachments to local folders?", default=True)
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    options = RenderOptions(
        inputs=inputs,
        output_dir=output,
        collapse_threshold=collapse,
        download_attachments=download_attachments,
        dry_run=args.dry_run,
        force=args.force,
        html=html_enabled,
        html_theme=html_theme,
        diff=getattr(args, "diff", False),
    )
    if download_attachments and env.drive is None:
        env.drive = DriveClient(ui)
    result = render_command(options, env)
    if json_output:
        payload = {
            "cmd": "render",
            "count": result.count,
            "out": str(result.output_dir),
            "files": [
                {
                    "output": str(f.output),
                    "slug": f.slug,
                    "attachments": f.attachments,
                    "stats": f.stats,
                    "html": str(f.html) if f.html else None,
                }
                for f in result.files
            ],
            "total_stats": result.total_stats,
        }
        print(json.dumps(payload, indent=2))
        return
    lines = [f"Rendered {result.count} file(s) → {result.output_dir}"]
    attachments_total = result.total_stats.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    skipped_total = result.total_stats.get("skipped", 0)
    if skipped_total:
        lines.append(f"Skipped: {skipped_total}")
    if "totalTokensApprox" in result.total_stats:
        total_tokens = int(result.total_stats["totalTokensApprox"])
        total_words = int(result.total_stats.get("totalWordsApprox", 0) or 0)
        if total_words:
            lines.append(f"Approx tokens: {total_tokens} (~{total_words} words)")
        else:
            lines.append(f"Approx tokens: {total_tokens}")
    for key, label in (
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ):
        value = result.total_stats.get(key)
        if value:
            lines.append(f"{label}: {int(value)}")
    for file in result.files:
        info = f"- {file.slug} (attachments: {file.attachments})"
        if file.html:
            info += " [+html]"
        lines.append(info)
    ui.summary("Render", lines)

    if getattr(args, "to_clipboard", False):
        if result.count != 1 or not result.files:
            console.print("[yellow]Clipboard copy skipped (requires exactly one rendered file).")
        else:
            target_file = result.files[0]
            target = target_file.output
            try:
                text = target.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - unlikely
                console.print(f"[red]Failed to read {target_file.slug}: {exc}")
            else:
                if write_clipboard_text(text):
                    console.print(f"[green]Copied {target_file.slug} to clipboard.")
                else:
                    console.print("[yellow]Clipboard support unavailable on this system.")


def copy_import_to_clipboard(ui: UI, results: Sequence[ImportResult]) -> None:
    console = ui.console
    written = [res for res in results if not res.skipped]
    if not written:
        console.print("[yellow]Clipboard copy skipped; no new files were written.")
        return
    if len(written) != 1:
        console.print("[yellow]Clipboard copy skipped (requires exactly one updated file).")
        return
    target = written[0]
    if target.document is not None:
        text = target.document.to_markdown()
    else:
        try:
            text = target.markdown_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - unlikely
            console.print(f"[red]Failed to read {target.slug}: {exc}")
            return
    if write_clipboard_text(text):
        console.print(f"[green]Copied {target.slug} to clipboard.")
    else:
        console.print("[yellow]Clipboard support unavailable on this system.")
