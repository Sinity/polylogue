#!/usr/bin/env python3
"""Polylogue: interactive-first CLI for AI chat log archives."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

from .cli_common import filter_chats, sk_select
from .commands import (
    CommandEnv,
    list_command,
    render_command,
    status_command,
    sync_command,
)
from .automation import REPO_ROOT, TARGETS, cron_snippet, describe_targets, systemd_snippet
from .drive_client import DEFAULT_FOLDER_NAME, DriveClient
from .importers import (
    ImportResult,
    import_chatgpt_export,
    import_claude_code_session,
    import_claude_export,
    import_codex_session,
)
from .importers.chatgpt import list_chatgpt_conversations
from .importers.claude_ai import list_claude_conversations
from .importers.claude_code import DEFAULT_PROJECT_ROOT, list_claude_code_sessions
from .local_sync import LocalSyncResult, sync_claude_code_sessions, sync_codex_sessions
from .options import ListOptions, RenderOptions, SyncOptions
from .ui import create_ui
from .settings import SETTINGS, reset_settings
from .config import CONFIG, CONFIG_PATH, DEFAULT_PATHS, CONFIG_ENV
from .doctor import run_doctor as doctor_run
from .util import add_run, parse_input_time_to_epoch, write_clipboard_text

try:  # pragma: no cover - optional dependency
    from watchfiles import watch as watch_directory
except Exception:  # pragma: no cover - watcher dependency optional
    watch_directory = None

DEFAULT_COLLAPSE = CONFIG.defaults.collapse_threshold
DEFAULT_RENDER_OUT = CONFIG.defaults.output_dirs.render
DEFAULT_SYNC_OUT = CONFIG.defaults.output_dirs.sync_drive
DEFAULT_CODEX_SYNC_OUT = CONFIG.defaults.output_dirs.sync_codex
DEFAULT_CLAUDE_CODE_SYNC_OUT = CONFIG.defaults.output_dirs.sync_claude_code
DEFAULT_CHATGPT_OUT = CONFIG.defaults.output_dirs.import_chatgpt
DEFAULT_CLAUDE_OUT = CONFIG.defaults.output_dirs.import_claude

SETTINGS.html_previews = CONFIG.defaults.html_previews
SETTINGS.html_theme = CONFIG.defaults.html_theme


def summarize_import(ui, title: str, results: List[ImportResult]) -> None:
    if not results:
        ui.summary(title, ["No files written."])
        return

    written = [res for res in results if not res.skipped]
    skipped = [res for res in results if res.skipped]

    if written:
        output_dir = written[0].markdown_path.parent
        lines = [f"{len(written)} file(s) → {output_dir}"]
    else:
        output_dir = results[0].markdown_path.parent
        lines = [f"No files written (existing files up to date in {output_dir})."]

    attachments_total = sum(len(res.document.attachments) for res in written if res.document)
    attachment_bytes = sum(res.document.metadata.get("attachmentBytes", 0) or 0 for res in written if res.document)
    diff_total = sum(1 for res in written if getattr(res, "diff_path", None))
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    if attachment_bytes:
        mb = attachment_bytes / (1024 * 1024)
        lines.append(f"Attachment size: {mb:.2f} MiB")
    if diff_total:
        lines.append(f"Diffs written: {diff_total}")
    stats_to_sum = [
        ("totalTokensApprox", "Approx tokens"),
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ]
    for key, label in stats_to_sum:
        total = 0
        for res in written:
            if not res.document:
                continue
            value = res.document.stats.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
        if total:
            lines.append(f"{label}: {total}")
    for res in written:
        att_count = len(res.document.attachments) if res.document else 0
        info = f"- {res.markdown_path.name} (attachments: {att_count})"
        if res.html_path:
            info += " [+html]"
        if getattr(res, "diff_path", None):
            info += " [+diff]"
        lines.append(info)
    if skipped:
        skipped_line = f"Skipped {len(skipped)} conversation(s)"
        reasons = {res.skip_reason for res in skipped if res.skip_reason}
        if reasons:
            skipped_line += f" ({'; '.join(sorted(reasons))})"
        lines.append(skipped_line)
    if not ui.plain:
        try:
            from rich.table import Table

            table = Table(title=title, show_lines=False)
            table.add_column("File")
            table.add_column("Attachments", justify="right")
            table.add_column("Attachment MiB", justify="right")
            table.add_column("Tokens", justify="right")
            for res in written:
                if res.document:
                    att_count = len(res.document.attachments)
                    att_bytes = res.document.metadata.get("attachmentBytes", 0) or 0
                    tokens = res.document.stats.get("totalTokensApprox", 0) or 0
                else:
                    att_count = 0
                    att_bytes = 0
                    tokens = 0
                table.add_row(
                    res.markdown_path.name,
                    str(att_count),
                    f"{att_bytes / (1024 * 1024):.2f}" if att_bytes else "0.00",
                    str(tokens),
                )
            ui.console.print(table)
        except Exception:
            pass
    ui.summary(title, lines)


def copy_import_to_clipboard(ui, results: List[ImportResult]) -> None:
    written = [res for res in results if not res.skipped]
    if not written:
        ui.console.print("[yellow]Clipboard copy skipped; no new files were written.")
        return
    if len(written) != 1:
        ui.console.print("[yellow]Clipboard copy skipped (requires exactly one updated file).")
        return
    target = written[0]
    if target.document is not None:
        text = target.document.to_markdown()
    else:
        try:
            text = target.markdown_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - unlikely
            ui.console.print(f"[red]Failed to read {target.markdown_path.name}: {exc}")
            return
    if write_clipboard_text(text):
        ui.console.print(f"[green]Copied {target.markdown_path.name} to clipboard.")
    else:
        ui.console.print("[yellow]Clipboard support unavailable on this system.")


def _log_local_sync(ui, title: str, result: LocalSyncResult) -> None:
    if result.written:
        summarize_import(ui, title, result.written)
    else:
        ui.console.print(f"[cyan]{title}: no new Markdown files.")
    if result.skipped:
        ui.console.print(f"[cyan]{title}: skipped {result.skipped} up-to-date session(s).")
    if result.pruned:
        ui.console.print(f"[cyan]{title}: pruned {result.pruned} path(s).")
    add_run(
        {
            "cmd": title.lower().replace(" ", "-"),
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": result.attachments,
            "attachmentBytes": result.attachment_bytes,
            "tokens": result.tokens,
            "diffs": result.diffs,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "duration": getattr(result, "duration", 0.0),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polylogue CLI")
    parser.add_argument("--plain", action="store_true", help="Disable interactive UI")
    sub = parser.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render")
    p_render.add_argument("input", type=Path, help="File or directory with provider JSON logs (e.g., Gemini)")
    p_render.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default {DEFAULT_RENDER_OUT})",
    )
    p_render.add_argument("--links-only", action="store_true", help="Link attachments instead of downloading")
    p_render.add_argument("--dry-run", action="store_true")
    p_render.add_argument("--force", action="store_true")
    p_render.add_argument("--collapse-threshold", type=int, default=None)
    p_render.add_argument("--json", action="store_true")
    p_render.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_render.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")
    p_render.add_argument("--diff", action="store_true", help="Write delta diff when output already exists")
    p_render.add_argument("--to-clipboard", action="store_true", help="Copy rendered Markdown to the clipboard when a single file is produced")

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME)
    p_sync.add_argument("--folder-id", type=str, default=None)
    p_sync.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default {DEFAULT_SYNC_OUT})",
    )
    p_sync.add_argument("--links-only", action="store_true")
    p_sync.add_argument("--since", type=str, default=None)
    p_sync.add_argument("--until", type=str, default=None)
    p_sync.add_argument("--name-filter", type=str, default=None)
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--force", action="store_true")
    p_sync.add_argument("--prune", action="store_true")
    p_sync.add_argument("--collapse-threshold", type=int, default=None)
    p_sync.add_argument("--json", action="store_true")
    p_sync.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_sync.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")
    p_sync.add_argument("--diff", action="store_true", help="Write delta diff when markdown updates")

    p_sync_codex = sub.add_parser("sync-codex")
    p_sync_codex.add_argument("--base-dir", type=Path, default=None)
    p_sync_codex.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default {DEFAULT_CODEX_SYNC_OUT})",
    )
    p_sync_codex.add_argument("--collapse-threshold", type=int, default=None)
    p_sync_codex.add_argument("--html", action="store_true")
    p_sync_codex.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_sync_codex.add_argument("--force", action="store_true", help="Re-render even if up-to-date")
    p_sync_codex.add_argument("--prune", action="store_true", help="Remove outputs for missing sessions")
    p_sync_codex.add_argument("--all", action="store_true", help="Process all sessions without prompting")
    p_sync_codex.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_sync_codex.add_argument("--diff", action="store_true", help="Write delta diff alongside updated files")

    p_sync_claude_code = sub.add_parser("sync-claude-code")
    p_sync_claude_code.add_argument("--base-dir", type=Path, default=None)
    p_sync_claude_code.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default {DEFAULT_CLAUDE_CODE_SYNC_OUT})",
    )
    p_sync_claude_code.add_argument("--collapse-threshold", type=int, default=None)
    p_sync_claude_code.add_argument("--html", action="store_true")
    p_sync_claude_code.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_sync_claude_code.add_argument("--force", action="store_true")
    p_sync_claude_code.add_argument("--prune", action="store_true")
    p_sync_claude_code.add_argument("--all", action="store_true")
    p_sync_claude_code.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_sync_claude_code.add_argument("--diff", action="store_true", help="Write delta diff alongside updated files")

    p_doctor = sub.add_parser("doctor", help="Check local data directories for common issues")
    p_doctor.add_argument("--codex-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_doctor.add_argument("--claude-code-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_doctor.add_argument("--limit", type=int, default=None, help="Limit number of files inspected per provider")
    p_doctor.add_argument("--json", action="store_true", help="Emit machine-readable report")

    p_stats = sub.add_parser("stats", help="Summarize Markdown output directories")
    p_stats.add_argument("--dir", type=Path, default=None, help="Directory containing Markdown exports")
    p_stats.add_argument("--json", action="store_true", help="Emit machine-readable stats")
    p_stats.add_argument("--since", type=str, default=None, help="Only include files modified on/after this date (YYYY-MM-DD or ISO)")
    p_stats.add_argument("--until", type=str, default=None, help="Only include files modified on/before this date")

    p_list = sub.add_parser("list")
    p_list.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME)
    p_list.add_argument("--folder-id", type=str, default=None)
    p_list.add_argument("--since", type=str, default=None)
    p_list.add_argument("--until", type=str, default=None)
    p_list.add_argument("--name-filter", type=str, default=None)
    p_list.add_argument("--json", action="store_true")

    p_watch = sub.add_parser("watch", help="Watch local session stores and sync on changes")
    watch_sub = p_watch.add_subparsers(dest="watch_target", required=True)

    p_watch_codex = watch_sub.add_parser("codex", help="Watch ~/.codex/sessions for changes")
    p_watch_codex.add_argument("--base-dir", type=Path, default=None)
    p_watch_codex.add_argument("--out", type=Path, default=None)
    p_watch_codex.add_argument("--collapse-threshold", type=int, default=None)
    p_watch_codex.add_argument("--html", action="store_true")
    p_watch_codex.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_watch_codex.add_argument("--debounce", type=float, default=2.0, help="Minimal seconds between sync runs")
    p_watch_codex.add_argument("--once", action="store_true", help="Run a single sync and exit")

    p_watch_claude = watch_sub.add_parser("claude-code", help="Watch ~/.claude/projects for changes")
    p_watch_claude.add_argument("--base-dir", type=Path, default=None)
    p_watch_claude.add_argument("--out", type=Path, default=None)
    p_watch_claude.add_argument("--collapse-threshold", type=int, default=None)
    p_watch_claude.add_argument("--html", action="store_true")
    p_watch_claude.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_watch_claude.add_argument("--debounce", type=float, default=2.0, help="Minimal seconds between sync runs")
    p_watch_claude.add_argument("--once", action="store_true", help="Run a single sync and exit")

    automation_choices = sorted(TARGETS.keys())
    p_automation = sub.add_parser("automation", help="Generate scheduler snippets")
    automation_sub = p_automation.add_subparsers(dest="automation_format", required=True)

    p_auto_systemd = automation_sub.add_parser("systemd", help="Emit a systemd service/timer pair")
    p_auto_systemd.add_argument("--target", choices=automation_choices, required=True)
    p_auto_systemd.add_argument("--interval", type=str, default="10m", help="Interval passed to OnUnitActiveSec")
    p_auto_systemd.add_argument("--boot-delay", type=str, default="2m", help="Delay before the first run (OnBootSec)")
    p_auto_systemd.add_argument("--working-dir", type=Path, default=None, help="Working directory for the service")
    p_auto_systemd.add_argument("--out", type=Path, default=None, help="Override --out argument for the sync command")
    p_auto_systemd.add_argument("--extra-arg", action="append", default=[], help="Additional argument to append to the sync command")
    p_auto_systemd.add_argument("--collapse-threshold", type=int, default=None, help="Override collapse threshold for the sync command")
    p_auto_systemd.add_argument("--html", action="store_true", help="Enable HTML generation in the sync command")

    p_auto_describe = automation_sub.add_parser("describe", help="Show automation target metadata")
    p_auto_describe.add_argument("--target", choices=automation_choices, default=None)

    p_auto_cron = automation_sub.add_parser("cron", help="Emit a cron entry")
    p_auto_cron.add_argument("--target", choices=automation_choices, required=True)
    p_auto_cron.add_argument("--schedule", type=str, default="*/30 * * * *", help="Cron schedule expression")
    p_auto_cron.add_argument("--log", type=str, default="$HOME/.cache/polylogue-sync.log", help="Log file path")
    p_auto_cron.add_argument("--state-home", type=str, default="$HOME/.local/state", help="Value for XDG_STATE_HOME")
    p_auto_cron.add_argument("--working-dir", type=Path, default=None, help="Working directory for the cron entry")
    p_auto_cron.add_argument("--out", type=Path, default=None, help="Override --out argument for the sync command")
    p_auto_cron.add_argument("--extra-arg", action="append", default=[], help="Additional argument to append to the sync command")
    p_auto_cron.add_argument("--collapse-threshold", type=int, default=None, help="Override collapse threshold for the sync command")
    p_auto_cron.add_argument("--html", action="store_true", help="Enable HTML generation in the sync command")

    p_status = sub.add_parser("status", help="Show cached Drive info and recent runs")
    p_status.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_status.add_argument("--watch", action="store_true", help="Continuously refresh the status output")
    p_status.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh while watching")

    p_import = sub.add_parser("import")
    import_sub = p_import.add_subparsers(dest="import_target", required=True)

    p_import_chatgpt = import_sub.add_parser("chatgpt", help="Convert a ChatGPT export to Markdown")
    p_import_chatgpt.add_argument("export_path", type=Path, help="Path to ChatGPT export .zip or directory")
    p_import_chatgpt.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Restrict to specific conversation id (repeatable)")
    p_import_chatgpt.add_argument("--all", action="store_true", help="Import all conversations without prompting")
    p_import_chatgpt.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory for Markdown files (default {DEFAULT_CHATGPT_OUT})",
    )
    p_import_chatgpt.add_argument("--collapse-threshold", type=int, default=None)
    p_import_chatgpt.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_import_chatgpt.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")
    p_import_chatgpt.add_argument("--to-clipboard", action="store_true", help="Copy the rendered Markdown to the clipboard when a single file is updated")

    p_import_claude = import_sub.add_parser("claude", help="Convert an Anthropic Claude export to Markdown")
    p_import_claude.add_argument("export_path", type=Path, help="Path to Claude export .zip or directory")
    p_import_claude.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Restrict to specific conversation id (repeatable)")
    p_import_claude.add_argument("--all", action="store_true", help="Import all conversations without prompting")
    p_import_claude.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory for Markdown files (default {DEFAULT_CLAUDE_OUT})",
    )
    p_import_claude.add_argument("--collapse-threshold", type=int, default=None)
    p_import_claude.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_import_claude.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")
    p_import_claude.add_argument("--to-clipboard", action="store_true", help="Copy the rendered Markdown to the clipboard when a single file is updated")

    p_import_claude_code = import_sub.add_parser("claude-code", help="Convert a Claude Code session to Markdown")
    p_import_claude_code.add_argument("session_id", type=str, help="Session UUID or suffix")
    p_import_claude_code.add_argument("--base-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_import_claude_code.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory for Markdown (default {DEFAULT_CLAUDE_CODE_SYNC_OUT})",
    )
    p_import_claude_code.add_argument("--collapse-threshold", type=int, default=None)
    p_import_claude_code.add_argument("--html", action="store_true", help="Also write HTML preview")
    p_import_claude_code.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML preview")
    p_import_claude_code.add_argument("--to-clipboard", action="store_true", help="Copy the rendered Markdown to the clipboard")

    p_import_codex = import_sub.add_parser("codex", help="Convert a Codex CLI session to Markdown")
    p_import_codex.add_argument("session_id", type=str, help="Codex session UUID (or suffix)")
    p_import_codex.add_argument("--base-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_import_codex.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory for Markdown (default {DEFAULT_CODEX_SYNC_OUT})",
    )
    p_import_codex.add_argument("--collapse-threshold", type=int, default=None, help="Fold responses longer than this many lines")
    p_import_codex.add_argument("--html", action="store_true", help="Also write HTML preview")
    p_import_codex.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML preview")
    p_import_codex.add_argument("--to-clipboard", action="store_true", help="Copy the rendered Markdown to the clipboard")

    return parser


def resolve_inputs(path: Path, plain: bool) -> Optional[List[Path]]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input path not found: {path}")
    candidates = [
        p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() == ".json"
    ]
    if len(candidates) <= 1 or plain:
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
    inputs = resolve_inputs(args.input, ui.plain)
    if inputs is None:
        ui.console.print("[yellow]Render cancelled.")
        return
    if not inputs:
        ui.console.print("No JSON files to render")
        return
    output = Path(args.out) if args.out else DEFAULT_RENDER_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    download_attachments = not args.links_only
    if not ui.plain and not args.links_only:
        download_attachments = ui.confirm("Download attachments to local folders?", default=True)
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
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
        info = f"- {file.output.name} (attachments: {file.attachments})"
        if file.html:
            info += " [+html]"
        lines.append(info)
    ui.summary("Render", lines)

    if getattr(args, "to_clipboard", False):
        if result.count != 1 or not result.files:
            ui.console.print("[yellow]Clipboard copy skipped (requires exactly one rendered file).")
        else:
            target = result.files[0].output
            try:
                text = target.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - unlikely
                ui.console.print(f"[red]Failed to read {target.name}: {exc}")
            else:
                if write_clipboard_text(text):
                    ui.console.print(f"[green]Copied {target.name} to clipboard.")
                else:
                    ui.console.print("[yellow]Clipboard support unavailable on this system.")


def run_list_cli(args: argparse.Namespace, env: CommandEnv, json_output: bool) -> None:
    options = ListOptions(
        folder_name=args.folder_name,
        folder_id=args.folder_id,
        since=args.since,
        until=args.until,
        name_filter=args.name_filter,
    )
    result = list_command(options, env)
    if json_output:
        print(
            json.dumps(
                {
                    "cmd": "list",
                    "folder_name": result.folder_name,
                    "folder_id": result.folder_id,
                    "count": len(result.files),
                    "files": result.files,
                },
                indent=2,
            )
        )
        return
    ui = env.ui
    ui.console.print(f"{len(result.files)} chat(s) in {result.folder_name}:")
    for c in result.files:
        ui.console.print(
            f"- {c.get('name')}  {c.get('modifiedTime', '')}  {c.get('id', '')}"
        )


def run_sync_cli(args: argparse.Namespace, env: CommandEnv, json_output: bool) -> None:
    ui = env.ui
    download_attachments = not args.links_only
    if not ui.plain and not args.links_only:
        download_attachments = ui.confirm("Download attachments for synced chats?", default=True)
    selected_ids: Optional[List[str]] = None
    if not ui.plain and not json_output:
        drive = env.drive or DriveClient(ui)
        env.drive = drive
        raw_chats = drive.list_chats(args.folder_name, args.folder_id)
        filtered = filter_chats(raw_chats, args.name_filter, args.since, args.until)
        if not filtered:
            ui.console.print("No chats to sync")
            return
        lines = [f"{c.get('name')}\t{c.get('modifiedTime')}\t{c.get('id')}" for c in filtered]
        selection = sk_select(lines, preview="printf '%s' {+}")
        if selection is None:
            ui.console.print("[yellow]Sync cancelled; no chats selected.")
            return
        if not selection:
            ui.console.print("[yellow]No chats selected; nothing to sync.")
            return
        selected_ids = [ln.split("\t")[-1] for ln in selection]

    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    options = SyncOptions(
        folder_name=args.folder_name,
        folder_id=args.folder_id,
        output_dir=Path(args.out) if args.out else DEFAULT_SYNC_OUT,
        collapse_threshold=args.collapse_threshold or DEFAULT_COLLAPSE,
        download_attachments=download_attachments,
        dry_run=args.dry_run,
        force=args.force,
        prune=args.prune,
        since=args.since,
        until=args.until,
        name_filter=args.name_filter,
        selected_ids=selected_ids,
        html=html_enabled,
        html_theme=html_theme,
        diff=getattr(args, "diff", False),
    )
    if download_attachments and env.drive is None:
        env.drive = DriveClient(ui)
    result = sync_command(options, env)
    if json_output:
        payload = {
            "cmd": "sync",
            "count": result.count,
            "out": str(result.output_dir),
            "folder_name": result.folder_name,
            "folder_id": result.folder_id,
            "files": [
                {
                    "id": item.id,
                    "name": item.name,
                    "output": str(item.output),
                    "attachments": item.attachments,
                    "stats": item.stats,
                    "html": str(item.html) if item.html else None,
                    "diff": str(item.diff) if getattr(item, "diff", None) else None,
                }
                for item in result.items
            ],
            "total_stats": result.total_stats,
        }
        print(json.dumps(payload, indent=2))
        return
    lines = [f"Synced {result.count} chat(s) → {result.output_dir}"]
    attachments_total = result.total_stats.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    skipped_total = result.total_stats.get("skipped", 0)
    if skipped_total:
        lines.append(f"Skipped: {skipped_total}")
    if "totalTokensApprox" in result.total_stats:
        total_tokens = int(result.total_stats["totalTokensApprox"])
        lines.append(f"Approx tokens: {total_tokens}")
    for key, label in (
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ):
        value = result.total_stats.get(key)
        if value:
            lines.append(f"{label}: {int(value)}")
    for item in result.items:
        info = f"- {Path(item.output).name} (attachments: {item.attachments})"
        if item.html:
            info += " [+html]"
        if getattr(item, "diff", None):
            info += " [+diff]"
        lines.append(info)
    ui.summary("Sync", lines)


def run_status_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui

    def emit() -> None:
        result = status_command(env)
        if getattr(args, "json", False):
            payload = {
                "credentials_present": result.credentials_present,
                "token_present": result.token_present,
                "state_path": str(result.state_path),
                "runs_path": str(result.runs_path),
                "recent_runs": result.recent_runs,
                "run_summary": result.run_summary,
                "provider_summary": result.provider_summary,
            }
            print(json.dumps(payload, indent=2))
            return

        if ui.plain:
            ui.console.print("Environment:")
            ui.console.print(f"  credentials.json: {'present' if result.credentials_present else 'missing'}")
            ui.console.print(f"  token.json: {'present' if result.token_present else 'missing'}")
            ui.console.print(f"  state cache: {result.state_path}")
            ui.console.print(f"  runs log: {result.runs_path}")
            if result.run_summary:
                ui.console.print("Run summary:")
                for cmd, stats in result.run_summary.items():
                    ui.console.print(
                        f"  {cmd}: runs={stats['count']} attachments={stats['attachments']} (~{stats['attachmentBytes'] / (1024 * 1024):.2f} MiB) tokens={stats['tokens']} diffs={stats['diffs']}"
                    )
                    if stats.get("last"):
                        ui.console.print(
                            f"    last={stats['last']} out={stats['last_out']} count={stats['last_count']} skipped={stats['skipped']} pruned={stats['pruned']}"
                        )
            if result.provider_summary:
                ui.console.print("Provider summary:")
                for provider, stats in result.provider_summary.items():
                    ui.console.print(
                        f"  {provider}: runs={stats['count']} attachments={stats['attachments']} (~{stats['attachmentBytes'] / (1024 * 1024):.2f} MiB) tokens={stats['tokens']} diffs={stats['diffs']}"
                    )
                    if stats.get("last"):
                        ui.console.print(
                            f"    last={stats['last']} out={stats['last_out']} commands={', '.join(stats['commands'])}"
                        )
        else:
            from rich.table import Table

            table = Table(title="Environment", show_lines=False)
            table.add_column("Item")
            table.add_column("Value")
            table.add_row("credentials.json", "present" if result.credentials_present else "missing")
            table.add_row("token.json", "present" if result.token_present else "missing")
            table.add_row("state cache", str(result.state_path))
            table.add_row("runs log", str(result.runs_path))
            ui.console.print(table)
            if result.run_summary:
                summary_table = Table(title="Run Summary", show_lines=False)
                summary_table.add_column("Command")
                summary_table.add_column("Runs", justify="right")
                summary_table.add_column("Attachments", justify="right")
                summary_table.add_column("Attachment MiB", justify="right")
                summary_table.add_column("Tokens", justify="right")
                summary_table.add_column("Diffs", justify="right")
                summary_table.add_column("Last Run", justify="left")
                for cmd, stats in result.run_summary.items():
                    summary_table.add_row(
                        cmd,
                        str(stats["count"]),
                        str(stats["attachments"]),
                        f"{stats['attachmentBytes'] / (1024 * 1024):.2f}",
                        str(stats["tokens"]),
                        str(stats["diffs"]),
                        (stats.get("last") or "-") + (f" → {stats.get('last_out')}" if stats.get("last_out") else ""),
                    )
                ui.console.print(summary_table)
            if result.provider_summary:
                provider_table = Table(title="Provider Summary", show_lines=False)
                provider_table.add_column("Provider")
                provider_table.add_column("Runs", justify="right")
                provider_table.add_column("Attachments", justify="right")
                provider_table.add_column("Attachment MiB", justify="right")
                provider_table.add_column("Tokens", justify="right")
                provider_table.add_column("Diffs", justify="right")
                provider_table.add_column("Last Run", justify="left")
                for provider, stats in result.provider_summary.items():
                    provider_table.add_row(
                        provider,
                        str(stats["count"]),
                        str(stats["attachments"]),
                        f"{stats['attachmentBytes'] / (1024 * 1024):.2f}",
                        str(stats["tokens"]),
                        str(stats["diffs"]),
                        (stats.get("last") or "-") + (f" → {stats.get('last_out')}" if stats.get("last_out") else ""),
                    )
                ui.console.print(provider_table)
        if not result.recent_runs:
            ui.console.print("Recent runs: (none)")
        else:
            ui.console.print("Recent runs (last 10):")
            for entry in result.recent_runs:
                ui.console.print(f"- {entry.get('cmd')} → {entry.get('out')}")

    if getattr(args, "watch", False):
        try:
            interval = max(getattr(args, "interval", 5.0), 0.5)
            while True:
                emit()
                if not getattr(args, "json", False):
                    ui.console.print("")
                time.sleep(interval)
        except KeyboardInterrupt:
            if not getattr(args, "json", False):
                ui.console.print("[cyan]Status watch stopped.")
    else:
        emit()


def interactive_menu(env: CommandEnv) -> None:
    ui = env.ui
    options = [
        "Render Local Logs",
        "Sync Drive Folder",
        "Sync Codex Sessions",
        "Sync Claude Code Sessions",
        "Doctor",
        "List Drive Chats",
        "View Recent Runs",
        "Settings",
        "Help",
        "Quit",
    ]
    while True:
        choice = ui.choose("Select an action", options)
        if choice == "Render Local Logs":
            prompt_render(env)
        elif choice == "Sync Drive Folder":
            prompt_sync(env)
        elif choice == "Sync Codex Sessions":
            prompt_sync_codex(env)
        elif choice == "Sync Claude Code Sessions":
            prompt_sync_claude_code(env)
        elif choice == "Doctor":
            prompt_doctor(env)
        elif choice == "List Drive Chats":
            prompt_list(env)
        elif choice == "View Recent Runs":
            status_args = argparse.Namespace(json=False, watch=False, interval=5.0)
            run_status_cli(status_args, env)
        elif choice == "Settings":
            settings_menu(env)
        elif choice == "Help":
            show_help(env)
        elif choice == "Quit" or choice is None:
            return


def run_import_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    target = args.import_target
    if target == "codex":
        run_import_codex(args, env)
    elif target == "chatgpt":
        run_import_chatgpt(args, env)
    elif target == "claude":
        run_import_claude(args, env)
    elif target == "claude-code":
        run_import_claude_code(args, env)
    else:
        raise SystemExit(f"Unknown import target: {target}")


def run_watch_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    if watch_directory is None:
        env.ui.console.print(
            "[red]The watchfiles package is not available. Enable it in your environment to use watcher commands."
        )
        return
    if args.watch_target == "codex":
        run_watch_codex(args, env)
    elif args.watch_target == "claude-code":
        run_watch_claude_code(args, env)
    else:
        raise SystemExit(f"Unknown watch target: {args.watch_target}")


def run_import_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme

    result = import_codex_session(
        args.session_id,
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=collapse,
        html=html_enabled,
        html_theme=html_theme,
    )

    lines = [f"Markdown: {result.markdown_path}"]
    if result.html_path:
        lines.append(f"HTML preview: {result.html_path}")
    if result.attachments_dir:
        lines.append(f"Attachments directory: {result.attachments_dir}")
    stats = result.document.stats
    attachments_total = stats.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachment count: {attachments_total}")
    tokens = stats.get("totalTokensApprox")
    if tokens is not None:
        lines.append(f"Approx tokens: {int(tokens)}")
    for key, label in (
        ("chunkCount", "Chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ):
        value = stats.get(key)
        if value:
            lines.append(f"{label}: {int(value)}")
    ui.summary("Codex Import", lines)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, [result])


def run_watch_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    debounce = max(0.5, args.debounce)

    ui.banner("Watching Codex sessions", str(base_dir))

    def sync_once() -> None:
        try:
            result = sync_codex_sessions(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                html=html_enabled,
                html_theme=html_theme,
                force=False,
                prune=False,
                diff=False,
                sessions=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            ui.console.print(f"[red]Codex sync failed: {exc}")
        else:
            _log_local_sync(ui, "Codex Watch", result)

    sync_once()
    if getattr(args, "once", False):
        return
    last_run = time.monotonic()
    try:
        for changes in watch_directory(base_dir, recursive=True):  # type: ignore[arg-type]
            if not any(Path(path).suffix == ".jsonl" for _, path in changes):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        ui.console.print("[cyan]Codex watcher stopped.")


def run_import_chatgpt(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CHATGPT_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_chatgpt_conversations(export_path)
        except Exception as exc:
            ui.console.print(f"[red]Failed to scan export: {exc}")
            return
        if not entries:
            ui.console.print("No conversations found in export.")
            return
        lines = [
            f"{entry.get('title') or '(untitled)'}\t{entry.get('update_time') or entry.get('create_time') or ''}\t{entry.get('id')}"
            for entry in entries
        ]
        selection = sk_select(
            lines,
            preview=None,
            header="Select conversations to import",
        )
        if selection is None:
            ui.console.print("[yellow]Import cancelled; no conversations selected.")
            return
        if not selection:
            ui.console.print("[yellow]No conversations selected; nothing to import.")
            return
        selected_ids = [line.split("\t")[-1] for line in selection]
    elif args.all:
        selected_ids = None

    try:
        results = import_chatgpt_export(
            export_path,
            output_dir=out_dir,
            collapse_threshold=collapse,
            html=html_enabled,
            html_theme=html_theme,
            selected_ids=selected_ids,
        )
    except Exception as exc:
        ui.console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(ui, "ChatGPT Import", results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_claude_conversations(export_path)
        except Exception as exc:
            ui.console.print(f"[red]Failed to scan export: {exc}")
            return
        if not entries:
            ui.console.print("No conversations found in export.")
            return
        lines = [
            f"{entry.get('title') or '(untitled)'}\t{entry.get('updated_at') or entry.get('created_at') or ''}\t{entry.get('id')}"
            for entry in entries
        ]
        selection = sk_select(
            lines,
            preview=None,
            header="Select conversations to import",
        )
        if selection is None:
            ui.console.print("[yellow]Import cancelled; no conversations selected.")
            return
        if not selection:
            ui.console.print("[yellow]No conversations selected; nothing to import.")
            return
        selected_ids = [line.split("\t")[-1] for line in selection]
    elif args.all:
        selected_ids = None

    try:
        results = import_claude_export(
            export_path,
            output_dir=out_dir,
            collapse_threshold=collapse,
            html=html_enabled,
            html_theme=html_theme,
            selected_ids=selected_ids,
        )
    except Exception as exc:
        ui.console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(ui, "Claude Import", results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_PROJECT_ROOT
    session_id = args.session_id

    if session_id in {"pick", "?"} or (session_id == "-" and not ui.plain):
        entries = list_claude_code_sessions(base_dir)
        if not entries:
            ui.console.print("No Claude Code sessions found.")
            return
        lines = [f"{entry['name']}\t{entry['workspace']}\t{entry['path']}" for entry in entries]
        selection = sk_select(lines, multi=False, header="Select Claude Code session")
        if not selection:
            ui.console.print("[yellow]Import cancelled; no session selected.")
            return
        session_id = selection[0].split("\t")[-1]

    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme

    kwargs = {}
    if args.base_dir:
        kwargs["base_dir"] = base_dir

    try:
        result = import_claude_code_session(
            session_id,
            output_dir=out_dir,
            collapse_threshold=collapse,
            html=html_enabled,
            html_theme=html_theme,
            **kwargs,
        )
    except Exception as exc:
        ui.console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(ui, "Claude Code Import", [result])
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, [result])


def run_automation_cli(args: argparse.Namespace, env: CommandEnv) -> None:  # noqa: D401
    """Print scheduler snippets for automation targets."""

    target = TARGETS[args.target]
    defaults = target.defaults or {}

    if args.automation_format == "describe":
        data = describe_targets(getattr(args, "target", None))
        print(json.dumps(data, indent=2))
        return

    working_dir_value = getattr(args, "working_dir", None)
    if working_dir_value is None and defaults.get("workingDir"):
        working_dir_value = defaults["workingDir"]
    working_dir = Path(working_dir_value) if working_dir_value else REPO_ROOT
    working_dir = working_dir.resolve()

    extra_args: List[str] = []
    out_value = getattr(args, "out", None)
    if out_value is None and defaults.get("outputDir"):
        out_value = defaults["outputDir"]
    if out_value:
        extra_args.extend(["--out", str(Path(out_value).resolve())])
    extra_args.extend(getattr(args, "extra_arg", []) or [])

    collapse_value = getattr(args, "collapse_threshold", None)
    html_override = True if getattr(args, "html", False) else None

    if args.automation_format == "systemd":
        snippet = systemd_snippet(
            target_key=args.target,
            interval=args.interval,
            working_dir=working_dir,
            extra_args=extra_args,
            boot_delay=args.boot_delay,
            collapse_threshold=collapse_value,
            html=html_override,
        )
    else:
        snippet = cron_snippet(
            target_key=args.target,
            schedule=args.schedule,
            working_dir=working_dir,
            log_path=args.log,
            extra_args=extra_args,
            state_env=args.state_home,
            collapse_threshold=collapse_value,
            html=html_override,
        )
    print(snippet, end="")


def run_watch_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_PROJECT_ROOT
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    debounce = max(0.5, args.debounce)

    ui.banner("Watching Claude Code sessions", str(base_dir))

    def sync_once() -> None:
        try:
            result = sync_claude_code_sessions(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                html=html_enabled,
                html_theme=html_theme,
                force=False,
                prune=False,
                diff=False,
                sessions=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            ui.console.print(f"[red]Claude Code sync failed: {exc}")
        else:
            _log_local_sync(ui, "Claude Code Watch", result)

    sync_once()
    if getattr(args, "once", False):
        return
    last_run = time.monotonic()
    try:
        for changes in watch_directory(base_dir, recursive=True):  # type: ignore[arg-type]
            if not any(Path(path).suffix == ".jsonl" for _, path in changes):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        ui.console.print("[cyan]Claude Code watcher stopped.")


def run_doctor_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    codex_dir = Path(args.codex_dir) if args.codex_dir else Path.home() / ".codex" / "sessions"
    claude_dir = Path(args.claude_code_dir) if args.claude_code_dir else DEFAULT_PROJECT_ROOT
    report = doctor_run(
        codex_dir=codex_dir,
        claude_code_dir=claude_dir,
        limit=args.limit,
    )

    sample_config = Path(__file__).resolve().parent.parent / "docs" / "polylogue.config.sample.jsonc"
    config_hint = {
        "cmd": "doctor",
        "checked": {k: int(v) for k, v in report.checked.items()},
        "issues": [
            {
                "provider": issue.provider,
                "path": str(issue.path),
                "message": issue.message,
                "severity": issue.severity,
            }
            for issue in report.issues
        ],
        "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
        "configEnv": CONFIG_ENV,
        "configCandidates": [str(p) for p in DEFAULT_PATHS],
        "configSample": str(sample_config),
    }

    if getattr(args, "json", False):
        print(json.dumps(config_hint, indent=2))
        return

    lines = [
        f"Codex sessions checked: {report.checked.get('codex', 0)}",
        f"Claude Code sessions checked: {report.checked.get('claude-code', 0)}",
    ]
    if CONFIG_PATH is None:
        candidates = ", ".join(str(p) for p in DEFAULT_PATHS)
        lines.append(
            f"No Polylogue config detected. Copy {sample_config} to one of [{candidates}] or set ${CONFIG_ENV}."
        )
    if not report.issues:
        lines.append("No issues detected.")
        ui.summary("Doctor", lines)
        return

    if not ui.plain:
        try:
            from rich.table import Table

            table = Table(title="Doctor Issues", show_lines=False)
            table.add_column("Provider")
            table.add_column("Severity")
            table.add_column("Path")
            table.add_column("Message")
            for issue in report.issues:
                table.add_row(issue.provider, issue.severity, str(issue.path), issue.message)
            ui.console.print(table)
        except Exception:
            pass
    lines.append(f"Found {len(report.issues)} issue(s):")
    for issue in report.issues:
        lines.append(f"- [{issue.severity}] {issue.provider}: {issue.path} — {issue.message}")
    ui.summary("Doctor", lines)


def run_stats_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    directory = Path(args.dir) if args.dir else DEFAULT_RENDER_OUT
    if not directory.exists():
        ui.console.print(f"[red]Directory not found: {directory}")
        return

    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        ui.summary("Stats", ["No Markdown files found."])
        return

    try:
        import frontmatter  # type: ignore
    except Exception:  # pragma: no cover
        frontmatter = None  # type: ignore

    since_epoch = parse_input_time_to_epoch(getattr(args, "since", None))
    until_epoch = parse_input_time_to_epoch(getattr(args, "until", None))

    def _load_metadata(path: Path) -> Dict[str, Any]:
        if frontmatter is not None:
            try:
                post = frontmatter.load(path)  # type: ignore[attr-defined]
                return dict(post.metadata)
            except Exception:
                pass
        # Minimal front matter parser: expects simple key: value pairs.
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return {}
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        meta: Dict[str, Any] = {}
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
        return meta

    totals: Dict[str, Any] = {
        "files": 0,
        "attachments": 0,
        "attachmentBytes": 0,
        "tokens": 0,
    }
    per_provider: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    filtered_out = 0

    for path in md_files:
        meta = _load_metadata(path)
        attachment_count = meta.get("attachmentCount") or meta.get("attachments")
        if isinstance(attachment_count, list):
            attachment_count = len(attachment_count)
        if attachment_count is None:
            attachment_count = 0
        attachment_bytes = meta.get("attachmentBytes") or 0
        tokens = meta.get("totalTokensApprox") or meta.get("tokensApprox") or 0
        provider = meta.get("sourcePlatform") or "unknown"

        timestamp = None
        for key in (
            "sourceModifiedTime",
            "sourceCreatedTime",
            "sourceModified",
            "sourceCreated",
        ):
            value = meta.get(key)
            epoch = parse_input_time_to_epoch(value) if value else None
            if epoch:
                timestamp = epoch
                break
        if timestamp is None:
            try:
                timestamp = path.stat().st_mtime
            except OSError:
                timestamp = None

        if since_epoch and (timestamp is None or timestamp < since_epoch):
            filtered_out += 1
            continue
        if until_epoch and (timestamp is None or timestamp > until_epoch):
            filtered_out += 1
            continue

        totals["files"] += 1
        totals["attachments"] += int(attachment_count)
        totals["attachmentBytes"] += int(attachment_bytes)
        totals["tokens"] += int(tokens)

        prov = per_provider.setdefault(
            provider,
            {"files": 0, "attachments": 0, "attachmentBytes": 0, "tokens": 0},
        )
        prov["files"] += 1
        prov["attachments"] += int(attachment_count)
        prov["attachmentBytes"] += int(attachment_bytes)
        prov["tokens"] += int(tokens)

        rows.append(
            {
                "file": path.name,
                "provider": provider,
                "attachments": int(attachment_count),
                "attachmentBytes": int(attachment_bytes),
                "tokens": int(tokens),
            }
        )

    if getattr(args, "json", False):
        payload = {
            "cmd": "stats",
            "directory": str(directory),
            "totals": totals,
            "providers": per_provider,
            "files": rows,
            "filteredOut": filtered_out,
        }
        print(json.dumps(payload, indent=2))
        return

    lines = [
        f"Directory: {directory}",
        f"Files: {totals['files']} Attachments: {totals['attachments']} (~{totals['attachmentBytes'] / (1024 * 1024):.2f} MiB) Tokens≈ {totals['tokens']}",
    ]
    if filtered_out:
        lines.append(f"Filtered out {filtered_out} file(s) outside date range.")
    if not ui.plain:
        try:
            from rich.table import Table

            table = Table(title="Provider Summary")
            table.add_column("Provider")
            table.add_column("Files", justify="right")
            table.add_column("Attachments", justify="right")
            table.add_column("Attachment MiB", justify="right")
            table.add_column("Tokens", justify="right")
            for provider, data in per_provider.items():
                table.add_row(
                    provider,
                    str(data["files"]),
                    str(data["attachments"]),
                    f"{data['attachmentBytes'] / (1024 * 1024):.2f}",
                    str(data["tokens"]),
                )
            ui.console.print(table)
        except Exception:
            pass
    ui.summary("Stats", lines)

def _collect_session_selection(ui, sessions: List[Path], header: str) -> Optional[List[Path]]:
    if not sessions:
        ui.console.print("No sessions found.")
        return None
    lines = [f"{path.stem}\t{path.parent.name}\t{path}" for path in sessions]
    selection = sk_select(lines, header=header)
    if selection is None:
        ui.console.print("[yellow]Sync cancelled; no sessions selected.")
        return None
    if not selection:
        ui.console.print("[yellow]No sessions selected; nothing to do.")
        return []
    return [Path(line.split("\t")[-1]) for line in selection]


def run_sync_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    force = args.force
    prune = args.prune
    diff_enabled = getattr(args, "diff", False)

    selected_paths: Optional[List[Path]] = None
    if not args.all and not ui.plain:
        sessions = sorted(base_dir.rglob("*.jsonl"))
        selection = _collect_session_selection(ui, sessions, "Select Codex sessions")
        if selection is None:
            return
        if not selection:
            return
        selected_paths = selection

    result = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=collapse,
        html=html_enabled,
        html_theme=html_theme,
        force=force,
        prune=prune,
        diff=diff_enabled,
        sessions=selected_paths,
    )

    attachments = result.attachments
    attachment_bytes = result.attachment_bytes
    tokens = result.tokens

    if getattr(args, "json", False):
        payload = {
            "cmd": "sync-codex",
            "count": len(result.written),
            "out": str(result.output_dir),
            "skipped": result.skipped,
            "pruned": result.pruned,
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokensApprox": tokens,
            "diffs": result.diffs,
            "files": [
                {
                    "output": str(item.markdown_path),
                    "attachments": len(item.document.attachments),
                    "attachmentBytes": item.document.metadata.get("attachmentBytes"),
                    "stats": item.document.stats,
                    "html": str(item.html_path) if item.html_path else None,
                    "diff": str(item.diff_path) if item.diff_path else None,
                }
                for item in result.written
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        summarize_import(ui, "Codex Sync", result.written)
        if result.skipped:
            ui.console.print(f"Skipped {result.skipped} up-to-date session(s).")
        if result.pruned:
            ui.console.print(f"Pruned {result.pruned} stale path(s).")

    add_run(
        {
            "cmd": "sync-codex",
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokens": tokens,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "diffs": result.diffs,
        }
    )


def run_sync_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_PROJECT_ROOT
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = args.html or SETTINGS.html_previews
    html_theme = args.html_theme or SETTINGS.html_theme
    force = args.force
    prune = args.prune

    selected_paths: Optional[List[Path]] = None
    if not args.all and not ui.plain:
        session_entries = list_claude_code_sessions(base_dir)
        sessions = [Path(entry["path"]) for entry in session_entries]
        selection = _collect_session_selection(ui, sessions, "Select Claude Code sessions")
        if selection is None:
            return
        if not selection:
            return
        selected_paths = selection

    result = sync_claude_code_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=collapse,
        html=html_enabled,
        html_theme=html_theme,
        force=force,
        prune=prune,
        diff=diff_enabled,
        sessions=selected_paths,
    )

    attachments = result.attachments
    attachment_bytes = result.attachment_bytes
    tokens = result.tokens

    if getattr(args, "json", False):
        payload = {
            "cmd": "sync-claude-code",
            "count": len(result.written),
            "out": str(result.output_dir),
            "skipped": result.skipped,
            "pruned": result.pruned,
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokensApprox": tokens,
            "diffs": result.diffs,
            "files": [
                {
                    "output": str(item.markdown_path),
                    "attachments": len(item.document.attachments),
                    "attachmentBytes": item.document.metadata.get("attachmentBytes"),
                    "stats": item.document.stats,
                    "html": str(item.html_path) if item.html_path else None,
                    "diff": str(item.diff_path) if item.diff_path else None,
                }
                for item in result.written
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        summarize_import(ui, "Claude Code Sync", result.written)
        if result.skipped:
            ui.console.print(f"Skipped {result.skipped} up-to-date session(s).")
        if result.pruned:
            ui.console.print(f"Pruned {result.pruned} stale path(s).")

    add_run(
        {
            "cmd": "sync-claude-code",
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokens": tokens,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "diffs": result.diffs,
        }
    )
def prompt_render(env: CommandEnv) -> None:
    ui = env.ui
    default_input = str(Path.cwd())
    user_input = ui.input("Directory or file to render", default=default_input)
    if not user_input:
        return
    path = Path(user_input).expanduser()
    if not path.exists():
        ui.console.print(f"[red]Path not found: {path}")
        return
    class Args:
        pass
    args = Args()
    args.input = path
    args.out = None
    args.links_only = False
    args.dry_run = False
    args.force = False
    args.collapse_threshold = None
    args.json = False
    args.html = SETTINGS.html_previews
    args.html_theme = SETTINGS.html_theme
    run_render_cli(args, env, json_output=False)


def prompt_sync(env: CommandEnv) -> None:
    ui = env.ui
    class Args:
        pass
    args = Args()
    args.folder_name = DEFAULT_FOLDER_NAME
    args.folder_id = None
    args.out = None
    args.links_only = False
    args.since = None
    args.until = None
    args.name_filter = None
    args.dry_run = False
    args.force = False
    args.prune = False
    args.collapse_threshold = None
    args.json = False
    args.html = SETTINGS.html_previews
    args.html_theme = SETTINGS.html_theme
    run_sync_cli(args, env, json_output=False)


def prompt_sync_codex(env: CommandEnv) -> None:
    ui = env.ui
    class Args:
        pass
    args = Args()
    args.base_dir = None
    args.out = None
    args.collapse_threshold = None
    args.html = SETTINGS.html_previews
    args.html_theme = SETTINGS.html_theme
    args.force = False
    args.prune = False
    args.all = False
    args.json = False
    args.diff = False
    run_sync_codex(args, env)


def prompt_sync_claude_code(env: CommandEnv) -> None:
    class Args:
        pass
    args = Args()
    args.base_dir = None
    args.out = None
    args.collapse_threshold = None
    args.html = SETTINGS.html_previews
    args.html_theme = SETTINGS.html_theme
    args.force = False
    args.prune = False
    args.all = False
    args.json = False
    args.diff = False
    run_sync_claude_code(args, env)


def prompt_doctor(env: CommandEnv) -> None:
    class Args:
        pass

    args = Args()
    args.codex_dir = None
    args.claude_code_dir = None
    args.limit = 25
    args.json = False
    run_doctor_cli(args, env)


def prompt_list(env: CommandEnv) -> None:
    class Args:
        pass
    args = Args()
    args.folder_name = DEFAULT_FOLDER_NAME
    args.folder_id = None
    args.since = None
    args.until = None
    args.name_filter = None
    args.json = False
    args.html = False
    args.html_theme = "light"
    run_list_cli(args, env, json_output=False)


def show_help(env: CommandEnv) -> None:
    ui = env.ui
    ui.console.print("Polylogue commands:")
    ui.console.print("  render  Render local provider JSON files to Markdown")
    ui.console.print("  sync    Sync Google Drive chats to local Markdown")
    ui.console.print("  list    List chats available in the configured Drive folder")
    ui.console.print("  status  Show cached Drive info and recent runs")
    ui.console.print("  watch   Watch Codex/Claude Code directories and sync on change")
    ui.console.print("  --plain Disable interactive UI for automation")
    ui.console.print("  --json  Emit machine-readable summaries when supported")


def settings_menu(env: CommandEnv) -> None:
    ui = env.ui
    while True:
        toggle_label = f"Toggle HTML previews ({'on' if SETTINGS.html_previews else 'off'})"
        theme_label = f"HTML theme ({SETTINGS.html_theme})"
        choices = [toggle_label, theme_label, "Reset defaults", "Back"]
        choice = ui.choose("Settings", choices)
        if choice is None or choice == "Back":
            return
        if choice == toggle_label:
            SETTINGS.html_previews = not SETTINGS.html_previews
            state = "enabled" if SETTINGS.html_previews else "disabled"
            ui.console.print(f"HTML previews {state}.")
        elif choice == theme_label:
            if ui.plain:
                ui.console.print("Set --html-theme when running in plain mode.")
                continue
            selection = ui.choose("Select HTML theme", ["light", "dark"])
            if selection:
                SETTINGS.html_theme = selection
                ui.console.print(f"HTML theme set to {selection}.")
        elif choice == "Reset defaults":
            reset_settings()
            ui.console.print("Settings reset to defaults.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ui = create_ui(args.plain)
    env = CommandEnv(ui=ui)

    if args.cmd is None:
        if ui.plain:
            parser.print_help()
            return
        ui.banner("Polylogue", "Render AI chat logs or sync providers")
        interactive_menu(env)
        return

    if args.cmd == "render":
        run_render_cli(args, env, json_output=getattr(args, "json", False))
    elif args.cmd == "sync":
        run_sync_cli(args, env, json_output=getattr(args, "json", False))
    elif args.cmd == "sync-codex":
        run_sync_codex(args, env)
    elif args.cmd == "sync-claude-code":
        run_sync_claude_code(args, env)
    elif args.cmd == "list":
        run_list_cli(args, env, json_output=getattr(args, "json", False))
    elif args.cmd == "automation":
        run_automation_cli(args, env)
    elif args.cmd == "status":
        run_status_cli(args, env)
    elif args.cmd == "import":
        run_import_cli(args, env)
    elif args.cmd == "watch":
        run_watch_cli(args, env)
    elif args.cmd == "doctor":
        run_doctor_cli(args, env)
    elif args.cmd == "stats":
        run_stats_cli(args, env)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
