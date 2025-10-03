#!/usr/bin/env python3
"""Gemini Markdown (gmd): interactive-first CLI for Gemini logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from chatmd.cli_common import filter_chats, sk_select
from chatmd.commands import (
    CommandEnv,
    list_command,
    render_command,
    status_command,
    sync_command,
)
from chatmd.drive_client import DEFAULT_FOLDER_NAME, DriveClient
from chatmd.importers import (
    ImportResult,
    import_chatgpt_export,
    import_claude_code_session,
    import_claude_export,
    import_codex_session,
)
from chatmd.importers.chatgpt import list_chatgpt_conversations
from chatmd.importers.claude_ai import list_claude_conversations
from chatmd.importers.claude_code import DEFAULT_PROJECT_ROOT, list_claude_code_sessions
from chatmd.local_sync import LocalSyncResult, sync_claude_code_sessions, sync_codex_sessions
from chatmd.options import ListOptions, RenderOptions, SyncOptions
from chatmd.ui import create_ui
from gmd_settings import SETTINGS, reset_settings

DEFAULT_RENDER_OUT = Path("gmd_out")
DEFAULT_SYNC_OUT = Path("gemini_synced")
DEFAULT_COLLAPSE = 25
DEFAULT_CODEX_SYNC_OUT = Path("codex_synced")
DEFAULT_CLAUDE_CODE_SYNC_OUT = Path("claude_code_synced")


def summarize_import(ui, title: str, results: List[ImportResult]) -> None:
    if not results:
        ui.summary(title, ["No files written."])
        return
    output_dir = results[0].markdown_path.parent
    lines = [f"{len(results)} file(s) → {output_dir}"]
    attachments_total = sum(len(res.document.attachments) for res in results)
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    stats_to_sum = [
        ("totalTokensApprox", "Approx tokens"),
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ]
    for key, label in stats_to_sum:
        total = 0
        for res in results:
            value = res.document.stats.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
        if total:
            lines.append(f"{label}: {total}")
    for res in results:
        info = f"- {res.markdown_path.name} (attachments: {len(res.document.attachments)})"
        if res.html_path:
            info += " [+html]"
        lines.append(info)
    ui.summary(title, lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini Markdown (gmd)")
    parser.add_argument("--plain", action="store_true", help="Disable interactive UI")
    sub = parser.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render")
    p_render.add_argument("input", type=Path, help="File or directory with Gemini JSON logs")
    p_render.add_argument("--out", type=Path, default=None, help="Output directory (default gmd_out)")
    p_render.add_argument("--links-only", action="store_true", help="Link attachments instead of downloading")
    p_render.add_argument("--dry-run", action="store_true")
    p_render.add_argument("--force", action="store_true")
    p_render.add_argument("--collapse-threshold", type=int, default=None)
    p_render.add_argument("--json", action="store_true")
    p_render.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_render.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME)
    p_sync.add_argument("--folder-id", type=str, default=None)
    p_sync.add_argument("--out", type=Path, default=None, help="Output directory (default gemini_synced)")
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

    p_sync_codex = sub.add_parser("sync-codex")
    p_sync_codex.add_argument("--base-dir", type=Path, default=None)
    p_sync_codex.add_argument("--out", type=Path, default=None)
    p_sync_codex.add_argument("--collapse-threshold", type=int, default=None)
    p_sync_codex.add_argument("--html", action="store_true")
    p_sync_codex.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_sync_codex.add_argument("--force", action="store_true", help="Re-render even if up-to-date")
    p_sync_codex.add_argument("--prune", action="store_true", help="Remove outputs for missing sessions")
    p_sync_codex.add_argument("--all", action="store_true", help="Process all sessions without prompting")

    p_sync_claude_code = sub.add_parser("sync-claude-code")
    p_sync_claude_code.add_argument("--base-dir", type=Path, default=None)
    p_sync_claude_code.add_argument("--out", type=Path, default=None)
    p_sync_claude_code.add_argument("--collapse-threshold", type=int, default=None)
    p_sync_claude_code.add_argument("--html", action="store_true")
    p_sync_claude_code.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"])
    p_sync_claude_code.add_argument("--force", action="store_true")
    p_sync_claude_code.add_argument("--prune", action="store_true")
    p_sync_claude_code.add_argument("--all", action="store_true")

    p_list = sub.add_parser("list")
    p_list.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME)
    p_list.add_argument("--folder-id", type=str, default=None)
    p_list.add_argument("--since", type=str, default=None)
    p_list.add_argument("--until", type=str, default=None)
    p_list.add_argument("--name-filter", type=str, default=None)
    p_list.add_argument("--json", action="store_true")

    sub.add_parser("status")

    p_import = sub.add_parser("import")
    import_sub = p_import.add_subparsers(dest="import_target", required=True)

    p_import_chatgpt = import_sub.add_parser("chatgpt", help="Convert a ChatGPT export to Markdown")
    p_import_chatgpt.add_argument("export_path", type=Path, help="Path to ChatGPT export .zip or directory")
    p_import_chatgpt.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Restrict to specific conversation id (repeatable)")
    p_import_chatgpt.add_argument("--all", action="store_true", help="Import all conversations without prompting")
    p_import_chatgpt.add_argument("--out", type=Path, default=None, help="Output directory for Markdown files")
    p_import_chatgpt.add_argument("--collapse-threshold", type=int, default=None)
    p_import_chatgpt.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_import_chatgpt.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")

    p_import_claude = import_sub.add_parser("claude", help="Convert an Anthropic Claude export to Markdown")
    p_import_claude.add_argument("export_path", type=Path, help="Path to Claude export .zip or directory")
    p_import_claude.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Restrict to specific conversation id (repeatable)")
    p_import_claude.add_argument("--all", action="store_true", help="Import all conversations without prompting")
    p_import_claude.add_argument("--out", type=Path, default=None, help="Output directory for Markdown files")
    p_import_claude.add_argument("--collapse-threshold", type=int, default=None)
    p_import_claude.add_argument("--html", action="store_true", help="Also write HTML previews")
    p_import_claude.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML previews")

    p_import_claude_code = import_sub.add_parser("claude-code", help="Convert a Claude Code session to Markdown")
    p_import_claude_code.add_argument("session_id", type=str, help="Session UUID or suffix")
    p_import_claude_code.add_argument("--base-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_import_claude_code.add_argument("--out", type=Path, default=None, help="Output directory for Markdown")
    p_import_claude_code.add_argument("--collapse-threshold", type=int, default=None)
    p_import_claude_code.add_argument("--html", action="store_true", help="Also write HTML preview")
    p_import_claude_code.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML preview")

    p_import_codex = import_sub.add_parser("codex", help="Convert a Codex CLI session to Markdown")
    p_import_codex.add_argument("session_id", type=str, help="Codex session UUID (or suffix)")
    p_import_codex.add_argument("--base-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_import_codex.add_argument("--out", type=Path, default=None, help="Output directory for Markdown")
    p_import_codex.add_argument("--collapse-threshold", type=int, default=None, help="Fold responses longer than this many lines")
    p_import_codex.add_argument("--html", action="store_true", help="Also write HTML preview")
    p_import_codex.add_argument("--html-theme", type=str, default=None, choices=["light", "dark"], help="Theme for HTML preview")

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
        lines.append(info)
    ui.summary("Sync", lines)


def run_status_cli(env: CommandEnv) -> None:
    result = status_command(env)
    ui = env.ui
    if ui.plain:
        ui.console.print("Environment:")
        ui.console.print(f"  credentials.json: {'present' if result.credentials_present else 'missing'}")
        ui.console.print(f"  token.json: {'present' if result.token_present else 'missing'}")
        ui.console.print(f"  state cache: {result.state_path}")
        ui.console.print(f"  runs log: {result.runs_path}")
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
    if not result.recent_runs:
        ui.console.print("Recent runs: (none)")
    else:
        ui.console.print("Recent runs (last 10):")
        for entry in result.recent_runs:
            ui.console.print(f"- {entry.get('cmd')} → {entry.get('out')}")


def interactive_menu(env: CommandEnv) -> None:
    ui = env.ui
    options = [
        "Render Local Logs",
        "Sync Drive Folder",
        "Sync Codex Sessions",
        "Sync Claude Code Sessions",
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
        elif choice == "List Drive Chats":
            prompt_list(env)
        elif choice == "View Recent Runs":
            run_status_cli(env)
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


def run_import_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    out_dir = Path(args.out) if args.out else DEFAULT_RENDER_OUT
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


def run_import_chatgpt(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_RENDER_OUT
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


def run_import_claude(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_RENDER_OUT
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

    out_dir = Path(args.out) if args.out else DEFAULT_RENDER_OUT
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
        sessions=selected_paths,
    )

    summarize_import(ui, "Codex Sync", result.written)
    if result.skipped:
        ui.console.print(f"Skipped {result.skipped} up-to-date session(s).")
    if result.pruned:
        ui.console.print(f"Pruned {result.pruned} stale path(s).")
    add_run({"cmd": "sync-codex", "count": len(result.written), "out": str(result.output_dir)})


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
        sessions=selected_paths,
    )

    summarize_import(ui, "Claude Code Sync", result.written)
    if result.skipped:
        ui.console.print(f"Skipped {result.skipped} up-to-date session(s).")
    if result.pruned:
        ui.console.print(f"Pruned {result.pruned} stale path(s).")
    add_run({"cmd": "sync-claude-code", "count": len(result.written), "out": str(result.output_dir)})
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
    run_sync_claude_code(args, env)


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
    ui.console.print("Gemini Markdown CLI commands:")
    ui.console.print("  render  Render local Gemini JSON files to Markdown")
    ui.console.print("  sync    Sync Google Drive chats to local Markdown")
    ui.console.print("  list    List chats available in the configured Drive folder")
    ui.console.print("  status  Show cached Drive info and recent runs")
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
        ui.banner("Gemini Markdown", "Render local logs or sync Google Drive chats")
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
    elif args.cmd == "status":
        run_status_cli(env)
    elif args.cmd == "import":
        run_import_cli(args, env)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
