#!/usr/bin/env python3
"""Gemini Markdown (gmd): interactive-first CLI for Gemini logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from geminimd.cli_common import filter_chats, sk_select
from geminimd.commands import (
    CommandEnv,
    list_command,
    render_command,
    status_command,
    sync_command,
)
from geminimd.drive_client import DEFAULT_FOLDER_NAME, DriveClient
from geminimd.options import ListOptions, RenderOptions, SyncOptions
from geminimd.ui import create_ui
from gmd_settings import SETTINGS, reset_settings

DEFAULT_RENDER_OUT = Path("gmd_out")
DEFAULT_SYNC_OUT = Path("gemini_synced")
DEFAULT_COLLAPSE = 25


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

    p_list = sub.add_parser("list")
    p_list.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME)
    p_list.add_argument("--folder-id", type=str, default=None)
    p_list.add_argument("--since", type=str, default=None)
    p_list.add_argument("--until", type=str, default=None)
    p_list.add_argument("--name-filter", type=str, default=None)
    p_list.add_argument("--json", action="store_true")

    sub.add_parser("status")

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
    selection = sk_select(lines, preview="bat --style=plain {}")
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
    args.html = False
    args.html_theme = "light"
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
    args.html = False
    args.html_theme = "light"
    run_sync_cli(args, env, json_output=False)


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
    elif args.cmd == "list":
        run_list_cli(args, env, json_output=getattr(args, "json", False))
    elif args.cmd == "status":
        run_status_cli(env)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
