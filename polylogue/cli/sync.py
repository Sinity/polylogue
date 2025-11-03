from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ..cli_common import filter_chats, sk_select
from ..commands import CommandEnv, list_command, sync_command
from ..drive_client import DriveClient
from ..importers.claude_code import list_claude_code_sessions
from ..local_sync import LocalSyncResult, sync_claude_code_sessions, sync_codex_sessions
from ..options import ListOptions, SyncOptions
from ..util import CLAUDE_CODE_PROJECT_ROOT, CODEX_SESSIONS_ROOT, add_run, path_order_key
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    DEFAULT_SYNC_OUT,
    default_sync_namespace,
    resolve_collapse_value,
    resolve_html_enabled,
    resolve_output_path,
    merge_with_defaults,
)
from .summaries import summarize_import


def _log_local_sync(ui, title: str, result: LocalSyncResult, *, provider: str) -> None:
    console = ui.console
    if result.written:
        summarize_import(ui, title, result.written)
    else:
        console.print(f"[cyan]{title}: no new Markdown files.")
    if result.skipped:
        console.print(f"[cyan]{title}: skipped {result.skipped} up-to-date session(s).")
    if result.pruned:
        console.print(f"[cyan]{title}: pruned {result.pruned} path(s).")
    add_run(
        {
            "cmd": title.lower().replace(" ", "-"),
            "provider": provider,
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": result.attachments,
            "attachmentBytes": result.attachment_bytes,
            "tokens": result.tokens,
            "words": result.words,
            "diffs": result.diffs,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "duration": getattr(result, "duration", 0.0),
        }
    )


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
    console = env.ui.console
    console.print(f"{len(result.files)} chat(s) in {result.folder_name}:")
    for chat in result.files:
        console.print(f"- {chat.get('name')}  {chat.get('modifiedTime', '')}  {chat.get('id', '')}")


def run_sync_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    settings = env.settings
    if provider == "drive":
        merged = merge_with_defaults(default_sync_namespace("drive", settings), args)
        _run_sync_drive(merged, env)
    elif provider == "codex":
        merged = merge_with_defaults(default_sync_namespace("codex", settings), args)
        _run_sync_codex(merged, env)
    elif provider == "claude-code":
        merged = merge_with_defaults(default_sync_namespace("claude-code", settings), args)
        _run_sync_claude_code(merged, env)
    else:
        raise SystemExit(f"Unsupported provider for sync: {provider}")


def _run_sync_drive(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    json_mode = getattr(args, "json", False)

    if getattr(args, "list_only", False):
        list_args = argparse.Namespace(
            folder_name=args.folder_name,
            folder_id=args.folder_id,
            since=args.since,
            until=args.until,
            name_filter=args.name_filter,
        )
        run_list_cli(list_args, env, json_output=json_mode)
        return

    download_attachments = not args.links_only
    if not ui.plain and not args.links_only:
        download_attachments = ui.confirm("Download attachments for synced chats?", default=True)

    selected_ids: Optional[List[str]] = None
    if not ui.plain and not json_mode:
        drive = env.drive or DriveClient(ui)
        env.drive = drive
        raw_chats = drive.list_chats(args.folder_name, args.folder_id)
        filtered = filter_chats(raw_chats, args.name_filter, args.since, args.until)
        if not filtered:
            console.print("No chats to sync")
            return
        lines = [f"{c.get('name') or '(untitled)'}\t{c.get('modifiedTime') or ''}\t{c.get('id')}" for c in filtered]
        selection = sk_select(lines, preview="printf '%s' {+}")
        if selection is None:
            console.print("[yellow]Sync cancelled; no chats selected.")
            return
        if not selection:
            console.print("[yellow]No chats selected; nothing to sync.")
            return
        selected_ids = [line.split("\t")[-1] for line in selection]

    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    options = SyncOptions(
        folder_name=args.folder_name,
        folder_id=args.folder_id,
        output_dir=resolve_output_path(args.out, DEFAULT_SYNC_OUT),
        collapse_threshold=resolve_collapse_value(args.collapse_threshold, DEFAULT_COLLAPSE),
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

    if json_mode:
        payload = {
            "cmd": "sync drive",
            "provider": "drive",
            "count": result.count,
            "out": str(result.output_dir),
            "folder_name": result.folder_name,
            "folder_id": result.folder_id,
            "files": [
                {
                    "id": item.id,
                    "name": item.name,
                    "output": str(item.output),
                    "slug": item.slug,
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
    for item in result.items:
        info = f"- {item.slug} (attachments: {item.attachments})"
        if item.html:
            info += " [+html]"
        if getattr(item, "diff", None):
            info += " [+diff]"
        lines.append(info)
    console.print("\n".join(lines))


def _collect_session_selection(ui, sessions: List[Path], header: str) -> Optional[List[Path]]:
    console = ui.console
    if not sessions:
        console.print("No sessions found.")
        return None
    name_width = min(max(len(path.stem) for path in sessions), 72)
    parent_width = min(max(len(path.parent.name) for path in sessions), 24)
    lines = [
        f"{path.stem[:name_width]:<{name_width}}\t{path.parent.name[:parent_width]:<{parent_width}}\t{path}"
        for path in sessions
    ]
    selection = sk_select(
        lines,
        header=f"{header} — tab to toggle, ctrl-a select all, enter accept",
        prompt="Sessions> ",
    )
    if selection is None:
        console.print("[yellow]Sync cancelled; no sessions selected.")
        return None
    if not selection:
        console.print("[yellow]No sessions selected; nothing to do.")
        return []
    return [Path(line.split("\t")[-1]) for line in selection]


def _run_sync_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CODEX_SESSIONS_ROOT
    out_dir = resolve_output_path(args.out, DEFAULT_CODEX_SYNC_OUT)
    collapse = resolve_collapse_value(args.collapse_threshold, DEFAULT_COLLAPSE)
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    force = args.force
    prune = args.prune
    diff_enabled = getattr(args, "diff", False)

    selected_paths: Optional[List[Path]] = None
    if not args.all and not ui.plain:
        sessions = sorted(base_dir.rglob("*.jsonl"), key=path_order_key, reverse=True)
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
    words = result.words

    if getattr(args, "json", False):
        files_payload = []
        for item in result.written:
            doc = item.document
            files_payload.append(
                {
                    "output": str(item.markdown_path),
                    "attachments": len(doc.attachments) if doc else 0,
                    "attachmentBytes": doc.metadata.get("attachmentBytes") if doc and doc.metadata else None,
                    "stats": doc.stats if doc and doc.stats else {},
                    "html": str(item.html_path) if item.html_path else None,
                    "diff": str(item.diff_path) if item.diff_path else None,
                }
            )
        payload = {
            "cmd": "sync codex",
            "provider": "codex",
            "count": len(result.written),
            "out": str(result.output_dir),
            "skipped": result.skipped,
            "pruned": result.pruned,
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokensApprox": tokens,
            "wordsApprox": words,
            "diffs": result.diffs,
            "files": files_payload,
        }
        print(json.dumps(payload, indent=2))
    else:
        summarize_import(ui, "Codex Sync", result.written)
        console = ui.console
        if result.skipped:
            console.print(f"Skipped {result.skipped} up-to-date session(s).")
        if result.pruned:
            console.print(f"Pruned {result.pruned} stale path(s).")

    add_run(
        {
            "cmd": "sync codex",
            "provider": "codex",
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokens": tokens,
            "words": words,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "diffs": result.diffs,
        }
    )


def _run_sync_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CLAUDE_CODE_PROJECT_ROOT
    out_dir = resolve_output_path(args.out, DEFAULT_CLAUDE_CODE_SYNC_OUT)
    collapse = resolve_collapse_value(args.collapse_threshold, DEFAULT_COLLAPSE)
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    force = args.force
    prune = args.prune
    diff_enabled = getattr(args, "diff", False)

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
    words = result.words

    if getattr(args, "json", False):
        files_payload = []
        for item in result.written:
            doc = item.document
            files_payload.append(
                {
                    "output": str(item.markdown_path),
                    "attachments": len(doc.attachments) if doc else 0,
                    "attachmentBytes": doc.metadata.get("attachmentBytes") if doc and doc.metadata else None,
                    "stats": doc.stats if doc and doc.stats else {},
                    "html": str(item.html_path) if item.html_path else None,
                    "diff": str(item.diff_path) if item.diff_path else None,
                }
            )
        payload = {
            "cmd": "sync claude-code",
            "provider": "claude-code",
            "count": len(result.written),
            "out": str(result.output_dir),
            "skipped": result.skipped,
            "pruned": result.pruned,
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokensApprox": tokens,
            "wordsApprox": words,
            "diffs": result.diffs,
            "files": files_payload,
        }
        print(json.dumps(payload, indent=2))
    else:
        summarize_import(ui, "Claude Code Sync", result.written)
        console = ui.console
        if result.skipped:
            console.print(f"Skipped {result.skipped} up-to-date session(s).")
        if result.pruned:
            console.print(f"Pruned {result.pruned} stale path(s).")

    add_run(
        {
            "cmd": "sync claude-code",
            "provider": "claude-code",
            "count": len(result.written),
            "out": str(result.output_dir),
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokens": tokens,
            "words": words,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "diffs": result.diffs,
        }
    )


__all__ = [
    "run_list_cli",
    "run_sync_cli",
    "_run_sync_drive",
    "_run_sync_codex",
    "_run_sync_claude_code",
    "_collect_session_selection",
    "_log_local_sync",
]
