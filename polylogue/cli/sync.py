from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ..cli_common import filter_chats, sk_select
from ..commands import CommandEnv, list_command, sync_command
from ..drive_client import DriveClient
from ..local_sync import (
    LocalSyncResult,
    LOCAL_SYNC_PROVIDER_NAMES,
    get_local_provider,
)
from ..options import ListOptions, SyncOptions
from ..util import add_run, format_run_brief, latest_run, path_order_key
from .context import (
    DEFAULT_COLLAPSE,
    DEFAULT_SYNC_OUT,
    default_sync_namespace,
    resolve_collapse_thresholds,
    resolve_html_enabled,
    resolve_output_path,
    merge_with_defaults,
)
from .summaries import summarize_import


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _apply_sync_prefs(args: argparse.Namespace, env: CommandEnv) -> argparse.Namespace:
    prefs = getattr(env, "prefs", {}) or {}
    sync_prefs = prefs.get("sync", {}) if isinstance(prefs, dict) else {}
    if not sync_prefs:
        return args

    def _apply_flag(flag: str, attr: str) -> None:
        if flag in sync_prefs and not getattr(args, attr, False) and _truthy(sync_prefs[flag]):
            setattr(args, attr, True)

    if "--html" in sync_prefs and getattr(args, "html_mode", "auto") == "auto":
        args.html_mode = "on" if _truthy(sync_prefs["--html"]) else "off"

    _apply_flag("--links-only", "links_only")
    _apply_flag("--diff", "diff")
    _apply_flag("--prune", "prune")
    _apply_flag("--watch", "watch")
    _apply_flag("--once", "once")
    _apply_flag("--attachment-ocr", "attachment_ocr")
    _apply_flag("--offline", "offline")
    return args


def _log_local_sync(ui, title: str, result: LocalSyncResult, *, provider: str, footer: Optional[List[str]] = None) -> None:
    console = ui.console
    if result.written:
        summarize_import(ui, title, result.written, extra_lines=footer)
    else:
        console.print(f"[cyan]{title}: no new Markdown files.")
        if footer:
            for line in footer:
                console.print(line)
    if result.skipped:
        console.print(f"[cyan]{title}: skipped {result.skipped} up-to-date item(s).")
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
    args = _apply_sync_prefs(args, env)
    if getattr(args, "offline", False) and provider == "drive":
        env.ui.console.print("[red]Drive sync does not support --offline.")
        raise SystemExit(1)
    if provider == "drive":
        merged = merge_with_defaults(default_sync_namespace("drive", settings), args)
        _run_sync_drive(merged, env)
    elif provider in LOCAL_SYNC_PROVIDER_NAMES:
        merged = merge_with_defaults(default_sync_namespace(provider, settings), args)
        if getattr(args, "max_disk", None):
            # Assume up to 20 MiB per session including attachments as a coarse estimate.
            projected = 20 * 1024 * 1024 * max(1, len(getattr(args, "sessions", []) or []))
            from ..util import preflight_disk_requirement

            preflight_disk_requirement(projected_bytes=projected, limit_gib=args.max_disk, ui=env.ui)
        _run_local_sync(provider, merged, env)
    else:
        env.ui.console.print(f"[red]Unsupported provider for sync: {provider}")
        raise SystemExit(1)


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
    if getattr(args, "offline", False):
        download_attachments = False

    previous_run_note = format_run_brief(latest_run(provider="drive", cmd="sync drive"))

    drive = env.drive or DriveClient(ui)
    env.drive = drive
    folder_id = drive.resolve_folder_id(args.folder_name, args.folder_id)
    raw_chats = drive.list_chats(args.folder_name, folder_id)
    filtered = filter_chats(raw_chats, args.name_filter, args.since, args.until)

    cli_ids = [item.strip() for item in getattr(args, "chat_ids", []) if item and item.strip()]
    selected_ids: Optional[List[str]] = cli_ids or None
    if selected_ids is None and not ui.plain and not json_mode:
        if not filtered:
            console.print("No chats to sync")
            return
        if not getattr(args, "all", False):
            lines = [
                f"{c.get('name') or '(untitled)'}\t{c.get('modifiedTime') or ''}\t{c.get('id')}"
                for c in filtered
            ]
            selection = sk_select(lines, preview="printf '%s' {+}", plain=ui.plain)
            if selection is None:
                console.print("[yellow]Sync cancelled; no chats selected.")
                return
            if not selection:
                console.print("[yellow]No chats selected; nothing to sync.")
                return
            selected_ids = [line.split("\t")[-1] for line in selection]

    settings = env.settings
    collapse_thresholds = resolve_collapse_thresholds(args, settings)
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    prefetched = filtered
    if selected_ids:
        selected_set = set(selected_ids)
        prefetched = [chat for chat in filtered if chat.get("id") in selected_set]
    options = SyncOptions(
        folder_name=args.folder_name,
        folder_id=folder_id,
        output_dir=resolve_output_path(args.out, DEFAULT_SYNC_OUT),
        collapse_threshold=collapse_thresholds["message"],
        collapse_thresholds=collapse_thresholds,
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
        prefetched_chats=prefetched,
        attachment_ocr=getattr(args, "attachment_ocr", False),
    )

    try:
        result = sync_command(options, env)
    except Exception as exc:
        console.print(f"[red]Drive sync failed: {exc}")
        console.print("[cyan]Run `polylogue doctor` and `polylogue config show --json` to verify credentials, tokens, and output directories.")
        raise

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
        if previous_run_note:
            payload["previousRun"] = previous_run_note
        print(json.dumps(payload, indent=2))
        return

    lines = [f"Synced {result.count} chat(s) → {result.output_dir}"]
    if getattr(args, "print_paths", False):
        lines.append("Written paths:")
        for item in result.items:
            lines.append(f"  {item.output}")
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
    if previous_run_note:
        lines.append(f"Previous run: {previous_run_note}")
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
        plain=ui.plain,
    )
    if selection is None:
        console.print("[yellow]Sync cancelled; no sessions selected.")
        return None
    if not selection:
        console.print("[yellow]No sessions selected; nothing to do.")
        return []
    return [Path(line.split("\t")[-1]) for line in selection]


def _run_local_sync(provider_name: str, args: argparse.Namespace, env: CommandEnv) -> None:
    provider = get_local_provider(provider_name)
    ui = env.ui
    previous_run_note = format_run_brief(latest_run(provider=provider.name, cmd=f"sync {provider.name}"))
    footer_lines = [f"Previous run: {previous_run_note}"] if previous_run_note else None
    if getattr(args, "diff", False) and not provider.supports_diff:
        ui.console.print(f"[red]{provider.title} does not support --diff output")
        raise SystemExit(1)
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else provider.default_base.expanduser()
    out_dir = resolve_output_path(args.out, provider.default_output)
    settings = env.settings
    collapse_thresholds = resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    force = args.force
    prune = args.prune
    diff_enabled = getattr(args, "diff", False)

    if provider.create_base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)

    selected_paths: Optional[List[Path]] = None
    cli_sessions = getattr(args, "sessions", None)
    if cli_sessions:
        selected_paths = [Path(path).expanduser() for path in cli_sessions if path]
    elif not args.all and not ui.plain:
        sessions = provider.list_sessions(base_dir)
        selection = _collect_session_selection(ui, sessions, f"Select {provider.title} sessions")
        if selection is None:
            return
        if not selection:
            return
        selected_paths = selection

    try:
        result = provider.sync_fn(
            base_dir=base_dir,
            output_dir=out_dir,
            collapse_threshold=collapse,
            collapse_thresholds=collapse_thresholds,
            html=html_enabled,
            html_theme=html_theme,
            force=force,
            prune=prune,
            diff=diff_enabled,
            sessions=selected_paths,
            registrar=env.registrar,
            ui=env.ui,
            attachment_ocr=getattr(args, "attachment_ocr", False),
        )
    except Exception as exc:
        ui.console.print(f"[red]{provider.title} sync failed: {exc}")
        raise SystemExit(1) from exc

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
            "cmd": f"sync {provider.name}",
            "provider": provider.name,
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
        if previous_run_note:
            payload["previousRun"] = previous_run_note
        print(json.dumps(payload, indent=2))
    else:
        summarize_import(ui, f"{provider.title} Sync", result.written, extra_lines=footer_lines)
        console = ui.console
        if result.skipped:
            console.print(f"Skipped {result.skipped} up-to-date item(s).")
        if result.pruned:
            console.print(f"Pruned {result.pruned} stale path(s).")

    add_run(
        {
            "cmd": f"sync {provider.name}",
            "provider": provider.name,
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
    "_run_local_sync",
    "_collect_session_selection",
    "_log_local_sync",
]
