#!/usr/bin/env python3
"""Polylogue: interactive-first CLI for AI chat log archives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import tempfile
import textwrap
import shlex
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..cli_common import choose_single_entry, filter_chats, sk_select
from ..commands import (
    CommandEnv,
    branches_command,
    search_command,
    status_command,
)
from ..automation import REPO_ROOT, TARGETS, cron_snippet, describe_targets, systemd_snippet
from ..drive_client import DEFAULT_FOLDER_NAME, DriveClient
from ..importers.claude_code import DEFAULT_PROJECT_ROOT
from ..options import BranchExploreOptions, SearchHit, SearchOptions, SyncOptions
from ..ui import create_ui
from .registry import CommandRegistry
from .arg_helpers import (
    add_branch_mode_option,
    add_collapse_option,
    add_diff_option,
    add_dry_run_option,
    add_force_option,
    add_html_option,
    add_out_option,
)
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CLAUDE_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    DEFAULT_OUTPUT_ROOTS,
    DEFAULT_RENDER_OUT,
    DEFAULT_SYNC_OUT,
    DEFAULT_CHATGPT_OUT,
    default_import_namespace,
    default_sync_namespace,
    resolve_html_settings,
)
from ..settings import ensure_settings_defaults, persist_settings, clear_persisted_settings
from ..config import CONFIG
from ..local_sync import LOCAL_SYNC_PROVIDER_NAMES
from .imports import (
    run_import_cli,
    run_import_chatgpt,
    run_import_claude,
    run_import_claude_code,
    run_import_codex,
)
from .render import copy_import_to_clipboard, run_render_cli
from .watch import run_watch_cli
from .status import run_status_cli, run_stats_cli
from .doctor import run_doctor_cli
from .automation_cli import run_automation_cli
from .migrate import run_migrate_cli
from .summaries import summarize_import
from .sync import (
    run_list_cli,
    run_sync_cli,
    _collect_session_selection,
    _log_local_sync,
    _run_sync_drive,
)
from ..util import CODEX_SESSIONS_ROOT, add_run, parse_input_time_to_epoch, write_clipboard_text
from ..branch_explorer import branch_diff, build_branch_html, format_branch_tree

SCRIPT_MODULE = "polylogue.cli"
COMMAND_REGISTRY = CommandRegistry()
_MENU_TIPS_SHOWN = False


def _legacy_candidates(root: Path) -> List[Path]:
    legacy: List[Path] = []
    for pattern in ("*.md", "*.html"):
        legacy.extend(path for path in root.glob(pattern) if path.is_file())
    for path in root.glob("*_attachments"):
        if path.exists():
            legacy.append(path)
    return sorted(legacy)


def run_prune_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    raw_dirs = args.dirs or []
    if raw_dirs:
        roots = [Path(path).expanduser() for path in raw_dirs]
    else:
        roots = list(DEFAULT_OUTPUT_ROOTS)
    seen: set[Path] = set()
    unique_roots: List[Path] = []
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)

    dry_run = bool(getattr(args, "dry_run", False))
    total_candidates = 0
    total_removed = 0

    for root in unique_roots:
        if not root.exists():
            continue
        legacy = _legacy_candidates(root)
        if not legacy:
            continue
        total_candidates += len(legacy)
        if dry_run:
            ui.console.print(f"[yellow][dry-run] Would prune {len(legacy)} path(s) in {root}")
            for path in legacy:
                ui.console.print(f"  rm {'-r ' if path.is_dir() else ''}{path}")
            continue
        removed_here = 0
        for path in legacy:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_here += 1
            except Exception as exc:
                ui.console.print(f"[red]Failed to remove {path}: {exc}")
        if removed_here:
            ui.console.print(f"[green]Pruned {removed_here} legacy path(s) in {root}")
            total_removed += removed_here

    summary_lines = [
        f"Roots scanned: {len(unique_roots)}",
        f"Legacy paths discovered: {total_candidates}",
    ]
    if dry_run:
        summary_lines.append("Dry run: no paths removed.")
    else:
        summary_lines.append(f"Paths removed: {total_removed}")
    ui.summary("Prune Legacy Outputs", summary_lines)


def run_inspect_branches(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    options = BranchExploreOptions(
        provider=getattr(args, "provider", None),
        slug=getattr(args, "slug", None),
        conversation_id=getattr(args, "conversation_id", None),
        min_branches=max(0, getattr(args, "min_branches", 1)),
    )
    result = branches_command(options)
    conversations = result.conversations
    if not conversations:
        target = []
        if options.slug:
            target.append(f"slug={options.slug}")
        if options.conversation_id:
            target.append(f"id={options.conversation_id}")
        if options.provider:
            target.append(f"provider={options.provider}")
        detail = f" ({', '.join(target)})" if target else ""
        ui.console.print(f"[yellow]No branchable conversations found{detail}.")
        return

    selected_conversations = conversations
    if not getattr(args, "no_picker", False) and len(conversations) > 1:
        def _format_conv(entry, idx):
            branch_total = len(entry.nodes)
            title = entry.title or entry.slug
            return f"{entry.provider}:{entry.conversation_id}\t{entry.slug}\tbranches={branch_total}\t{title}"

        chosen, cancelled = choose_single_entry(
            ui,
            conversations,
            format_line=_format_conv,
            header="idx\tprovider:id\tslug\tbranches\ttitle",
            prompt="branch>",
        )
        if cancelled:
            ui.console.print("[yellow]Branch explorer cancelled.")
            return
        if chosen is None:
            ui.console.print("[yellow]No conversation selected.")
            return
        selected_conversations = [chosen]

    multi_html = len(selected_conversations) > 1
    for conv in selected_conversations:
        title = conv.title or conv.slug
        header_lines = [
            f"Provider: {conv.provider}",
            f"Slug: {conv.slug}",
            f"Conversation ID: {conv.conversation_id}",
            f"Branches: {len(conv.nodes)} (canonical: {conv.canonical_branch_id or 'unknown'})",
        ]
        if conv.last_updated:
            header_lines.append(f"Last updated: {conv.last_updated}")
        if conv.conversation_path:
            header_lines.append(f"Canonical file: {conv.conversation_path}")
        ui.summary(title, header_lines)

        tree = format_branch_tree(conv, use_color=not ui.plain)
        if not tree.strip():
            ui.console.print("[yellow]No branch data recorded.")
        else:
            if not ui.plain:
                result = subprocess.run(
                    ["gum", "format"],
                    input=f"```\n{tree}\n```",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                output = result.stdout.strip()
                if output:
                    ui.console.print(output)
                else:
                    ui.console.print(tree)
            else:
                ui.console.print(tree)

        html_path = None
        html_enabled, html_explicit = resolve_html_settings(args, settings)
        html_out = getattr(args, "html_out", None)
        should_auto_html = html_enabled and not html_explicit and html_out is None and conv.branch_count > 1
        force_html = html_out is not None or (html_explicit and html_enabled)
        if force_html or should_auto_html:
            html_path = _generate_branch_html(
                conv,
                target=_resolve_html_output_path(conv, html_out, multi_html),
                theme=args.theme or settings.html_theme,
                ui=ui,
                auto=should_auto_html and not force_html,
            )

        branch_id = getattr(args, "branch", None)
        diff_requested = bool(getattr(args, "diff", False) or branch_id)
        if diff_requested:
            if not branch_id and not ui.plain:
                branch_id = _prompt_branch_choice(ui, conv)
            if not branch_id:
                non_canonical = [node.branch_id for node in conv.nodes.values() if not node.is_canonical]
                if getattr(args, "diff", False) and non_canonical:
                    branch_id = non_canonical[0]
        if branch_id:
            _display_branch_diff_for_id(conv, branch_id, ui)

        html_path = _prompt_branch_followups(ui, conv, args, html_path, settings)


def _generate_branch_html(conversation, target: Optional[Path], theme: str, ui, *, auto: bool) -> Optional[Path]:
    if target is None:
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        html_text = build_branch_html(conversation, theme=theme)
        target.write_text(html_text, encoding="utf-8")
        if auto:
            ui.console.print(f"[green]Auto-generated branch explorer → {target}")
        else:
            ui.console.print(f"[green]Wrote branch explorer to {target}")
        return target
    except Exception as exc:
        action = "auto-generate" if auto else "write"
        ui.console.print(f"[red]Failed to {action} HTML explorer: {exc}")
        return None


def _resolve_html_output_path(conversation, requested: Optional[Path], multi: bool) -> Optional[Path]:
    target = requested
    if target is None:
        if conversation.conversation_dir:
            target = conversation.conversation_dir / "branches.html"
        else:
            target = Path(f"{conversation.slug}-branches.html")
    if multi:
        suffix = target.suffix or ".html"
        stem = target.stem or "branches"
        target = target.parent / f"{stem}-{conversation.slug}{suffix}"
    if not target.suffix:
        target = target.with_suffix(".html")
    return target


def _prompt_branch_choice(ui, conversation) -> Optional[str]:
    candidates = [node for node in conversation.nodes.values() if not node.is_canonical]
    if not candidates:
        return None

    def _format(node, idx):
        delta = node.divergence_index + 1 if node.divergence_index else 0
        preview = node.divergence_snippet or ""
        role = node.divergence_role or ""
        return f"{node.branch_id}\tdelta#{delta}\t{role}: {preview}"

    selection, cancelled = choose_single_entry(
        ui,
        candidates,
        format_line=_format,
        header="idx\tbranch\tdelta\tpreview",
        prompt="branch>",
    )
    if cancelled:
        return None
    if selection is None:
        return candidates[0].branch_id
    return selection.branch_id


def _display_branch_diff_for_id(conversation, branch_id: str, ui) -> None:
    diff_text = branch_diff(conversation, branch_id)
    if diff_text is None:
        ui.console.print(f"[yellow]Unable to diff branch {branch_id}; ensure it exists and is not canonical.")
        return
    if not diff_text.strip():
        ui.console.print(f"[cyan]Branch {branch_id} matches the canonical transcript.")
        return
    _display_diff(diff_text, ui)


def _prompt_branch_followups(ui, conversation, args, html_path: Optional[Path], settings) -> Optional[Path]:
    if getattr(ui, "plain", False):
        return html_path

    current_html = html_path
    while True:
        options: List[str] = []
        if conversation.branch_count > 1:
            options.append("Diff a branch")
        if not current_html:
            options.append("Write HTML explorer")
        if current_html is not None:
            options.append("Show HTML path")
        options.append("Done")

        choice = ui.choose("Next action?", options)
        if not choice or choice == "Done":
            break
        if choice.startswith("Diff"):
            branch_choice = _prompt_branch_choice(ui, conversation)
            if branch_choice:
                _display_branch_diff_for_id(conversation, branch_choice, ui)
        elif choice.startswith("Write"):
            target = _resolve_html_output_path(conversation, getattr(args, "html_out", None), False)
            current_html = _generate_branch_html(
                conversation,
                target=target,
                theme=args.theme or settings.html_theme,
                ui=ui,
                auto=False,
            ) or current_html
        elif choice.startswith("Show") and current_html is not None:
            ui.console.print(f"[cyan]Branch explorer → {current_html}")
    return current_html


def _display_diff(diff_text: str, ui) -> None:
    if not diff_text.strip():
        ui.console.print("[cyan]No diff to display.")
        return
    if not ui.plain:
        subprocess.run(["gum", "pager"], input=diff_text, text=True, check=True)
        return
    ui.console.print(diff_text)


def run_inspect_search(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    if args.with_attachments and args.without_attachments:
        ui.console.print("[red]Choose only one of --with-attachments or --without-attachments.")
        return
    has_attachments: Optional[bool]
    if args.with_attachments:
        has_attachments = True
    elif args.without_attachments:
        has_attachments = False
    else:
        has_attachments = None

    options = SearchOptions(
        query=args.query,
        limit=args.limit,
        provider=args.provider,
        slug=args.slug,
        conversation_id=args.conversation_id,
        branch_id=args.branch,
        model=args.model,
        since=args.since,
        until=args.until,
        has_attachments=has_attachments,
    )
    result = search_command(options, env)
    hits = result.hits

    if getattr(args, "json", False):
        payload = {
            "query": options.query,
            "count": len(hits),
            "hits": [
                {
                    "provider": hit.provider,
                    "conversationId": hit.conversation_id,
                    "slug": hit.slug,
                    "title": hit.title,
                    "branchId": hit.branch_id,
                    "messageId": hit.message_id,
                    "position": hit.position,
                    "timestamp": hit.timestamp,
                    "attachments": hit.attachment_count,
                    "score": hit.score,
                    "snippet": hit.snippet,
                    "conversationPath": str(hit.conversation_path) if hit.conversation_path else None,
                    "branchPath": str(hit.branch_path) if hit.branch_path else None,
                    "model": hit.model,
                }
                for hit in hits
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not hits:
        ui.console.print("[yellow]No results found.")
        return

    summary_lines = [f"Hits: {len(hits)} (limit {options.limit})"]
    provider_counts = Counter(hit.provider for hit in hits)
    if provider_counts:
        provider_overview = ", ".join(
            f"{provider}×{count}" for provider, count in provider_counts.most_common(3)
        )
        summary_lines.append(f"Providers: {provider_overview}")
    model_set = {hit.model for hit in hits if hit.model}
    if model_set:
        summary_lines.append("Models: " + ", ".join(sorted(model_set)))
    attachment_hits = sum(1 for hit in hits if hit.attachment_count)
    if attachment_hits:
        summary_lines.append(f"With attachments: {attachment_hits}")
    ui.summary("Search", summary_lines)

    selected_hits: List[SearchHit]
    if not ui.plain and not getattr(args, "no_picker", False) and len(hits) > 1:
        picked, cancelled = _run_search_picker(ui, hits)
        if cancelled:
            ui.console.print("[yellow]Search cancelled.")
            return
        selected_hits = [picked] if picked is not None else hits
    else:
        selected_hits = hits

    for hit in selected_hits:
        _render_search_hit(hit, ui)


def _run_search_picker(ui, hits: List[SearchHit]) -> Tuple[Optional[SearchHit], bool]:
    if not hits:
        return None, False
    data_payload = [
        {
            "title": hit.title or hit.slug,
            "provider": hit.provider,
            "slug": hit.slug,
            "branch": hit.branch_id,
            "score": hit.score,
            "snippet": hit.snippet,
            "body": hit.body,
            "timestamp": hit.timestamp,
            "attachments": hit.attachment_count,
            "model": hit.model,
        }
        for hit in hits
    ]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as handle:
            json.dump({"hits": data_payload}, handle)
            tmp_path = Path(handle.name)
        def _format(hit: SearchHit, idx: int) -> str:
            snippet = hit.snippet or hit.body
            snippet = snippet.replace("\n", " ")
            snippet = textwrap.shorten(snippet, width=72, placeholder="…")
            return f"{hit.provider}:{hit.slug} [{hit.branch_id}] score={hit.score:.3f} {snippet}"

        preview_cmd = _build_search_preview_command(tmp_path)
        selection, cancelled = choose_single_entry(
            ui,
            hits,
            format_line=_format,
            header="idx\tprovider:slug [branch]\tscore\tsnippet",
            prompt="search>",
            preview=preview_cmd,
        )
        if cancelled:
            return None, True
        if selection is None:
            return hits[0], False
        return selection, False
    finally:
        if tmp_path:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _build_search_preview_command(data_file: Path) -> str:
    python_cmd = f"{shlex.quote(sys.executable)} -m {SCRIPT_MODULE}"
    return (
        "bash -lc "
        f"\"{python_cmd} _search-preview --data-file {shlex.quote(str(data_file))} "
        "--index $(printf %s \\\"{}\\\" | awk '{print $1}')\""
    )


def _render_search_hit(hit: SearchHit, ui) -> None:
    header = f"{hit.provider}/{hit.slug} [{hit.branch_id}]"
    lines = [
        f"Score: {hit.score:.4f}",
        f"Message: {hit.message_id} (position {hit.position})",
    ]
    if hit.timestamp:
        lines.append(f"Timestamp: {hit.timestamp}")
    if hit.model:
        lines.append(f"Model: {hit.model}")
    lines.append(f"Attachments: {hit.attachment_count}")
    if hit.conversation_path:
        lines.append(f"Conversation path: {hit.conversation_path}")
    if hit.branch_path:
        lines.append(f"Branch path: {hit.branch_path}")
    if hit.snippet:
        lines.append(f"Snippet: {hit.snippet}")
    ui.summary(hit.title or header, lines)

    body = hit.body.strip()
    if not body:
        ui.console.print("[cyan](Message body empty)")
        return

    if not ui.plain:
        subprocess.run(["gum", "format"], input=body, text=True, check=True)
    else:
        ui.console.print(body)


def run_search_preview(args: argparse.Namespace) -> None:
    try:
        data = json.loads(args.data_file.read_text(encoding="utf-8"))
    except Exception:
        return
    hits = data.get("hits")
    if not isinstance(hits, list):
        return
    index = getattr(args, "index", -1)
    if not isinstance(index, int) or index < 0 or index >= len(hits):
        return
    payload = hits[index]
    title = payload.get("title") or payload.get("slug") or "Result"
    provider = payload.get("provider")
    branch = payload.get("branch")
    score = payload.get("score")
    snippet = payload.get("snippet") or ""
    body = payload.get("body") or ""
    timestamp = payload.get("timestamp") or ""
    parts = [
        f"{title}",
        "=" * len(title),
        "",
        f"Provider: {provider}   Branch: {branch}",
        f"Score: {score}",
    ]
    if timestamp:
        parts.append(f"Timestamp: {timestamp}")
    if snippet:
        parts.extend(["", f"Snippet: {snippet}"])
    if body:
        parts.extend(["", body])
    print("\n".join(str(part) for part in parts))


def _dispatch_render(args: argparse.Namespace, env: CommandEnv) -> None:
    run_render_cli(args, env, json_output=getattr(args, "json", False))


def _dispatch_sync(args: argparse.Namespace, env: CommandEnv) -> None:
    run_sync_cli(args, env)


def _dispatch_import(args: argparse.Namespace, env: CommandEnv) -> None:
    run_import_cli(args, env)


def _dispatch_inspect(args: argparse.Namespace, env: CommandEnv) -> None:
    inspect_cmd = getattr(args, "inspect_cmd", None)
    if inspect_cmd == "branches":
        run_inspect_branches(args, env)
    elif inspect_cmd == "search":
        run_inspect_search(args, env)
    elif inspect_cmd == "stats":
        run_stats_cli(args, env)
    else:
        raise SystemExit("inspect requires a sub-command (branches, search, stats)")


def _dispatch_watch(args: argparse.Namespace, env: CommandEnv) -> None:
    run_watch_cli(args, env)


def _dispatch_prune(args: argparse.Namespace, env: CommandEnv) -> None:
    run_prune_cli(args, env)


def _dispatch_doctor(args: argparse.Namespace, env: CommandEnv) -> None:
    run_doctor_cli(args, env)


def _dispatch_settings(args: argparse.Namespace, env: CommandEnv) -> None:
    from .settings_cli import run_settings_cli

    run_settings_cli(args, env)


def _dispatch_status(args: argparse.Namespace, env: CommandEnv) -> None:
    run_status_cli(args, env)


def _dispatch_automation(args: argparse.Namespace, env: CommandEnv) -> None:
    run_automation_cli(args, env)


def _dispatch_migrate(args: argparse.Namespace, env: CommandEnv) -> None:
    if getattr(args, "migrate_cmd", None) is None:
        raise SystemExit("migrate requires a sub-command (legacy)")
    run_migrate_cli(args, env)


def _dispatch_search_preview(args: argparse.Namespace, _env: CommandEnv) -> None:
    run_search_preview(args)


_REGISTRATION_COMPLETE = False


def _register_default_commands() -> None:
    global _REGISTRATION_COMPLETE
    if _REGISTRATION_COMPLETE:
        return

    def _ensure(name: str, handler: Callable[[argparse.Namespace, CommandEnv], None], help_text: str) -> None:
        if COMMAND_REGISTRY.resolve(name) is None:
            COMMAND_REGISTRY.register(name, handler, help_text=help_text)

    _ensure("render", _dispatch_render, "Render local provider JSON logs")
    _ensure("sync", _dispatch_sync, "Synchronize provider archives")
    _ensure("import", _dispatch_import, "Import provider exports into the archive")
    _ensure("inspect", _dispatch_inspect, "Inspect existing archives and stats")
    _ensure("watch", _dispatch_watch, "Watch local session stores and sync on changes")
    _ensure("prune", _dispatch_prune, "Remove legacy single-file outputs and attachments")
    _ensure("doctor", _dispatch_doctor, "Check local data directories for common issues")
    _ensure("status", _dispatch_status, "Show cached Drive info and recent runs")
    _ensure("settings", _dispatch_settings, "Show or update default preferences")
    _ensure("automation", _dispatch_automation, "Generate scheduler snippets")
    _ensure("migrate", _dispatch_migrate, "Migrate legacy cache files into SQLite")
    _ensure("_search-preview", _dispatch_search_preview, "Internal search preview helper")

    _REGISTRATION_COMPLETE = True


def build_parser() -> argparse.ArgumentParser:
    _register_default_commands()
    parser = argparse.ArgumentParser(description="Polylogue CLI")
    parser.add_argument("--plain", action="store_true", help="Disable interactive UI")
    sub = parser.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render", help="Render local provider JSON logs")
    p_render.add_argument("input", type=Path, help="File or directory with provider JSON logs (e.g., Gemini)")
    add_out_option(p_render, default_path=DEFAULT_RENDER_OUT)
    p_render.add_argument("--links-only", action="store_true", help="Link attachments instead of downloading")
    add_dry_run_option(p_render, help_text="Report actions without writing files")
    add_force_option(p_render, help_text="Overwrite conversations even if they appear up-to-date")
    add_collapse_option(p_render, help_text="Override collapse threshold")
    p_render.add_argument("--json", action="store_true")
    add_html_option(p_render)
    add_diff_option(p_render, help_text="Write delta diff when output already exists")
    p_render.add_argument("--to-clipboard", action="store_true", help="Copy rendered Markdown to the clipboard when a single file is produced")

    p_sync = sub.add_parser("sync", help="Synchronize provider archives")
    p_sync.add_argument(
        "provider",
        choices=["drive", *LOCAL_SYNC_PROVIDER_NAMES],
        help="Provider to synchronize",
    )
    add_out_option(
        p_sync,
        default_path=DEFAULT_SYNC_OUT,
        help_text="Override output directory (provider defaults from config are used otherwise)",
    )
    p_sync.add_argument("--links-only", action="store_true", help="Link attachments instead of downloading (Drive only)")
    add_dry_run_option(p_sync)
    add_force_option(p_sync, help_text="Re-render even if conversations are up-to-date")
    p_sync.add_argument("--prune", action="store_true", help="Remove outputs for conversations that vanished upstream")
    add_collapse_option(p_sync)
    add_html_option(p_sync)
    add_branch_mode_option(p_sync)
    add_diff_option(p_sync, help_text="Write delta diff alongside updated Markdown")
    p_sync.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_sync.add_argument("--base-dir", type=Path, default=None, help="Override local session directory (codex/claude-code)")
    p_sync.add_argument("--all", action="store_true", help="Process all local sessions without prompting")
    p_sync.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME, help="Drive folder name (drive provider)")
    p_sync.add_argument("--folder-id", type=str, default=None, help="Drive folder ID override")
    p_sync.add_argument("--since", type=str, default=None, help="Only include Drive chats updated on/after this timestamp")
    p_sync.add_argument("--until", type=str, default=None, help="Only include Drive chats updated on/before this timestamp")
    p_sync.add_argument("--name-filter", type=str, default=None, help="Regex filter for Drive chat names")
    p_sync.add_argument("--list-only", action="store_true", help="List Drive chats without syncing")

    p_import = sub.add_parser("import", help="Import provider exports into the archive")
    p_import.add_argument("provider", choices=["chatgpt", "claude", "claude-code", "codex"], help="Provider export format")
    p_import.add_argument("source", nargs="*", help="Export path or session identifier (depends on provider)")
    p_import.add_argument("--out", type=Path, default=None, help="Override output directory")
    add_collapse_option(p_import)
    add_html_option(p_import)
    add_branch_mode_option(p_import)
    add_force_option(p_import, help_text="Rewrite even if conversations appear up-to-date")
    p_import.add_argument("--all", action="store_true", help="Process every conversation in the export (ChatGPT/Claude)")
    p_import.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Specific conversation ID to import (repeatable)")
    p_import.add_argument("--base-dir", type=Path, default=None, help="Override source directory for codex/claude-code sessions")
    p_import.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_import.add_argument("--to-clipboard", action="store_true", help="Copy a single imported Markdown file to the clipboard")

    p_inspect = sub.add_parser("inspect", help="Inspect existing archives")
    inspect_sub = p_inspect.add_subparsers(dest="inspect_cmd", required=True)

    p_inspect_branches = inspect_sub.add_parser("branches", help="Explore branch graphs for conversations")
    p_inspect_branches.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p_inspect_branches.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p_inspect_branches.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p_inspect_branches.add_argument("--min-branches", type=int, default=1, help="Only include conversations with at least this many branches")
    p_inspect_branches.add_argument("--branch", type=str, default=None, help="Branch ID to inspect or diff against the canonical path")
    p_inspect_branches.add_argument("--diff", action="store_true", help="Display a unified diff between a branch and canonical transcript")
    p_inspect_branches.add_argument(
        "--html",
        dest="html_mode",
        nargs="?",
        const="on",
        default="auto",
        choices=["auto", "on", "off"],
        metavar="MODE",
        help="Branch HTML mode: on/off/auto (default auto)",
    )
    p_inspect_branches.add_argument("--html-out", type=Path, default=None, help="Write the branch explorer to this path")
    p_inspect_branches.add_argument("--theme", type=str, default=None, choices=["light", "dark"], help="Override HTML explorer theme")
    p_inspect_branches.add_argument("--no-picker", action="store_true", help="Skip interactive selection even when skim/gum are available")

    p_inspect_search = inspect_sub.add_parser("search", help="Search rendered transcripts")
    p_inspect_search.add_argument("query", type=str, help="FTS search query (SQLite syntax)")
    p_inspect_search.add_argument("--limit", type=int, default=20, help="Maximum number of hits to return")
    p_inspect_search.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p_inspect_search.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p_inspect_search.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p_inspect_search.add_argument("--branch", type=str, default=None, help="Restrict to a single branch ID")
    p_inspect_search.add_argument("--model", type=str, default=None, help="Filter by source model when recorded")
    p_inspect_search.add_argument("--since", type=str, default=None, help="Only include messages on/after this timestamp")
    p_inspect_search.add_argument("--until", type=str, default=None, help="Only include messages on/before this timestamp")
    p_inspect_search.add_argument("--with-attachments", action="store_true", help="Limit to messages with extracted attachments")
    p_inspect_search.add_argument("--without-attachments", action="store_true", help="Limit to messages without attachments")
    p_inspect_search.add_argument("--no-picker", action="store_true", help="Skip skim picker preview even when interactive")
    p_inspect_search.add_argument("--json", action="store_true", help="Emit machine-readable search results")

    p_inspect_stats = inspect_sub.add_parser("stats", help="Summarize Markdown output directories")
    p_inspect_stats.add_argument("--dir", type=Path, default=None, help="Directory containing Markdown exports")
    p_inspect_stats.add_argument("--json", action="store_true", help="Emit machine-readable stats")
    p_inspect_stats.add_argument("--since", type=str, default=None, help="Only include files modified on/after this date (YYYY-MM-DD or ISO)")
    p_inspect_stats.add_argument("--until", type=str, default=None, help="Only include files modified on/before this date")

    p_watch = sub.add_parser("watch", help="Watch local session stores and sync on changes")
    p_watch.add_argument("provider", choices=list(LOCAL_SYNC_PROVIDER_NAMES), help="Local provider to watch")
    p_watch.add_argument("--base-dir", type=Path, default=None, help="Override source directory")
    add_out_option(p_watch, default_path=DEFAULT_CODEX_SYNC_OUT, help_text="Override output directory")
    add_collapse_option(p_watch)
    add_html_option(p_watch, description="HTML preview mode while watching: on/off/auto (default auto)")
    add_branch_mode_option(p_watch)
    p_watch.add_argument("--debounce", type=float, default=2.0, help="Minimal seconds between sync runs")
    p_watch.add_argument("--once", action="store_true", help="Run a single sync pass and exit")

    p_prune = sub.add_parser("prune", help="Remove legacy single-file outputs and attachments")
    p_prune.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        type=Path,
        help="Root directory to prune (repeatable). Defaults to all configured output directories.",
    )
    p_prune.add_argument("--dry-run", action="store_true", help="Print planned actions without deleting files")

    p_doctor = sub.add_parser("doctor", help="Check local data directories for common issues")
    p_doctor.add_argument("--codex-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_doctor.add_argument("--claude-code-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_doctor.add_argument("--limit", type=int, default=None, help="Limit number of files inspected per provider")
    p_doctor.add_argument("--json", action="store_true", help="Emit machine-readable report")

    p_status = sub.add_parser("status", help="Show cached Drive info and recent runs")
    p_status.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_status.add_argument("--watch", action="store_true", help="Continuously refresh the status output")
    p_status.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh while watching")
    p_status.add_argument("--dump", type=str, default=None, help="Write recent runs to a file ('-' for stdout)")
    p_status.add_argument("--dump-limit", type=int, default=100, help="Number of runs to include when dumping")
    p_status.add_argument("--runs-limit", type=int, default=200, help="Number of historical runs to include in summaries")
    p_status.add_argument("--dump-only", action="store_true", help="Only perform the dump action without printing summaries")

    p_settings_cmd = sub.add_parser("settings", help="Show or update Polylogue defaults")
    p_settings_cmd.add_argument("--html", choices=["on", "off"], default=None, help="Enable or disable default HTML previews")
    p_settings_cmd.add_argument("--theme", choices=["light", "dark"], default=None, help="Set the default HTML theme")
    p_settings_cmd.add_argument("--reset", action="store_true", help="Reset to config defaults")
    p_settings_cmd.add_argument("--json", action="store_true", help="Emit settings as JSON")

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
    p_auto_systemd.add_argument("--status-log", type=Path, default=None, help="Dump polylogue status output to this path after each run")
    p_auto_systemd.add_argument("--status-limit", type=int, default=50, help="Number of runs to include when dumping status logs")
    add_html_option(
        p_auto_systemd,
        description="HTML mode for generated sync commands: on/off/auto (default auto, inherits target defaults)",
    )

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
    p_auto_cron.add_argument("--status-log", type=Path, default=None, help="Dump polylogue status output to this path after each run")
    p_auto_cron.add_argument("--status-limit", type=int, default=50, help="Number of runs to include when dumping status logs")
    add_html_option(
        p_auto_cron,
        description="HTML mode for generated sync commands: on/off/auto (default auto, inherits target defaults)",
    )

    p_migrate = sub.add_parser("migrate", help="Migrate legacy cache files into SQLite")
    migrate_sub = p_migrate.add_subparsers(dest="migrate_cmd", required=True)
    p_migrate_legacy = migrate_sub.add_parser("legacy", help="Import legacy state.json/runs.json")
    p_migrate_legacy.add_argument("--state-path", type=str, default=None, help="Path to legacy state.json")
    p_migrate_legacy.add_argument("--runs-path", type=str, default=None, help="Path to legacy runs.json")
    p_migrate_legacy.add_argument("--dry-run", action="store_true", help="Report actions without modifying the database")
    p_migrate_legacy.add_argument("--force", action="store_true", help="Replace existing run history when importing")

    p_search_preview = sub.add_parser("_search-preview", help=argparse.SUPPRESS)
    p_search_preview.add_argument("--data-file", type=Path, required=True)
    p_search_preview.add_argument("--index", type=int, required=True)

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


def _menu_structure() -> List[tuple[str, List[tuple[str, Callable[[CommandEnv], None], Optional[str]]]]]:
    return [
        (
            "Render & Import",
            [
                ("Render Local Logs", prompt_render, "render"),
                ("Import Provider Export", prompt_import, "import"),
            ],
        ),
        (
            "Sync & Inspect",
            [
                ("Sync Provider Archives", prompt_sync, "sync"),
                ("Inspect Branches", prompt_inspect_branches, "inspect"),
                ("Inspect Search", prompt_inspect_search, "inspect"),
                ("Inspect Stats", prompt_inspect_stats, "inspect"),
            ],
        ),
        (
            "Maintenance",
            [
                ("Prune Legacy Outputs", prompt_prune, "prune"),
                ("Doctor", prompt_doctor, "doctor"),
                ("Automation Snippets", prompt_automation, "automation"),
            ],
        ),
    ]


def _build_menu_options(env: CommandEnv) -> OrderedDict[str, Optional[Callable[[CommandEnv], None]]]:
    options: OrderedDict[str, Optional[Callable[[CommandEnv], None]]] = OrderedDict()
    for group, entries in _menu_structure():
        for label, handler, command_name in entries:
            info = COMMAND_REGISTRY.info(command_name) if command_name else None
            help_text = info.help_text if info and info.help_text else ""
            display = f"{group} ▸ {label}"
            if help_text:
                display = f"{display} — {help_text}"
            options[display] = handler
    options["Status ▸ View environment summary — Show cached Drive info and recent runs"] = _show_status_dashboard
    options["Settings ▸ Adjust defaults"] = settings_menu
    options["Help ▸ Command overview"] = show_help
    options["Quit"] = None
    return options


def _show_status_dashboard(env: CommandEnv) -> None:
    args = argparse.Namespace(json=False, watch=False, interval=5.0)
    run_status_cli(args, env)


def _display_status_snapshot(env: CommandEnv) -> None:
    status = status_command(env)
    env_lines = [
        f"Credentials: {'present' if status.credentials_present else 'missing'}",
        f"Token: {'present' if status.token_present else 'missing'}",
        f"Runs recorded: {len(status.recent_runs)}",
    ]
    if status.recent_runs:
        last_run = status.recent_runs[-1]
        env_lines.append(
            f"Last run → {last_run.get('cmd')} @ {last_run.get('timestamp') or '-'}"
        )
    env.ui.summary("Environment Snapshot", env_lines)

    provider_lines: List[str] = []
    for provider, stats in list(status.provider_summary.items())[:3]:
        provider_lines.append(
            f"{provider}: runs={stats['count']} diffs={stats['diffs']} retries={stats.get('retries', 0)}"
        )
    if not provider_lines:
        provider_lines = ["No provider runs recorded yet."]
    env.ui.summary("Providers", provider_lines)

    recent_lines = [
        f"{item.get('cmd')} · count={item.get('count', 0)} @ {item.get('timestamp') or '-'}"
        for item in status.recent_runs[-3:]
    ]
    if recent_lines:
        env.ui.summary("Recent Runs", recent_lines)

    quick_lines = [
        "Render Local Logs → Render & Import ▸ Render Local Logs",
        "Sync Drive → Sync & Inspect ▸ Sync Provider Archives",
        "Import ChatGPT → Render & Import ▸ Import Provider Export",
    ]
    env.ui.summary("Quick Actions", quick_lines)

    pref_lines = [
        f"HTML previews: {'on' if env.settings.html_previews else 'off'}",
        f"HTML theme: {env.settings.html_theme}",
    ]
    env.ui.summary("Preferences", pref_lines)


def interactive_menu(env: CommandEnv) -> None:
    ui = env.ui
    global _MENU_TIPS_SHOWN
    if not ui.plain and not _MENU_TIPS_SHOWN:
        ui.console.print("[cyan]Use the arrow keys to navigate, press Enter to select, and 'q' to back out of pickers.")
        _MENU_TIPS_SHOWN = True
    while True:
        ui.banner("Polylogue", "Render AI chat logs or sync providers")
        _display_status_snapshot(env)
        options = _build_menu_options(env)
        choice = ui.choose("Select an action", list(options.keys()))
        if choice is None:
            return
        handler = options.get(choice)
        if handler is None:
            return
        handler(env)


def prompt_render(env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
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
    args.html_mode = "on" if settings.html_previews else "off"
    run_render_cli(args, env, json_output=False)


def prompt_sync(env: CommandEnv) -> None:
    ui = env.ui
    provider = ui.choose("Sync which provider?", ["drive", "codex", "claude-code"])
    if not provider:
        return

    args = default_sync_namespace(provider, env.settings)
    run_sync_cli(args, env)


def prompt_import(env: CommandEnv) -> None:
    ui = env.ui
    provider = ui.choose("Import which provider?", ["chatgpt", "claude", "codex", "claude-code"])
    if not provider:
        return

    sources: List[str] = []
    base_dir: Optional[Path] = None
    all_flag = False
    conversation_ids: List[str] = []

    if provider in {"chatgpt", "claude"}:
        path_input = ui.input("Export path", default=str(Path.cwd()))
        if not path_input:
            return
        sources.append(path_input)
        all_flag = ui.confirm("Import all conversations?", default=False) if not ui.plain else False
    else:
        session_hint = ui.input("Session identifier (leave blank to pick interactively)", default="")
        if session_hint:
            sources.append(session_hint)
        if provider == "codex":
            base_dir_input = ui.input("Codex sessions directory", default=str(CODEX_SESSIONS_ROOT))
            base_dir = Path(base_dir_input) if base_dir_input else None
        else:
            base_dir_input = ui.input("Claude Code projects directory", default=str(DEFAULT_PROJECT_ROOT))
            base_dir = Path(base_dir_input) if base_dir_input else None

    args = default_import_namespace(
        provider=provider,
        sources=sources,
        base_dir=base_dir,
        all_flag=all_flag,
        conversation_ids=conversation_ids,
        settings=env.settings,
    )
    run_import_cli(args, env)


def prompt_inspect_branches(env: CommandEnv) -> None:
    ui = env.ui
    provider = ui.input("Filter provider (optional)", default="")
    slug = ui.input("Filter slug (optional)", default="")
    conversation_id = ui.input("Filter conversation ID (optional)", default="")
    args = argparse.Namespace(
        provider=provider or None,
        slug=slug or None,
        conversation_id=conversation_id or None,
        min_branches=1,
        branch=None,
        diff=False,
        html_mode="auto",
        html_out=None,
        theme=None,
        no_picker=False,
    )
    run_inspect_branches(args, env)


def prompt_inspect_search(env: CommandEnv) -> None:
    ui = env.ui
    query = ui.input("Search query", default="error OR failure")
    if not query:
        return
    args = argparse.Namespace(
        query=query,
        limit=20,
        provider=None,
        slug=None,
        conversation_id=None,
        branch=None,
        model=None,
        since=None,
        until=None,
        with_attachments=False,
        without_attachments=False,
        no_picker=False,
        json=False,
    )
    run_inspect_search(args, env)


def prompt_inspect_stats(env: CommandEnv) -> None:
    ui = env.ui
    default_dir = str(CONFIG.defaults.output_dirs.render.parent)
    directory = ui.input("Directory to summarize", default=default_dir)
    if not directory:
        return
    args = argparse.Namespace(
        dir=Path(directory),
        json=False,
        since=None,
        until=None,
    )
    run_stats_cli(args, env)






def prompt_prune(env: CommandEnv) -> None:
    args = argparse.Namespace(dirs=None, dry_run=False)
    run_prune_cli(args, env)


def prompt_doctor(env: CommandEnv) -> None:
    class Args:
        pass

    args = Args()
    args.codex_dir = None
    args.claude_code_dir = None
    args.limit = 25
    args.json = False
    run_doctor_cli(args, env)


def prompt_automation(env: CommandEnv) -> None:
    ui = env.ui
    format_label = ui.choose(
        "Generate which automation snippet?",
        ["Systemd unit", "Cron entry", "Describe targets"],
    )
    if not format_label:
        return

    format_map = {
        "Systemd unit": "systemd",
        "Cron entry": "cron",
        "Describe targets": "describe",
    }
    fmt = format_map[format_label]
    target_keys = sorted(TARGETS.keys())

    if fmt == "describe":
        selection = ui.choose("Select automation target", ["All targets"] + target_keys)
        if selection is None:
            return
        target = None if selection == "All targets" else selection
        args = argparse.Namespace(automation_format="describe", target=target)
        run_automation_cli(args, env)
        return

    target = ui.choose("Select automation target", target_keys)
    if not target:
        return

    defaults = TARGETS[target].defaults or {}
    working_dir_default = defaults.get("workingDir") or str(REPO_ROOT)
    extra_args_default = " ".join(defaults.get("extraArgs", []))
    collapse_default = defaults.get("collapseThreshold")
    html_default = defaults.get("html")

    if fmt == "systemd":
        interval = ui.input("Run interval (OnUnitActiveSec)", default=defaults.get("systemdInterval", "10m")) or "10m"
        boot_delay = ui.input("Boot delay (OnBootSec)", default="2m") or "2m"
        working_dir_input = ui.input("Working directory", default=working_dir_default) or working_dir_default
        extra_args_input = ui.input("Additional CLI arguments (space separated)", default=extra_args_default)
        collapse_input = ui.input(
            "Collapse threshold override (optional)",
            default=str(collapse_default) if collapse_default is not None else "",
        )
        html_enabled = ui.confirm(
            "Enable HTML output?",
            default=bool(html_default) if html_default is not None else False,
        )
        out_input = ui.input("Override --out path (optional)", default=str(defaults.get("outputDir") or ""))
        args = argparse.Namespace(
            automation_format="systemd",
            target=target,
            interval=interval,
            boot_delay=boot_delay,
            working_dir=Path(working_dir_input).expanduser(),
            out=Path(out_input).expanduser() if out_input else None,
            extra_arg=shlex.split(extra_args_input) if extra_args_input else [],
            collapse_threshold=int(collapse_input) if collapse_input else None,
            html=html_enabled,
        )
        run_automation_cli(args, env)
        return

    schedule = ui.input("Cron schedule", default="*/30 * * * *") or "*/30 * * * *"
    log_path = ui.input("Log path", default="$HOME/.cache/polylogue-sync.log") or "$HOME/.cache/polylogue-sync.log"
    state_home = ui.input("XDG_STATE_HOME value", default="$HOME/.local/state") or "$HOME/.local/state"
    working_dir_input = ui.input("Working directory", default=working_dir_default) or working_dir_default
    extra_args_input = ui.input("Additional CLI arguments (space separated)", default=extra_args_default)
    collapse_input = ui.input(
        "Collapse threshold override (optional)",
        default=str(collapse_default) if collapse_default is not None else "",
    )
    html_enabled = ui.confirm(
        "Enable HTML output?",
        default=bool(html_default) if html_default is not None else False,
    )
    out_input = ui.input("Override --out path (optional)", default=str(defaults.get("outputDir") or ""))
    args = argparse.Namespace(
        automation_format="cron",
        target=target,
        schedule=schedule,
        log=log_path,
        state_home=state_home,
        working_dir=Path(working_dir_input).expanduser(),
        out=Path(out_input).expanduser() if out_input else None,
        extra_arg=shlex.split(extra_args_input) if extra_args_input else [],
        collapse_threshold=int(collapse_input) if collapse_input else None,
        html=html_enabled,
    )
    run_automation_cli(args, env)


def show_help(env: CommandEnv) -> None:
    ui = env.ui
    ui.console.print("Polylogue commands:")
    ui.console.print("  render            Render local provider JSON files")
    ui.console.print("  sync <provider>   Sync Drive/Codex/Claude Code archives")
    ui.console.print("  import <provider> Import provider exports or sessions")
    ui.console.print("  inspect branches  Explore branch graphs")
    ui.console.print("  inspect search    Query transcripts via SQLite FTS")
    ui.console.print("  inspect stats     Summarize Markdown output directories")
    ui.console.print("  watch <provider>  Watch local session stores and sync on change")
    ui.console.print("  prune             Remove legacy single-file outputs")
    ui.console.print("  doctor            Check local data directories for issues")
    ui.console.print("  status            Show cached Drive info and recent runs")
    ui.console.print("  migrate legacy    Import legacy state.json/runs.json into SQLite")
    ui.console.print("  settings          Update default HTML/theme preferences")
    ui.console.print("  automation        Generate automation snippets")
    ui.console.print("  --plain           Disable interactive UI for automation")
    ui.console.print("  --json            Emit machine-readable summaries when supported")


def settings_menu(env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    while True:
        toggle_label = f"Toggle HTML previews ({'on' if settings.html_previews else 'off'})"
        theme_label = f"HTML theme ({settings.html_theme})"
        choices = [toggle_label, theme_label, "Reset defaults", "Back"]
        choice = ui.choose("Settings", choices)
        if choice is None or choice == "Back":
            return
        if choice == toggle_label:
            settings.html_previews = not settings.html_previews
            state = "enabled" if settings.html_previews else "disabled"
            ui.console.print(f"HTML previews {state}.")
            persist_settings(settings)
        elif choice == theme_label:
            if ui.plain:
                ui.console.print("Switch to interactive mode or adjust defaults in your config to change the HTML theme.")
                continue
            selection = ui.choose("Select HTML theme", ["light", "dark"])
            if selection:
                settings.html_theme = selection
                ui.console.print(f"HTML theme set to {selection}.")
                persist_settings(settings)
        elif choice == "Reset defaults":
            ensure_settings_defaults(settings)
            clear_persisted_settings()
            ui.console.print("Settings reset to defaults.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ui = create_ui(args.plain)
    env = CommandEnv(ui=ui)
    ensure_settings_defaults(env.settings)

    if args.cmd is None:
        if ui.plain:
            parser.print_help()
            return
        ui.banner("Polylogue", "Render AI chat logs or sync providers")
        interactive_menu(env)
        return

    _register_default_commands()
    cmd = args.cmd
    handler = COMMAND_REGISTRY.resolve(cmd)
    if handler is None:
        parser.print_help()
        return

    handler(args, env)


if __name__ == "__main__":
    main()
