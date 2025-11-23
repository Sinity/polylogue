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
from collections import Counter
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..cli_common import choose_single_entry, filter_chats, resolve_inputs, sk_select
from ..commands import (
    CommandEnv,
    branches_command,
    search_command,
    status_command,
)
from ..drive_client import DEFAULT_FOLDER_NAME, DriveClient
from ..importers.claude_code import DEFAULT_PROJECT_ROOT
from ..options import BranchExploreOptions, SearchHit, SearchOptions, SyncOptions
from ..ui import create_ui
from .completion_engine import CompletionEngine, Completion
from .registry import CommandRegistry
from .arg_helpers import (
    add_allow_dirty_option,
    add_collapse_option,
    add_diff_option,
    add_dry_run_option,
    add_force_option,
    add_html_option,
    add_out_option,
    create_output_parent,
    create_filter_parent,
)
from .editor import open_in_editor, get_editor
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
from ..config import CONFIG, CONFIG_PATH
from ..local_sync import LOCAL_SYNC_PROVIDER_NAMES, WATCHABLE_LOCAL_PROVIDER_NAMES
from .imports import (
    run_import_cli,
    run_import_chatgpt,
    run_import_claude,
    run_import_claude_code,
    run_import_codex,
)
from .dashboards import run_dashboards_cli
from .runs import run_runs_cli
from .index_cli import run_index_cli
from .watch import run_watch_cli
from .status import run_status_cli, run_stats_cli
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
_FORCE_PLAIN_VALUES = {"1", "true", "yes", "on"}


def _should_use_plain(force_interactive: bool) -> bool:
    if force_interactive:
        return False
    forced = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    if forced and forced.strip().lower() in _FORCE_PLAIN_VALUES:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())
PARSER_FORMATTER = argparse.ArgumentDefaultsHelpFormatter


def _add_command_parser(subparsers: argparse._SubParsersAction, name: str, **kwargs):
    kwargs.setdefault("formatter_class", PARSER_FORMATTER)
    return subparsers.add_parser(name, **kwargs)

# Command Examples for --examples flag
# Each command maps to list of (description, command_line) tuples
COMMAND_EXAMPLES = {
    "sync": [
        ("Sync all Drive chats", "polylogue sync drive --all"),
        ("Sync from specific Drive folder", "polylogue sync drive --folder-name 'Work Chats'"),
        ("Sync Codex sessions with preview", "polylogue sync codex --dry-run"),
        ("Sync Claude Code with diff tracking", "polylogue sync claude-code --diff"),
        ("Sync and prune deleted chats", "polylogue sync drive --all --prune"),
    ],
    "import": [
        ("Import ChatGPT export", "polylogue import chatgpt export.zip --html on"),
        ("Import with interactive picker", "polylogue import claude-code pick"),
        ("Import specific conversation", "polylogue import chatgpt export.zip --conversation-id abc123"),
        ("Import all from Claude export", "polylogue import claude conversations.zip --all"),
        ("Preview import without writing", "polylogue import codex session.jsonl --dry-run"),
    ],
    "search": [
        ("Search for term", "polylogue search 'error handling'"),
        ("Search with filters", "polylogue search 'API' --provider chatgpt --since 2024-01-01"),
        ("Search with limit", "polylogue search 'authentication' --limit 10"),
        ("Search and open in editor", "polylogue search 'TODO' --limit 1 --open"),
        ("Search with attachment filter", "polylogue search 'diagram' --with-attachments"),
    ],
    "browse": [
        ("Browse branch graph", "polylogue browse branches --provider claude"),
        ("View stats", "polylogue browse stats --provider chatgpt"),
        ("Get stats with time filter", "polylogue browse stats --since 2024-01-01 --until 2024-12-31"),
        ("Show status overview", "polylogue browse status"),
        ("Watch status continuously", "polylogue browse status --watch"),
        ("Show provider dashboards", "polylogue browse dashboards"),
        ("List recent runs", "polylogue browse runs --limit 20"),
    ],
    "watch": [
        ("Watch Codex sessions", "polylogue watch codex"),
        ("Watch Claude Code with HTML", "polylogue watch claude-code --html on"),
        ("One-shot watch run", "polylogue watch chatgpt --once"),
    ],
    "maintain": [
        ("Preview prune operation", "polylogue maintain prune --dry-run"),
        ("Prune legacy files", "polylogue maintain prune --dir ~/polylogue-data/chatgpt"),
        ("Check all providers", "polylogue maintain doctor"),
        ("Check with JSON output", "polylogue maintain doctor --json"),
        ("Validate indexes", "polylogue maintain index check"),
        ("Repair indexes", "polylogue maintain index check --repair"),
    ],
    "config": [
        ("Interactive setup wizard", "polylogue config init"),
        ("Force re-initialization", "polylogue config init --force"),
        ("Show current configuration", "polylogue config show"),
        ("Get configuration as JSON", "polylogue config show --json"),
        ("Enable HTML previews", "polylogue config set --html on"),
        ("Set dark theme", "polylogue config set --theme dark"),
        ("Reset to defaults", "polylogue config set --reset"),
    ],
}


def _collect_subparser_map(parser: argparse.ArgumentParser) -> Dict[str, argparse.ArgumentParser]:
    mapping: Dict[str, argparse.ArgumentParser] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            mapping.update(action.choices)
    return mapping


def _command_entries(parser: argparse.ArgumentParser) -> List[Tuple[str, str]]:
    choices = _collect_subparser_map(parser)
    entries: List[Tuple[str, str]] = []
    for name, subparser in choices.items():
        if name.startswith("_"):
            continue
        info = COMMAND_REGISTRY.info(name)
        description = subparser.description or (info.help_text if info else None) or ""
        description = " ".join(description.split())
        entries.append((name, description))
    return sorted(entries, key=lambda item: item[0])


def _print_command_listing(console, plain: bool, entries: List[Tuple[str, str]]) -> None:
    if not entries:
        return
    console.print("\nCommands:")
    if plain:
        width = max(len(name) for name, _ in entries) + 2
        for name, description in entries:
            if description:
                console.print(f"  {name.ljust(width)}{description}")
            else:
                console.print(f"  {name}")
        return
    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for name, description in entries:
        table.add_row(name, description)
    console.print(table)


def _legacy_candidates(root: Path) -> List[Path]:
    legacy: List[Path] = []
    for pattern in ("*.md", "*.html"):
        legacy.extend(path for path in root.glob(pattern) if path.is_file())
    for path in root.glob("*_attachments"):
        if path.exists():
            legacy.append(path)
    return sorted(legacy)


def run_help_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    parser = build_parser()
    topic = getattr(args, "topic", None)
    entries = _command_entries(parser)
    choices = {name: sub for name, sub in _collect_subparser_map(parser).items() if not name.startswith("_")}
    console = env.ui.console

    if topic:
        subparser = choices.get(topic)
        if subparser is None:
            console.print(f"[red]Unknown command: {topic}")
            available = ", ".join(sorted(choices)) or "<none>"
            console.print(f"Available commands: {available}")
            raise SystemExit(1)

        console.print(f"[cyan]polylogue {topic}[/cyan]")
        subparser.print_help()

        # Show all examples for this command
        examples = COMMAND_EXAMPLES.get(topic, [])
        if examples:
            console.print(f"\n[bold cyan]EXAMPLES[/bold cyan]\n")
            for desc, cmd in examples:
                console.print(f"  [dim]# {desc}[/dim]")
                console.print(f"  [green]$ {cmd}[/green]\n")
        return

    # No topic specified - show general help with quick examples
    parser.print_help()
    _print_command_listing(console, getattr(env.ui, "plain", False), entries)

    console.print("\n[bold cyan]QUICK EXAMPLES[/bold cyan]")
    console.print("[dim]Run 'polylogue help <command>' for full examples and details.[/dim]\n")

    key_commands = ["render", "sync", "import", "search", "watch"]
    for cmd in key_commands:
        if cmd not in COMMAND_EXAMPLES:
            continue
        examples = COMMAND_EXAMPLES[cmd]
        if not examples:
            continue
        # Show first example for each key command
        desc, cmdline = examples[0]
        console.print(f"  [dim]{desc}:[/dim] [green]{cmdline}[/green]")


def _completion_script(shell: str, commands: List[str], descriptions: Optional[Dict[str, str]] = None) -> str:
    joined = " ".join(commands)
    if shell == "bash":
        return textwrap.dedent(
            f"""
            _polylogue_complete() {{
                local cur prev
                COMPREPLY=()
                cur="${{COMP_WORDS[COMP_CWORD]}}"
                if [[ $COMP_CWORD -eq 1 ]]; then
                    COMPREPLY=( $(compgen -W \"{joined}\" -- \"$cur\") )
                    return
                fi
            }}
            complete -F _polylogue_complete polylogue
            """
        ).strip()
    # fish
    entries: List[str] = []
    for cmd in commands:
        desc = (descriptions or {}).get(cmd, "")
        if desc:
            desc_literal = desc.replace('"', '\\"')
            entries.append(f'complete -c polylogue -f -a "{cmd}" -d "{desc_literal}"')
        else:
            entries.append(f'complete -c polylogue -f -a "{cmd}"')
    entries = "\n".join(entries)
    return entries.strip()


def _zsh_dynamic_script() -> str:
    return textwrap.dedent(
        """
        #compdef polylogue

        _polylogue_complete() {
            local -a completions
            local IFS=$'\n'
            completions=($(polylogue _complete --shell zsh --cword $CURRENT -- "${words[@]}"))
            if [[ $? -ne 0 ]]; then
                return
            fi
            if [[ ${#completions[@]} -gt 0 ]]; then
                local first=${completions[1]}
                if [[ $first == "__PATH__" || $first == "__PATH__:"* ]]; then
                    _files
                    return
                fi
            fi
            _describe 'values' completions
        }

        compdef _polylogue_complete polylogue
        """
    ).strip()


def run_completions_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    parser = build_parser()
    entries = _command_entries(parser)
    commands = [name for name, _ in entries]
    descriptions = {name: desc for name, desc in entries if desc}
    if args.shell == "zsh":
        print(_zsh_dynamic_script())
        return
    script = _completion_script(args.shell, commands, descriptions)
    print(script)


def run_complete_cli(args: argparse.Namespace, env: Optional[CommandEnv] = None) -> None:
    env = env or CommandEnv(ui=create_ui(True))
    engine = CompletionEngine(env, build_parser())
    completions = engine.complete(args.shell, args.cword, args.words or [])
    for entry in completions:
        if entry.description:
            print(f"{entry.value}:{entry.description}")
        else:
            print(entry.value)


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

    # Collect all legacy paths first
    all_legacy: List[Tuple[Path, Path]] = []  # (root, legacy_path)
    for root in unique_roots:
        if not root.exists():
            continue
        legacy = _legacy_candidates(root)
        for path in legacy:
            all_legacy.append((root, path))
        total_candidates += len(legacy)

    if dry_run:
        for root, path in all_legacy:
            ui.console.print(f"[yellow][dry-run] Would prune: {path}")
    else:
        with ui.progress("Pruning legacy files", total=len(all_legacy)) as tracker:
            for root, path in all_legacy:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    total_removed += 1
                except Exception as exc:
                    ui.console.print(f"[red]Failed to remove {path}: {exc}")
                tracker.advance()

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
        html_out = getattr(args, "out", None)
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

        # Open in editor if --open flag is set
        if getattr(args, "open", False):
            # Prefer HTML if available, otherwise open markdown
            target_path = html_path if html_path else conv.conversation_path
            if target_path:
                if open_in_editor(Path(target_path)):
                    ui.console.print(f"[dim]Opened {target_path} in editor[/dim]")
                else:
                    editor = get_editor()
                    if not editor:
                        ui.console.print("[yellow]Warning: $EDITOR not set. Cannot open file.")
                    else:
                        ui.console.print(f"[yellow]Warning: Could not open {target_path} in editor")


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
            target = _resolve_html_output_path(conversation, getattr(args, "out", None), False)
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

    # Open in editor if --open flag is set and we have a single result
    if getattr(args, "open", False) and len(selected_hits) == 1:
        hit = selected_hits[0]
        target_path = hit.branch_path or hit.conversation_path
        if target_path:
            if open_in_editor(target_path):
                ui.console.print(f"[dim]Opened {target_path} in editor[/dim]")
            else:
                editor = get_editor()
                if not editor:
                    ui.console.print("[yellow]Warning: $EDITOR not set. Cannot open file.")
                else:
                    ui.console.print(f"[yellow]Warning: Could not open {target_path} in editor")
        else:
            ui.console.print("[yellow]Warning: No file path available for selected result")
    elif getattr(args, "open", False) and len(selected_hits) > 1:
        ui.console.print("[yellow]Warning: --open requires a single search result. Use --limit 1 or select one result.")


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


def _dispatch_sync(args: argparse.Namespace, env: CommandEnv) -> None:
    run_sync_cli(args, env)


def _dispatch_import(args: argparse.Namespace, env: CommandEnv) -> None:
    run_import_cli(args, env)


def _dispatch_search(args: argparse.Namespace, env: CommandEnv) -> None:
    """Search rendered transcripts."""
    run_inspect_search(args, env)


def _dispatch_watch(args: argparse.Namespace, env: CommandEnv) -> None:
    run_watch_cli(args, env)


def _dispatch_config(args: argparse.Namespace, env: CommandEnv) -> None:
    config_cmd = getattr(args, "config_cmd", None)
    if config_cmd == "init":
        from .init import run_init_cli
        run_init_cli(args, env)
    elif config_cmd == "set":
        from .settings_cli import run_settings_cli
        run_settings_cli(args, env)
    elif config_cmd == "show":
        _run_config_show(args, env)
    else:
        raise SystemExit("config requires a subcommand: init, set, show")


def _run_config_show(args: argparse.Namespace, env: CommandEnv) -> None:
    """Show current configuration (combines env + settings)."""
    import json
    from ..settings import SETTINGS_PATH
    from ..config import CONFIG_PATH

    ui = env.ui
    settings = env.settings
    defaults = CONFIG.defaults

    if getattr(args, "json", False):
        payload = {
            "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
            "settingsPath": str(SETTINGS_PATH),
            "html_previews": settings.html_previews,
            "html_theme": settings.html_theme,
            "collapse_threshold": settings.collapse_threshold,
            "output_dirs": {
                "render": str(defaults.output_dirs.render),
                "sync_drive": str(defaults.output_dirs.sync_drive),
                "sync_codex": str(defaults.output_dirs.sync_codex),
                "sync_claude_code": str(defaults.output_dirs.sync_claude_code),
                "import_chatgpt": str(defaults.output_dirs.import_chatgpt),
                "import_claude": str(defaults.output_dirs.import_claude),
            },
            "statePath": str(env.conversations.state_path),
            "runsDb": str(env.database.resolve_path()),
        }
        print(json.dumps(payload, indent=2))
        return

    summary_lines = [
        f"Config: {CONFIG_PATH or '(default)'}",
        f"Settings: {SETTINGS_PATH}",
        "",
        f"HTML previews: {'on' if settings.html_previews else 'off'}",
        f"HTML theme: {settings.html_theme}",
    ]
    if settings.collapse_threshold is not None:
        summary_lines.append(f"Collapse threshold: {settings.collapse_threshold}")

    summary_lines.extend([
        "",
        "Output directories:",
        f"  render: {defaults.output_dirs.render}",
        f"  sync/drive: {defaults.output_dirs.sync_drive}",
        f"  sync/codex: {defaults.output_dirs.sync_codex}",
        f"  sync/claude-code: {defaults.output_dirs.sync_claude_code}",
        f"  import/chatgpt: {defaults.output_dirs.import_chatgpt}",
        f"  import/claude: {defaults.output_dirs.import_claude}",
        "",
        f"State DB: {env.conversations.state_path}",
        f"Runs DB: {env.database.resolve_path()}",
    ])
    ui.summary("Configuration", summary_lines)


def _dispatch_browse(args: argparse.Namespace, env: CommandEnv) -> None:
    from .browse import run_browse_cli
    run_browse_cli(args, env)


def _dispatch_maintain(args: argparse.Namespace, env: CommandEnv) -> None:
    from .maintain import run_maintain_cli
    run_maintain_cli(args, env)


def _dispatch_search_preview(args: argparse.Namespace, _env: CommandEnv) -> None:
    run_search_preview(args)


def _dispatch_complete(args: argparse.Namespace, env: CommandEnv) -> None:
    run_complete_cli(args, env)


def _dispatch_help(args: argparse.Namespace, env: CommandEnv) -> None:
    run_help_cli(args, env)


def _dispatch_completions(args: argparse.Namespace, env: CommandEnv) -> None:
    run_completions_cli(args, env)


_REGISTRATION_COMPLETE = False


def _register_default_commands() -> None:
    global _REGISTRATION_COMPLETE
    if _REGISTRATION_COMPLETE:
        return

    def _ensure(
        name: str,
        handler: Callable[[argparse.Namespace, CommandEnv], None],
        help_text: str,
        aliases: Optional[List[str]] = None
    ) -> None:
        if COMMAND_REGISTRY.resolve(name) is None:
            COMMAND_REGISTRY.register(name, handler, help_text=help_text, aliases=aliases or [])

    # Core workflow commands (data ingestion & exploration)
    _ensure("search", _dispatch_search, "Search rendered transcripts", ["find", "f"])
    _ensure("import", _dispatch_import, "Import provider exports", ["i"])
    _ensure("sync", _dispatch_sync, "Sync provider archives", ["s"])
    _ensure("watch", _dispatch_watch, "Watch and auto-sync", ["w"])

    # Exploration
    _ensure("browse", _dispatch_browse, "Browse data (branches/stats/status/dashboards/runs)", ["b"])

    # Maintenance
    _ensure("maintain", _dispatch_maintain, "System maintenance (prune/doctor/index)", ["m"])

    # Configuration
    _ensure("config", _dispatch_config, "Configuration (init/set/show)", ["cfg"])

    # Meta
    _ensure("help", _dispatch_help, "Show command help", ["?"])
    _ensure("completions", _dispatch_completions, "Emit shell completion script")

    # Internal (no aliases needed)
    _ensure("_complete", _dispatch_complete, "Internal completion helper")
    _ensure("_search-preview", _dispatch_search_preview, "Internal search preview helper")

    _REGISTRATION_COMPLETE = True


def build_parser() -> argparse.ArgumentParser:
    _register_default_commands()
    parser = argparse.ArgumentParser(description="Polylogue CLI", formatter_class=PARSER_FORMATTER)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive UI even when stdout/stderr are not TTYs",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
    sub = parser.add_subparsers(dest="cmd")

    # Core workflow: sync
    p_sync = _add_command_parser(sub, "sync", help="Sync provider archives", description="Sync provider archives")
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
    add_allow_dirty_option(p_sync)
    p_sync.add_argument("--prune", action="store_true", help="Remove outputs for conversations that vanished upstream")
    add_collapse_option(p_sync)
    add_html_option(p_sync)
    add_diff_option(p_sync, help_text="Write delta diff alongside updated Markdown")
    p_sync.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_sync.add_argument(
        "--chat-id",
        dest="chat_ids",
        action="append",
        help="Drive chat/file ID to sync (repeatable)",
    )
    sync_selection_group = p_sync.add_mutually_exclusive_group()
    sync_selection_group.add_argument(
        "--session",
        dest="sessions",
        action="append",
        type=Path,
        help="Local session/export path to sync (repeatable; local providers)",
    )
    sync_selection_group.add_argument("--all", action="store_true", help="Process all available items without interactive selection")
    p_sync.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override local session/export directory",
    )
    p_sync.add_argument("--folder-name", type=str, default=DEFAULT_FOLDER_NAME, help="Drive folder name (drive provider)")
    p_sync.add_argument("--folder-id", type=str, default=None, help="Drive folder ID override")
    p_sync.add_argument("--since", type=str, default=None, help="Only include Drive chats updated on/after this timestamp")
    p_sync.add_argument("--until", type=str, default=None, help="Only include Drive chats updated on/before this timestamp")
    p_sync.add_argument("--name-filter", type=str, default=None, help="Regex filter for Drive chat names")
    p_sync.add_argument("--list-only", action="store_true", help="List Drive chats without syncing")

    # Core workflow: import
    p_import = _add_command_parser(sub, "import", help="Import provider exports", description="Import provider exports")
    p_import.add_argument("provider", choices=["chatgpt", "claude", "claude-code", "codex"], help="Provider export format")
    p_import.add_argument("source", nargs="*", help="Export path or session identifier (depends on provider); use 'pick', '?', or '-' to trigger interactive picker")
    add_out_option(p_import, default_path=Path("(provider-specific)"), help_text="Override output directory")
    add_collapse_option(p_import)
    add_html_option(p_import)
    add_dry_run_option(p_import)
    add_force_option(p_import, help_text="Rewrite even if conversations appear up-to-date")
    add_allow_dirty_option(p_import)
    import_selection_group = p_import.add_mutually_exclusive_group()
    import_selection_group.add_argument("--all", action="store_true", help="Process all available items without interactive selection")
    import_selection_group.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Specific conversation ID to import (repeatable)")
    p_import.add_argument("--base-dir", type=Path, default=None, help="Override source directory for codex/claude-code sessions")
    p_import.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_import.add_argument("--to-clipboard", action="store_true", help="Copy a single imported Markdown file to the clipboard")

    p_browse = _add_command_parser(sub, "browse", help="Browse data (branches/stats/status/dashboards/runs)", description="Explore rendered data and system status")
    browse_sub = p_browse.add_subparsers(dest="browse_cmd", required=True)

    p_browse_branches = _add_command_parser(browse_sub, "branches", help="Explore branch graphs for conversations", description="Explore branch graphs for conversations")
    p_browse_branches.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p_browse_branches.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p_browse_branches.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p_browse_branches.add_argument("--min-branches", type=int, default=1, help="Only include conversations with at least this many branches")
    p_browse_branches.add_argument("--branch", type=str, default=None, help="Branch ID to inspect or diff against the canonical path")
    p_browse_branches.add_argument("--diff", action="store_true", help="Display a unified diff between a branch and canonical transcript")
    p_browse_branches.add_argument(
        "--html",
        dest="html_mode",
        nargs="?",
        const="on",
        default="auto",
        choices=["auto", "on", "off"],
        metavar="MODE",
        help="Branch HTML mode: on/off/auto (default auto)",
    )
    p_browse_branches.add_argument("--out", type=Path, default=None, help="Write the branch explorer HTML to this path")
    p_browse_branches.add_argument("--theme", type=str, default=None, choices=["light", "dark"], help="Override HTML explorer theme")
    p_browse_branches.add_argument("--no-picker", action="store_true", help="Skip interactive selection even when skim/gum are available")
    p_browse_branches.add_argument("--open", action="store_true", help="Open result in $EDITOR after command completes")

    output_parent = create_output_parent()
    filter_parent = create_filter_parent()

    p_browse_stats = _add_command_parser(
        browse_sub,
        "stats",
        parents=[output_parent, filter_parent],
        help="Summarize Markdown output directories",
        description="Summarize Markdown output directories"
    )
    p_browse_stats.add_argument("--dir", type=Path, default=None, help="Directory containing Markdown exports")

    p_browse_status = _add_command_parser(browse_sub, "status", help="Show cached Drive info and recent runs", description="Show cached Drive info and recent runs")
    p_browse_status.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_browse_status.add_argument(
        "--json-lines",
        action="store_true",
        help="Stream newline-delimited JSON records (auto-enables --json, useful with --watch)",
    )
    status_mode_group = p_browse_status.add_mutually_exclusive_group()
    status_mode_group.add_argument("--watch", action="store_true", help="Continuously refresh the status output")
    status_mode_group.add_argument("--dump-only", action="store_true", help="Only perform the dump action without printing summaries")
    p_browse_status.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh while watching")
    p_browse_status.add_argument("--dump", type=str, default=None, help="Write recent runs to a file ('-' for stdout)")
    p_browse_status.add_argument("--dump-limit", type=int, default=100, help="Number of runs to include when dumping")
    p_browse_status.add_argument("--runs-limit", type=int, default=200, help="Number of historical runs to include in summaries")
    p_browse_status.add_argument(
        "--providers",
        type=str,
        default=None,
        help="Comma-separated provider filter (limits summaries, dumps, and JSON output)",
    )
    p_browse_status.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Write aggregated provider/run summary JSON to a file ('-' for stdout)",
    )
    p_browse_status.add_argument(
        "--summary-only",
        action="store_true",
        help="Only emit the summary JSON without printing tables",
    )

    p_browse_dashboards = _add_command_parser(browse_sub, "dashboards", help="Show provider dashboards", description="Rich dashboard of provider stats and recent runs")
    p_browse_dashboards.add_argument("--runs-limit", type=int, default=10, help="Number of recent runs to show")
    p_browse_dashboards.add_argument("--json", action="store_true", help="Emit dashboard data as JSON")

    p_browse_runs = _add_command_parser(browse_sub, "runs", help="List recent runs", description="List run history with filters")
    p_browse_runs.add_argument("--limit", type=int, default=50, help="Number of runs to display")
    p_browse_runs.add_argument("--providers", type=str, default=None, help="Comma-separated provider filter")
    p_browse_runs.add_argument("--commands", type=str, default=None, help="Comma-separated command filter")
    p_browse_runs.add_argument("--since", type=str, default=None, help="Only include runs on/after this timestamp (YYYY-MM-DD or ISO)")
    p_browse_runs.add_argument("--until", type=str, default=None, help="Only include runs on/before this timestamp")
    p_browse_runs.add_argument("--json", action="store_true", help="Emit runs as JSON")

    p_watch = _add_command_parser(sub, "watch", help="Watch local session stores and sync on changes", description="Watch local session stores and sync on changes")
    p_watch.add_argument("provider", choices=list(WATCHABLE_LOCAL_PROVIDER_NAMES), help="Local provider to watch")
    p_watch.add_argument("--base-dir", type=Path, default=None, help="Override source directory")
    add_out_option(p_watch, default_path=Path("(provider-specific)"), help_text="Override output directory")
    add_collapse_option(p_watch)
    add_html_option(p_watch, description="HTML preview mode while watching: on/off/auto (default auto)")
    add_dry_run_option(p_watch)
    p_watch.add_argument("--debounce", type=float, default=2.0, help="Minimal seconds between sync runs")
    p_watch.add_argument("--once", action="store_true", help="Run a single sync pass and exit")

    p_search = _add_command_parser(sub, "search", help="Search rendered transcripts", description="Search rendered transcripts")
    p_search.add_argument("query", type=str, help="FTS search query (SQLite syntax)")
    p_search.add_argument("--limit", type=int, default=20, help="Maximum number of hits to return")
    p_search.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p_search.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p_search.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p_search.add_argument("--branch", type=str, default=None, help="Restrict to a single branch ID")
    p_search.add_argument("--model", type=str, default=None, help="Filter by source model when recorded")
    p_search.add_argument("--since", type=str, default=None, help="Only include messages on/after this timestamp")
    p_search.add_argument("--until", type=str, default=None, help="Only include messages on/before this timestamp")
    search_attachment_group = p_search.add_mutually_exclusive_group()
    search_attachment_group.add_argument("--with-attachments", action="store_true", help="Limit to messages with extracted attachments")
    search_attachment_group.add_argument("--without-attachments", action="store_true", help="Limit to messages without attachments")
    p_search.add_argument("--no-picker", action="store_true", help="Skip skim picker preview even when interactive")
    p_search.add_argument("--json", action="store_true", help="Emit machine-readable search results")
    p_search.add_argument("--open", action="store_true", help="Open result file in $EDITOR after search")

    p_maintain = _add_command_parser(sub, "maintain", help="System maintenance (prune/doctor/index)", description="System maintenance and diagnostics")
    maintain_sub = p_maintain.add_subparsers(dest="maintain_cmd", required=True)

    p_maintain_prune = _add_command_parser(maintain_sub, "prune", help="Remove legacy single-file outputs and attachments", description="Remove legacy single-file outputs and attachments")
    p_maintain_prune.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        type=Path,
        help="Root directory to prune (repeatable). Defaults to all configured output directories.",
    )
    p_maintain_prune.add_argument("--dry-run", action="store_true", help="Print planned actions without deleting files")

    p_maintain_doctor = _add_command_parser(maintain_sub, "doctor", help="Check local data directories for common issues", description="Check local data directories for common issues")
    p_maintain_doctor.add_argument("--codex-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_maintain_doctor.add_argument("--claude-code-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_maintain_doctor.add_argument("--limit", type=int, default=None, help="Limit number of files inspected per provider")
    p_maintain_doctor.add_argument("--json", action="store_true", help="Emit machine-readable report")

    p_maintain_index = _add_command_parser(maintain_sub, "index", help="Index maintenance helpers", description="Inspect/repair Polylogue indexes")
    index_sub = p_maintain_index.add_subparsers(dest="subcmd", required=True)
    p_index_check = index_sub.add_parser("check", help="Validate SQLite/Qdrant indexes")
    p_index_check.add_argument("--repair", action="store_true", help="Attempt to rebuild missing SQLite FTS data")
    p_index_check.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant validation even when configured")
    p_index_check.add_argument("--json", action="store_true", help="Emit validation results as JSON")

    p_config = _add_command_parser(sub, "config", help="Configuration (init/set/show)", description="Configure Polylogue settings")
    config_sub = p_config.add_subparsers(dest="config_cmd", required=True)

    p_config_init = config_sub.add_parser("init", help="Interactive configuration setup", description="Interactive configuration setup wizard")
    p_config_init.add_argument("--force", action="store_true", help="Overwrite existing configuration")

    p_config_set = config_sub.add_parser("set", help="Update settings", description="Show or update Polylogue defaults")
    p_config_set.add_argument("--html", choices=["on", "off"], default=None, help="Enable or disable default HTML previews")
    p_config_set.add_argument("--theme", choices=["light", "dark"], default=None, help="Set the default HTML theme")
    p_config_set.add_argument("--collapse-threshold", type=int, default=None, help="Set the default collapse threshold for long outputs")
    p_config_set.add_argument("--reset", action="store_true", help="Reset to config defaults")
    p_config_set.add_argument("--json", action="store_true", help="Emit settings as JSON")

    p_config_show = config_sub.add_parser("show", help="Show configuration", description="Show resolved configuration and output paths")
    p_config_show.add_argument("--json", action="store_true", help="Emit environment info as JSON")

    p_help_cmd = _add_command_parser(sub, "help", help="Show help for a specific command", description="Show help for a specific command")
    p_help_cmd.add_argument("topic", nargs="?", help="Command name")

    p_completions = _add_command_parser(sub, "completions", help="Emit shell completion script", description="Emit shell completion script")
    p_completions.add_argument("--shell", choices=["bash", "zsh", "fish"], required=True)

    p_complete = _add_command_parser(sub, "_complete", help=argparse.SUPPRESS)
    p_complete.add_argument("--shell", required=True)
    p_complete.add_argument("--cword", type=int, required=True)
    p_complete.add_argument("words", nargs=argparse.REMAINDER)

    p_search_preview = _add_command_parser(sub, "_search-preview", help=argparse.SUPPRESS)
    p_search_preview.add_argument("--data-file", type=Path, required=True)
    p_search_preview.add_argument("--index", type=int, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    interactive = bool(getattr(args, "interactive", False))
    plain_mode = _should_use_plain(force_interactive=interactive)
    ui = create_ui(plain_mode)
    env = CommandEnv(ui=ui)
    ensure_settings_defaults(env.settings)

    # Enable verbose output if requested
    if getattr(args, "verbose", False):
        ui.console.print("[dim]Verbose mode enabled[/dim]")

    # Validate --allow-dirty requires --force
    if getattr(args, "allow_dirty", False) and not getattr(args, "force", False):
        ui.console.print("[red]Error: --allow-dirty requires --force")
        raise SystemExit(1)

    if args.cmd is None:
        parser.print_help()
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
