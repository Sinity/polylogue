#!/usr/bin/env python3
"""Polylogue: interactive-first CLI for AI chat log archives."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..version import POLYLOGUE_VERSION, SCHEMA_VERSION

from ..cli_common import choose_single_entry, filter_chats, resolve_inputs, sk_select
from ..commands import (
    CommandEnv,
    branches_command,
    search_command,
    status_command,
)
from ..drive_client import DEFAULT_FOLDER_NAME, DriveClient
from ..options import BranchExploreOptions, SearchHit, SearchOptions
from ..ui import create_ui
from .completion_engine import CompletionEngine, Completion
from .registry import CommandRegistry
from .arg_helpers import (
    add_collapse_option,
    add_diff_option,
    add_dry_run_option,
    add_force_option,
    add_html_option,
    add_out_option,
    create_output_parent,
    create_filter_parent,
    create_render_parent,
    create_write_parent,
)
from .editor import open_in_editor, get_editor
from .context import (
    DEFAULT_OUTPUT_ROOTS,
    DEFAULT_SYNC_OUT,
    DEFAULT_RENDER_OUT,
    default_import_namespace,
    default_sync_namespace,
    resolve_html_settings,
)
from .render import run_render_cli
from ..settings import ensure_settings_defaults, persist_settings, clear_persisted_settings
from ..config import CONFIG, CONFIG_PATH, DEFAULT_CREDENTIALS, DEFAULT_TOKEN
from ..local_sync import LOCAL_SYNC_PROVIDER_NAMES, WATCHABLE_LOCAL_PROVIDER_NAMES
from ..paths import STATE_HOME
from .imports import (
    run_import_cli,
    run_import_chatgpt,
    run_import_claude,
    run_import_claude_code,
    run_import_codex,
)
from .runs import run_runs_cli
from .index_cli import run_index_cli
from .watch import run_watch_cli
from .status import run_status_cli, run_stats_cli
from .attachments import run_attachments_cli
from .prefs import run_prefs_cli
from .summaries import summarize_import
from .sync import (
    run_list_cli,
    run_sync_cli,
    _collect_session_selection,
    _log_local_sync,
    _run_sync_drive,
)
from .open_helper import run_open_cli
from .env_cli import run_env_cli
from ..util import CODEX_SESSIONS_ROOT, add_run, parse_input_time_to_epoch, write_clipboard_text
from ..branch_explorer import branch_diff, build_branch_html, format_branch_tree

SCRIPT_MODULE = "polylogue.cli"
COMMAND_REGISTRY = CommandRegistry()


CLI_VERSION = POLYLOGUE_VERSION
_FORCE_PLAIN_VALUES = {"1", "true", "yes", "on"}


def _should_use_plain(args: argparse.Namespace) -> bool:
    if getattr(args, "interactive", False):
        return False
    if getattr(args, "plain", False):
        return True
    forced = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    if forced and forced.strip().lower() in _FORCE_PLAIN_VALUES:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())
PARSER_FORMATTER = argparse.ArgumentDefaultsHelpFormatter


def _add_command_parser(subparsers: argparse._SubParsersAction, name: str, **kwargs):
    if "epilog" not in kwargs:
        epilog = _examples_epilog(name)
        if epilog:
            kwargs["epilog"] = epilog
    kwargs.setdefault("formatter_class", PARSER_FORMATTER)
    return subparsers.add_parser(name, **kwargs)

# Command Examples for --examples flag
# Each command maps to list of (description, command_line) tuples
COMMAND_EXAMPLES = {
    "render": [
        ("Render a single export", "polylogue render export.json --out ~/polylogue-data/render"),
        ("Render with HTML previews", "polylogue render export.json --html on"),
        ("Render multiple exports", "polylogue render exports/ --diff"),
    ],
    "sync": [
        ("Sync all Drive chats", "polylogue sync drive --all"),
        ("Sync from specific Drive folder", "polylogue sync drive --folder-name 'Work Chats'"),
        ("Sync Codex sessions with preview", "polylogue sync codex --dry-run"),
        ("Sync Claude Code with diff tracking", "polylogue sync claude-code --diff"),
        ("Sync and prune deleted chats", "polylogue sync drive --all --prune"),
        ("Watch Codex and auto-sync", "polylogue sync codex --watch"),
        ("Watch Claude Code with HTML", "polylogue sync claude-code --watch --html on"),
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
        ("Search and open anchored result", "polylogue search 'TODO' --limit 1 --open"),
        ("Search with attachment filter", "polylogue search 'diagram' --with-attachments"),
    ],
    "browse": [
        ("Browse branch graph", "polylogue browse branches --provider claude"),
        ("View stats", "polylogue browse stats --provider chatgpt"),
        ("Get stats with time filter", "polylogue browse stats --since 2024-01-01 --until 2024-12-31"),
        ("Show status overview", "polylogue browse status"),
        ("Watch status continuously", "polylogue browse status --watch"),
        ("List recent runs", "polylogue browse runs --limit 20"),
    ],
    "maintain": [
        ("Preview prune operation", "polylogue maintain prune --dry-run"),
        ("Prune legacy files", "polylogue maintain prune --dir ~/polylogue-data/chatgpt"),
        ("Check all providers", "polylogue maintain doctor"),
        ("Check with JSON output", "polylogue maintain doctor --json"),
        ("Validate indexes", "polylogue maintain index check"),
        ("Repair indexes", "polylogue maintain index check --repair"),
        ("Restore a snapshot", "polylogue maintain restore --from /tmp/snap --to ~/.local/share/polylogue/archive --force"),
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
    "compare": [
        ("Compare coverage between two providers", "polylogue compare 'auth error' --provider-a codex --provider-b claude-code --limit 10"),
    ],
    "attachments": [
        ("Show top attachments by size", "polylogue attachments stats --sort size --limit 5"),
        ("Read attachment metadata from the index", "polylogue attachments stats --from-index --json"),
        ("Extract PDFs to a folder", "polylogue attachments extract --ext .pdf --out ~/desk/pdfs"),
    ],
    "status": [
        ("Watch status with JSON output", "polylogue status --json --watch --interval 10"),
        ("Dump recent runs quietly", "polylogue status --dump - --runs-limit 50 --quiet"),
        ("Summarize a single provider", "polylogue status --providers codex --summary -"),
    ],
    "prefs": [
        ("Persist a search default", "polylogue prefs set search --flag --limit --value 5"),
        ("List saved preferences", "polylogue prefs list"),
        ("Clear all saved defaults", "polylogue prefs clear"),
    ],
    "help": [
        ("Show all examples", "polylogue help --examples"),
        ("View examples for a single command", "polylogue help search --examples"),
    ],
}


def _examples_epilog(command: str) -> Optional[str]:
    examples = COMMAND_EXAMPLES.get(command)
    if not examples:
        return None
    lines = ["Examples:"]
    for desc, cmdline in examples:
        lines.append(f"  # {desc}")
        lines.append(f"  $ {cmdline}")
    return "\n".join(lines)


def _classify_exit_reason(exc: BaseException) -> str:
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return "io"
    if isinstance(exc, ValueError):
        return "schema"
    return "error"


def _record_failure(args: argparse.Namespace, exc: BaseException, *, phase: str = "cli") -> None:
    exit_reason = _classify_exit_reason(exc)
    os.environ["POLYLOGUE_EXIT_REASON"] = exit_reason
    try:
        STATE_HOME.mkdir(parents=True, exist_ok=True)
        provider = getattr(args, "provider", None) or getattr(args, "providers", None)
        file_hint = None
        for attr in ("input", "source", "dir"):
            value = getattr(args, attr, None)
            if value:
                file_hint = str(value)
                break
        hints = {
            "io": "Check file paths, permissions, and available disk space.",
            "schema": "Validate input/export schema and retry with updated tooling.",
            "error": "Re-run with --verbose for a traceback and file a bug if reproducible.",
        }
        record = {
            "id": f"fail-{int(time.time() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cmd": getattr(args, "cmd", None),
            "provider": provider,
            "file": file_hint,
            "phase": phase,
            "exception": exc.__class__.__name__,
            "message": str(exc),
            "exit_reason": exit_reason,
            "hint": hints.get(exit_reason),
            "cwd": str(Path.cwd()),
            "argv": sys.argv[1:],
        }
        path = STATE_HOME / "failures.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
    except Exception:
        return


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
    show_examples_only = getattr(args, "examples", False) and not topic
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
        return

    if show_examples_only:
        console.print("[bold cyan]EXAMPLES[/bold cyan]\n")
        for name in sorted(COMMAND_EXAMPLES):
            examples = COMMAND_EXAMPLES[name]
            if not examples:
                continue
            console.print(f"[green]{name}[/green]")
            for desc, cmdline in examples:
                console.print(f"  [dim]{desc}:[/dim] [green]{cmdline}[/green]")
            console.print("")
        return

    # No topic specified - show general help with quick examples
    parser.print_help()
    _print_command_listing(console, getattr(env.ui, "plain", False), entries)

    console.print("\n[bold cyan]QUICK EXAMPLES[/bold cyan]")
    console.print("[dim]Run 'polylogue help <command>' for full examples and details.[/dim]\n")

    key_commands = ["render", "sync", "import", "search", "browse"]
    for cmd in key_commands:
        if cmd not in COMMAND_EXAMPLES:
            continue
        examples = COMMAND_EXAMPLES[cmd]
        if not examples:
            continue
        # Show first example for each key command
        desc, cmdline = examples[0]
        console.print(f"  [dim]{desc}:[/dim] [green]{cmdline}[/green]")


def _bash_dynamic_script() -> str:
    return textwrap.dedent(
        """
        _polylogue_complete() {
            local IFS=$'\\n'
            local completions
            completions=$(polylogue _complete --shell bash --cword $COMP_CWORD -- "${COMP_WORDS[@]}" 2>/dev/null)
            if [[ $? -ne 0 ]]; then
                return
            fi
            local first=$(echo "$completions" | head -1)
            if [[ $first == "__PATH__" ]]; then
                COMPREPLY=( $(compgen -f -- "${COMP_WORDS[COMP_CWORD]}") )
                return
            fi
            COMPREPLY=( $(compgen -W "$completions" -- "${COMP_WORDS[COMP_CWORD]}") )
        }
        complete -F _polylogue_complete polylogue
        """
    ).strip()


def _fish_dynamic_script() -> str:
    return textwrap.dedent(
        """
        function __polylogue_complete
            set -l cmd (commandline -opc)
            set -l cword (count $cmd)
            polylogue _complete --shell fish --cword $cword -- $cmd 2>/dev/null | while read -l line
                if string match -q "__PATH__*" -- $line
                    __fish_complete_path
                    continue
                end
                if string match -q "*:*" -- $line
                    set -l parts (string split -m 1 ":" -- $line)
                    echo $parts[1]\\t$parts[2]
                else
                    echo $line
                end
            end
        end
        complete -c polylogue -f -a "(__polylogue_complete)"
        """
    ).strip()


def _completion_script(shell: str, commands: List[str], descriptions: Optional[Dict[str, str]] = None) -> str:
    # Deprecated fallback for bash/fish - all shells now use dynamic completions
    joined = " ".join(commands)
    if shell == "bash":
        return _bash_dynamic_script()
    # fish
    desc_map = descriptions or {}
    static_lines: List[str] = []
    for name in commands:
        if name.startswith("_"):
            continue
        desc = desc_map.get(name, "")
        if desc:
            escaped_desc = desc.replace('"', '\"')
            static_lines.append(f"complete -c polylogue -n '__fish_use_subcommand' -a '{name}' -d \"{escaped_desc}\"")
        else:
            static_lines.append(f"complete -c polylogue -n '__fish_use_subcommand' -a '{name}'")
    static_block = "\n".join(static_lines)
    return _fish_dynamic_script() + ("\n" + static_block if static_block else "")


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
        snapshot_path = None
        if all_legacy:
            from zipfile import ZipFile
            from ..paths import STATE_HOME
            snapshot_dir = STATE_HOME / "rollback"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"prune-{int(time.time())}.zip"
            try:
                with ZipFile(snapshot_path, "w") as zipf:
                    for _, path in all_legacy:
                        try:
                            if path.is_file():
                                zipf.write(path, arcname=path.name)
                        except Exception:
                            continue
                ui.console.print(f"[dim]Snapshot saved to {snapshot_path} before pruning.[/dim]")
            except Exception as exc:
                ui.console.print(f"[yellow]Snapshot failed: {exc}")
                snapshot_path = None
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
    prefs = getattr(env, "prefs", {})
    search_prefs = prefs.get("search", {}) if isinstance(prefs, dict) else {}

    def _pref_bool(val: str) -> bool:
        return str(val).lower() in {"1", "true", "yes", "on"}

    # Apply saved defaults when caller did not override
    if "--limit" in search_prefs and getattr(args, "limit", None) == 20:
        try:
            args.limit = int(search_prefs["--limit"])
        except Exception:
            pass
    if "--no-picker" in search_prefs and not getattr(args, "no_picker", False):
        if _pref_bool(search_prefs["--no-picker"]):
            args.no_picker = True
    if "--json" in search_prefs and not getattr(args, "json", False):
        if _pref_bool(search_prefs["--json"]):
            args.json = True
    if "--in-attachments" in search_prefs and not getattr(args, "in_attachments", False):
        if _pref_bool(search_prefs["--in-attachments"]):
            args.in_attachments = True

    if getattr(args, "from_stdin", False) or getattr(args, "query", None) == "-":
        data = sys.stdin.read()
        if not data.strip():
            ui.console.print("[red]Search query is empty; provide a query via stdin or argument.")
            raise SystemExit(1)
        args.query = data.strip()
    if args.limit <= 0:
        args.limit = 20
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
        in_attachments=getattr(args, "in_attachments", False),
        attachment_name=getattr(args, "attachment_name", None),
    )
    result = search_command(options, env)
    hits = result.hits

    export_fields = [field.strip() for field in (getattr(args, "fields", "") or "").split(",") if field.strip()]
    csv_target = getattr(args, "csv", None)
    json_lines = bool(getattr(args, "json_lines", False))
    if json_lines:
        setattr(args, "json", True)

    def _row(hit: SearchHit) -> Dict[str, Any]:
        return {
            "provider": hit.provider,
            "conversationId": hit.conversation_id,
            "conversation_id": hit.conversation_id,
            "slug": hit.slug,
            "title": hit.title,
            "branchId": hit.branch_id,
            "branch_id": hit.branch_id,
            "messageId": hit.message_id,
            "message_id": hit.message_id,
            "position": hit.position,
            "timestamp": hit.timestamp,
            "attachments": hit.attachment_count,
            "kind": hit.kind,
            "attachmentName": hit.attachment_name,
            "attachment_name": hit.attachment_name,
            "attachmentPath": str(hit.attachment_path) if hit.attachment_path else None,
            "attachment_path": str(hit.attachment_path) if hit.attachment_path else None,
            "attachmentBytes": hit.attachment_bytes,
            "attachment_bytes": hit.attachment_bytes,
            "attachmentMime": hit.attachment_mime,
            "attachment_mime": hit.attachment_mime,
            "attachmentTextBytes": hit.attachment_text_bytes,
            "attachment_text_bytes": hit.attachment_text_bytes,
            "ocrUsed": hit.ocr_used,
            "ocr_used": hit.ocr_used,
            "score": hit.score,
            "snippet": hit.snippet,
            "conversationPath": str(hit.conversation_path) if hit.conversation_path else None,
            "conversation_path": str(hit.conversation_path) if hit.conversation_path else None,
            "branchPath": str(hit.branch_path) if hit.branch_path else None,
            "branch_path": str(hit.branch_path) if hit.branch_path else None,
            "model": hit.model,
            "path": str(
                hit.attachment_path or hit.branch_path or hit.conversation_path
            )
            if (hit.attachment_path or hit.branch_path or hit.conversation_path)
            else None,
        }

    if getattr(args, "json", False):
        rows = [_row(hit) for hit in hits]
        if json_lines:
            for row in rows:
                payload = {k: row.get(k) for k in export_fields} if export_fields else row
                payload.setdefault("schemaVersion", SCHEMA_VERSION)
                payload.setdefault("polylogueVersion", CLI_VERSION)
                print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            return
        payload = {
            "query": options.query,
            "count": len(hits),
            "schemaVersion": SCHEMA_VERSION,
            "polylogueVersion": CLI_VERSION,
            "hits": [
                {k: row.get(k) for k in export_fields} if export_fields else row
                for row in rows
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if csv_target:
        import csv

        fieldnames = export_fields or [
            "provider",
            "conversationId",
            "slug",
            "branchId",
            "messageId",
            "position",
            "timestamp",
            "kind",
            "score",
            "model",
            "attachments",
            "attachmentName",
            "attachmentPath",
            "attachmentBytes",
            "attachmentMime",
            "attachmentTextBytes",
            "ocrUsed",
            "snippet",
            "path",
        ]
        rows = [{k: _row(hit).get(k) for k in fieldnames} for hit in hits]
        destination = Path(csv_target)
        if str(destination) == "-":
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            destination = destination.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            ui.console.print(f"[green]Wrote {len(rows)} search hit(s) to {destination}")
        return

    if not hits:
        ui.console.print("[yellow]No results found.")
        return

    summary_lines = [f"Hits: {len(hits)} (limit {options.limit})"]
    if options.in_attachments:
        summary_lines.append("Mode: attachment text")
    provider_counts = Counter(hit.provider for hit in hits)
    if provider_counts:
        provider_overview = ", ".join(
            f"{provider}×{count}" for provider, count in provider_counts.most_common(3)
        )
        summary_lines.append(f"Providers: {provider_overview}")
    model_set = {hit.model for hit in hits if hit.model}
    if model_set:
        summary_lines.append("Models: " + ", ".join(sorted(model_set)))
    attachment_results = sum(1 for hit in hits if hit.kind == "attachment")
    if attachment_results:
        summary_lines.append(f"Attachment hits: {attachment_results}")
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
        target_path = hit.attachment_path or hit.branch_path or hit.conversation_path
        line_hint = None
        anchor_label = None
        if target_path and hit.kind != "attachment" and hit.position is not None and hit.position >= 0:
            anchor_label = f"msg-{hit.position}"
            line_hint = _find_anchor_line(Path(target_path), anchor_label)
        if target_path:
            if open_in_editor(Path(target_path), line=line_hint):
                suffix = f" (line {line_hint})" if line_hint else ""
                label = f"{target_path}#{anchor_label}" if anchor_label else str(target_path)
                ui.console.print(f"[dim]Opened {label}{suffix} in editor[/dim]")
            else:
                editor = get_editor()
                if not editor:
                    ui.console.print("[yellow]Warning: $EDITOR not set. Cannot open file.")
                else:
                    label = f"{target_path}#{anchor_label}" if anchor_label else str(target_path)
                    ui.console.print(f"[yellow]Warning: Could not open {label} in editor")
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
            "kind": hit.kind,
            "attachmentName": hit.attachment_name,
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
            prefix = "ATT" if hit.kind == "attachment" else "MSG"
            branch_label = hit.branch_id or "-"
            return f"{prefix}:{hit.provider}:{hit.slug} [{branch_label}] score={hit.score:.3f} {snippet}"

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


def _find_anchor_line(path: Path, anchor: str) -> Optional[int]:
    """Return the 1-based line containing the given anchor id."""

    target = anchor.lstrip("#")
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for idx, line in enumerate(lines, start=1):
        if target in line:
            return idx
    return None


def _render_search_hit(hit: SearchHit, ui) -> None:
    branch_label = hit.branch_id or "-"
    header = f"{hit.provider}/{hit.slug} [{branch_label}]"
    lines = [
        f"Kind: {hit.kind}",
        f"Score: {hit.score:.4f}",
    ]
    if hit.kind != "attachment":
        lines.append(f"Message: {hit.message_id} (position {hit.position})")
    if hit.kind == "attachment":
        if hit.attachment_name:
            lines.append(f"Attachment: {hit.attachment_name}")
        if hit.attachment_bytes is not None:
            lines.append(f"Attachment size: {hit.attachment_bytes} bytes")
        if hit.attachment_mime:
            lines.append(f"MIME: {hit.attachment_mime}")
        if hit.attachment_path:
            lines.append(f"Attachment path: {hit.attachment_path}")
        lines.append(f"OCR used: {'yes' if hit.ocr_used else 'no'}")
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

    body = (hit.body or "").strip()
    if not body:
        ui.console.print("[cyan](No text available for this result)")
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
    kind = payload.get("kind") or "message"
    attachment_name = payload.get("attachmentName") or ""
    parts = [
        f"{title}",
        "=" * len(title),
        "",
        f"Kind: {kind}   Provider: {provider}   Branch: {branch}",
        f"Score: {score}",
    ]
    if attachment_name:
        parts.append(f"Attachment: {attachment_name}")
    if timestamp:
        parts.append(f"Timestamp: {timestamp}")
    if snippet:
        parts.extend(["", f"Snippet: {snippet}"])
    if body:
        parts.extend(["", body])
    print("\n".join(str(part) for part in parts))


def _compare_hits(provider: str, hits: List[SearchHit], fields: List[str]) -> Dict[str, Any]:
    total_attachments = sum(hit.attachment_count or 0 for hit in hits)
    models = sorted({hit.model for hit in hits if hit.model})
    payload_hits = []
    for hit in hits:
        row = {
            "provider": hit.provider,
            "slug": hit.slug,
            "branchId": hit.branch_id,
            "messageId": hit.message_id,
            "score": hit.score,
            "snippet": hit.snippet,
            "model": hit.model,
            "path": str(hit.branch_path or hit.conversation_path) if (hit.branch_path or hit.conversation_path) else None,
        }
        payload_hits.append({k: row.get(k) for k in fields})
    return {
        "provider": provider,
        "count": len(hits),
        "attachments": total_attachments,
        "models": models,
        "hits": payload_hits,
    }


def run_compare_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    fields = [f.strip() for f in (getattr(args, "fields", "") or "").split(",") if f.strip()]
    limit = max(1, getattr(args, "limit", 20))

    def _search(provider: str) -> List[SearchHit]:
        options = SearchOptions(
            query=args.query,
            limit=limit,
            provider=provider,
            slug=None,
            conversation_id=None,
            branch_id=None,
            model=None,
            since=None,
            until=None,
            has_attachments=None,
        )
        return search_command(options, env).hits

    hits_a = _search(args.provider_a)
    hits_b = _search(args.provider_b)

    if getattr(args, "json", False):
        payload = {
            "query": args.query,
            "limit": limit,
            "providers": [
                _compare_hits(args.provider_a, hits_a, fields),
                _compare_hits(args.provider_b, hits_b, fields),
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    lines = [f"Query: {args.query}", f"Limit: {limit}"]
    for name, hits in ((args.provider_a, hits_a), (args.provider_b, hits_b)):
        attachments_total = sum(hit.attachment_count or 0 for hit in hits)
        models = sorted({hit.model for hit in hits if hit.model})
        lines.append(
            f"{name}: {len(hits)} hit(s), attachments={attachments_total}, models={', '.join(models) if models else 'n/a'}"
        )
        for hit in hits[: min(3, len(hits))]:
            path_val = hit.branch_path or hit.conversation_path
            snippet = (hit.snippet or "").replace("\n", " ")
            snippet = textwrap.shorten(snippet, width=96, placeholder="…")
            lines.append(
                f"  - {hit.slug} [{hit.branch_id}] score={hit.score:.3f} {snippet} ({path_val})"
            )
    ui.summary("Provider Compare", lines)


def _dispatch_sync(args: argparse.Namespace, env: CommandEnv) -> None:
    if getattr(args, "watch", False):
        if args.provider == "drive":
            raise SystemExit("Drive does not support --watch; use local providers like codex/claude-code/chatgpt.")
        # Validate provider supports watch mode
        from ..local_sync import get_local_provider
        provider = get_local_provider(args.provider)
        if not provider.supports_watch:
            raise SystemExit(f"{provider.title} does not support watch mode (use --watch with codex, claude-code, or chatgpt)")
        run_watch_cli(args, env)
    else:
        if getattr(args, "offline", False) and args.provider == "drive":
            raise SystemExit("--offline is not supported for Drive; run without it or target a local provider.")
        if getattr(args, "root", None):
            label = args.root
            defaults = env.config.defaults
            roots = getattr(defaults, "roots", {}) or {}
            paths = roots.get(label)
            if not paths:
                raise SystemExit(f"Unknown root label '{label}'. Define it in config or use a known label.")
            env.config.defaults.output_dirs = paths
        if args.provider == "drive":
            from ..drive_client import DEFAULT_CREDENTIALS
            cred_path = DEFAULT_CREDENTIALS
            if env.config.drive and env.config.drive.credentials_path:
                cred_path = env.config.drive.credentials_path
            if not cred_path.exists():
                raise SystemExit(f"Drive sync requires credentials.json at {cred_path} (set drive.credentials_path in config).")
        if args.provider in {"chatgpt", "claude"}:
            exports_root = env.config.exports.chatgpt if args.provider == "chatgpt" else env.config.exports.claude
            if not exports_root.exists():
                raise SystemExit(f"{args.provider} exports directory not found: {exports_root} (set exports.{args.provider} in config).")
        run_sync_cli(args, env)


def _dispatch_import(args: argparse.Namespace, env: CommandEnv) -> None:
    run_import_cli(args, env)


def _dispatch_search(args: argparse.Namespace, env: CommandEnv) -> None:
    """Search rendered transcripts."""
    run_inspect_search(args, env)


def _dispatch_render(args: argparse.Namespace, env: CommandEnv) -> None:
    run_render_cli(args, env, json_output=getattr(args, "json", False))


def _dispatch_compare(args: argparse.Namespace, env: CommandEnv) -> None:
    run_compare_cli(args, env)


def _dispatch_config(args: argparse.Namespace, env: CommandEnv) -> None:
    config_cmd = getattr(args, "config_cmd", None)
    if not config_cmd:
        env.ui.console.print("[red]config requires a sub-command (init/set/show)")
        raise SystemExit(1)
    if config_cmd == "init":
        from .init import run_init_cli
        run_init_cli(args, env)
    elif config_cmd == "set":
        from .settings_cli import run_settings_cli
        run_settings_cli(args, env)
    else:  # show
        _run_config_show(args, env)


def _dispatch_attachments(args: argparse.Namespace, env: CommandEnv) -> None:
    if not getattr(args, "attachments_cmd", None):
        env.ui.console.print("[red]attachments requires a sub-command (stats/extract)")
        raise SystemExit(1)
    run_attachments_cli(args, env)


def _run_config_show(args: argparse.Namespace, env: CommandEnv) -> None:
    """Show current configuration (combines env + settings)."""
    import json
    from ..settings import SETTINGS_PATH
    from ..config import CONFIG_PATH

    ui = env.ui
    settings = env.settings
    defaults = CONFIG.defaults

    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    drive_cfg = getattr(env, "config", None).drive if hasattr(env, "config") else None
    credential_path = drive_cfg.credentials_path if drive_cfg else DEFAULT_CREDENTIALS
    token_path = drive_cfg.token_path if drive_cfg else DEFAULT_TOKEN

    if getattr(args, "json", False):
        payload = {
            "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
            "settingsPath": str(SETTINGS_PATH),
            "ui": {
                "html_previews": settings.html_previews,
                "html_theme": settings.html_theme,
                "collapse_threshold": settings.collapse_threshold,
            },
            "schemaVersion": SCHEMA_VERSION,
            "polylogueVersion": CLI_VERSION,
            "auth": {
                "credentialPath": str(credential_path) if credential_path else None,
                "tokenPath": str(token_path) if token_path else None,
                "env": {
                    "POLYLOGUE_CREDENTIAL_PATH": credential_env,
                    "POLYLOGUE_TOKEN_PATH": token_env,
                },
            },
            "outputs": {
                "render": str(defaults.output_dirs.render),
                "gemini": str(defaults.output_dirs.sync_drive),
                "codex": str(defaults.output_dirs.sync_codex),
                "claude_code": str(defaults.output_dirs.sync_claude_code),
                "chatgpt": str(defaults.output_dirs.import_chatgpt),
                "claude": str(defaults.output_dirs.import_claude),
            },
            "inputs": {
                "chatgpt": str(CONFIG.exports.chatgpt),
                "claude": str(CONFIG.exports.claude),
            },
            "index": {
                "backend": CONFIG.index.backend if CONFIG.index else "sqlite",
                "qdrant": {
                    "url": CONFIG.index.qdrant_url if CONFIG.index else None,
                    "api_key": CONFIG.index.qdrant_api_key if CONFIG.index else None,
                    "collection": CONFIG.index.qdrant_collection if CONFIG.index else None,
                    "vector_size": CONFIG.index.qdrant_vector_size if CONFIG.index else None,
                },
            },
            "statePath": str(env.conversations.state_path),
            "runsDb": str(env.database.resolve_path()),
        }
        print(json.dumps(payload))
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

    summary_lines.extend(
        [
            "",
            "Auth paths:",
            f"  credentials: {credential_env or credential_path}",
            f"  token: {token_env or token_path}",
        ]
    )

    summary_lines.extend([
        "",
        "Output directories:",
        f"  render: {defaults.output_dirs.render}",
        f"  gemini: {defaults.output_dirs.sync_drive}",
        f"  codex: {defaults.output_dirs.sync_codex}",
        f"  claude-code: {defaults.output_dirs.sync_claude_code}",
        f"  chatgpt: {defaults.output_dirs.import_chatgpt}",
        f"  claude: {defaults.output_dirs.import_claude}",
        "",
        f"State DB: {env.conversations.state_path}",
        f"Runs DB: {env.database.resolve_path()}",
    ])
    ui.summary("Configuration", summary_lines)


def _dispatch_inspect(args: argparse.Namespace, env: CommandEnv) -> None:
    """Legacy inspect dispatcher retained for compatibility.

    The modern entrypoints live under `browse` (branches/status/stats) and
    `search`. This helper keeps older code paths and tests working while
    steering callers toward the consolidated commands.
    """
    subcmd = getattr(args, "inspect_cmd", None)
    if not subcmd:
        env.ui.console.print("[red]inspect requires a sub-command (branches/search). Use 'polylogue browse' instead.")
        raise SystemExit(1)

    if subcmd == "branches":
        run_inspect_branches(args, env)
        return
    if subcmd == "search":
        run_inspect_search(args, env)
        return

    env.ui.console.print(f"[red]Unknown inspect sub-command: {subcmd}")
    raise SystemExit(1)


def _dispatch_browse(args: argparse.Namespace, env: CommandEnv) -> None:
    from .browse import run_browse_cli
    run_browse_cli(args, env)


def _dispatch_maintain(args: argparse.Namespace, env: CommandEnv) -> None:
    from .maintain import run_maintain_cli
    run_maintain_cli(args, env)


def _dispatch_status(args: argparse.Namespace, env: CommandEnv) -> None:
    run_status_cli(args, env)


def _dispatch_search_preview(args: argparse.Namespace, _env: CommandEnv) -> None:
    run_search_preview(args)


def _dispatch_complete(args: argparse.Namespace, env: CommandEnv) -> None:
    run_complete_cli(args, env)


def _dispatch_help(args: argparse.Namespace, env: CommandEnv) -> None:
    run_help_cli(args, env)


def _dispatch_completions(args: argparse.Namespace, env: CommandEnv) -> None:
    run_completions_cli(args, env)


def _dispatch_env(args: argparse.Namespace, env: CommandEnv) -> None:
    run_env_cli(args, env)


def _dispatch_prefs(args: argparse.Namespace, env: CommandEnv) -> None:
    run_prefs_cli(args, env)


def _dispatch_open(args: argparse.Namespace, env: CommandEnv) -> None:
    run_open_cli(args, env)


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
    _ensure("render", _dispatch_render, "Render JSON exports to Markdown/HTML", ["r"])
    _ensure("search", _dispatch_search, "Search rendered transcripts", ["find", "f"])
    _ensure("compare", _dispatch_compare, "Compare coverage between two providers for a query")
    _ensure("import", _dispatch_import, "Import provider exports", ["i"])
    _ensure("sync", _dispatch_sync, "Synchronize provider archives (use --watch for continuous mode)", ["s"])
    _ensure("status", _dispatch_status, "Show cached Drive info and recent runs")
    # Exploration
    _ensure("browse", _dispatch_browse, "Browse data (branches/stats/status/runs)", ["b"])

    # Maintenance
    _ensure("maintain", _dispatch_maintain, "System maintenance (prune/doctor/index)", ["m"])
    _ensure("prefs", _dispatch_prefs, "Manage per-command preference defaults")
    _ensure("open", _dispatch_open, "Open or print paths from the latest run")

    # Configuration
    _ensure("config", _dispatch_config, "Configuration (init/set/show)", ["cfg"])
    _ensure("attachments", _dispatch_attachments, "Attachment utilities (stats/extract)")
    _ensure("env", _dispatch_env, "Check environment configuration")

    # Meta
    _ensure("help", _dispatch_help, "Show command help", ["?"])
    _ensure("completions", _dispatch_completions, "Emit shell completion script")

    # Internal (no aliases needed)
    _ensure("_complete", _dispatch_complete, "Internal completion helper")
    _ensure("_search-preview", _dispatch_search_preview, "Internal search preview helper")

    _REGISTRATION_COMPLETE = True


def _configure_status_parser(parser: argparse.ArgumentParser, *, require_provider: bool = False) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    parser.add_argument(
        "--json-lines",
        action="store_true",
        help="Stream newline-delimited JSON records (auto-enables --json, useful with --watch)",
    )
    parser.add_argument(
        "--json-verbose",
        action="store_true",
        help="Allow status to print tables/logs alongside JSON/JSONL output",
    )
    status_mode_group = parser.add_mutually_exclusive_group()
    status_mode_group.add_argument("--watch", action="store_true", help="Continuously refresh the status output")
    status_mode_group.add_argument("--dump-only", action="store_true", help="Only perform the dump action without printing summaries")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh while watching")
    parser.add_argument("--dump", type=str, default=None, help="Write recent runs to a file ('-' for stdout)")
    parser.add_argument("--dump-limit", type=int, default=100, help="Number of runs to include when dumping")
    parser.add_argument("--runs-limit", type=int, default=200, help="Number of historical runs to include in summaries")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        required=require_provider,
        help="Comma-separated provider filter (limits summaries, dumps, and JSON output)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress table output (useful with --json-lines)")
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Write aggregated provider/run summary JSON to a file ('-' for stdout)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only emit the summary JSON without printing tables",
    )


def build_parser() -> argparse.ArgumentParser:
    _register_default_commands()
    parser = argparse.ArgumentParser(description="Polylogue CLI", formatter_class=PARSER_FORMATTER)
    parser.add_argument("--version", action="version", version=f"polylogue {CLI_VERSION}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--profile-sql", action="store_true", help="Log SQL timings and top queries")
    parser.add_argument("--profile-io", action="store_true", help="Log IO/attachment timings during runs")
    parser.add_argument("--max-disk", type=float, default=None, help="Abort if projected disk use exceeds this many GiB (approx)")
    parser.add_argument("--root", type=str, default=None, help="Named root label to use when configs support multi-root archives")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", action="store_true", help="Force interactive UI even when stdout/stderr are not TTYs")
    mode_group.add_argument("--plain", action="store_true", help="Force plain/CI-safe output even when running in a TTY")
    sub = parser.add_subparsers(dest="cmd")

    render_parents = [create_render_parent(), create_write_parent()]
    p_render = _add_command_parser(
        sub,
        "render",
        parents=render_parents,
        help="Render JSON exports to Markdown/HTML",
        description="Render provider exports to Markdown/HTML",
        epilog=_examples_epilog("render"),
    )
    p_render.add_argument("input", type=Path, help="Input JSON file or directory containing exports")
    add_out_option(
        p_render,
        default_path=DEFAULT_RENDER_OUT,
        help_text="Output directory for rendered Markdown/HTML",
    )
    p_render.add_argument("--links-only", action="store_true", help="Link attachments without downloading")
    p_render.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when indexing attachment text",
    )
    add_diff_option(p_render)
    p_render.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_render.add_argument("--print-paths", action="store_true", help="List written files after rendering")
    p_render.add_argument("--to-clipboard", action="store_true", help="Copy single rendered file to clipboard")

    # Core workflow: sync
    p_sync = _add_command_parser(
        sub,
        "sync",
        help="Synchronize provider archives",
        description="Synchronize provider archives",
        epilog=_examples_epilog("sync"),
    )
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
    p_sync.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when indexing attachment text",
    )
    add_dry_run_option(p_sync)
    add_force_option(p_sync, help_text="Re-render even if conversations are up-to-date")
    p_sync.add_argument("--prune", action="store_true", help="Remove outputs for conversations that vanished upstream")
    add_collapse_option(p_sync)
    add_html_option(p_sync)
    add_diff_option(p_sync, help_text="Write delta diff alongside updated Markdown")
    p_sync.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_sync.add_argument("--print-paths", action="store_true", help="List written files after sync")
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
    p_sync.add_argument("--offline", action="store_true", help="Skip network-dependent steps (Drive disallowed)")
    # Watch mode flags (local providers only)
    p_sync.add_argument("--watch", action="store_true", help="Watch for changes and sync continuously (local providers only)")
    p_sync.add_argument("--debounce", type=float, default=2.0, help="Minimal seconds between sync runs in watch mode (default: 2.0)")
    p_sync.add_argument("--stall-seconds", type=float, default=60.0, help="Warn when watch makes no progress for this many seconds")
    p_sync.add_argument("--once", action="store_true", help="In watch mode, run a single sync pass and exit")
    p_sync.add_argument("--snapshot", action="store_true", help="Create a rollback snapshot of the output directory before watching")

    # Core workflow: import
    p_import = _add_command_parser(
        sub,
        "import",
        help="Import provider exports",
        description="Import provider exports",
        epilog=_examples_epilog("import"),
    )
    p_import.add_argument("provider", choices=["chatgpt", "claude", "claude-code", "codex"], help="Provider export format")
    p_import.add_argument("source", nargs="*", help="Export path or session identifier (depends on provider); use 'pick', '?', or '-' to trigger interactive picker")
    add_out_option(p_import, default_path=Path("(provider-specific)"), help_text="Override output directory")
    add_collapse_option(p_import)
    add_html_option(p_import)
    p_import.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when importing",
    )
    add_dry_run_option(p_import)
    add_force_option(p_import, help_text="Rewrite even if conversations appear up-to-date")
    p_import.add_argument("--print-paths", action="store_true", help="List written files after import")
    import_selection_group = p_import.add_mutually_exclusive_group()
    import_selection_group.add_argument("--all", action="store_true", help="Process all available items without interactive selection")
    import_selection_group.add_argument("--conversation-id", dest="conversation_ids", action="append", help="Specific conversation ID to import (repeatable)")
    p_import.add_argument("--base-dir", type=Path, default=None, help="Override source directory for codex/claude-code sessions")
    p_import.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p_import.add_argument("--to-clipboard", action="store_true", help="Copy a single imported Markdown file to the clipboard")

    p_browse = _add_command_parser(
        sub,
        "browse",
        help="Browse data (branches/stats/status/runs)",
        description="Explore rendered data and system status",
        epilog=_examples_epilog("browse"),
    )
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
    p_browse_stats.add_argument("--ignore-legacy", action="store_true", help="Ignore legacy *.md files alongside conversation.md")
    p_browse_stats.add_argument(
        "--sort",
        choices=["tokens", "attachments", "attachment-bytes", "words", "recent"],
        default="tokens",
        help="Sort per-file rows before display/export",
    )
    p_browse_stats.add_argument("--limit", type=int, default=0, help="Limit the number of file rows shown/exported (0 shows all)")
    p_browse_stats.add_argument("--csv", type=str, default=None, help="Write per-file rows to CSV ('-' for stdout)")
    p_browse_stats.add_argument("--json-verbose", action="store_true", help="Print warnings/logs alongside --json/--json-lines output")

    p_browse_status = _add_command_parser(
        browse_sub,
        "status",
        help="Show cached Drive info and recent runs",
        description="Show cached Drive info and recent runs",
    )
    _configure_status_parser(p_browse_status)

    p_browse_runs = _add_command_parser(browse_sub, "runs", help="List recent runs", description="List run history with filters")
    p_browse_runs.add_argument("--limit", type=int, default=50, help="Number of runs to display")
    p_browse_runs.add_argument("--providers", type=str, default=None, help="Comma-separated provider filter")
    p_browse_runs.add_argument("--commands", type=str, default=None, help="Comma-separated command filter")
    p_browse_runs.add_argument("--since", type=str, default=None, help="Only include runs on/after this timestamp (YYYY-MM-DD or ISO)")
    p_browse_runs.add_argument("--until", type=str, default=None, help="Only include runs on/before this timestamp")
    p_browse_runs.add_argument("--json", action="store_true", help="Emit runs as JSON")

    p_search = _add_command_parser(
        sub,
        "search",
        help="Search rendered transcripts",
        description="Search rendered transcripts",
        epilog=_examples_epilog("search"),
    )
    p_search.add_argument("query", type=str, help="FTS search query (SQLite syntax); use --from-stdin to read from stdin")
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
    p_search.add_argument("--in-attachments", action="store_true", help="Search within attachment text when indexed")
    p_search.add_argument("--attachment-name", type=str, default=None, help="Filter by attachment filename substring")
    p_search.add_argument("--no-picker", action="store_true", help="Skip skim picker preview even when interactive")
    p_search.add_argument("--json", action="store_true", help="Emit machine-readable search results")
    p_search.add_argument("--json-lines", action="store_true", help="Emit newline-delimited JSON hits (implies --json and disables tables)")
    p_search.add_argument("--csv", type=str, default=None, help="Write search hits to CSV ('-' for stdout)")
    p_search.add_argument(
        "--fields",
        type=str,
        default="provider,conversationId,slug,branchId,messageId,position,timestamp,score,model,attachments,snippet,conversationPath,branchPath",
        help="Comma-separated fields to include in CSV/JSONL output",
    )
    p_search.add_argument("--from-stdin", action="store_true", help="Read the search query from stdin (ignores positional query if present)")
    p_search.add_argument("--open", action="store_true", help="Open result file in $EDITOR after search")

    p_maintain = _add_command_parser(
        sub,
        "maintain",
        help="System maintenance (prune/doctor/index)",
        description="System maintenance and diagnostics",
        epilog=_examples_epilog("maintain"),
    )
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

    p_maintain_restore = _add_command_parser(
        maintain_sub,
        "restore",
        help="Restore a snapshot directory",
        description="Restore a previously snapshotted output directory",
    )
    p_maintain_restore.add_argument("--from", dest="src", type=Path, required=True, help="Snapshot directory to restore from")
    p_maintain_restore.add_argument("--to", dest="dest", type=Path, required=True, help="Destination output directory")
    p_maintain_restore.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    p_maintain_restore.add_argument("--json", action="store_true", help="Emit restoration summary as JSON")

    p_config = _add_command_parser(
        sub,
        "config",
        help="Configuration (init/set/show)",
        description="Configure Polylogue settings",
        epilog=_examples_epilog("config"),
    )
    config_sub = p_config.add_subparsers(dest="config_cmd", required=True)

    p_config_init = config_sub.add_parser("init", help="Interactive configuration setup", description="Interactive configuration setup wizard")
    p_config_init.add_argument("--force", action="store_true", help="Overwrite existing configuration")

    p_config_set = config_sub.add_parser("set", help="Update settings", description="Show or update Polylogue defaults")
    p_config_set.add_argument("--html", choices=["on", "off"], default=None, help="Enable or disable default HTML previews")
    p_config_set.add_argument("--theme", choices=["light", "dark"], default=None, help="Set the default HTML theme")
    p_config_set.add_argument("--collapse-threshold", type=int, default=None, help="Set the default collapse threshold for long outputs")
    p_config_set.add_argument("--output-root", type=Path, default=None, help="Set the output root for archives (overrides config.json)")
    p_config_set.add_argument("--input-root", type=Path, default=None, help="Set the inbox/input root for provider exports (overrides config.json)")
    p_config_set.add_argument("--reset", action="store_true", help="Reset to config defaults")
    p_config_set.add_argument("--json", action="store_true", help="Emit settings as JSON")

    p_config_show = config_sub.add_parser("show", help="Show configuration", description="Show resolved configuration and output paths")
    p_config_show.add_argument("--json", action="store_true", help="Emit environment info as JSON")

    p_attachments = _add_command_parser(
        sub,
        "attachments",
        help="Attachment utilities (stats/extract)",
        description="Inspect and extract attachments",
        epilog=_examples_epilog("attachments"),
    )
    attachments_sub = p_attachments.add_subparsers(dest="attachments_cmd", required=True)

    p_att_stats = _add_command_parser(attachments_sub, "stats", help="Summarize attachments", description="Summarize attachment counts/bytes")
    p_att_stats.add_argument("--dir", type=Path, default=None, help="Root directory containing archives (defaults to all output roots)")
    p_att_stats.add_argument("--ext", type=str, default=None, help="Filter by file extension (e.g., .png)")
    p_att_stats.add_argument("--hash", action="store_true", help="Hash attachments to compute deduped totals")
    p_att_stats.add_argument("--sort", choices=["size", "name"], default="size", help="Sort field for top rows")
    p_att_stats.add_argument("--limit", type=int, default=10, help="Limit number of files displayed (0 for all)")
    p_att_stats.add_argument("--csv", type=str, default=None, help="Write attachment rows to CSV ('-' for stdout)")
    p_att_stats.add_argument("--json", action="store_true", help="Emit stats as JSON")
    p_att_stats.add_argument("--json-lines", action="store_true", help="Emit per-attachment JSONL (implies --json)")
    p_att_stats.add_argument(
        "--from-index",
        action="store_true",
        help="Read attachment metadata from the index DB (includes text/OCR stats when available)",
    )

    p_att_extract = _add_command_parser(attachments_sub, "extract", help="Copy attachments to a directory", description="Extract attachments by extension")
    p_att_extract.add_argument("--dir", type=Path, default=None, help="Root directory containing archives (defaults to all output roots)")
    p_att_extract.add_argument("--ext", type=str, required=True, help="File extension to extract (e.g., .pdf)")
    p_att_extract.add_argument("--out", type=Path, required=True, help="Destination directory for extracted files")
    p_att_extract.add_argument("--limit", type=int, default=0, help="Limit number of files extracted (0 for all)")
    p_att_extract.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files in destination")
    p_att_extract.add_argument("--json", action="store_true", help="Emit extraction summary as JSON")

    p_help_cmd = _add_command_parser(sub, "help", help="Show help for a specific command", description="Show help for a specific command")
    p_help_cmd.add_argument("topic", nargs="?", help="Command name")
    p_help_cmd.add_argument("--examples", action="store_true", help="Show all examples for the topic (or all commands if none specified)")

    p_completions = _add_command_parser(sub, "completions", help="Emit shell completion script", description="Emit shell completion script")
    p_completions.add_argument("--shell", choices=["bash", "zsh", "fish"], required=True)

    p_complete = _add_command_parser(sub, "_complete", help=argparse.SUPPRESS)
    p_complete.add_argument("--shell", required=True)
    p_complete.add_argument("--cword", type=int, required=True)
    p_complete.add_argument("words", nargs=argparse.REMAINDER)

    p_env = _add_command_parser(sub, "env", help="Check environment and config paths", description="Validate Polylogue environment and configuration")
    p_env.add_argument("--json", action="store_true", help="Emit machine-readable output")

    p_search_preview = _add_command_parser(sub, "_search-preview", help=argparse.SUPPRESS)
    p_search_preview.add_argument("--data-file", type=Path, required=True)
    p_search_preview.add_argument("--index", type=int, required=True)

    p_status = _add_command_parser(
        sub,
        "status",
        help="Show cached Drive info and recent runs",
        description="Show cached Drive info and recent runs",
    )
    _configure_status_parser(p_status, require_provider=False)

    p_compare = _add_command_parser(
        sub,
        "compare",
        help="Compare coverage between two providers for a query",
        description="Run the same search against two providers and summarize differences",
    )
    p_compare.add_argument("query", type=str, help="Search query to compare")
    p_compare.add_argument("--provider-a", required=True, help="First provider slug")
    p_compare.add_argument("--provider-b", required=True, help="Second provider slug")
    p_compare.add_argument("--limit", type=int, default=20, help="Maximum hits per provider")
    p_compare.add_argument("--json", action="store_true", help="Emit machine-readable comparison summary")
    p_compare.add_argument("--fields", type=str, default="provider,slug,branchId,messageId,score,snippet,model,path", help="Fields for JSON export")

    p_prefs = _add_command_parser(
        sub,
        "prefs",
        help="Manage per-command preference defaults",
        description="List, set, or clear saved CLI defaults",
        epilog=_examples_epilog("prefs"),
    )
    prefs_sub = p_prefs.add_subparsers(dest="prefs_cmd", required=True)
    _add_command_parser(prefs_sub, "list", help="List stored preferences", description="List stored preferences")
    p_prefs_set = _add_command_parser(prefs_sub, "set", help="Set a preference", description="Set a preference")
    p_prefs_set.add_argument("command", type=str, help="Command name (e.g., search, sync)")
    p_prefs_set.add_argument("--flag", required=True, help="Flag name, e.g., --limit")
    p_prefs_set.add_argument("--value", required=True, help="Value for the flag")
    p_prefs_clear = _add_command_parser(prefs_sub, "clear", help="Clear preferences", description="Clear preferences")
    p_prefs_clear.add_argument("command", type=str, nargs="?", help="Command name to clear; omit to clear all")

    p_open = _add_command_parser(
        sub,
        "open",
        help="Open or print paths from the latest run",
        description="Open or print the most recent output path",
    )
    p_open.add_argument("--provider", type=str, default=None, help="Filter by provider for last run")
    p_open.add_argument("--command", type=str, default=None, help="Filter by command (e.g., sync codex)")
    p_open.add_argument("--print", dest="print_only", action="store_true", help="Only print the path without opening")
    p_open.add_argument("--json", action="store_true", help="Emit JSON with the last run info")
    p_open.add_argument("--fallback", type=Path, default=None, help="Fallback path if no run is found")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plain_mode = _should_use_plain(args)
    ui = create_ui(plain_mode)
    env = CommandEnv(ui=ui)
    # Load prefs into env for downstream defaults
    try:
        from .prefs import _load_prefs

        env.prefs = _load_prefs()
    except Exception:
        env.prefs = {}
    ensure_settings_defaults(env.settings)

    if getattr(args, "profile_io", False):
        os.environ["POLYLOGUE_PROFILE_IO"] = "1"
    if getattr(args, "profile_sql", False):
        os.environ["POLYLOGUE_PROFILE_SQL"] = "1"

    if getattr(args, "allow_dirty", False) and not getattr(args, "force", False):
        ui.console.print("--allow-dirty requires --force")
        raise SystemExit(1)

    # Enable verbose output if requested
    if getattr(args, "verbose", False):
        ui.console.print("[dim]Verbose mode enabled[/dim]")

    if args.cmd is None:
        parser.print_help()
        return

    _register_default_commands()
    cmd = args.cmd
    handler = COMMAND_REGISTRY.resolve(cmd)
    if handler is None:
        parser.print_help()
        return

    try:
        handler(args, env)
    except SystemExit:
        raise
    except Exception as exc:
        _record_failure(args, exc)
        raise


if __name__ == "__main__":
    main()
