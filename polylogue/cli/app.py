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
import webbrowser
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..version import POLYLOGUE_VERSION, SCHEMA_VERSION
from ..schema import stamp_payload

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
from .inbox import run_inbox_cli
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

# Command modules
from .commands import (
    sync as sync_cmd,
    render as render_cmd,
    config as config_cmd,
    attachments as attachments_cmd,
    import_cmd,
    search as search_cmd,
    maintain as maintain_cmd,
    status as status_cmd,
    browse as browse_cmd,
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
from ..schema import stamp_payload
from .imports import (
    run_import_cli,
    run_import_chatgpt,
    run_import_claude,
    run_import_claude_code,
    run_import_codex,
)
from .reprocess import run_reprocess
from .render_force import run_render_force
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


class ExamplesHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter that appends command examples from COMMAND_EXAMPLES."""

    def format_help(self) -> str:  # pragma: no cover - formatting path
        help_text = super().format_help()
        # Only append when the parser name exists in COMMAND_EXAMPLES
        prog = getattr(self, "_prog", None) or getattr(getattr(self, "_parser", None), "prog", "")
        # Map parser prog tokens like "polylogue search" to the subcommand key
        if prog:
            tokens = prog.split()
            if len(tokens) >= 2:
                cmd = tokens[1]
                examples = COMMAND_EXAMPLES.get(cmd)
                if examples:
                    lines = ["\nExamples:"]
                    for desc, cmdline in examples:
                        lines.append(f"  # {desc}")
                        lines.append(f"  $ {cmdline}")
                    help_text = help_text.rstrip() + "\n" + "\n".join(lines) + "\n"
        return help_text


def _add_command_parser(subparsers: argparse._SubParsersAction, name: str, **kwargs):
    if "epilog" not in kwargs:
        epilog = _examples_epilog(name)
        if epilog:
            kwargs["epilog"] = epilog
    kwargs.setdefault("formatter_class", ExamplesHelpFormatter)
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
        record = stamp_payload(
            {
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
        )
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
                payload = stamp_payload({k: row.get(k) for k in export_fields} if export_fields else row)
                print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            return
        payload = stamp_payload(
            {
                "query": options.query,
                "count": len(hits),
                "hits": [
                    {k: row.get(k) for k in export_fields} if export_fields else row
                    for row in rows
                ],
            }
        )
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
            target_obj = Path(target_path)
            is_html = target_obj.suffix.lower() == ".html"
            if is_html and anchor_label:
                fragment = f"#${anchor_label}" if anchor_label else ""
                try:
                    webbrowser.open(target_obj.as_uri() + f"#{anchor_label}")
                    ui.console.print(f"[dim]Opened {target_obj}#{anchor_label} in browser[/dim]")
                    return
                except Exception:
                    pass
            if open_in_editor(target_obj, line=line_hint):
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
        payload = stamp_payload(
            {
                "query": args.query,
                "limit": limit,
                "providers": [
                    _compare_hits(args.provider_a, hits_a, fields),
                    _compare_hits(args.provider_b, hits_b, fields),
                ],
            }
        )
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
    # Check if --force flag is set to regenerate from database
    if getattr(args, "force", False):
        # Extract provider and conversation_id from input if provided
        provider = None
        conversation_id = None
        output_dir = getattr(args, "out", None)

        # Call render_force command
        exit_code = run_render_force(
            env,
            provider=provider,
            conversation_id=conversation_id,
            output_dir=output_dir
        )
        raise SystemExit(exit_code)

    # Normal render flow
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
        roots_map = getattr(env.config.defaults, "roots", {}) or {}
        payload = stamp_payload(
            {
                "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
                "settingsPath": str(SETTINGS_PATH),
                "ui": {
                    "html_previews": settings.html_previews,
                    "html_theme": settings.html_theme,
                    "collapse_threshold": settings.collapse_threshold,
                },
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
                    "roots": {label: vars(paths) for label, paths in roots_map.items()},
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
        )
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
    ])
    roots_map = getattr(env.config.defaults, "roots", {}) or {}
    if roots_map:
        summary_lines.append("  labeled roots:")
        for label, paths in roots_map.items():
            summary_lines.append(f"    {label}: render={paths.render} codex={paths.sync_codex}")
    summary_lines.extend([
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


def _dispatch_reprocess(args: argparse.Namespace, env: CommandEnv) -> None:
    """Reprocess failed imports with optional fallback parser."""
    provider = getattr(args, "provider", None)
    use_fallback = getattr(args, "fallback", False)

    exit_code = run_reprocess(
        env,
        provider=provider,
        use_fallback=use_fallback
    )
    raise SystemExit(exit_code)


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
    _ensure("reprocess", _dispatch_reprocess, "Reprocess failed imports")
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
    parser.add_argument("--top", type=int, default=0, help="Show top runs by attachments/tokens")
    parser.add_argument("--inbox", action="store_true", help="Include inbox coverage counts in summaries")
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

    # Helper dict for command modules
    add_helpers = {
        "out_option": add_out_option,
        "dry_run_option": add_dry_run_option,
        "force_option": add_force_option,
        "collapse_option": add_collapse_option,
        "html_option": add_html_option,
        "diff_option": add_diff_option,
        "examples_epilog": _examples_epilog,
    }

    # Setup command parsers using modular command files
    render_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    sync_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    import_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    browse_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    search_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    maintain_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    config_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    attachments_cmd.setup_parser(sub, _add_command_parser, add_helpers)
    status_cmd.setup_parser(sub, _add_command_parser, add_helpers)

    # Utility/meta commands (not yet modularized)
    # Reprocess command for failed imports
    p_reprocess = _add_command_parser(
        sub,
        "reprocess",
        help="Reprocess failed imports",
        description="Reprocess imports that failed during initial parsing",
        epilog=_examples_epilog("reprocess"),
    )
    p_reprocess.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Filter by provider (chatgpt, claude, etc.)",
    )
    p_reprocess.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback heuristic parser for extraction",
    )
    p_reprocess.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary",
    )

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
