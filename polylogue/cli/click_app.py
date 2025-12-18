"""Click-based CLI entrypoint.

This preserves existing behaviours by converting Click parameters into
simple attribute namespaces and deferring to existing `run_*_cli` helpers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import click
from rich.table import Table

from ..commands import CommandEnv
from ..ui import create_ui
from .click_introspect import click_command_entries
from .completions_cli import run_complete_cli, run_completions_cli
from . import (
    attachments,
    browse as browse_cmd,
    imports,
    maintain as maintain_cli,
    open_helper,
    prefs as prefs_cmd,
    reprocess,
)
from .config_cli import run_config_show
from .search_cli import run_search_cli, run_search_preview
from .compare_cli import run_compare_cli
from .click_options import OptionalValueChoiceOption
from .examples import COMMAND_EXAMPLES
from .env_cli import run_env_cli


def _build_env(plain: bool) -> CommandEnv:
    return CommandEnv(ui=create_ui(plain))

_FORCE_PLAIN_VALUES = {"1", "true", "yes", "on"}


def _should_use_plain(*, plain: bool, interactive: bool) -> bool:
    if interactive:
        return False
    if plain:
        return True
    forced = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    if forced and forced.strip().lower() in _FORCE_PLAIN_VALUES:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())


def _print_command_listing(console, plain: bool, entries: Sequence[Tuple[str, str]]) -> None:
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
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for name, description in entries:
        table.add_row(name, description)
    console.print(table)


def _print_examples(console) -> None:
    console.print("[bold cyan]EXAMPLES[/bold cyan]\n")
    for name in sorted(COMMAND_EXAMPLES):
        examples = COMMAND_EXAMPLES[name]
        if not examples:
            continue
        console.print(f"[green]{name}[/green]")
        for desc, cmdline in examples:
            console.print(f"  [dim]{desc}:[/dim] [green]{cmdline}[/green]")
        console.print("")


def _print_quick_examples(console) -> None:
    console.print("\n[bold cyan]QUICK EXAMPLES[/bold cyan]")
    console.print("[dim]Run 'polylogue help <command>' for full examples and details.[/dim]\n")
    for cmd in ["render", "sync", "import", "search", "browse"]:
        examples = COMMAND_EXAMPLES.get(cmd)
        if not examples:
            continue
        desc, cmdline = examples[0]
        console.print(f"  [dim]{desc}:[/dim] [green]{cmdline}[/green]")


def _run_click_help(env: CommandEnv, ctx: click.Context, topic: Optional[str], examples: bool) -> None:
    console = env.ui.console
    root_ctx = ctx.find_root()
    group: click.Group = root_ctx.command  # type: ignore[assignment]
    entries = click_command_entries(group)
    show_examples_only = examples and not topic

    if topic:
        command = group.get_command(root_ctx, topic)
        if command is None or command.hidden:
            available = ", ".join(name for name, _ in entries) or "<none>"
            console.print(f"[red]Unknown command: {topic}")
            console.print(f"Available commands: {available}")
            raise SystemExit(1)
        sub_ctx = click.Context(command, info_name=topic, parent=root_ctx)
        console.print(f"[cyan]polylogue {topic}[/cyan]")
        console.print(command.get_help(sub_ctx))
        return

    if show_examples_only:
        _print_examples(console)
        return

    console.print(group.get_help(root_ctx))
    _print_command_listing(console, getattr(env.ui, "plain", False), entries)
    _print_quick_examples(console)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--plain", is_flag=True, help="Force plain/CI-safe output even when running in a TTY")
@click.option("--interactive", is_flag=True, help="Force interactive UI even when stdout/stderr are not TTYs")
@click.pass_context
def cli(ctx: click.Context, plain: bool, interactive: bool) -> None:
    """Polylogue CLI (Click)."""
    use_plain = _should_use_plain(plain=plain, interactive=interactive)
    ctx.obj = _build_env(use_plain)


# ---------------------------- sync ----------------------------


@cli.command()
@click.argument("provider", type=click.Choice(["drive", "codex", "claude-code", "chatgpt", "claude"]))
@click.option("--out", type=click.Path(path_type=Path), help="Override output directory")
@click.option("--links-only", is_flag=True, help="Link attachments instead of downloading (Drive only)")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.option("--sanitize-html", is_flag=True, help="Mask emails/keys/tokens in synced Markdown/HTML outputs")
@click.option("--dry-run", is_flag=True, help="Report actions without writing files")
@click.option("--force", is_flag=True, help="Re-render even if conversations are up-to-date")
@click.option("--prune", is_flag=True, help="Remove outputs for conversations that vanished upstream")
@click.option("--prune-snapshot", is_flag=True, help="Snapshot outputs before pruning (STATE_HOME/rollback)")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option(
    "--html",
    "html_mode",
    cls=OptionalValueChoiceOption,
    is_flag=False,
    flag_value="on",
    type=click.Choice(["on", "off", "auto"]),
    default="auto",
    show_default=True,
    help="HTML preview mode: on/off/auto (default auto)",
)
@click.option("--diff", is_flag=True, help="Write delta diff alongside updated Markdown")
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--print-paths", is_flag=True, help="List written files after sync")
@click.option("--chat-id", "chat_ids", multiple=True, help="Drive chat/file ID to sync (repeatable)")
@click.option("--session", "sessions", multiple=True, type=click.Path(path_type=Path), help="Local session/export path to sync (repeatable)")
@click.option("--all", is_flag=True, help="Process all available items without interactive selection")
@click.option("--base-dir", type=click.Path(path_type=Path), help="Override local session/export directory")
@click.option("--folder-name", type=str, help="Drive folder name (drive provider)")
@click.option("--folder-id", type=str, help="Drive folder ID override")
@click.option("--since", type=str, help="Only include Drive chats updated on/after this timestamp")
@click.option("--until", type=str, help="Only include Drive chats updated on/before this timestamp")
@click.option("--name-filter", type=str, help="Regex filter for Drive chat names")
@click.option("--list-only", is_flag=True, help="List Drive chats without syncing")
@click.option("--offline", is_flag=True, help="Skip network-dependent steps (Drive disallowed)")
@click.option("--watch", is_flag=True, help="Watch for changes and sync continuously (local providers only)")
@click.option("--jobs", type=int, default=1, show_default=True, help="Parallelism for local providers (codex/claude-code)")
@click.option("--debounce", type=float, default=2.0, show_default=True, help="Minimal seconds between sync runs in watch mode")
@click.option("--stall-seconds", type=float, default=60.0, show_default=True, help="Warn when watch makes no progress for this many seconds")
@click.option("--fail-on-stall", is_flag=True, help="Exit with non-zero status when watch detects a stall")
@click.option("--tail", is_flag=True, help="Log changed paths as they are detected in watch mode")
@click.option("--once", is_flag=True, help="In watch mode, run a single sync pass and exit")
@click.option("--snapshot", is_flag=True, help="Create a rollback snapshot of the output directory before watching")
@click.option("--watch-plan", is_flag=True, help="Print the assembled watch command and exit (no watch run)")
@click.option("--drive-retries", type=int, help="Override Drive retry attempts (default: config or 3)")
@click.option("--drive-retry-base", type=float, help="Override Drive retry base delay seconds (default: config or 0.5)")
@click.option("--resume-from", type=int, help="Resume a previous run by run ID (reprocess failed items only)")
@click.option("--meta", multiple=True, help="Attach custom metadata key=value (repeatable)")
@click.option("--root", type=str, help="Named root label to use when configs support multi-root archives")
@click.option("--max-disk", type=float, help="Abort if projected disk use exceeds this many GiB (approx)")
@click.pass_obj
def sync(env: CommandEnv, **kwargs) -> None:
    """Synchronize provider archives (use --watch for continuous mode)."""
    from .sync import run_sync_cli

    run_sync_cli(SimpleNamespace(**kwargs), env)


# ---------------------------- import ----------------------------


@cli.command(name="import")
@click.argument("provider", type=click.Choice(["chatgpt", "claude", "claude-code", "codex"]))
@click.argument("source", nargs=-1, type=click.Path(path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Output directory")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option(
    "--html",
    "html_mode",
    cls=OptionalValueChoiceOption,
    is_flag=False,
    flag_value="on",
    type=click.Choice(["on", "off", "auto"]),
    default="auto",
    show_default=True,
    help="HTML preview mode: on/off/auto (default auto)",
)
@click.option("--force", is_flag=True, help="Regenerate markdown from database instead of reading source files")
@click.option("--all", is_flag=True, help="Process all conversations/sessions without selection")
@click.option("--conversation-id", "conversation_ids", multiple=True, help="Conversation ID filter (repeatable)")
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--print-paths", is_flag=True, help="List written files after import")
@click.option("--to-clipboard", is_flag=True, help="Copy single imported Markdown file to the clipboard")
@click.option("--base-dir", type=click.Path(path_type=Path), help="Base directory for local providers")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.option("--sanitize-html", is_flag=True, help="Mask emails/keys/tokens in imported Markdown/HTML outputs")
@click.option("--meta", multiple=True, help="Attach custom metadata key=value (repeatable)")
@click.pass_obj
def import_cmd_click(env: CommandEnv, **kwargs) -> None:
    """Import provider exports."""
    args = SimpleNamespace(**kwargs)
    imports.run_import_cli(args, env)


# ---------------------------- render ----------------------------


@cli.command()
@click.argument("input", type=click.Path(path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Output directory")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option(
    "--html",
    "html_mode",
    cls=OptionalValueChoiceOption,
    is_flag=False,
    flag_value="on",
    type=click.Choice(["on", "off", "auto"]),
    default="auto",
    show_default=True,
    help="HTML preview mode: on/off/auto (default auto)",
)
@click.option("--force", is_flag=True, help="Overwrite outputs even when up-to-date")
@click.option("--allow-dirty", is_flag=True, help="Allow overwriting files with local edits (requires --force)")
@click.option("--links-only", is_flag=True, help="Link attachments instead of downloading")
@click.option("--diff", is_flag=True, help="Write delta diff alongside updated Markdown")
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--print-paths", is_flag=True, help="List written files after rendering")
@click.option("--to-clipboard", is_flag=True, help="Copy a single rendered file to the clipboard")
@click.option("--dry-run", is_flag=True, help="Report actions without writing files")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.option("--max-disk", type=float, help="Abort if projected disk use exceeds this many GiB (approx)")
@click.option("--sanitize-html", is_flag=True, help="Mask emails/keys/tokens in rendered Markdown/HTML outputs")
@click.option("--meta", multiple=True, help="Attach custom metadata key=value (repeatable)")
@click.pass_obj
def render(env: CommandEnv, **kwargs) -> None:
    """Render JSON exports to Markdown/HTML."""
    if kwargs.get("allow_dirty") and not kwargs.get("force"):
        env.ui.console.print("--allow-dirty requires --force")
        raise SystemExit(1)
    from .render import run_render_cli
    from .render_force import run_render_force

    args = SimpleNamespace(**kwargs)
    if getattr(args, "force", False):
        exit_code = run_render_force(env, provider=None, conversation_id=None, output_dir=getattr(args, "out", None))
        raise SystemExit(exit_code)
    run_render_cli(args, env, json_output=getattr(args, "json", False))


# ---------------------------- verify ----------------------------


@cli.command()
@click.option("--provider", type=str, help="Comma-separated provider filter")
@click.option("--slug", type=str, help="Filter to a single slug")
@click.option("--conversation-id", "conversation_ids", multiple=True, help="Filter to a conversation ID (repeatable)")
@click.option("--limit", type=int, help="Limit number of conversations verified")
@click.option("--fix", is_flag=True, help="Rewrite conversation.md front matter into canonical form when possible")
@click.option(
    "--unknown",
    "unknown_policy",
    type=click.Choice(["ignore", "warn", "error"]),
    default="warn",
    show_default=True,
    help="How to handle unknown polylogue front-matter keys",
)
@click.option("--allow-polylogue-key", "allow_polylogue_keys", multiple=True, help="Allow additional polylogue metadata key (repeatable)")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--json", is_flag=True, help="Emit machine-readable report")
@click.pass_obj
def verify(env: CommandEnv, **kwargs) -> None:
    """Verify rendered outputs against the database and state store."""
    from .verify import run_verify_cli

    run_verify_cli(SimpleNamespace(**kwargs), env)


# ---------------------------- search ----------------------------


@cli.command()
@click.argument("query", required=False)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum number of hits to return")
@click.option("--provider", type=str, help="Filter by provider slug")
@click.option("--slug", type=str, help="Filter by conversation slug")
@click.option("--conversation-id", type=str, help="Filter by provider conversation id")
@click.option("--branch", type=str, help="Restrict to a single branch ID")
@click.option("--model", type=str, help="Filter by source model when recorded")
@click.option("--since", type=str, help="Only include messages on/after this timestamp")
@click.option("--until", type=str, help="Only include messages on/before this timestamp")
@click.option("--with-attachments", is_flag=True, help="Limit to messages with extracted attachments")
@click.option("--without-attachments", is_flag=True, help="Limit to messages without attachments")
@click.option("--in-attachments", is_flag=True, help="Search within attachment text when indexed")
@click.option("--attachment-name", type=str, help="Filter by attachment filename substring")
@click.option("--no-picker", is_flag=True, help="Skip skim picker preview even when interactive")
@click.option("--json", is_flag=True, help="Emit machine-readable search results")
@click.option("--json-lines", is_flag=True, help="Emit newline-delimited JSON hits (implies --json and disables tables)")
@click.option("--csv", type=str, help="Write search hits to CSV ('-' for stdout)")
@click.option("--fields", type=str, default="provider,conversationId,slug,branchId,messageId,position,timestamp,score,model,attachments,snippet,conversationPath,branchPath", show_default=True)
@click.option("--from-stdin", is_flag=True, help="Read the search query from stdin (ignores positional query if present)")
@click.option("--open", "open_result", is_flag=True, help="Open result file in $EDITOR after search")
@click.pass_obj
def search(env: CommandEnv, **kwargs) -> None:
    """Search rendered transcripts."""
    kwargs["open"] = kwargs.pop("open_result")
    args = SimpleNamespace(**kwargs)
    run_search_cli(args, env)


# ---------------------------- status ----------------------------


@cli.command()
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--json-lines", is_flag=True, help="Stream newline-delimited JSON records (auto-enables --json, useful with --watch)")
@click.option("--json-verbose", is_flag=True, help="Allow status to print tables/logs alongside JSON/JSONL output")
@click.option("--watch", is_flag=True, help="Continuously refresh the status output")
@click.option("--dump-only", is_flag=True, help="Only perform the dump action without printing summaries")
@click.option("--interval", type=float, default=5.0, show_default=True, help="Seconds between refresh while watching")
@click.option("--dump", type=str, help="Write recent runs to a file ('-' for stdout)")
@click.option("--dump-limit", type=int, default=100, show_default=True, help="Number of runs to include when dumping")
@click.option("--runs-limit", type=int, default=200, show_default=True, help="Number of historical runs to include in summaries")
@click.option("--top", type=int, default=0, show_default=True, help="Show top runs by attachments/tokens")
@click.option("--inbox", is_flag=True, help="Include inbox coverage counts in summaries")
@click.option("--providers", type=str, help="Comma-separated provider filter")
@click.option("--quiet", is_flag=True, help="Suppress table output (useful with --json-lines)")
@click.option("--summary", type=str, help="Write aggregated provider/run summary JSON to a file ('-' for stdout)")
@click.option("--summary-only", is_flag=True, help="Only emit the summary JSON without printing tables")
@click.pass_obj
def status(env: CommandEnv, **kwargs) -> None:
    """Show cached Drive info and recent runs."""
    from .status import run_status_cli

    run_status_cli(SimpleNamespace(**kwargs), env)


# ---------------------------- compare ----------------------------


@cli.command()
@click.argument("query", type=str)
@click.option("--provider-a", required=True, help="First provider slug")
@click.option("--provider-b", required=True, help="Second provider slug")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum hits per provider")
@click.option("--json", is_flag=True, help="Emit machine-readable comparison summary")
@click.option("--fields", type=str, default="provider,slug,branchId,messageId,score,snippet,model,path", show_default=True, help="Fields for JSON export")
@click.pass_obj
def compare(env: CommandEnv, **kwargs) -> None:
    """Compare coverage between two providers for a query."""
    args = SimpleNamespace(**kwargs)
    run_compare_cli(args, env)

# ---------------------------- browse ----------------------------


@cli.group(name="browse")
@click.pass_context
def browse_group(ctx: click.Context) -> None:
    """Browse commands (branches/stats/status/runs/inbox)."""


@browse_group.command(name="branches")
@click.option("--provider", type=str, help="Filter by provider slug")
@click.option("--slug", type=str, help="Filter by conversation slug")
@click.option("--conversation-id", type=str, help="Filter by provider conversation id")
@click.option("--min-branches", type=int, default=1, show_default=True, help="Only include conversations with at least this many branches")
@click.option("--branch", type=str, help="Branch ID to inspect or diff against the canonical path")
@click.option("--diff", is_flag=True, help="Display a unified diff between a branch and canonical transcript")
@click.option("--out", type=click.Path(path_type=Path), help="Write the branch explorer HTML to this path")
@click.option("--theme", type=click.Choice(["light", "dark"]), help="Override HTML explorer theme")
@click.option("--no-picker", is_flag=True, help="Skip interactive selection even when skim/gum are available")
@click.option("--open", "open_result", is_flag=True, help="Open result in $EDITOR after command completes")
@click.pass_obj
def browse_branches(env: CommandEnv, **kwargs) -> None:
    kwargs["open"] = kwargs.pop("open_result")
    args = SimpleNamespace(browse_cmd="branches", **kwargs)
    browse_cmd.run_browse_cli(args, env)


@browse_group.command(name="stats")
@click.option("--dir", type=click.Path(path_type=Path), help="Directory containing Markdown exports")
@click.option("--provider", type=str, help="Provider filter")
@click.option("--ignore-legacy", is_flag=True, help="Ignore legacy *.md files alongside conversation.md")
@click.option("--sort", type=click.Choice(["tokens", "attachments", "attachment-bytes", "words", "recent"]), default="tokens")
@click.option("--limit", type=int, default=0)
@click.option("--csv", type=str, help="Write per-file rows to CSV ('-' for stdout)")
@click.option("--json", is_flag=True)
@click.option("--json-lines", is_flag=True)
@click.option("--json-verbose", is_flag=True)
@click.option("--since", type=str, help="Only include files on/after this timestamp")
@click.option("--until", type=str, help="Only include files on/before this timestamp")
@click.pass_obj
def browse_stats(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(browse_cmd="stats", **kwargs)
    browse_cmd.run_browse_cli(args, env)


@browse_group.command(name="status")
@click.option("--json", is_flag=True)
@click.option("--json-lines", is_flag=True)
@click.option("--json-verbose", is_flag=True)
@click.option("--watch", is_flag=True)
@click.option("--interval", type=float, default=5.0, show_default=True)
@click.option("--dump", type=str, help="Write recent runs to a file ('-' for stdout)")
@click.option("--dump-limit", type=int, default=100, show_default=True)
@click.option("--runs-limit", type=int, default=200, show_default=True)
@click.option("--top", type=int, default=0, show_default=True)
@click.option("--inbox", is_flag=True)
@click.option("--providers", type=str, help="Comma-separated provider filter")
@click.option("--quiet", is_flag=True)
@click.option("--summary", type=str, help="Write aggregated provider/run summary JSON to a file ('-' for stdout)")
@click.option("--summary-only", is_flag=True)
@click.pass_obj
def browse_status(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(browse_cmd="status", **kwargs)
    browse_cmd.run_browse_cli(args, env)


@browse_group.command(name="runs")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--providers", type=str, help="Comma-separated provider filter")
@click.option("--commands", type=str, help="Comma-separated command filter")
@click.option("--since", type=str, help="Only include runs on/after this timestamp")
@click.option("--until", type=str, help="Only include runs on/before this timestamp")
@click.option("--json", is_flag=True)
@click.option("--json-lines", is_flag=True, help="Emit newline-delimited JSON records (implies --json)")
@click.option("--json-verbose", is_flag=True)
@click.pass_obj
def browse_runs(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(browse_cmd="runs", **kwargs)
    browse_cmd.run_browse_cli(args, env)


@browse_group.command(name="metrics")
@click.option("--providers", type=str, help="Comma-separated provider filter")
@click.option("--runs-limit", type=int, default=0, show_default=True, help="Limit historical runs considered (0 = all)")
@click.option("--json", is_flag=True, help="Emit JSON instead of Prometheus text format")
@click.option("--serve", is_flag=True, help="Serve metrics over HTTP at /metrics (Prometheus format)")
@click.option("--host", type=str, default="127.0.0.1", show_default=True, help="Host to bind when serving metrics")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to bind when serving metrics")
@click.pass_obj
def browse_metrics(env: CommandEnv, **kwargs) -> None:
    """Export Prometheus-friendly metrics from Polylogue state/run history."""
    args = SimpleNamespace(browse_cmd="metrics", **kwargs)
    browse_cmd.run_browse_cli(args, env)


@browse_group.command(name="inbox")
@click.option("--providers", type=str, default="chatgpt,claude", show_default=True, help="Comma-separated provider filter")
@click.option("--dir", type=click.Path(path_type=Path), help="Override inbox root for a generic scan")
@click.option("--quarantine", is_flag=True, help="Move unknown/malformed inbox items into a quarantine folder")
@click.option("--quarantine-dir", type=click.Path(path_type=Path), help="Target directory for quarantined items")
@click.option("--json", is_flag=True)
@click.pass_obj
def browse_inbox(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(browse_cmd="inbox", **kwargs)
    browse_cmd.run_browse_cli(args, env)


# ---------------------------- maintain ----------------------------


@cli.group()
@click.pass_context
def maintain(ctx: click.Context) -> None:
    """System maintenance and diagnostics."""


@maintain.command(name="prune")
@click.option("--dir", "dirs", multiple=True, type=click.Path(path_type=Path), help="Root directory to prune")
@click.option("--dry-run", is_flag=True, help="Print planned actions without deleting files")
@click.option("--max-disk", type=float, help="Abort if projected snapshot size exceeds this many GiB")
@click.pass_obj
def maintain_prune(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(maintain_cmd="prune", **kwargs)
    maintain_cli.run_maintain_cli(args, env)


@maintain.command(name="doctor")
@click.option("--codex-dir", type=click.Path(path_type=Path), help="Override Codex sessions directory")
@click.option("--claude-code-dir", type=click.Path(path_type=Path), help="Override Claude Code projects directory")
@click.option("--limit", type=int, help="Limit number of files inspected per provider")
@click.option("--json", is_flag=True, help="Emit machine-readable report")
@click.option("--json-verbose", is_flag=True, help="Emit JSON with verbose details")
@click.pass_obj
def maintain_doctor(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(maintain_cmd="doctor", **kwargs)
    maintain_cli.run_maintain_cli(args, env)


@maintain.command(name="index")
@click.argument("subcmd", type=click.Choice(["check"]))
@click.option("--repair", is_flag=True, help="Attempt to rebuild missing SQLite FTS data")
@click.option("--skip-qdrant", is_flag=True, help="Skip Qdrant validation even when configured")
@click.option("--json", is_flag=True, help="Emit validation results as JSON")
@click.option("--json-verbose", is_flag=True, help="Emit JSON with verbose details")
@click.pass_obj
def maintain_index(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(maintain_cmd="index", subcmd=kwargs.pop("subcmd"), **kwargs)
    maintain_cli.run_maintain_cli(args, env)


@maintain.command(name="restore")
@click.option("--from", "src", type=click.Path(path_type=Path), required=True, help="Snapshot directory to restore from")
@click.option("--to", "dest", type=click.Path(path_type=Path), required=True, help="Destination output directory")
@click.option("--force", is_flag=True, help="Overwrite destination if it exists")
@click.option("--json", is_flag=True, help="Emit restoration summary as JSON")
@click.option("--max-disk", type=float, help="Abort if projected snapshot size exceeds this many GiB")
@click.pass_obj
def maintain_restore(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(maintain_cmd="restore", **kwargs)
    maintain_cli.run_maintain_cli(args, env)


# ---------------------------- config/env/prefs ----------------------------


@cli.command()
@click.option("--json", is_flag=True, help="Emit environment info as JSON")
@click.pass_obj
def env(env: CommandEnv, **kwargs) -> None:
    """Check environment and config paths."""
    run_env_cli(SimpleNamespace(**kwargs), env)


@cli.command(name="help")
@click.argument("topic", required=False)
@click.option(
    "--examples",
    is_flag=True,
    help="Show all examples for the topic (or all commands if none specified)",
)
@click.pass_context
def help_cmd(ctx: click.Context, topic: Optional[str], examples: bool) -> None:  # type: ignore[func-returns-value]
    env = ctx.ensure_object(CommandEnv)
    _run_click_help(env, ctx, topic, examples)


@cli.command()
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), required=True)
@click.pass_obj
def completions(env: CommandEnv, **kwargs) -> None:
    """Emit shell completion script."""
    args = SimpleNamespace(**kwargs)
    run_completions_cli(args, env, cli)


@cli.command(name="_complete", hidden=True)
@click.option("--shell", required=True)
@click.option("--cword", type=int, required=True)
@click.argument("words", nargs=-1)
@click.pass_obj
def _complete(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(**kwargs)
    run_complete_cli(args, env, cli)


@cli.command(name="_search-preview", hidden=True)
@click.option("--data-file", type=click.Path(path_type=Path), required=True)
@click.option("--index", type=int, required=True)
@click.pass_obj
def _search_preview(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(**kwargs)
    run_search_preview(args)


@cli.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration (init/set/show)."""


@config.command(name="show")
@click.option("--json", is_flag=True, help="Emit environment info as JSON")
@click.pass_obj
def config_show(env: CommandEnv, **kwargs) -> None:
    run_config_show(SimpleNamespace(**kwargs), env)


@config.command(name="set")
@click.option("--html", type=click.Choice(["on", "off"]), help="Enable or disable default HTML previews")
@click.option("--theme", type=click.Choice(["light", "dark"]), help="Set the default HTML theme")
@click.option("--collapse-threshold", type=int, help="Set the default collapse threshold")
@click.option("--output-root", type=click.Path(path_type=Path), help="Set the output root for archives")
@click.option("--input-root", type=click.Path(path_type=Path), help="Set the inbox/input root for provider exports")
@click.option("--reset", is_flag=True, help="Reset to config defaults")
@click.option("--json", is_flag=True, help="Emit settings as JSON")
@click.pass_obj
def config_set(env: CommandEnv, **kwargs) -> None:
    from .settings_cli import run_settings_cli

    run_settings_cli(SimpleNamespace(**kwargs), env)


@config.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
@click.pass_obj
def config_init(env: CommandEnv, **kwargs) -> None:
    from .init import run_init_cli

    run_init_cli(SimpleNamespace(**kwargs), env)


@config.command(name="edit")
@click.pass_obj
def config_edit(env: CommandEnv, **kwargs) -> None:
    """Interactively edit configuration."""
    from .config_editor import run_config_edit_cli

    run_config_edit_cli(SimpleNamespace(**kwargs), env)


@cli.command()
@click.argument("subcmd", type=click.Choice(["list", "set", "clear"]))
@click.option("--command", "command_name", type=str, help="Command name (e.g., search, sync)")
@click.option("--flag", type=str, help="Flag name, e.g., --limit")
@click.option("--value", type=str, help="Value for the flag")
@click.option("--json", "json_mode", is_flag=True, help="Emit JSON output")
@click.pass_obj
def prefs(env: CommandEnv, subcmd: str, command_name: Optional[str], flag: Optional[str], value: Optional[str], json_mode: bool) -> None:  # type: ignore[func-returns-value]
    """Manage per-command preference defaults."""
    args = SimpleNamespace(prefs_cmd=subcmd, command=command_name, flag=flag, value=value, json=json_mode)
    prefs_cmd.run_prefs_cli(args, env)


# ---------------------------- attachments ----------------------------


@cli.group(name="attachments")
@click.pass_context
def attachments_group(ctx: click.Context) -> None:
    """Attachment utilities."""


@attachments_group.command(name="stats")
@click.option("--dir", type=click.Path(path_type=Path), help="Root directory containing archives")
@click.option("--provider", type=str, help="Filter by provider (comma-separated)")
@click.option("--since", type=str, help="Only include attachments on/after this timestamp (index only)")
@click.option("--until", type=str, help="Only include attachments on/before this timestamp (index only)")
@click.option("--ext", type=str, help="Filter by file extension")
@click.option("--hash", "hash", is_flag=True, help="Hash attachments to compute deduped totals")
@click.option("--sort", type=click.Choice(["size", "name"]), default="size")
@click.option("--limit", type=int, default=10)
@click.option("--csv", type=str, help="Write attachment rows to CSV")
@click.option("--json", is_flag=True)
@click.option("--json-lines", is_flag=True)
@click.option("--from-index", is_flag=True, help="Read attachment metadata from the index DB")
@click.option("--clean-orphans", is_flag=True, help="Remove on-disk attachments not referenced by the index DB (requires --from-index)")
@click.option("--dry-run", is_flag=True, help="Preview actions without deleting files (used with --clean-orphans)")
@click.pass_obj
def attachments_stats(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(attachments_cmd="stats", **kwargs)
    attachments.run_attachments_cli(args, env)


@attachments_group.command(name="extract")
@click.option("--dir", type=click.Path(path_type=Path), help="Root directory containing archives")
@click.option("--ext", type=str, required=True, help="File extension to extract (e.g., .pdf)")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Destination directory")
@click.option("--limit", type=int, default=0, help="Limit number of files extracted (0 for all)")
@click.option("--overwrite", is_flag=True, help="Allow overwriting existing files in destination")
@click.option("--json", is_flag=True, help="Emit extraction summary as JSON")
@click.option("--json-lines", is_flag=True, help="Emit per-file JSONL")
@click.pass_obj
def attachments_extract(env: CommandEnv, **kwargs) -> None:
    args = SimpleNamespace(attachments_cmd="extract", **kwargs)
    attachments.run_attachments_cli(args, env)


# ---------------------------- misc ----------------------------


@cli.command()
@click.argument("provider", required=False)
@click.option("--command", "cmd", type=str, help="Filter by command (e.g., sync codex)")
@click.option("--print", "print_only", is_flag=True, help="Only print the path without opening")
@click.option("--json", is_flag=True, help="Emit JSON with the last run info")
@click.option("--fallback", type=click.Path(path_type=Path), help="Fallback path if no run is found")
@click.pass_obj
def open(env: CommandEnv, **kwargs) -> None:  # type: ignore[func-returns-value]
    """Open or print paths from the latest run."""
    args = SimpleNamespace(**kwargs)
    open_helper.run_open_cli(args, env)


@cli.command()
@click.option("--provider", type=str, help="Provider filter for reprocess")
@click.option("--fallback", is_flag=True, help="Use fallback parser")
@click.pass_obj
def reprocess_cmd(env: CommandEnv, **kwargs) -> None:
    """Reprocess failed imports."""
    args = SimpleNamespace(**kwargs)
    reprocess.run_reprocess_cli(args, env)


def main() -> None:  # pragma: no cover
    try:
        cli.main(prog_name="polylogue", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        raise SystemExit(1) from exc


__all__ = ["cli", "main"]
