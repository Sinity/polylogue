"""Click-based CLI entrypoint that wraps existing command dispatchers.

This preserves current behaviours by converting Click parameters into
argparse-style Namespaces and deferring to the existing dispatch
functions under polylogue.cli.commands.*.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import click

from ..commands import CommandEnv
from ..ui import create_ui
from . import (
    attachments,
    browse,
    config as config_cmd,
    env_cli,
    imports,
    maintain,
    open_helper,
    prefs as prefs_cmd,
    render as render_cmd,
    reprocess,
    search as search_cmd,
    status as status_cmd,
    sync as sync_cmd,
)


def _build_env(plain: bool) -> CommandEnv:
    return CommandEnv(ui=create_ui(plain))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--plain", is_flag=True, help="Force plain/CI-safe output even when running in a TTY")
@click.option("--interactive", is_flag=True, help="Force interactive UI even when stdout/stderr are not TTYs")
@click.pass_context
def cli(ctx: click.Context, plain: bool, interactive: bool) -> None:
    """Polylogue CLI (Click)."""
    use_plain = plain or (not sys.stdout.isatty() or not sys.stderr.isatty()) and not interactive
    ctx.obj = _build_env(use_plain)


# ---------------------------- sync ----------------------------


@cli.command()
@click.argument("provider", type=click.Choice(["drive", "codex", "claude-code", "chatgpt", "claude"]))
@click.option("--out", type=click.Path(path_type=Path), help="Override output directory")
@click.option("--links-only", is_flag=True, help="Link attachments instead of downloading (Drive only)")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.option("--dry-run", is_flag=True, help="Report actions without writing files")
@click.option("--force", is_flag=True, help="Re-render even if conversations are up-to-date")
@click.option("--prune", is_flag=True, help="Remove outputs for conversations that vanished upstream")
@click.option("--prune-snapshot", is_flag=True, help="Snapshot outputs before pruning (STATE_HOME/rollback)")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option("--html", "html_mode", type=click.Choice(["on", "off", "auto"]), default="auto", show_default=True)
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
@click.option("--debounce", type=float, default=2.0, show_default=True, help="Minimal seconds between sync runs in watch mode")
@click.option("--stall-seconds", type=float, default=60.0, show_default=True, help="Warn when watch makes no progress for this many seconds")
@click.option("--fail-on-stall", is_flag=True, help="Exit with non-zero status when watch detects a stall")
@click.option("--tail", is_flag=True, help="Log changed paths as they are detected in watch mode")
@click.option("--once", is_flag=True, help="In watch mode, run a single sync pass and exit")
@click.option("--snapshot", is_flag=True, help="Create a rollback snapshot of the output directory before watching")
@click.option("--watch-plan", is_flag=True, help="Print the assembled watch command and exit (no watch run)")
@click.option("--drive-retries", type=int, help="Override Drive retry attempts (default: config or 3)")
@click.option("--drive-retry-base", type=float, help="Override Drive retry base delay seconds (default: config or 0.5)")
@click.option("--root", type=str, help="Named root label to use when configs support multi-root archives")
@click.option("--max-disk", type=float, help="Abort if projected disk use exceeds this many GiB (approx)")
@click.pass_obj
def sync(env: CommandEnv, **kwargs) -> None:
    args = Namespace(**kwargs)
    sync_cmd.dispatch(args, env)


# ---------------------------- import ----------------------------


@cli.command(name="import")
@click.argument("provider", type=click.Choice(["chatgpt", "claude", "claude-code", "codex"]))
@click.argument("source", nargs=-1, type=click.Path(path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Output directory")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option("--html", "html_mode", type=click.Choice(["on", "off", "auto"]), default="auto", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite outputs even when up-to-date")
@click.option("--all", is_flag=True, help="Process all conversations/sessions without selection")
@click.option("--conversation-id", "conversation_ids", multiple=True, help="Conversation ID filter (repeatable)")
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--to-clipboard", is_flag=True, help="Copy single imported Markdown file to the clipboard")
@click.option("--base-dir", type=click.Path(path_type=Path), help="Base directory for local providers")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.pass_obj
def import_cmd_click(env: CommandEnv, **kwargs) -> None:
    args = Namespace(**kwargs)
    imports.run_import_cli(args, env)


# ---------------------------- render ----------------------------


@cli.command()
@click.argument("input", type=click.Path(path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Output directory")
@click.option("--collapse-threshold", type=int, default=None, help="Collapse threshold override")
@click.option("--html", "html_mode", type=click.Choice(["on", "off", "auto"]), default="auto", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite outputs even when up-to-date")
@click.option("--links-only", is_flag=True, help="Link attachments instead of downloading")
@click.option("--diff", is_flag=True, help="Write delta diff alongside updated Markdown")
@click.option("--json", is_flag=True, help="Emit machine-readable summary")
@click.option("--print-paths", is_flag=True, help="List written files after rendering")
@click.option("--to-clipboard", is_flag=True, help="Copy a single rendered file to the clipboard")
@click.option("--dry-run", is_flag=True, help="Report actions without writing files")
@click.option("--attachment-ocr", is_flag=True, help="Attempt OCR on image attachments when indexing attachment text")
@click.option("--max-disk", type=float, help="Abort if projected disk use exceeds this many GiB (approx)")
@click.pass_obj
def render(env: CommandEnv, **kwargs) -> None:
    args = Namespace(**kwargs)
    render_cmd.dispatch(args, env)


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
    kwargs["open"] = kwargs.pop("open_result")
    args = Namespace(**kwargs)
    search_cmd.dispatch(args, env)


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
    args = Namespace(**kwargs)
    status_cmd.dispatch(args, env)


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
    args = Namespace(maintain_cmd="prune", **kwargs)
    maintain.run_maintain_cli(args, env)


@maintain.command(name="doctor")
@click.option("--codex-dir", type=click.Path(path_type=Path), help="Override Codex sessions directory")
@click.option("--claude-code-dir", type=click.Path(path_type=Path), help="Override Claude Code projects directory")
@click.option("--limit", type=int, help="Limit number of files inspected per provider")
@click.option("--json", is_flag=True, help="Emit machine-readable report")
@click.option("--json-verbose", is_flag=True, help="Emit JSON with verbose details")
@click.pass_obj
def maintain_doctor(env: CommandEnv, **kwargs) -> None:
    args = Namespace(maintain_cmd="doctor", **kwargs)
    maintain.run_maintain_cli(args, env)


# ---------------------------- config/env/prefs ----------------------------


@cli.command()
@click.option("--json", is_flag=True, help="Emit environment info as JSON")
@click.pass_obj
def env(env: CommandEnv, **kwargs) -> None:
    run_env_cli(Namespace(**kwargs), env)


@cli.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration (init/set/show)."""


@config.command(name="show")
@click.option("--json", is_flag=True, help="Emit environment info as JSON")
@click.pass_obj
def config_show(env: CommandEnv, **kwargs) -> None:
    args = Namespace(config_cmd="show", **kwargs)
    config_cmd.dispatch(args, env)


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
    args = Namespace(config_cmd="set", **kwargs)
    config_cmd.dispatch(args, env)


@config.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
@click.pass_obj
def config_init(env: CommandEnv, **kwargs) -> None:
    args = Namespace(config_cmd="init", **kwargs)
    config_cmd.dispatch(args, env)


@cli.command()
@click.argument("prefs_cmd", type=click.Choice(["list", "set", "clear"]))
@click.option("--command", "command_name", type=str, help="Command name (e.g., search, sync)")
@click.option("--flag", type=str, help="Flag name, e.g., --limit")
@click.option("--value", type=str, help="Value for the flag")
@click.option("--json", is_flag=True, help="Emit JSON output")
@click.pass_obj
def prefs(env: CommandEnv, prefs_cmd: str, command_name: Optional[str], flag: Optional[str], value: Optional[str], json: bool) -> None:  # type: ignore[func-returns-value]
    args = Namespace(prefs_cmd=prefs_cmd, command=command_name, flag=flag, value=value, json=json)
    prefs_cmd_mod = prefs_cmd  # quiet lint
    prefs_cmd = prefs_cmd_mod
    prefs_cmd.dispatch(args, env)  # type: ignore[attr-defined]


# ---------------------------- attachments ----------------------------


@cli.group()
@click.pass_context
def attachments_group(ctx: click.Context) -> None:
    """Attachment utilities."""


@attachments_group.command(name="stats")
@click.option("--dir", type=click.Path(path_type=Path), help="Root directory containing archives")
@click.option("--ext", type=str, help="Filter by file extension")
@click.option("--hash", "hash_mode", is_flag=True, help="Hash attachments to compute deduped totals")
@click.option("--sort", type=click.Choice(["size", "name"]), default="size")
@click.option("--limit", type=int, default=10)
@click.option("--csv", type=str, help="Write attachment rows to CSV")
@click.option("--json", is_flag=True)
@click.option("--json-lines", is_flag=True)
@click.option("--from-index", is_flag=True, help="Read attachment metadata from the index DB")
@click.pass_obj
def attachments_stats(env: CommandEnv, **kwargs) -> None:
    args = Namespace(attachments_cmd="stats", **kwargs)
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
    args = Namespace(attachments_cmd="extract", **kwargs)
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
    args = Namespace(**kwargs)
    open_helper.run_open_cli(args, env)


@cli.command()
@click.option("--provider", type=str, help="Provider filter for reprocess")
@click.option("--fallback", is_flag=True, help="Use fallback parser")
@click.pass_obj
def reprocess_cmd(env: CommandEnv, **kwargs) -> None:
    args = Namespace(**kwargs)
    reprocess.run_reprocess_cli(args, env)


def main() -> None:  # pragma: no cover
    cli()


__all__ = ["cli", "main"]
