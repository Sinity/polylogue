from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, cast

from ..cli_common import sk_select
from ..commands import CommandEnv
from ..importers import (
    import_chatgpt_export,
    import_claude_code_session,
    import_claude_export,
    import_codex_session,
)
from ..importers.chatgpt import list_chatgpt_conversations
from ..importers.claude_ai import list_claude_conversations
from ..importers.claude_code import DEFAULT_PROJECT_ROOT, list_claude_code_sessions
from ..settings import SETTINGS
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CHATGPT_OUT,
    DEFAULT_CLAUDE_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    resolve_html_enabled,
)
from .render import copy_import_to_clipboard
from .summaries import SummaryUI, summarize_import


def _console(ui) -> Any:
    return cast(Any, ui.console)


def run_import_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    sources = args.source or []

    def _ensure_path() -> Path:
        if not sources:
            raise SystemExit("Provide an export path for this import.")
        return Path(sources[0])

    if provider == "chatgpt":
        export_path = _ensure_path()
        ns = argparse.Namespace(
            export_path=export_path,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            html_mode=args.html_mode,
            force=args.force,
            conversation_ids=args.conversation_ids or [],
            all=args.all,
            json=args.json,
            to_clipboard=args.to_clipboard,
        )
        run_import_chatgpt(ns, env)
    elif provider == "claude":
        export_path = _ensure_path()
        ns = argparse.Namespace(
            export_path=export_path,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            html_mode=args.html_mode,
            force=args.force,
            conversation_ids=args.conversation_ids or [],
            all=args.all,
            json=args.json,
            to_clipboard=args.to_clipboard,
        )
        run_import_claude(ns, env)
    elif provider == "claude-code":
        session_id = sources[0] if sources else "pick"
        ns = argparse.Namespace(
            session_id=session_id,
            base_dir=args.base_dir,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            html_mode=args.html_mode,
            force=args.force,
            json=args.json,
            to_clipboard=args.to_clipboard,
        )
        run_import_claude_code(ns, env)
    elif provider == "codex":
        session_id = sources[0] if sources else "pick"
        ns = argparse.Namespace(
            session_id=session_id,
            base_dir=args.base_dir,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            html_mode=args.html_mode,
            force=args.force,
            json=args.json,
            to_clipboard=args.to_clipboard,
        )
        run_import_codex(ns, env)
    else:
        raise SystemExit(f"Unsupported provider for import: {provider}")


def run_import_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = _console(ui)
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme

    result = import_codex_session(
        args.session_id,
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=collapse,
        html=html_enabled,
        html_theme=html_theme,
        force=getattr(args, "force", False),
    )

    lines = [f"Markdown: {result.markdown_path}"]
    if result.html_path:
    doc = result.document
    if doc is None:
        console.print("[red]Import produced no document; aborting.")
        return

        lines.append(f"HTML preview: {result.html_path}")
    if result.attachments_dir:
        lines.append(f"Attachments directory: {result.attachments_dir}")
    stats = doc.stats
    attachments_total = stats.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachment count: {attachments_total}")
    tokens = stats.get("totalTokensApprox")
    if tokens is not None:
        words = stats.get("totalWordsApprox") or 0
        if words:
            lines.append(f"Approx tokens: {int(tokens)} (~{int(words)} words)")
        else:
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
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, [result])


def run_import_chatgpt(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = _console(ui)
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CHATGPT_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_chatgpt_conversations(export_path)
        except Exception as exc:
            console.print(f"[red]Failed to scan export: {exc}")
            return
        if not entries:
            console.print("No conversations found in export.")
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
            console.print("[yellow]Import cancelled; no conversations selected.")
            return
        if not selection:
            console.print("[yellow]No conversations selected; nothing to import.")
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
            force=getattr(args, "force", False),
        )
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(cast(SummaryUI, ui), "ChatGPT Import", results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = _console(ui)
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_claude_conversations(export_path)
        except Exception as exc:
            console.print(f"[red]Failed to scan export: {exc}")
            return
        if not entries:
            console.print("No conversations found in export.")
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
            console.print("[yellow]Import cancelled; no conversations selected.")
            return
        if not selection:
            console.print("[yellow]No conversations selected; nothing to import.")
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
            force=getattr(args, "force", False),
        )
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(cast(SummaryUI, ui), "Claude Import", results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = _console(ui)
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_PROJECT_ROOT
    session_id = args.session_id

    if session_id in {"pick", "?"} or (session_id == "-" and not ui.plain):
        entries = list_claude_code_sessions(base_dir)
        if not entries:
            console.print("No Claude Code sessions found.")
            return
        lines = [f"{entry['name']}\t{entry['workspace']}\t{entry['path']}" for entry in entries]
        selection = sk_select(lines, multi=False, header="Select Claude Code session")
        if not selection:
            console.print("[yellow]Import cancelled; no session selected.")
            return
        session_id = selection[0].split("\t")[-1]

    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme

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
            force=getattr(args, "force", False),
            **kwargs,
        )
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}")
        return

    summarize_import(cast(SummaryUI, ui), "Claude Code Import", [result])
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, [result])


__all__ = [
    "run_import_cli",
    "run_import_codex",
    "run_import_chatgpt",
    "run_import_claude",
    "run_import_claude_code",
]
