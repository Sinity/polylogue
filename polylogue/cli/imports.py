from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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
from ..importers.base import ImportResult
from ..pipeline_runner import Pipeline, PipelineContext
from ..util import CODEX_SESSIONS_ROOT
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CHATGPT_OUT,
    DEFAULT_CLAUDE_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    resolve_html_enabled,
)
from .render import copy_import_to_clipboard
from .summaries import summarize_import


class ImportExecuteStage:
    def run(self, context: PipelineContext) -> None:
        env: CommandEnv = context.env
        func = context.get("import_callable")
        if func is None:
            env.ui.console.print("[red]Import callable not configured.")
            context.abort()
            return
        kwargs: Dict[str, object] = context.get("import_kwargs", {})
        error_message = context.get("import_error_message", "Import failed")
        try:
            results = func(**kwargs)
        except Exception as exc:
            env.ui.console.print(f"[red]{error_message}: {exc}")
            context.set("error", exc)
            context.abort()
            return
        if results is None:
            result_list: List[ImportResult] = []
        elif isinstance(results, ImportResult):
            result_list = [results]
        else:
            result_list = list(results)
        context.set("import_results", result_list)


class ImportSummarizeStage:
    def __init__(self, title: str) -> None:
        self.title = title

    def run(self, context: PipelineContext) -> None:
        results: List[ImportResult] = context.get("import_results", [])
        ui = context.env.ui
        if not results:
            ui.console.print(f"[yellow]{self.title}: no conversations imported.")
            return
        summarize_import(ui, self.title, results)


def _emit_import_json(results: List[ImportResult]) -> None:
    payload = {
        "count": len(results),
        "results": [
            {
                "slug": res.slug,
                "markdown": str(res.markdown_path),
                "html": str(res.html_path) if res.html_path else None,
                "attachmentsDir": str(res.attachments_dir) if res.attachments_dir else None,
                "diff": str(res.diff_path) if res.diff_path else None,
                "skipped": res.skipped,
                "skipReason": res.skip_reason,
                "dirty": res.dirty,
                "contentHash": res.content_hash,
            }
            for res in results
        ],
    }
    print(json.dumps(payload, indent=2))


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
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CODEX_SESSIONS_ROOT
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme

    pipeline = Pipeline(
        [
            ImportExecuteStage(),
            ImportSummarizeStage("Codex Import"),
        ]
    )
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_codex_session,
            "import_kwargs": {
                "session_id": args.session_id,
                "base_dir": base_dir,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "html": html_enabled,
                "html_theme": html_theme,
                "force": getattr(args, "force", False),
                "branch_mode": getattr(args, "branch_export", "full"),
                "registrar": env.registrar,
            },
            "import_error_message": "Import failed",
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_chatgpt(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CHATGPT_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
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

    pipeline = Pipeline(
        [
            ImportExecuteStage(),
            ImportSummarizeStage("ChatGPT Import"),
        ]
    )
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_chatgpt_export,
            "import_kwargs": {
                "export_path": export_path,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "html": html_enabled,
                "html_theme": html_theme,
                "selected_ids": selected_ids,
                "force": getattr(args, "force", False),
                "branch_mode": getattr(args, "branch_export", "full"),
                "registrar": env.registrar,
            },
            "import_error_message": "Import failed",
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    export_path = Path(args.export_path)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_OUT
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
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

    pipeline = Pipeline(
        [
            ImportExecuteStage(),
            ImportSummarizeStage("Claude Import"),
        ]
    )
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_claude_export,
            "import_kwargs": {
                "export_path": export_path,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "html": html_enabled,
                "html_theme": html_theme,
                "selected_ids": selected_ids,
                "force": getattr(args, "force", False),
                "branch_mode": getattr(args, "branch_export", "full"),
                "registrar": env.registrar,
            },
            "import_error_message": "Import failed",
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


def run_import_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else DEFAULT_PROJECT_ROOT
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
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme

    kwargs = {}
    if args.base_dir:
        kwargs["base_dir"] = base_dir

    pipeline = Pipeline(
        [
            ImportExecuteStage(),
            ImportSummarizeStage("Claude Code Import"),
        ]
    )
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_claude_code_session,
            "import_kwargs": {
                "session_id": session_id,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "html": html_enabled,
                "html_theme": html_theme,
                "force": getattr(args, "force", False),
                "branch_mode": getattr(args, "branch_export", "full"),
                "registrar": env.registrar,
                **kwargs,
            },
            "import_error_message": "Import failed",
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


__all__ = [
    "run_import_cli",
    "run_import_codex",
    "run_import_chatgpt",
    "run_import_claude",
    "run_import_claude_code",
]
