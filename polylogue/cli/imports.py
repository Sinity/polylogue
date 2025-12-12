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
from ..util import CODEX_SESSIONS_ROOT, format_run_brief, latest_run, path_order_key
from ..schema import stamp_payload
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CHATGPT_OUT,
    DEFAULT_CLAUDE_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    resolve_collapse_thresholds,
    resolve_collapse_value,
    resolve_html_enabled,
)
from .json_output import safe_json_handler
from .render import copy_import_to_clipboard
from .summaries import summarize_import


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _apply_import_prefs(args: argparse.Namespace, env: CommandEnv) -> None:
    prefs = getattr(env, "prefs", {}) or {}
    import_prefs = prefs.get("import", {}) if isinstance(prefs, dict) else {}
    if not import_prefs:
        return
    if "--html" in import_prefs and getattr(args, "html_mode", "auto") == "auto":
        args.html_mode = "on" if _truthy(import_prefs["--html"]) else "off"
    if "--attachment-ocr" in import_prefs and _truthy(import_prefs["--attachment-ocr"]):
        setattr(args, "attachment_ocr", True)
    if "--sanitize-html" in import_prefs and _truthy(import_prefs["--sanitize-html"]):
        setattr(args, "sanitize_html", True)


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
        from .app import _record_failure
        try:
            results = func(**kwargs)
        except Exception as exc:
            env.ui.console.print(f"[red]{error_message}: {exc}")
            _record_failure(argparse.Namespace(**kwargs), exc, phase="import")
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
        footer = context.get("summary_footer", [])
        summarize_import(ui, self.title, results, extra_lines=footer)


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
    print(json.dumps(stamp_payload(payload), indent=2))


def _emit_print_paths(results: List[ImportResult], ui) -> None:
    ui.console.print("Written paths:")
    for res in results:
        ui.console.print(f"  {res.markdown_path}")
        if res.html_path:
            ui.console.print(f"  {res.html_path}")


def _build_summary_footer(provider: str, cmd: str) -> List[str]:
    note = format_run_brief(latest_run(provider=provider, cmd=cmd))
    return [f"Previous run: {note}"] if note else []


def run_import_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    sources = args.source or []
    _apply_import_prefs(args, env)
    collapse_thresholds = resolve_collapse_thresholds(args, env.settings)

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
            collapse_thresholds=collapse_thresholds,
            html_mode=args.html_mode,
            force=args.force,
            conversation_ids=args.conversation_ids or [],
            all=args.all,
            json=args.json,
            to_clipboard=args.to_clipboard,
            attachment_ocr=args.attachment_ocr,
            sanitize_html=getattr(args, "sanitize_html", False),
        )
        run_import_chatgpt(ns, env)
    elif provider == "claude":
        export_path = _ensure_path()
        ns = argparse.Namespace(
            export_path=export_path,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            collapse_thresholds=collapse_thresholds,
            html_mode=args.html_mode,
            force=args.force,
            conversation_ids=args.conversation_ids or [],
            all=args.all,
            json=args.json,
            to_clipboard=args.to_clipboard,
            attachment_ocr=args.attachment_ocr,
            sanitize_html=getattr(args, "sanitize_html", False),
        )
        run_import_claude(ns, env)
    elif provider == "claude-code":
        session_id = sources[0] if sources else "pick"
        ns = argparse.Namespace(
            session_id=session_id,
            base_dir=args.base_dir,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            collapse_thresholds=collapse_thresholds,
            html_mode=args.html_mode,
            force=args.force,
            json=args.json,
            to_clipboard=args.to_clipboard,
            attachment_ocr=args.attachment_ocr,
            sanitize_html=getattr(args, "sanitize_html", False),
        )
        run_import_claude_code(ns, env)
    elif provider == "codex":
        session_id = sources[0] if sources else "pick"
        ns = argparse.Namespace(
            session_id=session_id,
            base_dir=args.base_dir,
            out=args.out,
            collapse_threshold=args.collapse_threshold,
            collapse_thresholds=collapse_thresholds,
            html_mode=args.html_mode,
            force=args.force,
            json=args.json,
            to_clipboard=args.to_clipboard,
            attachment_ocr=args.attachment_ocr,
            sanitize_html=getattr(args, "sanitize_html", False),
        )
        run_import_codex(ns, env)
    else:
        raise SystemExit(f"Unsupported provider for import: {provider}")


@safe_json_handler
def run_import_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CODEX_SESSIONS_ROOT
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = env.settings
    collapse_thresholds = getattr(args, "collapse_thresholds", None) or resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    session_target = args.session_id
    if (session_target in {None, "", "pick", "?"}) and not ui.plain:
        from ..cli_common import choose_single_entry
        candidates = sorted(base_dir.expanduser().rglob("*.jsonl"), key=path_order_key, reverse=True)
        if not candidates:
            console.print("No Codex sessions found.")
            return

        def _format_session(path, idx):
            return f"{path.stem}\t{path.parent}\t{path}"

        chosen, cancelled = choose_single_entry(
            ui, candidates, format_line=_format_session, header="Select Codex session", prompt="session>"
        )
        if cancelled:
            console.print("[yellow]Import cancelled; no session selected.")
            return
        if chosen is None:
            console.print("[yellow]No session selected.")
            return
        session_target = str(chosen)
    elif not session_target:
        console.print("[yellow]Provide a Codex session file or run interactively.")
        return

    pipeline = Pipeline(
        [
            ImportExecuteStage(),
            ImportSummarizeStage("Codex Import"),
        ]
    )
    footer = _build_summary_footer("codex", "import codex")
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_codex_session,
            "import_kwargs": {
                "session_id": session_target,
                "base_dir": base_dir,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "collapse_thresholds": collapse_thresholds,
                "html": html_enabled,
                "html_theme": html_theme,
                "force": getattr(args, "force", False),
                "allow_dirty": getattr(args, "allow_dirty", False),
                "registrar": env.registrar,
                "attachment_ocr": getattr(args, "attachment_ocr", False),
                "sanitize_html": getattr(args, "sanitize_html", False),
            },
            "import_error_message": "Import failed",
            "summary_footer": footer,
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
        return
    if getattr(args, "print_paths", False):
        _emit_print_paths(results, env.ui)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


@safe_json_handler
def run_import_chatgpt(args: argparse.Namespace, env: CommandEnv) -> None:
    from .json_output import JSONModeError

    ui = env.ui
    console = ui.console
    export_path = Path(args.export_path)

    # Validate export file exists
    if not export_path.exists():
        json_mode = getattr(args, "json", False)
        if json_mode:
            raise JSONModeError(
                "file_not_found", f"Export file not found: {export_path}", path=str(export_path)
            )
        else:
            console.print(f"[red]Error: Export file not found: {export_path}")
            console.print(f"[dim]Check that the path is correct and the file exists.")
            raise SystemExit(1)

    out_dir = Path(args.out) if args.out else DEFAULT_CHATGPT_OUT
    settings = env.settings
    collapse_thresholds = getattr(args, "collapse_thresholds", None) or resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_chatgpt_conversations(export_path)
        except Exception as exc:
            json_mode = getattr(args, "json", False)
            if json_mode:
                raise JSONModeError(
                    "invalid_export",
                    f"Failed to scan ChatGPT export: {exc}",
                    path=str(export_path),
                    hint="Ensure the export file is a valid ChatGPT conversations.json or .zip export",
                )
            else:
                console.print(f"[red]Failed to scan ChatGPT export: {exc}")
                console.print("Hint: Ensure the export file is a valid ChatGPT conversations.json or .zip export")
                raise SystemExit(1)
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
            plain=ui.plain,
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
    footer = _build_summary_footer("chatgpt", "import chatgpt")
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_chatgpt_export,
            "import_kwargs": {
                "export_path": export_path,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "collapse_thresholds": collapse_thresholds,
                "html": html_enabled,
                "html_theme": html_theme,
                "selected_ids": selected_ids,
                "force": getattr(args, "force", False),
                "allow_dirty": getattr(args, "allow_dirty", False),
                "registrar": env.registrar,
                "attachment_ocr": getattr(args, "attachment_ocr", False),
                "sanitize_html": getattr(args, "sanitize_html", False),
            },
            "import_error_message": "Import failed",
            "summary_footer": footer,
            "print_paths": getattr(args, "print_paths", False),
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
        return
    if ctx.get("print_paths"):
        _emit_print_paths(results, env.ui)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


@safe_json_handler
def run_import_claude(args: argparse.Namespace, env: CommandEnv) -> None:
    from .json_output import JSONModeError

    ui = env.ui
    console = ui.console
    export_path = Path(args.export_path)

    # Validate export file exists
    if not export_path.exists():
        json_mode = getattr(args, "json", False)
        if json_mode:
            raise JSONModeError(
                "file_not_found", f"Export file not found: {export_path}", path=str(export_path)
            )
        else:
            console.print(f"[red]Error: Export file not found: {export_path}")
            console.print(f"[dim]Check that the path is correct and the file exists.")
            raise SystemExit(1)

    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_OUT
    settings = env.settings
    collapse_thresholds = getattr(args, "collapse_thresholds", None) or resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    selected_ids = args.conversation_ids[:] if args.conversation_ids else None

    if not args.all and not selected_ids and not ui.plain:
        try:
            entries = list_claude_conversations(export_path)
        except Exception as exc:
            json_mode = getattr(args, "json", False)
            if json_mode:
                raise JSONModeError(
                    "invalid_export",
                    f"Failed to scan Claude export: {exc}",
                    path=str(export_path),
                    hint="Ensure the export file is a valid Claude conversations.json or .zip export",
                )
            else:
                console.print(f"[red]Failed to scan Claude export: {exc}")
                console.print("Hint: Ensure the export file is a valid Claude conversations.json or .zip export")
                raise SystemExit(1)
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
            plain=ui.plain,
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
    footer = _build_summary_footer("claude", "import claude")
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_claude_export,
            "import_kwargs": {
                "export_path": export_path,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "collapse_thresholds": collapse_thresholds,
                "html": html_enabled,
                "html_theme": html_theme,
                "selected_ids": selected_ids,
                "force": getattr(args, "force", False),
                "registrar": env.registrar,
                "attachment_ocr": getattr(args, "attachment_ocr", False),
                "sanitize_html": getattr(args, "sanitize_html", False),
            },
            "import_error_message": "Import failed",
            "summary_footer": footer,
            "print_paths": getattr(args, "print_paths", False),
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
        return
    if ctx.get("print_paths"):
        _emit_print_paths(results, env.ui)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


@safe_json_handler
def run_import_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else DEFAULT_PROJECT_ROOT
    session_id = args.session_id

    if session_id in {"pick", "?"} or (session_id == "-" and not ui.plain):
        from ..cli_common import choose_single_entry
        entries = list_claude_code_sessions(base_dir)
        if not entries:
            console.print("No Claude Code sessions found.")
            return

        def _format_session(entry, idx):
            return f"{entry['name']}\t{entry['workspace']}\t{entry['path']}"

        chosen, cancelled = choose_single_entry(
            ui, entries, format_line=_format_session, header="Select Claude Code session", prompt="session>"
        )
    if cancelled:
        console.print("[yellow]Import cancelled; no session selected.")
        return
    if chosen is None:
        console.print("[yellow]No session selected.")
        return
    session_id = chosen["path"]

    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    settings = env.settings
    collapse_thresholds = getattr(args, "collapse_thresholds", None) or resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
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
    footer = _build_summary_footer("claude-code", "import claude-code")
    ctx = PipelineContext(
        env=env,
        options=args,
        data={
            "import_callable": import_claude_code_session,
            "import_kwargs": {
                "session_id": session_id,
                "output_dir": out_dir,
                "collapse_threshold": collapse,
                "collapse_thresholds": collapse_thresholds,
                "html": html_enabled,
                "html_theme": html_theme,
                "force": getattr(args, "force", False),
                "allow_dirty": getattr(args, "allow_dirty", False),
                "registrar": env.registrar,
                "attachment_ocr": getattr(args, "attachment_ocr", False),
                "sanitize_html": getattr(args, "sanitize_html", False),
                **kwargs,
            },
            "import_error_message": "Import failed",
            "summary_footer": footer,
            "print_paths": getattr(args, "print_paths", False),
        },
    )
    pipeline.run(ctx)
    if ctx.aborted:
        return
    results: List[ImportResult] = ctx.get("import_results", [])
    if getattr(args, "json", False):
        _emit_import_json(results)
        return
    if ctx.get("print_paths"):
        _emit_print_paths(results, env.ui)
    if getattr(args, "to_clipboard", False):
        copy_import_to_clipboard(ui, results)


__all__ = [
    "run_import_cli",
    "run_import_codex",
    "run_import_chatgpt",
    "run_import_claude",
    "run_import_claude_code",
]
