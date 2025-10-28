from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List, Optional, cast

from ..local_sync import sync_claude_code_sessions, sync_codex_sessions
from ..settings import SETTINGS
from .context import DEFAULT_CLAUDE_CODE_SYNC_OUT, DEFAULT_CODEX_SYNC_OUT, DEFAULT_COLLAPSE, resolve_html_enabled
from .summaries import summarize_import
from .sync import _log_local_sync
from ..util import add_run

watch_directory: Any
try:  # pragma: no cover - optional dependency
    from watchfiles import watch as watch_directory  # type: ignore[assignment]
except Exception:  # pragma: no cover
    watch_directory = None

watch_directory = cast(Any, watch_directory)


def _console(ui) -> Any:
    return cast(Any, ui.console)


def run_watch_cli(args: argparse.Namespace, env) -> None:
    if watch_directory is None:
        env.ui.console.print(
            "[red]The watchfiles package is not available. Enable it in your environment to use watcher commands."
        )
        return

    provider = getattr(args, "provider", None)
    if provider == "codex":
        run_watch_codex(args, env)
    elif provider == "claude-code":
        run_watch_claude_code(args, env)
    else:
        raise SystemExit(f"Unsupported provider for watch: {provider}")


def run_watch_codex(args: argparse.Namespace, env) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else Path.home() / ".codex" / "sessions"
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme
    debounce = max(0.5, args.debounce)

    console = _console(ui)
    ui.banner("Watching Codex sessions", str(base_dir))

    def sync_once() -> None:
        try:
            result = sync_codex_sessions(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                html=html_enabled,
                html_theme=html_theme,
                force=False,
                prune=False,
                diff=False,
                sessions=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]Codex sync failed: {exc}")
        else:
            _log_local_sync(ui, "Codex Watch", result, provider="codex")

    sync_once()
    if getattr(args, "once", False):
        return
    last_run = time.monotonic()
    try:
        for changes in watch_directory(base_dir, recursive=True):
            if not any(Path(path).suffix == ".jsonl" for _, path in changes):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print("[cyan]Codex watcher stopped.")


def run_watch_claude_code(args: argparse.Namespace, env) -> None:
    ui = env.ui
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_CLAUDE_CODE_SYNC_OUT
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    html_enabled = resolve_html_enabled(args)
    html_theme = SETTINGS.html_theme
    debounce = max(0.5, args.debounce)

    console = _console(ui)
    ui.banner("Watching Claude Code sessions", str(base_dir))

    def sync_once() -> None:
        try:
            result = sync_claude_code_sessions(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                html=html_enabled,
                html_theme=html_theme,
                force=False,
                prune=False,
                diff=False,
                sessions=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]Claude Code sync failed: {exc}")
        else:
            _log_local_sync(ui, "Claude Code Watch", result, provider="claude-code")

    sync_once()
    if getattr(args, "once", False):
        return
    last_run = time.monotonic()
    try:
        for changes in watch_directory(base_dir, recursive=True):
            if not any(Path(path).suffix == ".jsonl" for _, path in changes):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print("[cyan]Claude Code watcher stopped.")


__all__ = [
    "run_watch_cli",
    "run_watch_codex",
    "run_watch_claude_code",
]
