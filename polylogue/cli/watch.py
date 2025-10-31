from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Set, Tuple

from watchfiles import watch as _watch_directory

from ..commands import CommandEnv
from ..local_sync import sync_claude_code_sessions, sync_codex_sessions
from ..util import CLAUDE_CODE_PROJECT_ROOT, CODEX_SESSIONS_ROOT
from .context import DEFAULT_CLAUDE_CODE_SYNC_OUT, DEFAULT_CODEX_SYNC_OUT, DEFAULT_COLLAPSE, resolve_html_enabled
from .sync import _log_local_sync

WatchChange = Tuple[Any, str]
WatchBatch = Iterable[Set[WatchChange]]
WatchDirectoryFn = Callable[..., WatchBatch]


def run_watch_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    if provider == "codex":
        run_watch_codex(args, env)
    elif provider == "claude-code":
        run_watch_claude_code(args, env)
    else:
        raise SystemExit(f"Unsupported provider for watch: {provider}")


def run_watch_codex(args: argparse.Namespace, env: CommandEnv) -> None:
    watch_fn = _watch_directory
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CODEX_SESSIONS_ROOT
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CODEX_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    debounce = max(0.5, args.debounce)

    console = ui.console
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
        for changes in watch_fn(base_dir, recursive=True):
            if not any(Path(path).suffix == ".jsonl" for _, path in changes):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print("[cyan]Codex watcher stopped.")


def run_watch_claude_code(args: argparse.Namespace, env: CommandEnv) -> None:
    watch_fn = _watch_directory
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else CLAUDE_CODE_PROJECT_ROOT
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DEFAULT_CLAUDE_CODE_SYNC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = args.collapse_threshold or DEFAULT_COLLAPSE
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    debounce = max(0.5, args.debounce)

    console = ui.console
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
        for changes in watch_fn(base_dir, recursive=True):
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
