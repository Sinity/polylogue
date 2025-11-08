from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Set, Tuple

try:  # pragma: no cover - optional dependency
    from watchfiles import watch as _watch_directory
except ImportError:  # pragma: no cover - used in non-watch environments
    def _watch_directory(*_args, **_kwargs):
        yield from ()

from ..commands import CommandEnv
from ..local_sync import sync_claude_code_sessions, sync_codex_sessions
from ..util import CLAUDE_CODE_PROJECT_ROOT, CODEX_SESSIONS_ROOT
from .context import (
    DEFAULT_CLAUDE_CODE_SYNC_OUT,
    DEFAULT_CODEX_SYNC_OUT,
    DEFAULT_COLLAPSE,
    resolve_collapse_value,
    resolve_html_enabled,
    resolve_output_path,
)
from .sync import _log_local_sync

WatchChange = Tuple[Any, str]
WatchBatch = Iterable[Set[WatchChange]]
WatchDirectoryFn = Callable[..., WatchBatch]


def run_watch_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    if provider == "codex":
        _run_watch_sessions(
            args,
            env,
            provider="codex",
            base_default=CODEX_SESSIONS_ROOT,
            out_default=DEFAULT_CODEX_SYNC_OUT,
            banner="Watching Codex sessions",
            log_title="Codex Watch",
            sync_fn=sync_codex_sessions,
        )
    elif provider == "claude-code":
        _run_watch_sessions(
            args,
            env,
            provider="claude-code",
            base_default=CLAUDE_CODE_PROJECT_ROOT,
            out_default=DEFAULT_CLAUDE_CODE_SYNC_OUT,
            banner="Watching Claude Code sessions",
            log_title="Claude Code Watch",
            sync_fn=sync_claude_code_sessions,
        )
    else:
        raise SystemExit(f"Unsupported provider for watch: {provider}")


def _run_watch_sessions(
    args: argparse.Namespace,
    env: CommandEnv,
    *,
    provider: str,
    base_default: Path,
    out_default: Path,
    banner: str,
    log_title: str,
    sync_fn,
) -> None:
    watch_fn = _watch_directory
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else base_default
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve_output_path(args.out, out_default)
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = resolve_collapse_value(args.collapse_threshold, DEFAULT_COLLAPSE)
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    debounce = max(0.5, args.debounce)

    console = ui.console
    ui.banner(banner, str(base_dir))

    def sync_once() -> None:
        try:
            result = sync_fn(
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
            console.print(f"[red]{log_title} failed: {exc}")
        else:
            _log_local_sync(ui, log_title, result, provider=provider)

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
        console.print(f"[cyan]{log_title} stopped.")


__all__ = ["run_watch_cli"]
