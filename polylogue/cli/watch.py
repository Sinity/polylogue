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
from ..local_sync import get_local_provider
from .context import (
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
    provider_name = getattr(args, "provider", None)
    provider = get_local_provider(provider_name)
    if not provider.supports_watch:
        raise SystemExit(f"{provider.title} does not support watch mode")
    _run_watch_sessions(args, env, provider)


def _run_watch_sessions(
    args: argparse.Namespace,
    env: CommandEnv,
    provider,
) -> None:
    watch_fn = _watch_directory
    ui = env.ui
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else provider.default_base
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve_output_path(args.out, provider.default_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    collapse = resolve_collapse_value(args.collapse_threshold, DEFAULT_COLLAPSE)
    settings = env.settings
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    debounce = max(0.5, args.debounce)

    console = ui.console
    ui.banner(provider.watch_banner, str(base_dir))

    def sync_once() -> None:
        try:
            result = provider.sync_fn(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                html=html_enabled,
                html_theme=html_theme,
                force=False,
                prune=False,
                diff=False,
                sessions=None,
                branch_mode=getattr(args, "branch_export", "full"),
                registrar=env.registrar,
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]{provider.watch_log_title} failed: {exc}")
        else:
            _log_local_sync(ui, provider.watch_log_title, result, provider=provider.name)

    sync_once()
    if getattr(args, "once", False):
        return
    last_run = time.monotonic()
    try:
        for changes in watch_fn(base_dir, recursive=True):
            if provider.watch_suffixes and not any(
                Path(path).suffix in provider.watch_suffixes for _, path in changes
            ):
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                continue
            sync_once()
            last_run = now
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print(f"[cyan]{provider.watch_log_title} stopped.")


__all__ = ["run_watch_cli"]
