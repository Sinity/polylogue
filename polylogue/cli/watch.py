from __future__ import annotations

import argparse
import time
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

try:
    from watchfiles import watch as _watch_directory
except ImportError:  # pragma: no cover - fallback for environments without watchfiles
    from .._vendor.watchfiles import watch as _watch_directory

from ..commands import CommandEnv
from ..local_sync import get_local_provider
from .context import resolve_collapse_thresholds, resolve_html_enabled, resolve_output_path
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
    if getattr(args, "offline", False):
        ui.console.print("[yellow]Offline mode enabled; network-dependent steps will be skipped.")
    base_exists = base_dir.exists()
    if not base_exists:
        if getattr(provider, "create_base_dir", False):
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            hint = "(pass --base-dir to use a different location)" if not args.base_dir else ""
            ui.console.print(f"[red]Base directory does not exist: {base_dir} {hint}")
            raise SystemExit(1)
    out_dir = resolve_output_path(args.out, provider.default_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir: Optional[Path] = None
    if getattr(args, "snapshot", False):
        try:
            tmp_root = Path(tempfile.mkdtemp(prefix="polylogue-watch-rollback-"))
            shutil.copytree(out_dir, tmp_root / out_dir.name, dirs_exist_ok=True)
            snapshot_dir = tmp_root
            ui.console.print(f"[dim]Snapshot created at {tmp_root}[/dim]")
        except Exception as exc:
            ui.console.print(f"[yellow]Snapshot failed: {exc}")
    settings = env.settings
    collapse_thresholds = resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    debounce = max(0.5, args.debounce)
    force = bool(getattr(args, "force", False))
    prune = bool(getattr(args, "prune", False))
    diff_requested = bool(getattr(args, "diff", False))
    supports_diff = bool(getattr(provider, "supports_diff", False))
    diff_enabled = diff_requested and supports_diff
    if diff_requested and not supports_diff:
        ui.console.print(f"[yellow]{provider.title} does not support --diff; ignoring for watch mode.")

    console = ui.console
    ui.banner(provider.watch_banner, str(base_dir))

    def sync_once(changed: Optional[Iterable[Path]] = None) -> None:
        try:
            session_override = None
            if changed:
                session_override = [Path(p) for p in changed]
            result = provider.sync_fn(
                base_dir=base_dir,
                output_dir=out_dir,
                collapse_threshold=collapse,
                collapse_thresholds=collapse_thresholds,
                html=html_enabled,
                html_theme=html_theme,
                force=force,
                prune=prune,
                diff=diff_enabled,
                sessions=session_override,
                registrar=env.registrar,
                ui=ui,
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]{provider.watch_log_title} failed: {exc}")
        else:
            _log_local_sync(ui, provider.watch_log_title, result, provider=provider.name)

    sync_once()
    if getattr(args, "once", False):
        return
    start_ts = time.monotonic()
    last_run = start_ts
    skipped_events: List[Path] = []
    stall_seconds = getattr(args, "stall_seconds", 60.0)
    last_progress = start_ts
    try:
        for changes in watch_fn(base_dir, recursive=True):
            relevant: List[Path] = []
            for _, changed_path in changes:
                path_obj = Path(changed_path)
                if provider.watch_suffixes:
                    if path_obj.suffix not in provider.watch_suffixes:
                        continue
                relevant.append(path_obj)
            if provider.watch_suffixes and not relevant:
                continue
            now = time.monotonic()
            if now - last_run < debounce:
                skipped_events.extend(relevant)
                continue
            elapsed = now - last_progress
            if stall_seconds and elapsed > stall_seconds:
                console.print(
                    f"[yellow]No sync progress for {stall_seconds}s; check watcher input or increase --stall-seconds."
                )
            sync_once(relevant)
            last_run = now
            last_progress = now
            if skipped_events:
                total_skipped = len(skipped_events)
                console.print(
                    f"[yellow]Skipped {total_skipped} change(s) during debounce window."  # pragma: no cover - logging path
                )
                if total_skipped <= 5:
                    for path in skipped_events:
                        console.print(f"  {path}")
                else:
                    for path in skipped_events[:3]:
                        console.print(f"  {path}")
                    console.print(f"  ... {total_skipped - 3} more")
                skipped_events.clear()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print(f"[cyan]{provider.watch_log_title} stopped.")


__all__ = ["run_watch_cli"]
