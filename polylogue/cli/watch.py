from __future__ import annotations

import time
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
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


def run_watch_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    provider_name = getattr(args, "provider", None)
    provider = get_local_provider(provider_name)
    if not provider.supports_watch:
        raise SystemExit(f"{provider.title} does not support watch mode")
    if getattr(args, "watch_plan", False):
        cmd = ["polylogue", "sync", provider_name, "--watch"]
        if args.base_dir:
            cmd.extend(["--base-dir", str(Path(args.base_dir).expanduser())])
        if args.out:
            cmd.extend(["--out", str(Path(args.out).expanduser())])
        if args.debounce is not None:
            cmd.extend(["--debounce", str(args.debounce)])
        if getattr(args, "stall_seconds", None) is not None:
            cmd.extend(["--stall-seconds", str(args.stall_seconds)])
        if getattr(args, "once", False):
            cmd.append("--once")
        if getattr(args, "snapshot", False):
            cmd.append("--snapshot")
        if getattr(args, "diff", False):
            cmd.append("--diff")
        if getattr(args, "prune", False):
            cmd.append("--prune")
        if getattr(args, "attachment_ocr", False):
            cmd.append("--attachment-ocr")
        if getattr(args, "sanitize_html", False):
            cmd.append("--sanitize-html")
        env.ui.console.print(" ".join(cmd))
        return
    _run_watch_sessions(args, env, provider)


def _run_watch_sessions(
    args: SimpleNamespace,
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
        # In one-shot mode (or when provider opts in), create the base dir so
        # we can still run an initial sync without forcing users to pre-create it.
        if getattr(args, "once", False) or getattr(provider, "create_base_dir", False):
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                base_exists = True
            except OSError:
                base_exists = False
        if not base_exists:
            hint_parts = []
            if not args.base_dir:
                hint_parts.append(f"pass --base-dir to override (current default: {base_dir})")
            else:
                hint_parts.append("double-check the path or create it before watching")
            hint = "; ".join(hint_parts)
            ui.console.print(f"[red]Base directory does not exist: {base_dir}[/red]")
            if hint:
                ui.console.print(f"[yellow]{hint}[/yellow]")
            raise SystemExit(1)
    out_dir = resolve_output_path(args.out, provider.default_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir: Optional[Path] = None
    if getattr(args, "snapshot", False):
        from ..util import preflight_disk_requirement

        try:
            total_bytes = 0
            for path in out_dir.rglob("*"):
                try:
                    if path.is_file():
                        total_bytes += path.stat().st_size
                except Exception:
                    continue
            preflight_disk_requirement(projected_bytes=total_bytes, limit_gib=getattr(args, "max_disk", None), ui=ui)
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
                attachment_ocr=getattr(args, "attachment_ocr", False),
                sanitize_html=getattr(args, "sanitize_html", False),
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]{provider.watch_log_title} failed: {exc}")
        else:
            _log_local_sync(
                ui,
                provider.watch_log_title,
                result,
                provider=provider.name,
                redacted=getattr(args, "sanitize_html", False),
            )

    sync_once()
    if getattr(args, "once", False):
        return
    start_ts = time.monotonic()
    last_run = start_ts
    skipped_events: List[Path] = []
    skipped_total = 0
    stall_seconds = getattr(args, "stall_seconds", 60.0)
    last_progress = start_ts
    fail_on_stall = bool(getattr(args, "fail_on_stall", False))
    stalled = False
    log_tail = bool(getattr(args, "tail", False))
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
            if log_tail and relevant:
                console.print("[dim]Changes:[/dim]")
                for path in relevant:
                    console.print(f"  {path}")
            if now - last_run < debounce:
                skipped_events.extend(relevant)
                skipped_total += len(relevant)
                continue
            elapsed = now - last_progress
            if stall_seconds and elapsed > stall_seconds:
                console.print(
                    f"[yellow]No sync progress for {stall_seconds}s; check watcher input or increase --stall-seconds."
                )
                stalled = True
                if fail_on_stall:
                    import os

                    os.environ["POLYLOGUE_EXIT_REASON"] = "partial"
                    raise SystemExit(2)
            sync_once(relevant)
            last_run = now
            last_progress = now
            if skipped_events:
                total_skipped = len(skipped_events)
                console.print(
                    f"[yellow]Skipped {total_skipped} change(s) during debounce window."  # pragma: no cover - logging path
                )
                sample = skipped_events[:5]
                for path in sample:
                    console.print(f"  {path}")
                if total_skipped > len(sample):
                    console.print(f"  ... {total_skipped - len(sample)} more")
                skipped_events.clear()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        console.print(f"[cyan]{provider.watch_log_title} stopped.")
    if skipped_total:
        console.print(f"[dim]Total skipped during debounce: {skipped_total}")


__all__ = ["run_watch_cli"]
