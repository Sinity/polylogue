from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Tuple

from ..commands import CommandEnv
from ..paths import STATE_HOME
from ..util import preflight_disk_requirement
from .context import resolve_output_roots


def _legacy_candidates(root: Path) -> List[Path]:
    legacy: List[Path] = []
    for pattern in ("*.md", "*.html"):
        legacy.extend(path for path in root.glob(pattern) if path.is_file())
    for path in root.glob("*_attachments"):
        if path.exists():
            legacy.append(path)
    return sorted(legacy)


def run_prune_cli(args: object, env: CommandEnv) -> None:
    ui = env.ui
    raw_dirs = getattr(args, "dirs", None) or []
    if raw_dirs:
        roots = [Path(path).expanduser() for path in raw_dirs]
    else:
        roots = resolve_output_roots(env.config)
    seen: set[Path] = set()
    unique_roots: List[Path] = []
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)

    dry_run = bool(getattr(args, "dry_run", False))
    total_candidates = 0
    total_removed = 0

    # Collect all legacy paths first
    all_legacy: List[Tuple[Path, Path]] = []  # (root, legacy_path)
    total_bytes = 0
    for root in unique_roots:
        if not root.exists():
            continue
        legacy = _legacy_candidates(root)
        for path in legacy:
            all_legacy.append((root, path))
            try:
                if path.is_file():
                    total_bytes += path.stat().st_size
                elif path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_file():
                            total_bytes += child.stat().st_size
            except Exception:
                continue
        total_candidates += len(legacy)

    if dry_run:
        for _root, path in all_legacy:
            ui.console.print(f"[yellow][dry-run] Would prune: {path}")
    else:
        if all_legacy:
            preflight_disk_requirement(projected_bytes=total_bytes, limit_gib=getattr(args, "max_disk", None), ui=ui)
            from zipfile import ZipFile

            snapshot_dir = STATE_HOME / "rollback"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"prune-{int(time.time())}.zip"
            try:
                with ZipFile(snapshot_path, "w") as zipf:
                    for _, path in all_legacy:
                        try:
                            if path.is_file():
                                zipf.write(path, arcname=path.name)
                        except Exception:
                            continue
                ui.console.print(f"[dim]Snapshot saved to {snapshot_path} before pruning.[/dim]")
            except Exception as exc:
                ui.console.print(f"[yellow]Snapshot failed: {exc}")
        with ui.progress("Pruning legacy files", total=len(all_legacy)) as tracker:
            for _root, path in all_legacy:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    total_removed += 1
                except Exception as exc:
                    ui.console.print(f"[red]Failed to remove {path}: {exc}")
                tracker.advance()

    summary_lines = [
        f"Roots scanned: {len(unique_roots)}",
        f"Legacy paths discovered: {total_candidates}",
    ]
    if dry_run:
        summary_lines.append("Dry run: no paths removed.")
    else:
        summary_lines.append(f"Paths removed: {total_removed}")
    ui.summary("Prune Legacy Outputs", summary_lines)


__all__ = ["run_prune_cli"]
