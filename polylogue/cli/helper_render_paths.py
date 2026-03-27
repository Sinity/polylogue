"""Render-output path helpers."""

from __future__ import annotations

from pathlib import Path


def latest_render_path(render_root: Path) -> Path | None:
    if not render_root.exists():
        return None
    candidates = list(render_root.rglob("conversation.md")) + list(render_root.rglob("conversation.html"))
    if not candidates:
        return None
    latest: Path | None = None
    latest_mtime = 0.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = path
        except OSError:
            continue
    return latest


__all__ = ["latest_render_path"]
