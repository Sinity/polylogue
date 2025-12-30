from __future__ import annotations

from pathlib import Path


def asset_path(archive_root: Path, attachment_id: str) -> Path:
    safe_id = attachment_id.strip()
    prefix = safe_id[:2] if len(safe_id) >= 2 else safe_id.ljust(2, "_")
    return archive_root / "assets" / prefix / safe_id


__all__ = ["asset_path"]
