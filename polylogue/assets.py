from __future__ import annotations

import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def asset_path(archive_root: Path, attachment_id: str) -> Path:
    raw = attachment_id.strip()
    safe_id = _SAFE_ID_RE.sub("_", raw)
    if not safe_id or safe_id != raw:
        digest = sha256(raw.encode("utf-8")).hexdigest()[:32]
        safe_id = f"att-{digest}"
    prefix = safe_id[:2] if len(safe_id) >= 2 else safe_id.ljust(2, "_")
    return archive_root / "assets" / prefix / safe_id


def write_asset(archive_root: Path, asset_id: str, content: bytes) -> Path:
    """Write asset content atomically to prevent corruption.

    Uses write-to-temp-then-rename pattern for atomicity. This ensures that:
    - Multiple concurrent writes to the same asset don't corrupt the file
    - Partial writes are never visible
    - The operation is as atomic as the filesystem allows

    Args:
        archive_root: Root directory for asset storage
        asset_id: Identifier for the asset
        content: Binary content to write

    Returns:
        Path to the written asset file

    Raises:
        OSError: If file operations fail
    """
    final_path = asset_path(archive_root, asset_id)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename
    # This prevents partial writes from corrupting the file even with concurrent access
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(
            dir=final_path.parent, prefix=f".{asset_id}.", text=False
        )
        os.write(fd, content)
        os.close(fd)
        fd = None
        # Atomic rename (on same filesystem)
        os.replace(temp_path, final_path)
    except Exception:
        if fd is not None:
            os.close(fd)
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    return final_path


__all__ = ["asset_path", "write_asset"]
