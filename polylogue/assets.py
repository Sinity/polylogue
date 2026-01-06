from __future__ import annotations

import re
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


__all__ = ["asset_path"]
