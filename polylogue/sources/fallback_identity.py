"""Source-path fallback identities shared by retained parsing routes."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

_SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)


def fallback_session_id(source_path: str | None, raw_id: str) -> str:
    """Preserve source naming rules when a provider record has no native ID.

    ZIP member paths use the final colon-delimited coordinate. Claude agent
    filenames carry parser meaning, and Drive cache names already have a
    provider-defined identity. Other acquisition hash suffixes are removed.
    """
    if not source_path:
        return raw_id
    if source_path.startswith("drive:"):
        return unquote(source_path.rsplit("/", 1)[-1].removesuffix(".json"))
    normalized = source_path.replace("\\", "/")
    entry_path = normalized.rsplit(":", 1)[-1]
    stem = Path(entry_path).stem
    if not stem:
        return raw_id
    if stem.startswith("agent-") or "/drive-cache/" in normalized:
        return stem
    cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
    return cleaned or stem


__all__ = ["fallback_session_id"]
