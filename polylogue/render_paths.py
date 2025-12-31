from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paths import is_within_root, safe_path_component


def render_root(archive_root: Path, provider: str, conversation_id: str) -> Path:
    safe_provider = safe_path_component(provider, fallback="provider")
    safe_conversation = safe_path_component(conversation_id, fallback="conversation")
    return archive_root / "render" / safe_provider / safe_conversation


def legacy_render_root(archive_root: Path, provider: str, conversation_id: str) -> Optional[Path]:
    base = archive_root / "render"
    candidate = archive_root / "render" / provider / conversation_id
    if is_within_root(candidate, base):
        return candidate
    return None


__all__ = ["render_root", "legacy_render_root"]
