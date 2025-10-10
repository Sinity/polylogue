from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .render import MarkdownDocument


def update_index(
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    path: Path,
    document: MarkdownDocument,
    metadata: Dict[str, Any],
) -> None:
    backend = os.environ.get("POLYLOGUE_INDEX_BACKEND", "sqlite").strip().lower()
    if backend in ("", "sqlite"):
        from .index_sqlite import update_sqlite_index

        update_sqlite_index(
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            path=path,
            document=document,
            metadata=metadata,
        )
        return
    if backend in ("none", "disabled"):
        return
    if backend == "qdrant":
        try:
            from .index_qdrant import update_qdrant_index
        except ImportError:  # pragma: no cover - optional dependency
            return
        update_qdrant_index(
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            path=path,
            document=document,
            metadata=metadata,
        )
        return
    raise ValueError(f"Unknown POLYLOGUE_INDEX_BACKEND '{backend}'")
