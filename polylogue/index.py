from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .render import MarkdownDocument


class IndexBackend(Protocol):
    def update(
        self,
        *,
        provider: str,
        conversation_id: str,
        slug: str,
        path: Path,
        document: MarkdownDocument,
        metadata: Dict[str, Any],
    ) -> None:
        ...


class SqliteBackend:
    """Legacy shim: SQLite indexing now happens inside conversation_registrar."""

    def update(self, **_: Any) -> None:
        return


class NullBackend:
    def update(self, **_: Any) -> None:  # pragma: no cover - intentionally no-op
        return


class QdrantBackend:
    def __init__(self) -> None:
        try:
            from .index_qdrant import update_qdrant_index
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "POLYLOGUE_INDEX_BACKEND=qdrant requires the 'qdrant-client' package. "
                "Install it with: pip install 'polylogue[vector]'"
            ) from exc
        self._update = update_qdrant_index

    def update(
        self,
        *,
        provider: str,
        conversation_id: str,
        slug: str,
        path: Path,
        document: MarkdownDocument,
        metadata: Dict[str, Any],
    ) -> None:
        self._update(
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            path=path,
            document=document,
            metadata=metadata,
        )


_BACKEND_CACHE: Optional[IndexBackend] = None


def _resolve_backend() -> IndexBackend:
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE

    backend_name = os.environ.get("POLYLOGUE_INDEX_BACKEND", "sqlite").strip().lower()
    if backend_name in ("", "sqlite"):
        _BACKEND_CACHE = SqliteBackend()
    elif backend_name in ("none", "disabled"):
        _BACKEND_CACHE = NullBackend()
    elif backend_name == "qdrant":
        _BACKEND_CACHE = QdrantBackend()
    else:
        raise ValueError(f"Unknown POLYLOGUE_INDEX_BACKEND '{backend_name}'")
    return _BACKEND_CACHE


def update_index(
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    path: Path,
    document: MarkdownDocument,
    metadata: Dict[str, Any],
) -> None:
    backend = _resolve_backend()
    backend.update(
        provider=provider,
        conversation_id=conversation_id,
        slug=slug,
        path=path,
        document=document,
        metadata=metadata,
    )
