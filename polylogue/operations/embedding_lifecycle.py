"""Operation boundary for embedding generation lifecycle recovery."""

from __future__ import annotations

from pathlib import Path


def ensure_embedding_lifecycle_startup(archive_root_path: Path) -> Path:
    """Adopt or recover the archive-owned embedding generation pointer."""
    from polylogue.storage.embeddings.generations import ensure_embedding_lifecycle

    return ensure_embedding_lifecycle(archive_root_path)
