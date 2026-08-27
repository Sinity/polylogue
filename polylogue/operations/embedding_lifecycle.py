"""Operation boundary for embedding generation lifecycle recovery."""

from __future__ import annotations

from pathlib import Path


def ensure_embedding_lifecycle_startup(archive_root_path: Path) -> Path:
    """Adopt or recover the archive-owned embedding generation pointer."""
    from polylogue.storage.embeddings.generations import ensure_embedding_lifecycle

    result = ensure_embedding_lifecycle(archive_root_path)
    from polylogue.daemon.embedding_backlog import recover_embedding_catchup_receipts

    recover_embedding_catchup_receipts(archive_root_path)
    return result
