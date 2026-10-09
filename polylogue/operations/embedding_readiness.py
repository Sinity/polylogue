"""Typed observation boundary for daemon embedding readiness."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from polylogue.storage.embeddings.status_payload import EmbeddingStatusPayload, embedding_status_payload


class EmbeddingReadinessUnavailableError(RuntimeError):
    """The embedding readiness reader could not measure its archive tiers."""


def read_embedding_readiness(db_file: Path, *, detail: bool = False) -> EmbeddingStatusPayload:
    """Return measured status or preserve the original cause as a domain failure."""
    try:
        return embedding_status_payload(
            SimpleNamespace(config=SimpleNamespace(db_path=db_file)),
            include_retrieval_bands=False,
            include_detail=detail,
        )
    except (sqlite3.Error, OSError) as exc:
        raise EmbeddingReadinessUnavailableError("embedding readiness is unreadable") from exc
