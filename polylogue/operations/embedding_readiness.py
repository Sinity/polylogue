"""Typed observation boundary for daemon embedding readiness."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path
from types import SimpleNamespace

from polylogue.storage.archive_identity import ArchiveLocation, ArchiveLocationError
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.status_payload import EmbeddingStatusPayload, embedding_status_payload


class EmbeddingReadinessUnavailableError(RuntimeError):
    """The embedding readiness reader could not measure its archive tiers."""


def _file_currency(path: Path, *, header_bytes: int) -> tuple[object, ...]:
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError("embedding readiness requires a regular tier file")
            header = stream.read(header_bytes)
            identity = os.fstat(stream.fileno())
    except FileNotFoundError:
        return (str(path), "absent")
    return (
        str(path),
        identity.st_dev,
        identity.st_ino,
        identity.st_size,
        identity.st_mtime_ns,
        identity.st_ctime_ns,
        header.hex(),
    )


def embedding_readiness_currency(root: Path, *, enabled: bool, has_key: bool, model: str, dimension: int) -> str:
    """Bind cached measurements to policy, recipe and the actual purchased tier.

    Only file identities and SQLite/WAL headers are read. The configured tier
    can resolve into its own physical directory; its WAL belongs beside that
    leaf, rather than beside Source or the configured link.
    """
    try:
        tier = ArchiveLocation.resolve(root).configured_tier("embeddings")
        path = tier.resolved_path
        currency = (
            str(root.absolute()),
            str(tier.configured_path),
            enabled,
            has_key,
            EmbeddingRecipe.current(model=model, dimensions=dimension).recipe_hash.hex(),
            _file_currency(path, header_bytes=100),
            _file_currency(path.with_name(path.name + "-wal"), header_bytes=32),
        )
    except (OSError, ArchiveLocationError) as exc:
        raise EmbeddingReadinessUnavailableError("embedding readiness currency is unreadable") from exc
    return hashlib.sha256(json.dumps(currency, separators=(",", ":")).encode()).hexdigest()


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
