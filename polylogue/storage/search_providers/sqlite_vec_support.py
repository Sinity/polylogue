"""Shared support types and constants for the sqlite-vec provider."""

from __future__ import annotations

import struct

from polylogue.errors import DatabaseError
from polylogue.logging import get_logger

logger = get_logger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4"
DEFAULT_DIMENSION = 1024
BATCH_SIZE = 128


class SqliteVecError(DatabaseError):
    """Raised when sqlite-vec operations fail."""


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize float vector to binary format for sqlite-vec."""
    return struct.pack(f"<{len(vector)}f", *vector)


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "SqliteVecError",
    "VOYAGE_API_URL",
    "_serialize_f32",
    "logger",
]
