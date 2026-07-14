"""Shared support types and constants for the sqlite-vec provider."""

from __future__ import annotations

import struct

from polylogue.core.errors import DatabaseError
from polylogue.logging import get_logger

logger = get_logger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4"
DEFAULT_DIMENSION = 1024
BATCH_SIZE = 128

# Rough cost estimation: voyage-4 is $0.10 / 1M tokens.
# Average chat message is ~500 tokens (English prose, including code blocks).
# This is intentionally approximate — Voyage does not return token counts.
ESTIMATED_TOKENS_PER_MESSAGE = 500
VOYAGE_4_COST_PER_1M_TOKENS = 0.10


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
