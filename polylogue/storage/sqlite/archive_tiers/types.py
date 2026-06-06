"""Typed archive-tier primitives."""

from __future__ import annotations

from enum import StrEnum


class ArchiveTier(StrEnum):
    """Durability tiers in the split archive file set."""

    SOURCE = "source"
    INDEX = "index"
    EMBEDDINGS = "embeddings"
    USER = "user"
    OPS = "ops"


__all__ = ["ArchiveTier"]
