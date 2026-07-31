"""Typed archive-tier primitives."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ArchiveTier(StrEnum):
    """Durability tiers in the split archive file set."""

    SOURCE = "source"
    INDEX = "index"
    EMBEDDINGS = "embeddings"
    USER = "user"
    OPS = "ops"


# polylogue-u6tl: these two live here (a dependency-free leaf module) rather
# than in archive.py, where they conceptually belong, so that index.py's DDL
# can import them for `literal_check` without a cycle -- archive.py
# transitively imports insights.archive -> storage.repair ->
# storage.raw_authority -> storage.sqlite.migration_runner ->
# storage.sqlite.archive_tiers (this package), so `index.py` (loaded by that
# package's __init__) importing anything from archive.py at module scope
# deadlocks partial initialization. archive.py re-exports both names for
# existing call sites.
DelegationMappingState = Literal["resolved", "unresolved", "edge_only", "quarantined"]
DelegationResultStatus = Literal["ok", "error", "unknown"]


__all__ = ["ArchiveTier", "DelegationMappingState", "DelegationResultStatus"]
