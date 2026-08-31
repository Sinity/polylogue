"""Identity hashes for rebuildable SQLite tiers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from enum import StrEnum


class DerivedTier(StrEnum):
    INDEX = "index"
    OPS = "ops"


DERIVED_SCHEMA_META_DDL = """
CREATE TABLE IF NOT EXISTS schema_identity (
    tier TEXT PRIMARY KEY,
    identity TEXT NOT NULL
) STRICT;
"""


def _canonical_digest(parts: dict[str, object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def derived_schema_identity(tier: DerivedTier) -> str:
    """Return the current identity for a rebuildable tier."""
    if tier is DerivedTier.INDEX:
        from polylogue.sources.origin_specs import (
            lowering_fingerprint,
            materializer_fingerprint,
            replay_routing_fingerprint,
        )
        from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
        from polylogue.storage.sqlite.runtime_indexes import runtime_index_ddl

        return _canonical_digest(
            {
                "tier": tier.value,
                "ddl": INDEX_DDL,
                "runtime_index_ddl": runtime_index_ddl(),
                "lowering_fingerprint": lowering_fingerprint(),
                "materializer_fingerprint": materializer_fingerprint(),
                "replay_routing_fingerprint": replay_routing_fingerprint(),
            }
        )
    if tier is DerivedTier.OPS:
        from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL

        return _canonical_digest({"tier": tier.value, "ddl": OPS_DDL})
    raise ValueError(f"unsupported derived tier: {tier}")


def read_schema_identity(conn: sqlite3.Connection, tier: DerivedTier) -> str | None:
    row = conn.execute("SELECT identity FROM schema_identity WHERE tier = ?", (tier.value,)).fetchone()
    return None if row is None else str(row[0])


__all__ = [
    "DERIVED_SCHEMA_META_DDL",
    "DerivedTier",
    "derived_schema_identity",
    "read_schema_identity",
]
