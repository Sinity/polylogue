"""Identity hashes for rebuildable SQLite tiers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from enum import StrEnum


def _normalize_schema_sql(value: str | None) -> str:
    """Normalize one sqlite_master SQL definition for semantic comparison."""
    source = value or ""
    tokens: list[tuple[str, bool]] = []
    i = 0
    pending_space = False
    length = len(source)

    while i < length:
        char = source[i]
        if char.isspace():
            pending_space = True
            i += 1
            continue
        if source.startswith("--", i):
            pending_space = True
            newline = source.find("\n", i + 2)
            i = length if newline < 0 else newline + 1
            continue
        if source.startswith("/*", i):
            pending_space = True
            end = source.find("*/", i + 2)
            i = length if end < 0 else end + 2
            continue

        if char in "'\"`":
            quote = char
            start = i
            i += 1
            while i < length:
                if source[i] == quote:
                    i += 1
                    if i < length and source[i] == quote:
                        i += 1
                        continue
                    break
                i += 1
            tokens.append((source[start:i], pending_space))
            pending_space = False
            continue
        if char == "[":
            start = i
            closing = source.find("]", i + 1)
            i = length if closing < 0 else closing + 1
            tokens.append((source[start:i], pending_space))
            pending_space = False
            continue
        if char.isalnum() or char in "_$":
            start = i
            i += 1
            while i < length and (source[i].isalnum() or source[i] in "_$"):
                i += 1
            tokens.append((source[start:i].lower(), pending_space))
            pending_space = False
            continue

        operator = next(
            (
                candidate
                for candidate in ("->>", "<<", ">>", "||", "->", "<=", ">=", "!=", "<>", "==")
                if source.startswith(candidate, i)
            ),
            char,
        )
        tokens.append((operator, pending_space))
        pending_space = False
        i += len(operator)

    rendered: list[str] = []
    previous_word = False
    for token, had_space in tokens:
        word = token[0].isalnum() or token[0] in "_'\"`["
        if rendered and (had_space or previous_word) and previous_word and word:
            rendered.append(" ")
        rendered.append(token)
        previous_word = word
    return "".join(rendered).strip()


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


def _identity_ddl(ddl: str) -> str:
    """Exclude the identity stamp table from the identity it stores."""
    return ddl.removesuffix(DERIVED_SCHEMA_META_DDL)


def _semantic_manifest_fingerprint(tier: DerivedTier) -> str:
    """Return the canonical SQLite-object manifest for one derived tier.

    Rendering the declared DDL into SQLite makes the identity depend on the
    objects SQLite actually creates, rather than on comments or formatting in
    the source script.  The index manifest also applies the runtime indexes,
    so that component remains covered without hashing its raw SQL separately.
    """
    # During archive-tiers package initialization, ``schema_manifest`` may be
    # only partially imported because schema dispositions bootstrap a
    # prototype.  Once initialization is complete, use its cached canonical
    # renderer so normal opens do not render a second in-memory database.
    manifest_module = sys.modules.get("polylogue.storage.sqlite.schema_manifest")
    canonical_manifest = getattr(manifest_module, "canonical_schema_manifest", None)
    if canonical_manifest is not None:
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        return str(canonical_manifest(ArchiveTier(tier.value)).fingerprint)

    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

    archive_tier = ArchiveTier(tier.value)
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ARCHIVE_DDL_BY_TIER[archive_tier])
        if archive_tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
            conn.executescript(DERIVED_SCHEMA_META_DDL)
        if archive_tier is ArchiveTier.INDEX:
            ensure_runtime_indexes_sync(conn)
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[archive_tier]}")
        rows = conn.execute(
            """SELECT type, name, sql FROM sqlite_master
               WHERE name NOT LIKE 'sqlite_%'
                 AND name != 'schema_identity'
                 AND sql IS NOT NULL
               ORDER BY type, name"""
        ).fetchall()
        fts_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING fts5%' COLLATE NOCASE"
        ).fetchall()
        shadow_names = {
            f"{name}{suffix}" for (name,) in fts_rows for suffix in ("_data", "_idx", "_content", "_docsize", "_config")
        }
        objects = tuple(
            (str(kind), str(name), _normalize_schema_sql(sql))
            for kind, name, sql in rows
            if str(name) not in shadow_names
        )
        payload = {
            "tier": archive_tier.value,
            "version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
            "objects": objects,
        }
        return hashlib.sha256((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    finally:
        conn.close()


def derived_schema_identity(tier: DerivedTier) -> str:
    """Return the current identity for a rebuildable tier."""
    if tier is DerivedTier.INDEX:
        from polylogue.sources.origin_specs import (
            lowering_fingerprint,
            materializer_fingerprint,
            replay_routing_fingerprint,
        )

        return _canonical_digest(
            {
                "tier": tier.value,
                "semantic_schema_manifest": _semantic_manifest_fingerprint(tier),
                "lowering_fingerprint": lowering_fingerprint(),
                "materializer_fingerprint": materializer_fingerprint(),
                "replay_routing_fingerprint": replay_routing_fingerprint(),
            }
        )
    if tier is DerivedTier.OPS:
        return _canonical_digest({"tier": tier.value, "semantic_schema_manifest": _semantic_manifest_fingerprint(tier)})
    raise ValueError(f"unsupported derived tier: {tier}")


def read_schema_identity(conn: sqlite3.Connection, tier: DerivedTier) -> str | None:
    """Return the stamped identity, or None when the tier declares none.

    A tier written before the ledger existed carries no ledger table at all.
    That is an absent identity, which this signature already admits -- not a
    malformed database, and not something to raise about.
    """

    try:
        row = conn.execute("SELECT identity FROM schema_identity WHERE tier = ?", (tier.value,)).fetchone()
    except sqlite3.OperationalError:
        if not _table_exists(conn, "schema_identity"):
            return None
        raise
    return None if row is None else str(row[0])


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


__all__ = [
    "DERIVED_SCHEMA_META_DDL",
    "DerivedTier",
    "derived_schema_identity",
    "read_schema_identity",
]
