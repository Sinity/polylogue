"""Canonical semantic manifests for SQLite archive tiers.

``PRAGMA user_version`` is a useful migration cursor, but it is not a proof
that the objects belonging to that version are present.  This module compares
the normalized ``sqlite_master`` projection with the same projection produced
by the canonical create route.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_WS = re.compile(r"\s+")
_FTS_BULK_GUARD = re.compile(
    r"\s+(?:and|when) not exists \(select 1 from derived_refresh_guard "
    r"where guard_name = '[^']+'\)"
)


@dataclass(frozen=True, slots=True)
class _SameVersionSchemaVariant:
    introduced_version: int
    object_names: tuple[tuple[str, str], ...]
    transformation: str = "remove_fts_bulk_guard"


_INDEX_SAME_VERSION_SCHEMA_VARIANTS = (
    _SameVersionSchemaVariant(
        introduced_version=63,
        object_names=(
            ("trigger", "blocks_command_trigram_ai"),
            ("trigger", "blocks_command_trigram_ad"),
            ("trigger", "blocks_command_trigram_au"),
        ),
    ),
)


def _sql(value: str | None) -> str:
    return _WS.sub(" ", (value or "").strip()).lower()


#: The exact table suffixes FTS5 creates to back one virtual table. Matched
#: exactly rather than by prefix: ``messages_fts_identity`` is a declared
#: table of ours that shares the ``messages_fts`` prefix, and dropping it from
#: the manifest would hide real drift in it.
_FTS5_SHADOW_SUFFIXES = ("_data", "_idx", "_content", "_docsize", "_config")


def _fts_shadow_names(conn: sqlite3.Connection) -> frozenset[str]:
    """Return the shadow tables FTS5 owns for this database's FTS5 tables."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING fts5%' COLLATE NOCASE"
    ).fetchall()
    return frozenset(f"{name}{suffix}" for (name,) in rows for suffix in _FTS5_SHADOW_SUFFIXES)


def _projection(conn: sqlite3.Connection) -> tuple[tuple[str, str, str], ...]:
    rows = conn.execute(
        """SELECT type, name, sql FROM sqlite_master
           WHERE name NOT LIKE 'sqlite_%'
             AND name != 'schema_identity'
             AND sql IS NOT NULL
           ORDER BY type, name"""
    ).fetchall()
    # FTS5 shadow tables are storage FTS5 owns and reshapes across SQLite
    # builds; only the virtual table that declares them is our contract, so
    # comparing them reports library drift as archive schema drift.
    shadow_names = _fts_shadow_names(conn)
    return tuple((str(kind), str(name), _sql(sql)) for kind, name, sql in rows if str(name) not in shadow_names)


@dataclass(frozen=True, slots=True)
class SchemaManifest:
    tier: str
    version: int
    objects: tuple[tuple[str, str, str], ...]
    fingerprint: str

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection, tier: ArchiveTier) -> SchemaManifest:
        objects = _projection(conn)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        payload = {"tier": tier.value, "version": version, "objects": objects}
        fingerprint = hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        return cls(tier.value, version, objects, fingerprint)


@functools.cache
def _canonical_schema_manifest(tier: ArchiveTier, version: int, ddl: str) -> SchemaManifest:
    """Render one tier's declared DDL and return its semantic manifest.

    Cached because rendering executes the whole tier DDL into a fresh
    in-memory database and every read open asserts against the result.
    ``SchemaManifest`` is immutable, so one rendering is safely shared. The
    DDL is part of the key rather than re-read inside: it is a module
    constant in production, but keying on it means a caller that substitutes
    a different schema gets that schema's manifest and not a stale hit.
    """
    conn = sqlite3.connect(":memory:")
    try:
        if tier is ArchiveTier.EMBEDDINGS:
            from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

            loaded, error = try_load_sqlite_vec(conn)
            if not loaded:
                raise RuntimeError(f"cannot render embeddings schema without sqlite-vec: {error}")
        conn.executescript(ddl)
        if tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
            from polylogue.storage.sqlite.archive_tiers.schema_identity import DERIVED_SCHEMA_META_DDL

            conn.executescript(DERIVED_SCHEMA_META_DDL)
        if tier is ArchiveTier.INDEX:
            from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

            ensure_runtime_indexes_sync(conn)
        conn.execute(f"PRAGMA user_version = {version}")
        return SchemaManifest.from_connection(conn, tier)
    finally:
        conn.close()


def canonical_schema_manifest(tier: ArchiveTier, *, version: int | None = None) -> SchemaManifest:
    """Render one tier's declared DDL and return its semantic manifest."""
    return _canonical_schema_manifest(
        tier,
        int(ARCHIVE_VERSION_BY_TIER[tier] if version is None else version),
        ARCHIVE_DDL_BY_TIER[tier],
    )


def schema_manifest_diff(expected: SchemaManifest, actual: SchemaManifest) -> dict[str, object]:
    expected_by_name = {(kind, name): sql for kind, name, sql in expected.objects}
    actual_by_name = {(kind, name): sql for kind, name, sql in actual.objects}
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    variant_objects: dict[tuple[str, str], set[str]] = {}
    if expected.tier == ArchiveTier.INDEX.value and expected.version == actual.version:
        for variant in _INDEX_SAME_VERSION_SCHEMA_VARIANTS:
            if expected.version < variant.introduced_version:
                continue
            for object_name in variant.object_names:
                canonical = expected_by_name.get(object_name)
                if canonical is not None and variant.transformation == "remove_fts_bulk_guard":
                    variant_objects.setdefault(object_name, set()).add(_FTS_BULK_GUARD.sub("", canonical))

    wrong = sorted(
        (kind, name, expected_by_name[(kind, name)], actual_by_name[(kind, name)])
        for kind, name in set(expected_by_name) & set(actual_by_name)
        if expected_by_name[(kind, name)] != actual_by_name[(kind, name)]
        and actual_by_name[(kind, name)] not in variant_objects.get((kind, name), set())
    )
    return {"missing": missing, "extra": extra, "wrong_definition": wrong}


#: The message FTS surface is a derived read model inside the derived index:
#: contentless, trigger-maintained, and rebuildable from ``blocks``. Its
#: absence degrades search; the rest of the index stays readable, so a read
#: open admits a manifest diff confined to these objects.
#:
#: Membership is declared, not matched on the ``messages_fts`` name prefix.
#: ``messages_fts_identity`` is a plain declared table of ours rather than
#: FTS5 storage, and a later object sharing the prefix would otherwise be
#: admitted before anyone decided its absence is survivable.
MESSAGE_FTS_DEGRADABLE_OBJECTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("table", "messages_fts"),
        ("table", "messages_fts_identity"),
        ("trigger", "messages_fts_ai"),
        ("trigger", "messages_fts_ad"),
        ("trigger", "messages_fts_au"),
    }
)


def schema_manifest_diff_is_message_fts_only(diff: Mapping[str, object]) -> bool:
    """Report whether a non-empty manifest diff is confined to message FTS."""

    if diff.get("version"):
        return False
    objects: list[tuple[str, str]] = []
    for key in ("missing", "extra", "wrong_definition"):
        entries = diff.get(key)
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                return False
            objects.append((str(entry[0]), str(entry[1])))
    return bool(objects) and all(entry in MESSAGE_FTS_DEGRADABLE_OBJECTS for entry in objects)


def assert_schema_manifest(conn: sqlite3.Connection, tier: ArchiveTier) -> SchemaManifest:
    expected = canonical_schema_manifest(tier)
    actual = SchemaManifest.from_connection(conn, tier)
    diff = schema_manifest_diff(expected, actual)
    if actual.version != expected.version:
        diff["version"] = {"expected": expected.version, "actual": actual.version}
    if any(diff.values()):
        raise RuntimeError(f"{tier.value} schema semantic manifest mismatch: {json.dumps(diff, sort_keys=True)}")
    return actual


__all__ = [
    "MESSAGE_FTS_DEGRADABLE_OBJECTS",
    "SchemaManifest",
    "assert_schema_manifest",
    "canonical_schema_manifest",
    "schema_manifest_diff",
    "schema_manifest_diff_is_message_fts_only",
]
