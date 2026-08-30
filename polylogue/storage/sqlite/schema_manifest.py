"""Canonical semantic manifests for SQLite archive tiers.

``PRAGMA user_version`` is a useful migration cursor, but it is not a proof
that the objects belonging to that version are present.  This module compares
the normalized ``sqlite_master`` projection with the same projection produced
by the canonical create route.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_WS = re.compile(r"\s+")


def _sql(value: str | None) -> str:
    return _WS.sub(" ", (value or "").strip()).lower()


def _projection(conn: sqlite3.Connection) -> tuple[tuple[str, str, str], ...]:
    rows = conn.execute(
        """SELECT type, name, sql FROM sqlite_master
           WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
           ORDER BY type, name"""
    ).fetchall()
    return tuple((str(kind), str(name), _sql(sql)) for kind, name, sql in rows)


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


def canonical_schema_manifest(tier: ArchiveTier, *, version: int | None = None) -> SchemaManifest:
    """Render one tier's declared DDL and return its semantic manifest."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ARCHIVE_DDL_BY_TIER[tier])
        if tier is ArchiveTier.INDEX:
            from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

            ensure_runtime_indexes_sync(conn)
        conn.execute(f"PRAGMA user_version = {int(ARCHIVE_VERSION_BY_TIER[tier] if version is None else version)}")
        return SchemaManifest.from_connection(conn, tier)
    finally:
        conn.close()


def schema_manifest_diff(expected: SchemaManifest, actual: SchemaManifest) -> dict[str, object]:
    expected_by_name = {(kind, name): sql for kind, name, sql in expected.objects}
    actual_by_name = {(kind, name): sql for kind, name, sql in actual.objects}
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    wrong = sorted(
        (kind, name, expected_by_name[(kind, name)], actual_by_name[(kind, name)])
        for kind, name in set(expected_by_name) & set(actual_by_name)
        if expected_by_name[(kind, name)] != actual_by_name[(kind, name)]
    )
    return {"missing": missing, "extra": extra, "wrong_definition": wrong}


def assert_schema_manifest(conn: sqlite3.Connection, tier: ArchiveTier) -> SchemaManifest:
    expected = canonical_schema_manifest(tier)
    actual = SchemaManifest.from_connection(conn, tier)
    diff = schema_manifest_diff(expected, actual)
    if actual.version != expected.version:
        diff["version"] = {"expected": expected.version, "actual": actual.version}
    if any(diff.values()):
        raise RuntimeError(f"{tier.value} schema semantic manifest mismatch: {json.dumps(diff, sort_keys=True)}")
    return actual


__all__ = ["SchemaManifest", "assert_schema_manifest", "canonical_schema_manifest", "schema_manifest_diff"]
