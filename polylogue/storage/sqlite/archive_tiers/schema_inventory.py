"""Read-only six-tier schema and population census.

The canonical DDL is the declaration authority.  This module compares that
authority with one physical archive tuple and returns ephemeral evidence.  It
does not persist counts, dispositions, or task state.  In particular, a
failed count is an error, never an empty population.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

SchemaObjectType = Literal["table", "index", "trigger", "view", "column"]
GeneratedKind = Literal["stored", "virtual"]


class SchemaCensusError(RuntimeError):
    """The schema or population census cannot prove a complete result."""


@dataclass(frozen=True, slots=True)
class SchemaObject:
    """One declared or physical SQLite object, including table columns."""

    tier: ArchiveTier
    object_type: SchemaObjectType
    name: str
    table_name: str
    definition_sha256: str
    generated_kind: GeneratedKind | None = None
    virtual: bool = False

    @property
    def object_ref(self) -> str:
        if self.object_type == "column":
            return f"{self.tier.value}:column:{self.table_name}.{self.name}"
        return f"{self.tier.value}:{self.object_type}:{self.name}"


@dataclass(frozen=True, slots=True)
class TierSchemaCensus:
    """Schema and row evidence for one archive tier."""

    tier: ArchiveTier
    expected_version: int
    actual_version: int | None
    schema_identity: str
    file_identity: str | None
    file_sha256: str | None
    file_size: int | None
    objects: tuple[SchemaObject, ...]
    row_counts: tuple[tuple[str, int], ...]
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.errors and self.actual_version == self.expected_version


@dataclass(frozen=True, slots=True)
class SchemaCensus:
    """Ephemeral complete-tuple evidence."""

    archive_identity_digest: str
    observed_at_ns: int
    tiers: tuple[TierSchemaCensus, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(error for tier in self.tiers for error in tier.errors)

    @property
    def complete(self) -> bool:
        return not self.errors and len(self.tiers) == len(ArchiveTier) and all(tier.complete for tier in self.tiers)

    def to_payload(self) -> dict[str, object]:
        """Return privacy-safe machine-readable evidence."""
        return {
            "format": "polylogue-schema-census/v1",
            "archive_identity_digest": self.archive_identity_digest,
            "observed_at_ns": self.observed_at_ns,
            "complete": self.complete,
            "errors": list(self.errors),
            "tiers": [
                {
                    "tier": tier.tier.value,
                    "expected_version": tier.expected_version,
                    "actual_version": tier.actual_version,
                    "schema_identity": tier.schema_identity,
                    "file_identity": tier.file_identity,
                    "file_sha256": tier.file_sha256,
                    "file_size": tier.file_size,
                    "objects": [
                        {
                            "object_ref": obj.object_ref,
                            "object_type": obj.object_type,
                            "name": obj.name,
                            "table_name": obj.table_name,
                            "definition_sha256": obj.definition_sha256,
                            "generated_kind": obj.generated_kind,
                            "virtual": obj.virtual,
                        }
                        for obj in tier.objects
                    ],
                    "row_counts": dict(tier.row_counts),
                    "errors": list(tier.errors),
                }
                for tier in self.tiers
            ],
        }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _catalog_rows(connection: sqlite3.Connection) -> list[sqlite3.Row | tuple[object, ...]]:
    return connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
          AND type IN ('table', 'index', 'trigger', 'view')
        ORDER BY type, name
        """
    ).fetchall()


def _pragma_rows(connection: sqlite3.Connection, pragma: str, name: str) -> list[list[object]]:
    return [list(row) for row in connection.execute(f"PRAGMA {pragma}({_quote(name)})").fetchall()]


def _object_definition(
    connection: sqlite3.Connection,
    *,
    object_type: str,
    name: str,
    table_name: str,
    sql: str | None,
) -> str:
    payload: dict[str, object] = {
        "type": object_type,
        "name": name,
        "table_name": table_name,
        "sql": sql or "",
    }
    if object_type == "table":
        payload["table_xinfo"] = _pragma_rows(connection, "table_xinfo", name)
        payload["foreign_key_list"] = _pragma_rows(connection, "foreign_key_list", name)
    elif object_type == "index":
        payload["index_xinfo"] = _pragma_rows(connection, "index_xinfo", name)
    return _sha256(payload)


def _objects_from_connection(connection: sqlite3.Connection, tier: ArchiveTier) -> tuple[SchemaObject, ...]:
    objects: list[SchemaObject] = []
    for raw_type, raw_name, raw_table_name, raw_sql in _catalog_rows(connection):
        object_type = str(raw_type)
        name = str(raw_name)
        table_name = str(raw_table_name)
        sql = str(raw_sql) if raw_sql is not None else None
        objects.append(
            SchemaObject(
                tier=tier,
                object_type=object_type,  # type: ignore[arg-type]
                name=name,
                table_name=table_name,
                definition_sha256=_object_definition(
                    connection,
                    object_type=object_type,
                    name=name,
                    table_name=table_name,
                    sql=sql,
                ),
                virtual=sql is not None and sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE"),
            )
        )
        if object_type != "table":
            continue
        for row in _pragma_rows(connection, "table_xinfo", name):
            column_name = str(row[1])
            hidden = cast(int, row[6])
            generated_kind: GeneratedKind | None = None
            if hidden == 2:
                generated_kind = "virtual"
            elif hidden == 3:
                generated_kind = "stored"
            objects.append(
                SchemaObject(
                    tier=tier,
                    object_type="column",
                    name=column_name,
                    table_name=name,
                    definition_sha256=_sha256({"table": name, "column": row}),
                    generated_kind=generated_kind,
                    virtual=generated_kind == "virtual",
                )
            )
    return tuple(objects)


def _canonical_connection(tier: ArchiveTier) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    if tier is ArchiveTier.EMBEDDINGS:
        loaded, error = try_load_sqlite_vec(connection)
        if not loaded:
            connection.close()
            raise SchemaCensusError(f"canonical {tier.value} tier unavailable: sqlite-vec: {error or 'not loadable'}")
    try:
        initialize_archive_tier(connection, tier)
    except Exception:
        connection.close()
        raise
    connection.execute("PRAGMA query_only = ON")
    return connection


def canonical_schema_objects(tier: ArchiveTier) -> tuple[SchemaObject, ...]:
    """Derive the complete declaration universe from fresh canonical DDL."""
    connection = _canonical_connection(tier)
    try:
        return _objects_from_connection(connection, tier)
    finally:
        connection.close()


def _schema_identity(tier: ArchiveTier, objects: tuple[SchemaObject, ...]) -> str:
    return _sha256(
        {
            "tier": tier.value,
            "version": ARCHIVE_VERSION_BY_TIER[tier],
            "objects": [{"object_ref": obj.object_ref, "definition_sha256": obj.definition_sha256} for obj in objects],
        }
    )


def _open_read_only(path: Path, *, tier: ArchiveTier) -> sqlite3.Connection:
    if not path.is_file():
        raise SchemaCensusError(f"tier file is missing: {path.name}")
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        if tier is ArchiveTier.EMBEDDINGS:
            loaded, error = try_load_sqlite_vec(connection)
            if not loaded:
                connection.close()
                raise SchemaCensusError(f"embeddings: sqlite-vec unavailable: {error or 'not loadable'}")
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as exc:
        raise SchemaCensusError(f"tier file is unreadable: {path.name}: {exc}") from exc


def _count_tables(connection: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    counts: list[tuple[str, int]] = []
    for raw_type, raw_name, _raw_table_name, _raw_sql in _catalog_rows(connection):
        if str(raw_type) != "table":
            continue
        name = str(raw_name)
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {_quote(name)}").fetchone()
            if row is None or row[0] is None:
                raise SchemaCensusError(f"row count returned no value: {name}")
            counts.append((name, int(row[0])))
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise SchemaCensusError(f"row count failed for {name}: {exc}") from exc
    return tuple(counts)


def _tier_path(location: ArchiveLocation, tier: ArchiveTier) -> TierFileIdentity:
    return location.active_tier(tier.value)


def _archive_identity(tiers: tuple[TierSchemaCensus, ...]) -> str:
    return _sha256(
        {
            "tiers": [
                {
                    "tier": tier.tier.value,
                    "file_identity": tier.file_identity,
                    "file_sha256": tier.file_sha256,
                    "schema_identity": tier.schema_identity,
                    "actual_version": tier.actual_version,
                }
                for tier in tiers
            ]
        }
    )


def capture_schema_census(
    archive_root: Path,
    *,
    observed_at_ns: int,
    count_rows: bool = True,
    hash_files: bool = True,
) -> SchemaCensus:
    """Capture one complete six-tier tuple without opening a write handle.

    Missing, unreadable, or failed-count tiers are represented as errors and
    make the returned census incomplete.  Callers must not reinterpret those
    errors as an empty PASS.
    """
    location = ArchiveLocation.resolve(archive_root)
    tiers: list[TierSchemaCensus] = []
    for tier in ArchiveTier:
        errors: list[str] = []
        try:
            canonical = canonical_schema_objects(tier)
        except SchemaCensusError as exc:
            canonical = ()
            errors.append(str(exc))
        canonical_identity = _schema_identity(tier, canonical)
        identity = _tier_path(location, tier)
        actual_version: int | None = None
        physical: tuple[SchemaObject, ...] = ()
        row_counts: tuple[tuple[str, int], ...] = ()
        file_sha256: str | None = None
        file_size: int | None = None
        connection: sqlite3.Connection | None = None
        if not identity.exists:
            errors.append(f"{tier.value}: tier file is missing")
        else:
            try:
                file_size = identity.resolved_path.stat().st_size
                if hash_files:
                    file_sha256 = _file_sha256(identity.resolved_path)
                connection = _open_read_only(identity.resolved_path, tier=tier)
                actual_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                physical = _objects_from_connection(connection, tier)
                if count_rows:
                    row_counts = _count_tables(connection)
            except SchemaCensusError as exc:
                errors.append(f"{tier.value}: {exc}")
            except (OSError, sqlite3.Error) as exc:
                errors.append(f"{tier.value}: unreadable census: {exc}")
            finally:
                if connection is not None:
                    connection.close()
        canonical_refs = {obj.object_ref for obj in canonical}
        physical_refs = {obj.object_ref for obj in physical}
        missing = sorted(canonical_refs - physical_refs)
        unexpected = sorted(physical_refs - canonical_refs)
        if missing:
            errors.append(f"{tier.value}: missing declared objects: {', '.join(missing)}")
        if unexpected:
            errors.append(f"{tier.value}: unexpected objects: {', '.join(unexpected)}")
        canonical_by_ref = {obj.object_ref: obj for obj in canonical}
        physical_by_ref = {obj.object_ref: obj for obj in physical}
        changed = sorted(
            object_ref
            for object_ref in canonical_refs & physical_refs
            if canonical_by_ref[object_ref].definition_sha256 != physical_by_ref[object_ref].definition_sha256
        )
        if changed:
            errors.append(f"{tier.value}: changed declared objects: {', '.join(changed)}")
        if actual_version != ARCHIVE_VERSION_BY_TIER[tier] and actual_version is not None:
            errors.append(
                f"{tier.value}: schema version {actual_version} does not match canonical {ARCHIVE_VERSION_BY_TIER[tier]}"
            )
        tiers.append(
            TierSchemaCensus(
                tier=tier,
                expected_version=ARCHIVE_VERSION_BY_TIER[tier],
                actual_version=actual_version,
                schema_identity=canonical_identity,
                file_identity=identity.stable_id if identity.exists else None,
                file_sha256=file_sha256,
                file_size=file_size,
                objects=physical,
                row_counts=row_counts,
                errors=tuple(errors),
            )
        )
    return SchemaCensus(
        archive_identity_digest=_archive_identity(tuple(tiers)),
        observed_at_ns=observed_at_ns,
        tiers=tuple(tiers),
    )


def assert_complete_census(census: SchemaCensus) -> None:
    """Fail closed when a census omitted a tier or declared object."""
    if not census.complete:
        detail = "; ".join(census.errors) or "tier denominator is incomplete"
        raise SchemaCensusError(detail)


__all__ = [
    "SchemaCensus",
    "SchemaCensusError",
    "SchemaObject",
    "TierSchemaCensus",
    "assert_complete_census",
    "canonical_schema_objects",
    "capture_schema_census",
]
