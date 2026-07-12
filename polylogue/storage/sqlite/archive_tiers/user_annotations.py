"""Durable user-tier persistence for annotation schemas and batch provenance.

Writer module: user.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import cast

from polylogue.annotations.batch import AnnotationBatch, AnnotationBatchError
from polylogue.annotations.schema import AnnotationSchema, AnnotationSchemaError
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.json import loads as json_loads
from polylogue.core.refs import normalize_object_ref_text


@dataclass(frozen=True, slots=True)
class DurableAnnotationSchema:
    """One immutable schema definition resolved from ``user.db``."""

    schema: AnnotationSchema
    definition_json: str
    definition_sha256: str
    registered_at_ms: int


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _schema_from_row(row: sqlite3.Row) -> DurableAnnotationSchema:
    definition_json = str(row["definition_json"])
    stored_fingerprint = str(row["definition_sha256"])
    actual_fingerprint = hashlib.sha256(definition_json.encode("utf-8")).hexdigest()
    if actual_fingerprint != stored_fingerprint:
        raise AnnotationSchemaError(
            f"durable annotation schema {row['schema_id']!r}@v{row['schema_version']} has a fingerprint mismatch"
        )
    schema = AnnotationSchema.from_canonical_definition_json(definition_json)
    if schema.schema_id != str(row["schema_id"]) or schema.version != int(row["schema_version"]):
        raise AnnotationSchemaError("durable annotation schema row identity disagrees with its definition JSON")
    if schema.definition_fingerprint != stored_fingerprint:
        raise AnnotationSchemaError("durable annotation schema definition does not match its stored fingerprint")
    return DurableAnnotationSchema(
        schema=schema,
        definition_json=definition_json,
        definition_sha256=stored_fingerprint,
        registered_at_ms=int(row["registered_at_ms"]),
    )


def read_durable_annotation_schema(
    conn: sqlite3.Connection,
    schema_id: str,
    version: int | None = None,
) -> DurableAnnotationSchema | None:
    """Read one schema version, defaulting to the highest durable version."""

    conn.row_factory = sqlite3.Row
    if version is None:
        row = conn.execute(
            """
            SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            FROM annotation_schemas
            WHERE schema_id = ?
            ORDER BY schema_version DESC
            LIMIT 1
            """,
            (schema_id,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            FROM annotation_schemas
            WHERE schema_id = ? AND schema_version = ?
            """,
            (schema_id, version),
        ).fetchone()
    return _schema_from_row(row) if row is not None else None


def list_durable_annotation_schemas(conn: sqlite3.Connection) -> tuple[DurableAnnotationSchema, ...]:
    """List every durable schema definition in stable identity order."""

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
        FROM annotation_schemas
        ORDER BY schema_id, schema_version
        """
    ).fetchall()
    return tuple(_schema_from_row(row) for row in rows)


def persist_annotation_schema(
    conn: sqlite3.Connection,
    schema: AnnotationSchema,
    *,
    registered_at_ms: int,
) -> DurableAnnotationSchema:
    """Persist one immutable definition; identical reuse is idempotent, drift fails closed."""

    if not _is_nonnegative_int(registered_at_ms):
        raise AnnotationSchemaError("registered_at_ms cannot be negative")
    conn.execute("PRAGMA foreign_keys = ON")
    existing = read_durable_annotation_schema(conn, schema.schema_id, schema.version)
    if existing is not None:
        if existing.definition_json != schema.canonical_definition_json():
            raise AnnotationSchemaError(
                f"annotation schema {schema.qualified_id!r} already exists with an incompatible durable definition"
            )
        return existing
    definition_json = schema.canonical_definition_json()
    conn.execute(
        """
        INSERT INTO annotation_schemas (
            schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (schema.schema_id, schema.version, definition_json, schema.definition_fingerprint, registered_at_ms),
    )
    persisted = read_durable_annotation_schema(conn, schema.schema_id, schema.version)
    if persisted is None:  # pragma: no cover - INSERT followed by same-connection SELECT
        raise AnnotationSchemaError(f"annotation schema {schema.qualified_id!r} was not persisted")
    return persisted


def _json_array_text(value: tuple[object, ...]) -> str:
    return json.dumps(list(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_document_text(value: JSONDocument) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    parsed = json_loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise AnnotationBatchError(f"{context} must decode to an array of strings")
    return tuple(cast(list[str], parsed))


def _load_document_tuple(value: object, *, context: str) -> tuple[JSONDocument, ...]:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    parsed = json_loads(value)
    if not isinstance(parsed, list):
        raise AnnotationBatchError(f"{context} must decode to an array")
    return tuple(require_json_document(item, context=context) for item in parsed)


def _load_document(value: object, *, context: str) -> JSONDocument:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    return require_json_document(json_loads(value), context=context)


def _batch_from_row(row: sqlite3.Row) -> AnnotationBatch:
    return AnnotationBatch(
        batch_id=str(row["batch_id"]),
        schema_id=str(row["schema_id"]),
        schema_version=int(row["schema_version"]),
        target_ref=str(row["target_ref"]),
        source_result_ref=str(row["source_result_ref"]),
        actor_ref=str(row["actor_ref"]),
        model_ref=str(row["model_ref"]),
        prompt_ref=str(row["prompt_ref"]),
        total_count=int(row["total_count"]),
        valid_count=int(row["valid_count"]),
        invalid_count=int(row["invalid_count"]),
        abstained_count=int(row["abstained_count"]),
        assertion_refs=_load_string_tuple(row["assertion_refs_json"], context="annotation batch assertion_refs"),
        validation_failures=_load_document_tuple(
            row["validation_failures_json"],
            context="annotation batch validation_failures",
        ),
        metadata=_load_document(row["metadata_json"], context="annotation batch metadata"),
        created_at_ms=int(row["created_at_ms"]),
    )


def read_annotation_batch(conn: sqlite3.Connection, batch_id: str) -> AnnotationBatch | None:
    """Read one immutable annotation batch by id."""

    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT batch_id, schema_id, schema_version, target_ref, source_result_ref,
               actor_ref, model_ref, prompt_ref, total_count, valid_count,
               invalid_count, abstained_count, assertion_refs_json,
               validation_failures_json, metadata_json, created_at_ms
        FROM annotation_batches
        WHERE batch_id = ?
        """,
        (batch_id,),
    ).fetchone()
    return _batch_from_row(row) if row is not None else None


def persist_annotation_batch(conn: sqlite3.Connection, batch: AnnotationBatch) -> AnnotationBatch:
    """Persist write-once batch provenance; incompatible id reuse fails closed."""

    conn.execute("PRAGMA foreign_keys = ON")
    schema = read_durable_annotation_schema(conn, batch.schema_id, batch.schema_version)
    if schema is None:
        raise AnnotationBatchError(f"annotation batch references unknown schema {batch.qualified_schema_id!r}")
    existing = read_annotation_batch(conn, batch.batch_id)
    if existing is not None:
        if existing != batch:
            raise AnnotationBatchError(
                f"annotation batch {batch.batch_ref!r} already exists with incompatible provenance"
            )
        return existing
    conn.execute(
        """
        INSERT INTO annotation_batches (
            batch_id, schema_id, schema_version, target_ref, source_result_ref,
            actor_ref, model_ref, prompt_ref, total_count, valid_count,
            invalid_count, abstained_count, assertion_refs_json,
            validation_failures_json, metadata_json, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch.batch_id,
            batch.schema_id,
            batch.schema_version,
            batch.target_ref,
            batch.source_result_ref,
            batch.actor_ref,
            batch.model_ref,
            batch.prompt_ref,
            batch.total_count,
            batch.valid_count,
            batch.invalid_count,
            batch.abstained_count,
            _json_array_text(tuple(batch.assertion_refs)),
            _json_array_text(tuple(batch.validation_failures)),
            _json_document_text(batch.metadata),
            batch.created_at_ms,
        ),
    )
    persisted = read_annotation_batch(conn, batch.batch_id)
    if persisted is None:  # pragma: no cover - INSERT followed by same-connection SELECT
        raise AnnotationBatchError(f"annotation batch {batch.batch_ref!r} was not persisted")
    return persisted


def list_annotation_batches(
    conn: sqlite3.Connection,
    *,
    schema_id: str | None = None,
    schema_version: int | None = None,
    target_ref: str | None = None,
    limit: int | None = None,
) -> tuple[AnnotationBatch, ...]:
    """List batches with optional schema/target filters, newest first."""

    if schema_version is not None and schema_id is None:
        raise AnnotationBatchError("schema_version filter requires schema_id")
    where: list[str] = []
    params: list[object] = []
    if schema_id is not None:
        where.append("schema_id = ?")
        params.append(schema_id)
    if schema_version is not None:
        where.append("schema_version = ?")
        params.append(schema_version)
    if target_ref is not None:
        where.append("target_ref = ?")
        params.append(normalize_object_ref_text(target_ref))
    query = """
        SELECT batch_id, schema_id, schema_version, target_ref, source_result_ref,
               actor_ref, model_ref, prompt_ref, total_count, valid_count,
               invalid_count, abstained_count, assertion_refs_json,
               validation_failures_json, metadata_json, created_at_ms
        FROM annotation_batches
    """
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY created_at_ms DESC, batch_id"
    if limit is not None:
        if not _is_nonnegative_int(limit):
            raise AnnotationBatchError("limit cannot be negative")
        query += " LIMIT ?"
        params.append(limit)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, tuple(params)).fetchall()
    return tuple(_batch_from_row(row) for row in rows)


__all__ = [
    "DurableAnnotationSchema",
    "list_annotation_batches",
    "list_durable_annotation_schemas",
    "persist_annotation_batch",
    "persist_annotation_schema",
    "read_annotation_batch",
    "read_durable_annotation_schema",
]
