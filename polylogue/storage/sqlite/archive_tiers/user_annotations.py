"""Durable user-tier persistence for annotation schemas and batch provenance.

Writer module: user.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from polylogue.annotations.batch import AnnotationBatch, AnnotationBatchError
from polylogue.annotations.schema import (
    BUILTIN_ANNOTATION_SCHEMAS,
    AnnotationSchema,
    AnnotationSchemaError,
    normalize_annotation_target_ref,
)
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.json import loads as json_loads
from polylogue.core.refs import ObjectRef, normalize_durable_object_ref_text, normalize_object_ref_text

if TYPE_CHECKING:
    from polylogue.annotations.import_spill import AnnotationImportSpill


def annotation_batch_provenance_digest(conn: sqlite3.Connection, batch_id: str) -> str | None:
    """Hash the existing immutable batch without decoding its collections."""
    from contextlib import closing

    from polylogue.annotations.batch import ANNOTATION_BATCH_PROVENANCE_FORMAT
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.core.digest import REFERENCE, canonical_bytes
    from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection
    from polylogue.storage.sqlite.literal_cells import stream_literal_blob

    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT rowid, batch_id, schema_id, schema_version, target_ref, source_result_ref,
                  actor_ref, model_ref, prompt_ref, total_count, valid_count,
                  invalid_count, abstained_count, created_at_ms,
                  length(CAST(assertion_refs_json AS BLOB)) AS refs_bytes,
                  length(CAST(validation_failures_json AS BLOB)) AS failures_bytes,
                  length(CAST(metadata_json AS BLOB)) AS metadata_bytes
           FROM annotation_batches WHERE batch_id=?""",
        (batch_id,),
    ).fetchone()
    if row is None:
        return None
    owner = native_sql_owner_for_connection(conn)
    if owner is None:
        raise AnnotationBatchError("batch digest requires its retained native read owner")
    fields = {
        key: row[key] for key in dict(row) if key not in {"rowid", "refs_bytes", "failures_bytes", "metadata_bytes"}
    }
    fields.update(
        format=ANNOTATION_BATCH_PROVENANCE_FORMAT, assertion_refs=None, validation_failures=None, metadata=None
    )
    columns = {
        "assertion_refs": ("assertion_refs_json", "refs_bytes"),
        "validation_failures": ("validation_failures_json", "failures_bytes"),
        "metadata": ("metadata_json", "metadata_bytes"),
    }
    digest = hashlib.sha256()
    digest.update(b"{")
    for ordinal, key in enumerate(sorted(fields)):
        check_compute_cancelled()
        if ordinal:
            digest.update(b",")
        digest.update(canonical_bytes(key, REFERENCE) + b":")
        if key in columns:
            column, length_key = columns[key]
            with (
                owner.readonly_blob("annotation_batches", column, int(row["rowid"])) as blob,
                closing(stream_literal_blob(blob, int(row[length_key]), check_compute_cancelled)) as chunks,
            ):
                for chunk in chunks:
                    digest.update(chunk)
        else:
            digest.update(canonical_bytes(fields[key], REFERENCE))
    digest.update(b"}")
    return digest.hexdigest()


def _require_durable_batch_ref(ref: str) -> None:
    """Admit stored batch coordinates without rewriting their provenance."""
    if ref.startswith(("phase:", "work_event:")):
        return
    try:
        canonical = normalize_durable_object_ref_text(ref)
        if canonical != ref:
            raise ValueError("annotation batch ref is not canonical")
    except ValueError as exc:
        raise AnnotationBatchError("annotation batch refs require canonical durable object coordinates") from exc


def _require_durable_batch_header(header: JSONDocument) -> None:
    for field in ("target_ref", "source_result_ref", "actor_ref", "model_ref", "prompt_ref"):
        value = header[field]
        if not isinstance(value, str):
            raise AnnotationBatchError(f"annotation batch {field} must be a string")
        _require_durable_batch_ref(value)


def persist_spilled_annotation_batch(conn: sqlite3.Connection, batch: AnnotationImportSpill) -> None:
    """Publish sealed exact batch evidence in its existing atomic User write."""
    from contextlib import closing

    from polylogue.storage.sqlite.literal_cells import write_literal_text

    _require_durable_batch_header(batch.header)
    with closing(batch.assertion_refs()) as refs:
        for ref in refs:
            _require_durable_batch_ref(ref)
    expected = batch.provenance_digest()
    existing = annotation_batch_provenance_digest(conn, str(batch.header["batch_id"]))
    if existing is not None:
        if existing != expected:
            raise AnnotationBatchError(
                f"annotation batch {batch.batch_ref!r} already exists with incompatible provenance"
            )
        return
    schema = read_durable_annotation_schema(
        conn, str(batch.header["schema_id"]), int(cast(int, batch.header["schema_version"]))
    )
    if schema is None or not schema.schema.accepts_target_kind(str(batch.header["target_ref"])):
        raise AnnotationBatchError("annotation batch schema or target is not admitted")
    fields = (
        "batch_id",
        "schema_id",
        "schema_version",
        "target_ref",
        "source_result_ref",
        "actor_ref",
        "model_ref",
        "prompt_ref",
        "total_count",
        "valid_count",
        "invalid_count",
        "abstained_count",
        "created_at_ms",
    )
    # The durable columns have JSON/count CHECKs. A zero-filled placeholder
    # cannot satisfy them: fill disposable TEXT cells first, then publish all
    # checked values in one INSERT. SQLite may materialize the native cells
    # for its CHECK/record write; Python only transfers bounded chunks.
    conn.execute("CREATE TEMP TABLE annotation_import_literals (refs TEXT, failures TEXT, metadata TEXT)")
    cursor = conn.execute("INSERT INTO temp.annotation_import_literals VALUES ('', '', '')")
    rowid = cursor.lastrowid
    cursor.close()
    assert rowid is not None
    for column, destination in (
        ("assertion_refs_json", "refs"),
        ("validation_failures_json", "failures"),
        ("metadata_json", "metadata"),
    ):
        with closing(batch.column_chunks(column)) as chunks:
            length = sum(len(chunk) for chunk in chunks)
        write_literal_text(
            conn,
            "annotation_import_literals",
            destination,
            rowid,
            schema="temp",
            byte_length=length,
            chunks=partial(batch.column_chunks, column),
        )
    conn.execute(
        """INSERT INTO annotation_batches (
               batch_id, schema_id, schema_version, target_ref, source_result_ref,
               actor_ref, model_ref, prompt_ref, total_count, valid_count,
               invalid_count, abstained_count, created_at_ms,
               assertion_refs_json, validation_failures_json, metadata_json
           ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, refs, failures, metadata
             FROM temp.annotation_import_literals WHERE rowid=?""",
        (*tuple(batch.header[key] for key in fields), rowid),
    )
    conn.execute("DROP TABLE temp.annotation_import_literals")
    if annotation_batch_provenance_digest(conn, str(batch.header["batch_id"])) != expected:
        raise AnnotationBatchError("annotation batch changed during durable publication")


@dataclass(frozen=True, slots=True)
class DurableAnnotationSchema:
    """One immutable schema definition resolved from ``user.db``."""

    schema: AnnotationSchema
    definition_json: str
    definition_sha256: str
    registered_at_ms: int


@dataclass(frozen=True, slots=True)
class AnnotationBatchIdentity:
    """Only the immutable batch operands required by one assertion writer."""

    schema_id: str
    schema_version: int
    target_ref: str
    actor_ref: str

    @property
    def qualified_schema_id(self) -> str:
        return f"{self.schema_id}@v{self.schema_version}"


def read_annotation_batch_identity(conn: sqlite3.Connection, batch_id: str) -> AnnotationBatchIdentity | None:
    row = conn.execute(
        "SELECT schema_id, schema_version, target_ref, actor_ref FROM annotation_batches WHERE batch_id=?",
        (batch_id,),
    ).fetchone()
    return AnnotationBatchIdentity(str(row[0]), int(row[1]), str(row[2]), str(row[3])) if row is not None else None


def annotation_batch_declares_assertion(conn: sqlite3.Connection, batch_id: str, assertion_ref: str) -> bool:
    """Check original durable membership without decoding its complete roster."""
    return bool(
        conn.execute(
            """SELECT EXISTS(SELECT 1 FROM annotation_batches b, json_each(b.assertion_refs_json) a
                         WHERE b.batch_id=? AND a.type='text' AND a.value=?)""",
            (batch_id, assertion_ref),
        ).fetchone()[0]
        == 1
    )


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
    *,
    schema: str | None = None,
) -> DurableAnnotationSchema | None:
    """Read one schema version, defaulting to the highest durable version."""

    if schema is not None and not schema.replace("_", "").isalnum():
        raise ValueError(f"invalid SQLite schema name: {schema!r}")
    table = f"{schema}.annotation_schemas" if schema is not None else "annotation_schemas"
    conn.row_factory = sqlite3.Row
    if version is None:
        row = conn.execute(
            f"""
            SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            FROM {table}
            WHERE schema_id = ?
            ORDER BY schema_version DESC
            LIMIT 1
            """,
            (schema_id,),
        ).fetchone()
    else:
        row = conn.execute(
            f"""
            SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            FROM {table}
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
    schema.require_live_targets()
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


def persist_builtin_annotation_schemas(
    conn: sqlite3.Connection,
    *,
    registered_at_ms: int = 0,
) -> tuple[DurableAnnotationSchema, ...]:
    """Ensure every packaged annotation construct has an immutable user-tier row.

    Built-in vocabulary growth is data-only: ``annotation_schemas`` is already
    the versioned registry and ``assertions.kind`` is plain ``TEXT``. Replaying
    this function on a same-version ``user.db`` therefore adds missing rows
    without a durable-tier schema migration, while ``persist_annotation_schema``
    still fails closed if an existing identity has drifted.
    """

    return tuple(
        persist_annotation_schema(conn, schema, registered_at_ms=registered_at_ms)
        for schema in BUILTIN_ANNOTATION_SCHEMAS
    )


def _json_array_text(value: tuple[object, ...]) -> str:
    return json.dumps(list(value), allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_document_text(value: JSONDocument) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json_value(value: str, *, context: str) -> object:
    try:
        return json_loads(value)
    except ValueError as exc:
        raise AnnotationBatchError(f"{context} is not valid finite JSON") from exc


def _load_string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    parsed = _load_json_value(value, context=context)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise AnnotationBatchError(f"{context} must decode to an array of strings")
    return tuple(cast(list[str], parsed))


def _load_document_tuple(value: object, *, context: str) -> tuple[JSONDocument, ...]:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    parsed = _load_json_value(value, context=context)
    if not isinstance(parsed, list):
        raise AnnotationBatchError(f"{context} must decode to an array")
    try:
        return tuple(require_json_document(item, context=context) for item in parsed)
    except TypeError as exc:
        raise AnnotationBatchError(f"{context} must contain only JSON objects") from exc


def _load_document(value: object, *, context: str) -> JSONDocument:
    if not isinstance(value, str):
        raise AnnotationBatchError(f"{context} is not stored as JSON text")
    parsed = _load_json_value(value, context=context)
    try:
        return require_json_document(parsed, context=context)
    except TypeError as exc:
        raise AnnotationBatchError(f"{context} must decode to a JSON object") from exc


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


@dataclass(frozen=True, slots=True)
class AnnotationBatchReadPage:
    """One immutable batch header and exact evidence window."""

    header: JSONDocument
    items: tuple[JSONDocument, ...]
    total: int
    offset: int
    next_offset: int | None


def read_annotation_batch_page(
    conn: sqlite3.Connection, batch_id: str, *, limit: int, offset: int, schema: str | None = None
) -> AnnotationBatchReadPage | None:
    """Decode only the selected evidence page on the caller's read transaction.

    SQLite parses the retained JSON values and may scan/sort their entries per
    offset page. Python does not load the full amplified failure document.
    """
    if limit < 1 or offset < 0:
        raise ValueError("annotation page limit must be positive and offset nonnegative")
    if schema is not None and not schema.replace("_", "").isalnum():
        raise ValueError("invalid SQLite schema")
    table = "annotation_batches" if schema is None else f"{schema}.annotation_batches"
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"""SELECT batch_id, schema_id, schema_version, target_ref, source_result_ref,
                  actor_ref, model_ref, prompt_ref, total_count, valid_count,
                  invalid_count, abstained_count, metadata_json, created_at_ms,
                  json_array_length(assertion_refs_json) AS refs_count,
                  json_array_length(validation_failures_json) AS failures_count
           FROM {table} WHERE batch_id = ?""",
        (batch_id,),
    ).fetchone()
    if row is None:
        return None
    if row["refs_count"] != row["valid_count"] or row["failures_count"] != row["invalid_count"]:
        raise AnnotationBatchError("annotation batch evidence counts disagree with its summary")
    header = require_json_document(
        {
            key: value
            for key, value in dict(row).items()
            if key not in {"metadata_json", "refs_count", "failures_count"}
        },
        context="annotation batch header",
    )
    header["metadata"] = _load_document(row["metadata_json"], context="annotation batch metadata")
    header["batch_ref"] = f"annotation-batch:{batch_id}"
    header["qualified_schema_id"] = f"{row['schema_id']}@v{row['schema_version']}"
    total = int(row["valid_count"]) + int(
        conn.execute(
            f"""SELECT COALESCE(SUM(CASE WHEN json_type(f.value, '$.errors') = 'array'
                                    AND json_array_length(f.value, '$.errors') > 0
                              THEN json_array_length(f.value, '$.errors') ELSE 1 END), 0)
           FROM {table} b, json_each(b.validation_failures_json) f
           WHERE b.batch_id = ?""",
            (batch_id,),
        ).fetchone()[0]
    )
    # Order only ordinal keys before the page cut. Amplified error documents
    # and repeated failure metadata are serialized only for selected items.
    selected = conn.execute(
        f"""WITH failures AS MATERIALIZED (
          SELECT CAST(f.key AS INTEGER) AS ordinal, f.value AS original,
                 json_remove(f.value, '$.errors') AS fields,
                 CASE WHEN json_type(f.value, '$.errors') = 'array'
                      THEN json_extract(f.value, '$.errors') ELSE '[]' END AS errors
          FROM {table} b, json_each(b.validation_failures_json) f WHERE b.batch_id = ?
        ), entries AS (
          SELECT 0 AS section, CAST(a.key AS INTEGER) AS ordinal, -1 AS error_ordinal
          FROM {table} b, json_each(b.assertion_refs_json) a WHERE b.batch_id = ?
          UNION ALL
          SELECT 1, f.ordinal, CAST(e.key AS INTEGER) FROM failures f, json_each(f.errors) e
          UNION ALL
          SELECT 1, f.ordinal, -1 FROM failures f WHERE json_array_length(f.errors) = 0
        ), selected AS MATERIALIZED (
          SELECT section, ordinal, error_ordinal FROM entries
          ORDER BY section, ordinal, error_ordinal LIMIT ? OFFSET ?
        ) SELECT CASE
          WHEN s.section = 0 THEN json_object('kind', 'assertion', 'ordinal', s.ordinal,
                                             'assertion_ref', json_extract(b.assertion_refs_json, '$[' || s.ordinal || ']'))
          WHEN s.error_ordinal = -1 THEN json_object('kind', 'validation-failure',
                                                    'failure_ordinal', s.ordinal, 'failure', json(f.original))
          ELSE json_object('kind', 'validation-error', 'failure_ordinal', s.ordinal,
                           'error_ordinal', s.error_ordinal, 'failure', json(f.fields),
                           'error', json(f.errors -> ('$[' || s.error_ordinal || ']')))
          END AS item
        FROM selected s LEFT JOIN failures f ON s.section = 1 AND s.ordinal = f.ordinal
                        JOIN {table} b ON b.batch_id = ?
        ORDER BY s.section, s.ordinal, s.error_ordinal""",
        (batch_id, batch_id, limit + 1, offset, batch_id),
    ).fetchall()
    items = tuple(_load_document(record[0], context="annotation batch page item") for record in selected[:limit])
    return AnnotationBatchReadPage(header, items, total, offset, offset + len(items) if len(selected) > limit else None)


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

    # AnnotationBatch is also used to decode historical provenance, so its
    # constructor keeps the retired positional grammar readable. Enforce the
    # durable rule only at this write boundary: new batches may retain a
    # positional block selector only after its creator bound it to a stable
    # block ID under the source read that authorized the write.
    durable_refs = (
        batch.target_ref,
        batch.source_result_ref,
        batch.actor_ref,
        batch.model_ref,
        batch.prompt_ref,
        *batch.assertion_refs,
    )
    for ref in durable_refs:
        if ref.startswith(("phase:", "work_event:")):
            continue
        try:
            normalized = normalize_durable_object_ref_text(ref)
        except ValueError as exc:
            raise AnnotationBatchError("annotation batch refs must use stable block identities") from exc
        if normalized != ref:
            raise AnnotationBatchError("annotation batch refs must be normalized before durable storage")

    candidate_provenance = batch.canonical_provenance_bytes()
    provenance = batch.provenance_document()
    _require_durable_batch_header(provenance)
    for ref in batch.assertion_refs:
        _require_durable_batch_ref(ref)
    provenance_metadata = require_json_document(
        provenance["metadata"],
        context="annotation batch provenance metadata",
    )
    provenance_failures_value = provenance["validation_failures"]
    if not isinstance(provenance_failures_value, list):  # pragma: no cover - internal snapshot invariant
        raise AnnotationBatchError("annotation batch provenance validation_failures must be an array")
    try:
        provenance_failures = tuple(
            require_json_document(item, context="annotation batch provenance validation failure")
            for item in provenance_failures_value
        )
    except TypeError as exc:  # pragma: no cover - internal snapshot invariant
        raise AnnotationBatchError(
            "annotation batch provenance validation_failures must contain only JSON objects"
        ) from exc
    conn.execute("PRAGMA foreign_keys = ON")
    schema = read_durable_annotation_schema(conn, batch.schema_id, batch.schema_version)
    if schema is None:
        raise AnnotationBatchError(f"annotation batch references unknown schema {batch.qualified_schema_id!r}")
    if not schema.schema.accepts_target_kind(batch.target_ref):
        raise AnnotationBatchError(
            f"annotation batch target_ref {batch.target_ref!r} is incompatible with "
            f"schema target_ref_kinds {list(schema.schema.target_ref_kinds)}"
        )
    try:
        normalized_source_ref = normalize_object_ref_text(batch.source_result_ref)
        source_ref = ObjectRef.parse(normalized_source_ref)
    except ValueError as exc:
        raise AnnotationBatchError("annotation batch source_result_ref must be a valid ObjectRef") from exc
    if normalized_source_ref != batch.source_result_ref or source_ref.kind != "result-set":
        raise AnnotationBatchError("annotation batch source_result_ref must use the 'result-set' ObjectRef kind")
    try:
        normalized_assertion_refs = tuple(normalize_object_ref_text(ref) for ref in batch.assertion_refs)
        assertion_ref_kinds = tuple(ObjectRef.parse(ref).kind for ref in normalized_assertion_refs)
    except ValueError as exc:
        raise AnnotationBatchError("annotation batch assertion_refs must be valid ObjectRefs") from exc
    if normalized_assertion_refs != batch.assertion_refs or any(kind != "assertion" for kind in assertion_ref_kinds):
        raise AnnotationBatchError("annotation batch assertion_refs must use normalized 'assertion' ObjectRefs")
    if len(set(normalized_assertion_refs)) != len(normalized_assertion_refs):
        raise AnnotationBatchError("annotation batch assertion_refs must be unique after normalization")
    existing = read_annotation_batch(conn, batch.batch_id)
    if existing is not None:
        if existing.canonical_provenance_bytes() != candidate_provenance:
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
            _json_array_text(provenance_failures),
            _json_document_text(provenance_metadata),
            batch.created_at_ms,
        ),
    )
    persisted = read_annotation_batch(conn, batch.batch_id)
    if persisted is None:  # pragma: no cover - INSERT followed by same-connection SELECT
        raise AnnotationBatchError(f"annotation batch {batch.batch_ref!r} was not persisted")
    if persisted.canonical_provenance_bytes() != candidate_provenance:
        raise AnnotationBatchError(f"annotation batch {batch.batch_ref!r} changed during durable persistence")
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
        params.append(normalize_annotation_target_ref(target_ref))
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
    "persist_builtin_annotation_schemas",
    "persist_annotation_schema",
    "read_annotation_batch",
    "read_annotation_batch_page",
    "AnnotationBatchReadPage",
    "read_durable_annotation_schema",
]
