"""Preserve and restore content-addressed embedding vectors across a rebuild."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.durable_fs import sync_directory, write_once
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_VECTOR_TABLES = (
    "message_embeddings",
    "message_embeddings_auxiliary",
    "message_embeddings_chunks",
    "message_embeddings_info",
    "message_embeddings_rowids",
    "message_embeddings_vector_chunks00",
)
_CURRENT_HASH_COLUMN = "vector_derivation_hash"
_LEGACY_HASH_COLUMN = "embedding_input_hash"
_META_FIELDS = ("model", "dimension", "embedded_at_ms", "recipe_hash", "output_contract_hash")
# Hashes per IN list. Bounded by the connection's own variable limit, which is
# 999 on a default SQLite build and must never be assumed larger.
_MAX_HASH_BATCH = 500


class RestoreMissReason(StrEnum):
    """Why a wanted hash did not restore."""

    METADATA_ABSENT = "metadata_absent"
    METADATA_INCOMPLETE = "metadata_incomplete"
    VECTOR_ABSENT = "vector_absent"


@dataclass(frozen=True, slots=True)
class RestoreMiss:
    input_hash: str
    reason: RestoreMissReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class EmbeddingPreservationReceipt:
    source: str
    copy: str
    metadata_rows: int
    vector_rows: int
    table_set_digest: str
    restored_hashes: int = 0
    misses: tuple[RestoreMiss, ...] = ()


def _connect(path: Path, *, readonly: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True) if readonly else sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        raise RuntimeError("sqlite-vec is required for embedding preservation") from error
    return conn


def _table_digest(conn: sqlite3.Connection) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for table in ("message_embeddings_meta", *_VECTOR_TABLES):
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'shadow') AND name = ? LIMIT 1", (table,)
        ).fetchone()
        count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) if present else 0
        counts[table] = count
        digest.update(table.encode())
        digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest(), counts


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _hash_column(conn: sqlite3.Connection, table: str) -> str:
    columns = _columns(conn, table)
    if _CURRENT_HASH_COLUMN in columns:
        return _CURRENT_HASH_COLUMN
    if _LEGACY_HASH_COLUMN in columns:
        return _LEGACY_HASH_COLUMN
    raise RuntimeError(f"{table} has no supported embedding hash column")


def _hash_batches(conn: sqlite3.Connection, values: Sequence[bytes]) -> Iterator[Sequence[bytes]]:
    """Chunk host parameters below this connection's own variable limit."""
    size = max(1, min(_MAX_HASH_BATCH, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _receipt_path(copy_path: Path) -> Path:
    return copy_path.with_suffix(copy_path.suffix + ".receipt.json")


def _fsync_file(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _fsync_directory(path: Path) -> None:
    sync_directory(path)


def preserve_embedding_vectors(source: str | Path, destination: str | Path) -> EmbeddingPreservationReceipt:
    """Checkpoint-copy an embeddings database and record its vector population.

    The copy is built in a private temporary file and renamed into place only
    once the backup has finished and the receipt has been derived from the
    finished copy, so a file at the destination path is always a whole copy
    that its receipt describes.
    """
    source_path = Path(source).absolute()
    destination_path = Path(destination).absolute()
    if source_path == destination_path:
        raise ValueError("embedding preservation source and copy must differ")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    handle, partial_name = tempfile.mkstemp(
        dir=destination_path.parent, prefix=f".{destination_path.name}.", suffix=".partial"
    )
    os.close(handle)
    partial = Path(partial_name)
    try:
        with (
            closing(_connect(source_path, readonly=True)) as source_conn,
            closing(sqlite3.connect(partial)) as copy_conn,
        ):
            source_conn.backup(copy_conn)
            # The copy inherits the source's journal mode, and a rename moves
            # only the main file: the archived copy is made self-contained so
            # it can never be separated from a WAL holding its content.
            copy_conn.execute("PRAGMA journal_mode=DELETE").fetchall()
        with closing(_connect(partial, readonly=True)) as copy_reader:
            digest, counts = _table_digest(copy_reader)
        _fsync_file(partial)
        os.replace(partial, destination_path)
        _fsync_directory(destination_path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    receipt = EmbeddingPreservationReceipt(
        source=str(source_path),
        copy=str(destination_path),
        metadata_rows=counts["message_embeddings_meta"],
        vector_rows=counts["message_embeddings"],
        table_set_digest=digest,
    )
    write_once(
        _receipt_path(destination_path),
        (json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return receipt


@dataclass(frozen=True, slots=True)
class _PreservedMetadata:
    """A preserved row in the shape the current tier requires."""

    model: str
    dimension: int
    embedded_at_ms: int | None
    recipe_hash: bytes
    output_contract_hash: bytes


def _validated_metadata(fields: dict[str, object]) -> _PreservedMetadata | str:
    """The row as the current tier requires it, or the field that disqualifies it.

    ``message_embeddings_meta`` is complete by schema: a preserved row whose
    model, dimension, or derivation identity is absent describes an output
    nobody can vouch for, so it cannot stand in for a fresh embedding.
    """
    model = fields.get("model")
    if not isinstance(model, str) or not model:
        return "model"
    dimension = fields.get("dimension")
    if not isinstance(dimension, int) or dimension != EMBEDDING_DIMENSION:
        return "dimension"
    identities: dict[str, bytes] = {}
    for name in ("recipe_hash", "output_contract_hash"):
        value = fields.get(name)
        if not isinstance(value, (bytes, bytearray, memoryview)) or len(value) != 32:
            return name
        identities[name] = bytes(value)
    embedded_at_ms = fields.get("embedded_at_ms")
    return _PreservedMetadata(
        model=model,
        dimension=dimension,
        embedded_at_ms=embedded_at_ms if isinstance(embedded_at_ms, int) else None,
        recipe_hash=identities["recipe_hash"],
        output_contract_hash=identities["output_contract_hash"],
    )


def _preserved_metadata(
    conn: sqlite3.Connection, hash_column: str, projection: Sequence[str], batch: Sequence[bytes]
) -> dict[bytes, dict[str, object]]:
    columns = ", ".join((hash_column, *projection))
    placeholders = ",".join("?" for _ in batch)
    rows = conn.execute(
        f"SELECT {columns} FROM message_embeddings_meta WHERE {hash_column} IN ({placeholders})",
        tuple(batch),
    ).fetchall()
    return {bytes(row[0]): dict(zip(projection, row[1:], strict=True)) for row in rows}


def _preserved_vectors(
    conn: sqlite3.Connection, hash_column: str, batch: Sequence[bytes]
) -> dict[bytes, tuple[object, object]]:
    addresses = [value.hex() for value in batch]
    placeholders = ",".join("?" for _ in addresses)
    rows = conn.execute(
        f"SELECT {hash_column}, embedding, model FROM message_embeddings WHERE {hash_column} IN ({placeholders})",
        addresses,
    ).fetchall()
    return {bytes.fromhex(str(row[0])): (row[1], row[2]) for row in rows}


def restore_embedding_vectors(
    destination: str | Path,
    preserved_copy: str | Path,
    input_hashes: set[bytes],
) -> EmbeddingPreservationReceipt:
    """Import preserved vectors for ``input_hashes`` into a fresh embeddings DB.

    Metadata and vectors are write-once by input hash and are written together
    in one transaction: a metadata row is the tier's reuse signal, so it may
    never exist without the vector at its address. A hash counts as restored
    only once both rows are present; every other outcome is an enumerated miss
    carrying its cause. Refs and lifecycle rows remain owned by the fresh
    database and are created by normal convergence.
    """
    destination_path = Path(destination).absolute()
    copy_path = Path(preserved_copy).absolute()
    wanted = sorted(input_hashes)
    restored = 0
    misses: list[RestoreMiss] = []
    with (
        closing(_connect(destination_path, readonly=False)) as target,
        closing(_connect(copy_path, readonly=True)) as source,
    ):
        meta_hash_column = _hash_column(source, "message_embeddings_meta")
        vector_hash_column = _hash_column(source, "message_embeddings")
        projection = [name for name in _META_FIELDS if name in _columns(source, "message_embeddings_meta")]
        for batch in _hash_batches(source, wanted):
            preserved = _preserved_metadata(source, meta_hash_column, projection, batch)
            vectors = _preserved_vectors(source, vector_hash_column, batch)
            for value in batch:
                fields = preserved.get(value)
                if fields is None:
                    misses.append(RestoreMiss(value.hex(), RestoreMissReason.METADATA_ABSENT))
                    continue
                record = _validated_metadata(fields)
                if isinstance(record, str):
                    misses.append(RestoreMiss(value.hex(), RestoreMissReason.METADATA_INCOMPLETE, record))
                    continue
                vector = vectors.get(value)
                if vector is None:
                    misses.append(RestoreMiss(value.hex(), RestoreMissReason.VECTOR_ABSENT))
                    continue
                target.execute(
                    "INSERT OR IGNORE INTO message_embeddings (vector_derivation_hash, embedding, model) "
                    "VALUES (?, ?, ?)",
                    (value.hex(), vector[0], vector[1]),
                )
                target.execute(
                    "INSERT OR IGNORE INTO message_embeddings_meta "
                    "(vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        value,
                        record.model,
                        record.dimension,
                        record.embedded_at_ms,
                        record.recipe_hash,
                        record.output_contract_hash,
                    ),
                )
                restored += 1
            target.commit()
        digest, counts = _table_digest(source)
    return EmbeddingPreservationReceipt(
        source=str(copy_path),
        copy=str(destination_path),
        metadata_rows=counts["message_embeddings_meta"],
        vector_rows=counts["message_embeddings"],
        table_set_digest=digest,
        restored_hashes=restored,
        misses=tuple(misses),
    )


def delete_preserved_copy(path: str | Path, *, receipt_path: str | Path | None = None) -> None:
    """Delete a preservation copy only when an AC2 receipt proves it is this copy.

    The receipt must name this file and carry the table-set digest the copy
    still has, so a receipt filed for one copy can never authorize deleting
    another, nor a copy that has changed since it was proven.
    """
    copy_path = Path(path).absolute()
    if not copy_path.is_file() or copy_path.is_symlink():
        raise FileNotFoundError(copy_path)
    receipt = Path(receipt_path).absolute() if receipt_path is not None else _receipt_path(copy_path)
    if not receipt.is_file() or receipt.is_symlink():
        raise FileNotFoundError(receipt)
    try:
        proof = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("preservation deletion receipt is not valid JSON") from exc
    if proof.get("ac2_passed") is not True:
        raise ValueError("preservation copy requires an AC2-passed receipt before deletion")
    named = proof.get("copy")
    if not isinstance(named, str) or Path(named).absolute() != copy_path:
        raise ValueError("preservation receipt names a different copy")
    with closing(_connect(copy_path, readonly=True)) as conn:
        digest, _counts = _table_digest(conn)
    if digest != proof.get("table_set_digest"):
        raise ValueError("preservation copy no longer matches its receipt digest")
    copy_path.unlink()
    receipt.unlink()


__all__ = [
    "EmbeddingPreservationReceipt",
    "RestoreMiss",
    "RestoreMissReason",
    "delete_preserved_copy",
    "preserve_embedding_vectors",
    "restore_embedding_vectors",
]
