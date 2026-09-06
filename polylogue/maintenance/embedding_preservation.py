"""Preserve and restore content-addressed embedding vectors across a rebuild."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.durable_fs import atomic_replace, sync_directory, write_once
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
#: Receipt schema for the AC2 reuse proof that authorizes deleting a copy.
AC2_RECEIPT_SCHEMA = "polylogue.embedding-reuse-verification.v1"
#: Share of recomputed hashes that must resolve against preserved vectors.
DEFAULT_MINIMUM_HIT_RATE = 0.95


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


def _connect(path: Path, *, readonly: bool, immutable: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro" + ("&immutable=1" if immutable else "")
    conn = sqlite3.connect(uri, uri=True) if readonly else sqlite3.connect(path)
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


def ac2_receipt_path(copy_path: str | Path) -> Path:
    """Where a copy's reuse proof lives.

    Distinct from the preservation receipt so proving reuse cannot overwrite
    the record of what was copied.
    """
    path = Path(copy_path).absolute()
    return path.with_suffix(path.suffix + ".ac2.json")


def _fsync_file(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _fsync_directory(path: Path) -> None:
    sync_directory(path)


def preserve_embedding_vectors(
    source: str | Path, destination: str | Path, *, immutable: bool = False
) -> EmbeddingPreservationReceipt:
    """Checkpoint-copy an embeddings database and record its vector population.

    The copy is built in a private temporary file and renamed into place only
    once the backup has finished and the receipt has been derived from the
    finished copy, so a file at the destination path is always a whole copy
    that its receipt describes.

    ``immutable`` opens the source without creating or touching its sidecar
    files, which a source about to be discarded requires and a live one
    forbids: an immutable read of a database another writer is changing
    returns torn pages.
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
            closing(_connect(source_path, readonly=True, immutable=immutable)) as source_conn,
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


def _tier_output_contract_hash(*, model: str, dimension: int) -> bytes:
    """The output contract every vector at this dimension shares.

    ``EmbeddingRecipe.output_contract`` is a function of dimensions,
    element type, and the output schema version alone -- never of the input
    text, the recipe labels, or the writer. A preserved row therefore carries
    its own output identity in its ``dimension`` column plus the float32
    element type its vector table declares, so an absent column is a value
    the older writer did not record, not an output nobody can vouch for.
    """
    from polylogue.storage.embeddings.identity import EmbeddingRecipe

    return EmbeddingRecipe.current(model=model, dimensions=dimension).output_contract_hash


def _validated_metadata(fields: dict[str, object]) -> _PreservedMetadata | str:
    """The row as the current tier requires it, or the field that disqualifies it.

    A preserved row whose model, dimension, or recipe identity is absent
    describes an embedding nobody can vouch for and cannot stand in for a
    fresh one. An absent output contract is reconstructed from the row's own
    dimension; a present one that is malformed still disqualifies the row.
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
        if value is None and name == "output_contract_hash":
            identities[name] = _tier_output_contract_hash(model=model, dimension=dimension)
            continue
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


class ReuseMissReason(StrEnum):
    """Why a recomputed hash did not resolve against a preserved vector."""

    #: The fresh corpus produced text this archive never embedded. Parser
    #: changes legitimately move content, so this is the expected miss class.
    NOT_PRESERVED = "not_preserved"
    #: Preserved metadata names an address whose vector row is absent.
    PRESERVED_VECTOR_ABSENT = "preserved_vector_absent"
    #: Restore did not carry the metadata row into the fresh database.
    RESTORE_METADATA_ABSENT = "restore_metadata_absent"
    #: Restore did not carry the vector row into the fresh database.
    RESTORE_VECTOR_ABSENT = "restore_vector_absent"


@dataclass(frozen=True, slots=True)
class ReuseMiss:
    input_hash: str
    reason: ReuseMissReason
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingReuseVerification:
    """AC2: how much of the rebuilt corpus resolved against preserved vectors."""

    copy: str
    destination: str
    model: str
    recomputed_hashes: int
    hit_hashes: int
    minimum_hit_rate: float
    table_set_digest: str
    misses: tuple[ReuseMiss, ...]

    @property
    def hit_rate(self) -> float:
        return self.hit_hashes / self.recomputed_hashes if self.recomputed_hashes else 0.0

    @property
    def ac2_passed(self) -> bool:
        """A corpus with nothing to embed proves no reuse and cannot pass."""
        return self.recomputed_hashes > 0 and self.hit_rate >= self.minimum_hit_rate

    def as_receipt(self) -> dict[str, object]:
        """The proof shape :func:`delete_preserved_copy` requires.

        ``copy`` and ``table_set_digest`` name the exact preserved file this
        measurement read, so the proof cannot authorize deleting another copy
        or one that has changed since.
        """
        counts: dict[str, int] = {}
        for miss in self.misses:
            counts[miss.reason.value] = counts.get(miss.reason.value, 0) + 1
        return {
            "schema": AC2_RECEIPT_SCHEMA,
            "copy": self.copy,
            "destination": self.destination,
            "model": self.model,
            "recomputed_hashes": self.recomputed_hashes,
            "hit_hashes": self.hit_hashes,
            "hit_rate": self.hit_rate,
            "minimum_hit_rate": self.minimum_hit_rate,
            "table_set_digest": self.table_set_digest,
            "misses_by_reason": counts,
            "misses": [asdict(miss) for miss in self.misses],
            "ac2_passed": self.ac2_passed,
        }


def archive_tier_paths(root: str | Path) -> tuple[Path, Path]:
    """The archive's embeddings tier and its active index generation, in that order."""
    from polylogue.storage.archive_identity import ArchiveLocation

    location = ArchiveLocation.resolve(Path(root).absolute())
    return location.configured_root / "embeddings.db", location.active_index_path


def recomputed_vector_hashes(index_db: str | Path, *, model: str) -> dict[str, bytes]:
    """Vector addresses the rebuilt archive will ask the embedder for.

    The relation is the production embedder's own message selection and hash
    expression, so a measurement taken here answers for the route that will
    actually spend money, not a reimplementation of it.
    """
    from polylogue.storage.embeddings.materialization import archive_embeddable_messages_relation

    # A plain read-only open, never immutable: this index is the rebuilt
    # archive's, which convergence may still be writing.
    with closing(_connect(Path(index_db).absolute(), readonly=True)) as conn:
        relation = archive_embeddable_messages_relation(conn, alias="embeddable", model=model)
        rows = conn.execute(
            f"SELECT embeddable.message_id, embeddable.vector_derivation_hash FROM {relation} "
            "WHERE embeddable.vector_derivation_hash IS NOT NULL"
        ).fetchall()
    return {str(row[0]): bytes(row[1]) for row in rows}


def _present_hashes(conn: sqlite3.Connection, wanted: Sequence[bytes]) -> tuple[set[bytes], set[bytes]]:
    """Hashes with a metadata row and hashes with a vector row, in that order."""
    metadata: set[bytes] = set()
    vectors: set[bytes] = set()
    meta_column = _hash_column(conn, "message_embeddings_meta")
    vector_column = _hash_column(conn, "message_embeddings")
    for batch in _hash_batches(conn, wanted):
        placeholders = ",".join("?" for _ in batch)
        metadata.update(
            bytes(row[0])
            for row in conn.execute(
                f"SELECT {meta_column} FROM message_embeddings_meta WHERE {meta_column} IN ({placeholders})",
                tuple(batch),
            )
        )
        vectors.update(
            bytes.fromhex(str(row[0]))
            for row in conn.execute(
                f"SELECT {vector_column} FROM message_embeddings WHERE {vector_column} IN ({placeholders})",
                tuple(value.hex() for value in batch),
            )
        )
    return metadata, vectors


def verify_embedding_reuse(
    destination: str | Path,
    preserved_copy: str | Path,
    recomputed: Mapping[str, bytes],
    *,
    model: str,
    minimum_hit_rate: float = DEFAULT_MINIMUM_HIT_RATE,
    receipt_path: str | Path | None = None,
) -> EmbeddingReuseVerification:
    """Measure AC2: reuse of preserved vectors by the rebuilt archive.

    A hash counts as a hit only when the fresh database holds both its
    metadata and its vector, so a restore that wrote half a pair is a miss
    rather than a reuse the embedder cannot actually take. Every other
    outcome is enumerated with the cause that distinguishes a legitimate
    content change from a preservation or restore defect.
    """
    if not 0.0 <= minimum_hit_rate <= 1.0:
        raise ValueError("minimum_hit_rate must be between zero and one")
    destination_path = Path(destination).absolute()
    copy_path = Path(preserved_copy).absolute()
    representative: dict[bytes, str] = {}
    for message_id, value in recomputed.items():
        representative.setdefault(bytes(value), str(message_id))
    wanted = sorted(representative)
    with (
        closing(_connect(destination_path, readonly=True)) as target,
        closing(_connect(copy_path, readonly=True)) as source,
    ):
        preserved_meta, preserved_vectors = _present_hashes(source, wanted)
        target_meta, target_vectors = _present_hashes(target, wanted)
        digest, _counts = _table_digest(source)
    misses: list[ReuseMiss] = []
    hits = 0
    for value in wanted:
        message_id = representative[value]
        if value not in preserved_meta:
            misses.append(ReuseMiss(value.hex(), ReuseMissReason.NOT_PRESERVED, message_id))
        elif value not in preserved_vectors:
            misses.append(ReuseMiss(value.hex(), ReuseMissReason.PRESERVED_VECTOR_ABSENT, message_id))
        elif value not in target_meta:
            misses.append(ReuseMiss(value.hex(), ReuseMissReason.RESTORE_METADATA_ABSENT, message_id))
        elif value not in target_vectors:
            misses.append(ReuseMiss(value.hex(), ReuseMissReason.RESTORE_VECTOR_ABSENT, message_id))
        else:
            hits += 1
    verification = EmbeddingReuseVerification(
        copy=str(copy_path),
        destination=str(destination_path),
        model=model,
        recomputed_hashes=len(wanted),
        hit_hashes=hits,
        minimum_hit_rate=minimum_hit_rate,
        table_set_digest=digest,
        misses=tuple(misses),
    )
    if receipt_path is not None:
        payload = (json.dumps(verification.as_receipt(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_replace(Path(receipt_path).absolute(), payload)
    return verification


def delete_preserved_copy(path: str | Path, *, receipt_path: str | Path | None = None) -> None:
    """Delete a preservation copy only when an AC2 receipt proves it is this copy.

    The receipt must name this file and carry the table-set digest the copy
    still has, so a receipt filed for one copy can never authorize deleting
    another, nor a copy that has changed since it was proven.
    """
    copy_path = Path(path).absolute()
    if not copy_path.is_file() or copy_path.is_symlink():
        raise FileNotFoundError(copy_path)
    receipt = Path(receipt_path).absolute() if receipt_path is not None else ac2_receipt_path(copy_path)
    if not receipt.is_file() or receipt.is_symlink():
        raise FileNotFoundError(receipt)
    try:
        proof = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("preservation deletion receipt is not valid JSON") from exc
    if proof.get("schema") != AC2_RECEIPT_SCHEMA or proof.get("ac2_passed") is not True:
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
    "AC2_RECEIPT_SCHEMA",
    "DEFAULT_MINIMUM_HIT_RATE",
    "EmbeddingPreservationReceipt",
    "EmbeddingReuseVerification",
    "RestoreMiss",
    "RestoreMissReason",
    "ReuseMiss",
    "ReuseMissReason",
    "ac2_receipt_path",
    "archive_tier_paths",
    "delete_preserved_copy",
    "preserve_embedding_vectors",
    "recomputed_vector_hashes",
    "restore_embedding_vectors",
    "verify_embedding_reuse",
]
