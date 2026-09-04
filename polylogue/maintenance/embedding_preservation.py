"""Preserve and restore content-addressed embedding vectors across a rebuild."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class EmbeddingPreservationReceipt:
    source: str
    copy: str
    metadata_rows: int
    vector_rows: int
    table_set_digest: str
    restored_hashes: int = 0
    missing_hashes: tuple[str, ...] = ()


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


def _hash_column(conn: sqlite3.Connection, table: str) -> str:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    if _CURRENT_HASH_COLUMN in columns:
        return _CURRENT_HASH_COLUMN
    if _LEGACY_HASH_COLUMN in columns:
        return _LEGACY_HASH_COLUMN
    raise RuntimeError(f"{table} has no supported embedding hash column")


def preserve_embedding_vectors(source: str | Path, destination: str | Path) -> EmbeddingPreservationReceipt:
    """Checkpoint-copy an embeddings database and record its vector population."""
    source_path = Path(source).absolute()
    destination_path = Path(destination).absolute()
    if source_path == destination_path:
        raise ValueError("embedding preservation source and copy must differ")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(source_path, readonly=True) as source_conn:
        digest, counts = _table_digest(source_conn)
        if destination_path.exists():
            raise FileExistsError(destination_path)
        with sqlite3.connect(destination_path) as copy_conn:
            source_conn.backup(copy_conn)
    receipt = EmbeddingPreservationReceipt(
        source=str(source_path),
        copy=str(destination_path),
        metadata_rows=counts["message_embeddings_meta"],
        vector_rows=counts["message_embeddings"],
        table_set_digest=digest,
    )
    destination_path.with_suffix(destination_path.suffix + ".receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def restore_embedding_vectors(
    destination: str | Path,
    preserved_copy: str | Path,
    input_hashes: set[bytes],
) -> EmbeddingPreservationReceipt:
    """Import preserved vectors for ``input_hashes`` into a fresh embeddings DB.

    Metadata and vectors are write-once by input hash. Refs and lifecycle rows
    remain owned by the fresh database and are created by normal convergence.
    """
    destination_path = Path(destination).absolute()
    copy_path = Path(preserved_copy).absolute()
    wanted = sorted(input_hashes)
    with _connect(destination_path, readonly=False) as target, _connect(copy_path, readonly=True) as source:
        source_meta_hash = _hash_column(source, "message_embeddings_meta")
        source_vector_hash = _hash_column(source, "message_embeddings")
        if wanted:
            placeholders = ",".join("?" for _ in wanted)
            rows = source.execute(
                f"SELECT {source_meta_hash}, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash "
                f"FROM message_embeddings_meta WHERE {source_meta_hash} IN ({placeholders})",
                wanted,
            ).fetchall()
            found = {bytes(row[0]) for row in rows}
            for row in rows:
                target.execute(
                    "INSERT OR IGNORE INTO message_embeddings_meta "
                    "(vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )
                vector = source.execute(
                    f"SELECT embedding, model FROM message_embeddings WHERE {source_vector_hash} = ?",
                    (bytes(row[0]).hex(),),
                ).fetchone()
                if vector is None:
                    continue
                target.execute(
                    "INSERT OR IGNORE INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                    (bytes(row[0]).hex(), vector[0], vector[1]),
                )
            target.commit()
        else:
            found = set()
        digest, counts = _table_digest(source)
    missing = tuple(value.hex() for value in wanted if value not in found)
    return EmbeddingPreservationReceipt(
        source=str(copy_path),
        copy=str(destination_path),
        metadata_rows=counts["message_embeddings_meta"],
        vector_rows=counts["message_embeddings"],
        table_set_digest=digest,
        restored_hashes=len(found),
        missing_hashes=missing,
    )


def delete_preserved_copy(path: str | Path, *, receipt_path: str | Path | None = None) -> None:
    """Delete a preservation copy only when an AC2 receipt authorizes it."""
    copy_path = Path(path).absolute()
    if not copy_path.is_file() or copy_path.is_symlink():
        raise FileNotFoundError(copy_path)
    if receipt_path is None:
        receipt_path = copy_path.with_suffix(copy_path.suffix + ".receipt.json")
    receipt = Path(receipt_path).absolute()
    if not receipt.is_file() or receipt.is_symlink():
        raise FileNotFoundError(receipt)
    try:
        proof = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("preservation deletion receipt is not valid JSON") from exc
    if proof.get("ac2_passed") is not True:
        raise ValueError("preservation copy requires an AC2-passed receipt before deletion")
    copy_path.unlink()
    receipt.unlink()


__all__ = [
    "EmbeddingPreservationReceipt",
    "delete_preserved_copy",
    "preserve_embedding_vectors",
    "restore_embedding_vectors",
]
