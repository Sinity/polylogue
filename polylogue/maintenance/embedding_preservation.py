"""Preserve and restore content-addressed embedding vectors across a rebuild."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_CURRENT_HASH_COLUMN = "vector_derivation_hash"
_LEGACY_HASH_COLUMN = "embedding_input_hash"


@dataclass(frozen=True, slots=True)
class EmbeddingPreservationReceipt:
    source: str
    copy: str
    metadata_rows: int
    vector_rows: int
    table_set_digest: str
    table_counts: dict[str, int]
    restored_hashes: int = 0
    missing_hashes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmbeddingReuseMiss:
    input_hash: str
    cause: str
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingReuseVerification:
    preserved_copy: str
    destination: str
    recomputed_hashes: int
    restored_hashes: int
    hit_rate: float
    minimum_hit_rate: float
    table_set_digest: str
    misses: tuple[EmbeddingReuseMiss, ...]

    @property
    def ac2_passed(self) -> bool:
        return self.recomputed_hashes > 0 and self.hit_rate >= self.minimum_hit_rate

    def as_receipt(self) -> dict[str, object]:
        return {
            "schema": "polylogue.embedding-reuse-verification.v1",
            "preserved_copy": self.preserved_copy,
            "destination": self.destination,
            "recomputed_hashes": self.recomputed_hashes,
            "restored_hashes": self.restored_hashes,
            "hit_rate": self.hit_rate,
            "minimum_hit_rate": self.minimum_hit_rate,
            "table_set_digest": self.table_set_digest,
            "misses": [asdict(miss) for miss in self.misses],
            "ac2_passed": self.ac2_passed,
        }


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
    tables = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'shadow') AND name LIKE 'message_embeddings%' "
            "ORDER BY name"
        ).fetchall()
    )
    for table in tables:
        count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        counts[table] = count
        digest.update(table.encode())
        digest.update(b"\0")
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
    with _connect(destination_path, readonly=True) as copy_conn:
        copied_digest, copied_counts = _table_digest(copy_conn)
    if copied_digest != digest or copied_counts != counts:
        raise RuntimeError("embedding preservation copy does not match its source receipt")
    receipt = EmbeddingPreservationReceipt(
        source=str(source_path),
        copy=str(destination_path),
        metadata_rows=counts["message_embeddings_meta"],
        vector_rows=counts["message_embeddings"],
        table_set_digest=digest,
        table_counts=counts,
    )
    destination_path.with_suffix(destination_path.suffix + ".receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def restore_embedding_vectors(
    destination: str | Path,
    preserved_copy: str | Path,
    input_hashes: set[bytes],
    *,
    recipe: EmbeddingRecipe | None = None,
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
            found: set[bytes] = set()
            for row in rows:
                vector = source.execute(
                    f"SELECT embedding, model FROM message_embeddings WHERE {source_vector_hash} = ?",
                    (bytes(row[0]).hex(),),
                ).fetchone()
                if vector is None:
                    continue
                found.add(bytes(row[0]))
                if source_meta_hash == _LEGACY_HASH_COLUMN:
                    fallback_recipe = recipe or EmbeddingRecipe.current(model=str(row[1]), dimensions=int(row[2]))
                    metadata_row = (*row[:4], fallback_recipe.recipe_hash, fallback_recipe.output_contract_hash)
                else:
                    metadata_row = row
                target.execute(
                    "INSERT OR IGNORE INTO message_embeddings_meta "
                    "(vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    metadata_row,
                )
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
        table_counts=counts,
        restored_hashes=len(found),
        missing_hashes=missing,
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'shadow') AND name = ? LIMIT 1", (table,)
        ).fetchone()
        is not None
    )


def _complete_vector_hashes(
    conn: sqlite3.Connection,
    hashes: Iterable[bytes],
    *,
    metadata_column: str,
    vector_column: str,
) -> tuple[set[bytes], set[bytes]]:
    wanted = sorted(set(hashes))
    if not wanted:
        return set(), set()
    placeholders = ",".join("?" for _ in wanted)
    metadata = {
        bytes(row[0])
        for row in conn.execute(
            f"SELECT {metadata_column} FROM message_embeddings_meta WHERE {metadata_column} IN ({placeholders})",
            wanted,
        ).fetchall()
    }
    vectors = {
        bytes.fromhex(str(row[0]))
        for row in conn.execute(
            f"SELECT {vector_column} FROM message_embeddings "
            f"WHERE {vector_column} IN ({','.join('?' for _ in wanted)})",
            [value.hex() for value in wanted],
        ).fetchall()
    }
    return metadata, vectors


def verify_embedding_reuse(
    destination: str | Path,
    preserved_copy: str | Path,
    recomputed_hashes: Mapping[str, bytes] | Iterable[bytes],
    *,
    minimum_hit_rate: float = 0.95,
    receipt_path: str | Path | None = None,
) -> EmbeddingReuseVerification:
    """Verify post-build vector reuse and enumerate every non-hit.

    A mapping supplies message IDs and proves that convergence re-minted the
    corresponding refs. An iterable verifies vector reuse without requiring a
    message-to-hash relation. A hit requires complete metadata and vector rows
    in both the preserved copy and the fresh destination.
    """
    if not 0.0 <= minimum_hit_rate <= 1.0:
        raise ValueError("minimum_hit_rate must be between zero and one")
    has_message_ids = isinstance(recomputed_hashes, Mapping)
    candidates: tuple[tuple[str | None, bytes], ...]
    if has_message_ids:
        mapping = cast(Mapping[str, bytes], recomputed_hashes)
        candidates = tuple((str(message_id), bytes(value)) for message_id, value in mapping.items())
    else:
        values = cast(Iterable[bytes], recomputed_hashes)
        candidates = tuple((None, bytes(value)) for value in values)
    destination_path = Path(destination).absolute()
    copy_path = Path(preserved_copy).absolute()
    with _connect(destination_path, readonly=True) as target, _connect(copy_path, readonly=True) as source:
        source_meta_column = _hash_column(source, "message_embeddings_meta")
        source_vector_column = _hash_column(source, "message_embeddings")
        target_meta_column = _hash_column(target, "message_embeddings_meta")
        target_vector_column = _hash_column(target, "message_embeddings")
        hashes = [value for _message_id, value in candidates]
        source_meta, source_vectors = _complete_vector_hashes(
            source,
            hashes,
            metadata_column=source_meta_column,
            vector_column=source_vector_column,
        )
        target_meta, target_vectors = _complete_vector_hashes(
            target,
            hashes,
            metadata_column=target_meta_column,
            vector_column=target_vector_column,
        )
        target_refs: dict[str, bytes] = {}
        has_refs = _table_exists(target, "message_embedding_refs")
        if has_refs and has_message_ids:
            target_ref_column = _hash_column(target, "message_embedding_refs")
            target_refs = {
                str(row[0]): bytes(row[1])
                for row in target.execute(
                    f"SELECT message_id, {target_ref_column} FROM message_embedding_refs"
                ).fetchall()
            }
        table_set_digest, _counts = _table_digest(source)

    misses: list[EmbeddingReuseMiss] = []
    hits = 0
    for message_id, input_hash in candidates:
        hash_text = input_hash.hex()
        if len(input_hash) != 32:
            misses.append(EmbeddingReuseMiss(hash_text, "invalid_input_hash", message_id))
        elif input_hash not in source_meta:
            misses.append(EmbeddingReuseMiss(hash_text, "content_changed_or_new", message_id))
        elif input_hash not in source_vectors:
            misses.append(EmbeddingReuseMiss(hash_text, "preserved_vector_missing", message_id))
        elif input_hash not in target_meta:
            misses.append(EmbeddingReuseMiss(hash_text, "restore_metadata_missing", message_id))
        elif input_hash not in target_vectors:
            misses.append(EmbeddingReuseMiss(hash_text, "restore_vector_missing", message_id))
        elif has_message_ids and not has_refs:
            misses.append(EmbeddingReuseMiss(hash_text, "reference_table_missing", message_id))
        elif has_message_ids and target_refs.get(str(message_id)) != input_hash:
            misses.append(EmbeddingReuseMiss(hash_text, "reference_not_reminted", message_id))
        else:
            hits += 1
    total = len(candidates)
    verification = EmbeddingReuseVerification(
        preserved_copy=str(copy_path),
        destination=str(destination_path),
        recomputed_hashes=total,
        restored_hashes=hits,
        hit_rate=hits / total if total else 1.0,
        minimum_hit_rate=minimum_hit_rate,
        table_set_digest=table_set_digest,
        misses=tuple(misses),
    )
    if receipt_path is not None:
        receipt = Path(receipt_path).absolute()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(verification.as_receipt(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


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
    if not isinstance(proof, dict) or (
        proof.get("schema") != "polylogue.embedding-reuse-verification.v1"
        or proof.get("preserved_copy") != str(copy_path)
        or proof.get("ac2_passed") is not True
    ):
        raise ValueError("preservation copy requires an AC2-passed receipt before deletion")
    copy_path.unlink()
    receipt.unlink()


__all__ = [
    "EmbeddingPreservationReceipt",
    "EmbeddingReuseMiss",
    "EmbeddingReuseVerification",
    "delete_preserved_copy",
    "preserve_embedding_vectors",
    "restore_embedding_vectors",
    "verify_embedding_reuse",
]
