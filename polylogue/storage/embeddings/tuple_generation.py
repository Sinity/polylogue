"""Tuple-bound construction for inactive embedding generations.

An embedding candidate is a member of an :class:`ArchiveTupleManifest`.  This
module deliberately has no active pointer, retention receipt, or generation
ordering logic.  The tuple manifest supplies the source and index bindings;
the candidate-local record supplies the exact recipe and construction seal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.archive_tuple_location import (
    ArchiveTupleAllocator,
    InactiveTierDestination,
    validate_inactive_destination,
)
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.search_providers.sqlite_vec_support import _serialize_f32
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDINGS_SCHEMA_VERSION,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_METADATA_FILENAME = "embeddings-generation.json"
_HEX_DIGEST_LENGTH = 64


class EmbeddingTupleGenerationError(RuntimeError):
    """Raised when tuple-bound embedding construction cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class EmbeddingTupleGeneration:
    """Authenticated construction metadata for one tuple's embeddings member."""

    tuple_id: str
    owner_id: str
    generation_id: str
    recipe_identity: dict[str, object]
    recipe_hash: str
    output_contract_hash: str
    source_generation: str
    index_generation: str
    schema_version: int
    physical_root: str
    sealed: bool
    membership_digest: str

    @property
    def metadata_path(self) -> Path:
        return Path(self.physical_root) / _METADATA_FILENAME

    def _payload(self) -> dict[str, object]:
        return {
            "tuple_id": self.tuple_id,
            "owner_id": self.owner_id,
            "generation_id": self.generation_id,
            "recipe_identity": self.recipe_identity,
            "recipe_hash": self.recipe_hash,
            "output_contract_hash": self.output_contract_hash,
            "source_generation": self.source_generation,
            "index_generation": self.index_generation,
            "schema_version": self.schema_version,
            "physical_root": self.physical_root,
            "sealed": self.sealed,
            "membership_digest": self.membership_digest,
        }

    @property
    def metadata_digest(self) -> str:
        encoded = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingPartitionRow:
    """One vector and its content address for a bounded candidate partition."""

    vector_derivation_hash: bytes
    embedding: Sequence[float]
    embedded_at_ms: int


@dataclass(frozen=True, slots=True)
class EmbeddingPartitionReceipt:
    """Result of an idempotent partition publication."""

    inserted_count: int
    vector_derivation_hashes: tuple[str, ...]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_metadata(metadata: EmbeddingTupleGeneration) -> None:
    path = metadata.metadata_path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = metadata._payload()
    payload["metadata_digest"] = metadata.metadata_digest
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _destination_context(destination: InactiveTierDestination) -> tuple[InactiveTierDestination, Any]:
    archive_root = destination.archive_root
    location = ArchiveLocation.resolve(archive_root)
    validate_inactive_destination(destination, location, expected_tier=ArchiveTier.EMBEDDINGS)
    tuple_location = ArchiveTupleAllocator(location).load(destination.tuple_id)
    return tuple_location.embeddings, tuple_location.manifest


def _recipe_identity(recipe: EmbeddingRecipe) -> dict[str, object]:
    return cast(dict[str, object], json.loads(recipe.identity().canonical_bytes()))


def _new_metadata(
    destination: InactiveTierDestination,
    manifest: Any,
    recipe: EmbeddingRecipe,
    *,
    sealed: bool = False,
    membership_digest: str = "",
) -> EmbeddingTupleGeneration:
    if recipe.dimensions != EMBEDDING_DIMENSION:
        raise EmbeddingTupleGenerationError(
            f"embedding recipe dimensions {recipe.dimensions} do not match schema {EMBEDDING_DIMENSION}"
        )
    return EmbeddingTupleGeneration(
        tuple_id=manifest.tuple_id,
        owner_id=manifest.owner_id,
        generation_id=manifest.embeddings_generation,
        recipe_identity=_recipe_identity(recipe),
        recipe_hash=recipe.recipe_hash.hex(),
        output_contract_hash=recipe.output_contract_hash.hex(),
        source_generation=manifest.source_generation,
        index_generation=manifest.index_generation,
        schema_version=EMBEDDINGS_SCHEMA_VERSION,
        physical_root=str(destination.candidate_root),
        sealed=sealed,
        membership_digest=membership_digest,
    )


def _read_metadata(destination: InactiveTierDestination) -> EmbeddingTupleGeneration:
    path = destination.candidate_root / _METADATA_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("metadata is not an object")
        digest = payload.pop("metadata_digest")
        metadata = EmbeddingTupleGeneration(**cast(dict[str, Any], payload))
        if str(digest) != metadata.metadata_digest:
            raise ValueError("metadata digest mismatch")
        if metadata.metadata_path != path:
            raise ValueError("metadata physical root mismatch")
        if metadata.tuple_id != destination.tuple_id or metadata.owner_id != destination.owner_id:
            raise ValueError("metadata tuple identity mismatch")
        if metadata.generation_id != destination.generation_id:
            raise ValueError("metadata generation identity mismatch")
        if metadata.schema_version != EMBEDDINGS_SCHEMA_VERSION:
            raise ValueError("metadata schema version mismatch")
        if metadata.sealed and len(metadata.membership_digest) != _HEX_DIGEST_LENGTH:
            raise ValueError("sealed metadata has no membership digest")
        return metadata
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EmbeddingTupleGenerationError(f"malformed embedding tuple metadata: {path}") from exc


def _assert_recipe(metadata: EmbeddingTupleGeneration, recipe: EmbeddingRecipe) -> None:
    if (
        metadata.recipe_identity != _recipe_identity(recipe)
        or metadata.recipe_hash != recipe.recipe_hash.hex()
        or metadata.output_contract_hash != recipe.output_contract_hash.hex()
    ):
        raise EmbeddingTupleGenerationError("embedding recipe does not match tuple metadata")


def _assert_bindings(
    metadata: EmbeddingTupleGeneration,
    *,
    source_generation: str | None = None,
    index_generation: str | None = None,
) -> None:
    if source_generation is not None and source_generation != metadata.source_generation:
        raise EmbeddingTupleGenerationError("source generation does not match tuple metadata")
    if index_generation is not None and index_generation != metadata.index_generation:
        raise EmbeddingTupleGenerationError("index generation does not match tuple metadata")


def _assert_manifest_bindings(metadata: EmbeddingTupleGeneration, manifest: Any) -> None:
    if (
        metadata.tuple_id != manifest.tuple_id
        or metadata.owner_id != manifest.owner_id
        or metadata.generation_id != manifest.embeddings_generation
        or metadata.source_generation != manifest.source_generation
        or metadata.index_generation != manifest.index_generation
    ):
        raise EmbeddingTupleGenerationError("embedding metadata does not match tuple bindings")


def _membership_digest(conn: sqlite3.Connection, recipe: EmbeddingRecipe) -> str:
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        raise EmbeddingTupleGenerationError("embedding candidate requires sqlite-vec") from error
    rows = conn.execute(
        "SELECT vector_derivation_hash, model, dimension FROM message_embeddings_meta ORDER BY vector_derivation_hash"
    ).fetchall()
    digest = hashlib.sha256()
    # The vector address is a function of the provider request; a meta row
    # whose recipe/output labels differ from the current recipe is still the
    # same request's vector. Only model/dimension drift or missing bytes are
    # conflicts.
    for vector_hash, model, dimension in rows:
        value = bytes(vector_hash)
        if len(value) != 32 or str(model) != recipe.model or int(dimension) != recipe.dimensions:
            raise EmbeddingTupleGenerationError("embedding membership has an incompatible vector contract")
        if (
            conn.execute(
                "SELECT 1 FROM message_embeddings WHERE vector_derivation_hash = ? LIMIT 1", (value.hex(),)
            ).fetchone()
            is None
        ):
            raise EmbeddingTupleGenerationError("embedding membership is missing vector bytes")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    digest.update(len(rows).to_bytes(8, "big"))
    return digest.hexdigest()


def prepare_inactive_embedding_generation(
    destination: InactiveTierDestination,
    *,
    recipe: EmbeddingRecipe,
    source_generation: str,
    index_generation: str,
) -> EmbeddingTupleGeneration:
    """Create or resume one tuple-bound, inactive embeddings database."""
    destination, manifest = _destination_context(destination)
    if source_generation != manifest.source_generation:
        raise EmbeddingTupleGenerationError("source generation does not match tuple manifest")
    if index_generation != manifest.index_generation:
        raise EmbeddingTupleGenerationError("index generation does not match tuple manifest")
    metadata_path = destination.candidate_root / _METADATA_FILENAME
    if metadata_path.exists():
        metadata = _read_metadata(destination)
        _assert_manifest_bindings(metadata, manifest)
        _assert_recipe(metadata, recipe)
        _assert_bindings(metadata, source_generation=source_generation, index_generation=index_generation)
        return metadata

    initialize_archive_database(destination.path, ArchiveTier.EMBEDDINGS, inactive_destination=destination)
    with sqlite3.connect(destination.path) as conn:
        if conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] != 0:
            raise EmbeddingTupleGenerationError("candidate has vectors but no authenticated metadata")
    metadata = _new_metadata(destination, manifest, recipe)
    _write_metadata(metadata)
    return metadata


def publish_embedding_partition(
    destination: InactiveTierDestination,
    *,
    recipe: EmbeddingRecipe,
    rows: Sequence[EmbeddingPartitionRow],
    max_rows: int = 128,
    source_generation: str | None = None,
    index_generation: str | None = None,
) -> EmbeddingPartitionReceipt:
    """Publish at most ``max_rows`` vectors, safely repeating a prior partition."""
    destination, manifest = _destination_context(destination)
    metadata = _read_metadata(destination)
    _assert_manifest_bindings(metadata, manifest)
    _assert_recipe(metadata, recipe)
    _assert_bindings(metadata, source_generation=source_generation, index_generation=index_generation)
    if metadata.sealed:
        raise EmbeddingTupleGenerationError("cannot publish into a sealed embedding generation")
    if max_rows <= 0 or len(rows) > max_rows:
        raise ValueError("embedding partition exceeds bounded row limit")

    hashes: list[str] = []
    with sqlite3.connect(destination.path) as conn:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise EmbeddingTupleGenerationError("embedding candidate requires sqlite-vec") from error
        inserted = 0
        for row in rows:
            value = bytes(row.vector_derivation_hash)
            if len(value) != 32:
                raise EmbeddingTupleGenerationError("embedding vector identity must be SHA-256")
            if len(row.embedding) != recipe.dimensions or any(not math.isfinite(float(item)) for item in row.embedding):
                raise ValueError("embedding vector does not match recipe dimensions")
            hashes.append(value.hex())
            existing = conn.execute(
                "SELECT model, dimension FROM message_embeddings_meta WHERE vector_derivation_hash = ?",
                (value,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing[0]) != recipe.model
                    or int(existing[1]) != recipe.dimensions
                    or conn.execute(
                        "SELECT 1 FROM message_embeddings WHERE vector_derivation_hash = ? LIMIT 1", (value.hex(),)
                    ).fetchone()
                    is None
                ):
                    raise EmbeddingTupleGenerationError("existing vector conflicts with tuple recipe")
                continue
            if (
                conn.execute(
                    "SELECT 1 FROM message_embeddings WHERE vector_derivation_hash = ? LIMIT 1", (value.hex(),)
                ).fetchone()
                is not None
            ):
                raise EmbeddingTupleGenerationError("vector bytes exist without authenticated metadata")
            conn.execute(
                "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                (value.hex(), _serialize_f32(list(row.embedding)), recipe.model),
            )
            conn.execute(
                "INSERT INTO message_embeddings_meta (vector_derivation_hash, model, dimension, embedded_at_ms, "
                "recipe_hash, output_contract_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    value,
                    recipe.model,
                    recipe.dimensions,
                    row.embedded_at_ms,
                    recipe.recipe_hash,
                    recipe.output_contract_hash,
                ),
            )
            inserted += 1
    _fsync_file(destination.path)
    _fsync_directory(destination.path.parent)
    return EmbeddingPartitionReceipt(inserted_count=inserted, vector_derivation_hashes=tuple(hashes))


def seal_inactive_embedding_generation(
    destination: InactiveTierDestination,
    *,
    recipe: EmbeddingRecipe,
    source_generation: str,
    index_generation: str,
) -> EmbeddingTupleGeneration:
    """Authenticate all candidate rows and atomically publish its construction seal."""
    destination, manifest = _destination_context(destination)
    metadata = _read_metadata(destination)
    _assert_manifest_bindings(metadata, manifest)
    _assert_recipe(metadata, recipe)
    _assert_bindings(metadata, source_generation=source_generation, index_generation=index_generation)
    with sqlite3.connect(destination.path) as conn:
        membership_digest = _membership_digest(conn, recipe)
    if metadata.sealed:
        if metadata.membership_digest != membership_digest:
            raise EmbeddingTupleGenerationError("sealed embedding membership changed")
        return metadata
    _fsync_file(destination.path)
    _fsync_directory(destination.path.parent)
    sealed = EmbeddingTupleGeneration(
        **{
            **asdict(metadata),
            "sealed": True,
            "membership_digest": membership_digest,
        }
    )
    _write_metadata(sealed)
    return sealed


__all__ = [
    "EmbeddingPartitionReceipt",
    "EmbeddingPartitionRow",
    "EmbeddingTupleGeneration",
    "EmbeddingTupleGenerationError",
    "prepare_inactive_embedding_generation",
    "publish_embedding_partition",
    "seal_inactive_embedding_generation",
]
