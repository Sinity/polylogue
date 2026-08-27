from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.archive_tuple_location import ArchiveTupleAllocator, ArchiveTupleLocation
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.tuple_generation import (
    EmbeddingPartitionRow,
    EmbeddingTupleGenerationError,
    prepare_inactive_embedding_generation,
    publish_embedding_partition,
    seal_inactive_embedding_generation,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _candidate(tmp_path: Path) -> ArchiveTupleLocation:
    initialize_active_archive_root(tmp_path)
    return ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path)).allocate(
        owner_id="embedding-builder",
        source_generation="source-v1",
        index_generation="index-v1",
        embeddings_generation="embeddings-v1",
    )


def _recipe(model: str = "test-model") -> EmbeddingRecipe:
    return EmbeddingRecipe.current(model=model, dimensions=1024)


def _row(value: int, *, recipe: EmbeddingRecipe) -> EmbeddingPartitionRow:
    return EmbeddingPartitionRow(
        vector_derivation_hash=bytes([value]) * 32,
        embedding=[float(value), 0.0, 1.0] + [0.0] * 1021,
        embedded_at_ms=1000 + value,
    )


def test_tuple_embedding_construction_is_bound_and_sealed_without_embedding_registry(tmp_path: Path) -> None:
    """Removing tuple binding, partition bounds, or seal validation makes this red."""
    candidate = _candidate(tmp_path)
    recipe = _recipe()

    generation = prepare_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )

    assert generation.generation_id == "embeddings-v1"
    assert generation.recipe_hash == recipe.recipe_hash.hex()
    assert generation.source_generation == "source-v1"
    assert generation.index_generation == "index-v1"
    assert generation.sealed is False
    assert not (tmp_path / ".embeddings-generations").exists()

    rows = (_row(1, recipe=recipe), _row(2, recipe=recipe))
    with pytest.raises(ValueError, match="bounded"):
        publish_embedding_partition(candidate.embeddings, recipe=recipe, rows=rows, max_rows=1)

    first = publish_embedding_partition(candidate.embeddings, recipe=recipe, rows=rows, max_rows=2)
    second = publish_embedding_partition(candidate.embeddings, recipe=recipe, rows=rows, max_rows=2)
    assert first.inserted_count == 2
    assert second.inserted_count == 0

    sealed = seal_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )
    assert sealed.sealed is True
    assert sealed.membership_digest

    with sqlite3.connect(candidate.embeddings.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 2
    assert not (tmp_path / ".embeddings-generations").exists()


def test_tuple_embedding_rejects_wrong_binding_or_recipe(tmp_path: Path) -> None:
    """Removing either admission check must allow a foreign candidate write."""
    candidate = _candidate(tmp_path)
    recipe = _recipe()
    prepare_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )

    with pytest.raises(EmbeddingTupleGenerationError, match="source generation"):
        publish_embedding_partition(
            candidate.embeddings,
            recipe=recipe,
            rows=(_row(1, recipe=recipe),),
            source_generation="source-v2",
        )
    with pytest.raises(EmbeddingTupleGenerationError, match="recipe"):
        publish_embedding_partition(
            candidate.embeddings,
            recipe=_recipe("other-model"),
            rows=(_row(1, recipe=recipe),),
        )


def test_tuple_embedding_rejects_tampered_metadata(tmp_path: Path) -> None:
    """Removing the metadata seal must allow an unauthenticated binding change."""
    candidate = _candidate(tmp_path)
    recipe = _recipe()
    prepare_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )
    metadata_path = candidate.candidate_root / "embeddings-generation.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["source_generation"] = "source-foreign"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EmbeddingTupleGenerationError, match="malformed"):
        seal_inactive_embedding_generation(
            candidate.embeddings,
            recipe=recipe,
            source_generation="source-v1",
            index_generation="index-v1",
        )


def test_sealed_tuple_embedding_is_immutable_and_reprepare_is_idempotent(tmp_path: Path) -> None:
    """Removing the sealed-state guard must permit post-seal mutation."""
    candidate = _candidate(tmp_path)
    recipe = _recipe()
    prepare_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )
    seal_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    )

    assert prepare_inactive_embedding_generation(
        candidate.embeddings,
        recipe=recipe,
        source_generation="source-v1",
        index_generation="index-v1",
    ).sealed
    with pytest.raises(EmbeddingTupleGenerationError, match="sealed"):
        publish_embedding_partition(
            candidate.embeddings,
            recipe=recipe,
            rows=(_row(1, recipe=recipe),),
        )
