"""Neutral retained-vector fixture for embedding lifecycle startup laws."""

from __future__ import annotations

import struct
from contextlib import closing
from pathlib import Path

from polylogue.storage.embeddings.identity import EmbeddingRecipe
from tests.infra.embedding_backup_fixture import connect_vector_fixture


def seed_retained_vector(path: Path) -> None:
    recipe = EmbeddingRecipe.current(model="voyage-4", dimensions=1024)
    address = b"\x13" * 32
    vector = struct.pack("<1024f", *([0.125] * 1024))
    with closing(connect_vector_fixture(path)) as conn, conn:
        conn.execute(
            "INSERT INTO message_embeddings(vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
            (address.hex(), vector, recipe.model),
        )
        conn.execute(
            "INSERT INTO message_embeddings_meta(vector_derivation_hash, model, dimension, embedded_at_ms, "
            "recipe_hash, output_contract_hash) VALUES (?, ?, ?, 0, ?, ?)",
            (address, recipe.model, 1024, recipe.recipe_hash, recipe.output_contract_hash),
        )
