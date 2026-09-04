from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.embedding_preservation import (
    delete_preserved_copy,
    preserve_embedding_vectors,
    restore_embedding_vectors,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_HASH = b"h" * 32
_OTHER = b"o" * 32
_MISSING = b"m" * 32


def _db(path: Path, *, vector: bytes = _HASH) -> None:
    initialize_archive_database(path, ArchiveTier.EMBEDDINGS)
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error))
    conn.execute(
        "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
        (vector.hex(), b"\x00" * (1024 * 4), "test"),
    )
    conn.execute(
        "INSERT INTO message_embeddings_meta (vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash) "
        "VALUES (?, 'test', 1024, ?, ?)",
        (vector, b"a" * 32, b"b" * 32),
    )
    conn.commit()
    conn.close()


def test_preserve_restore_and_proof_deletion(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(source)
    _db(fresh, vector=_OTHER)

    before = preserve_embedding_vectors(source, preserved)
    assert before.metadata_rows == before.vector_rows == 1
    assert before.table_set_digest

    restored = restore_embedding_vectors(fresh, preserved, {_HASH})
    assert restored.restored_hashes == 1
    assert restored.missing_hashes == ()
    with sqlite3.connect(fresh) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE vector_derivation_hash = ?", (_HASH,)
            ).fetchone()[0]
            == 1
        )

    proof = preserved.with_suffix(preserved.suffix + ".proof.json")
    proof.write_text(json.dumps({"ac2_passed": True}))
    delete_preserved_copy(preserved, receipt_path=proof)
    assert not preserved.exists()
    assert not proof.exists()


def test_missing_preserved_hash_is_enumerated(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(source)
    _db(fresh, vector=_OTHER)
    preserve_embedding_vectors(source, preserved)
    result = restore_embedding_vectors(fresh, preserved, {_HASH, _MISSING})
    assert result.restored_hashes == 1
    assert result.missing_hashes == (_MISSING.hex(),)
