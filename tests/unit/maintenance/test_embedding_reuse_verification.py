"""AC2/AC3 coverage: reuse measurement and the proof that authorizes deletion."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.maintenance.embedding_preservation import (
    AC2_RECEIPT_SCHEMA,
    ReuseMissReason,
    delete_preserved_copy,
    preserve_embedding_vectors,
    recomputed_vector_hashes,
    verify_embedding_reuse,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_MISSING = b"m" * 32
_VECTOR = b"\x00" * (1024 * 4)
_RECIPE = b"a" * 32
_CONTRACT = b"b" * 32
_MODEL = "voyage-4"
_PROSE = (
    "the preserved corpus prose belonging to the first message",
    "the preserved corpus prose belonging to the second message",
)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error))
    return conn


def _index_db(path: Path, prose: tuple[str, ...]) -> None:
    """Minimal index tier carrying embeddable authored prose, one message per entry."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                content_hash TEXT,
                text TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, 'conv-1', 'user', 'message', 'human_authored', 8, NULL, ?)",
            [(f"msg-{index}", value) for index, value in enumerate(prose)],
        )
        conn.commit()


def _embeddings_db(path: Path, hashes: tuple[bytes, ...]) -> None:
    initialize_archive_database(path, ArchiveTier.EMBEDDINGS)
    with closing(_open(path)) as conn:
        for value in hashes:
            conn.execute(
                "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                (value.hex(), _VECTOR, _MODEL),
            )
            conn.execute(
                "INSERT INTO message_embeddings_meta "
                "(vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash) "
                "VALUES (?, ?, 1024, ?, ?)",
                (value, _MODEL, _RECIPE, _CONTRACT),
            )
        conn.commit()


def _reuse_fixture(tmp_path: Path, *, restored: tuple[bytes, ...] | None = None) -> tuple[Path, Path, dict[str, bytes]]:
    """A preserved copy, plus a fresh embeddings tier holding ``restored``."""
    index = tmp_path / "index.db"
    _index_db(index, _PROSE)
    recomputed = recomputed_vector_hashes(index, model=_MODEL)
    wanted = tuple(sorted(recomputed.values()))

    source = tmp_path / "source.db"
    _embeddings_db(source, wanted)
    preserved = tmp_path / "preserved.db"
    preserve_embedding_vectors(source, preserved)

    fresh = tmp_path / "fresh.db"
    _embeddings_db(fresh, wanted if restored is None else restored)
    return fresh, preserved, recomputed


def test_recomputed_hashes_come_from_the_production_embeddable_relation(tmp_path: Path) -> None:
    """Red if the address is computed any other way: it must equal what the embedder sends."""
    from polylogue.storage.embeddings.identity import vector_derivation_hash

    index = tmp_path / "index.db"
    _index_db(index, _PROSE)

    recomputed = recomputed_vector_hashes(index, model=_MODEL)

    assert set(recomputed) == {"msg-0", "msg-1"}
    assert recomputed["msg-0"] == vector_derivation_hash(model=_MODEL, input_text=_PROSE[0])


def test_full_reuse_passes_ac2_and_authorizes_deletion(tmp_path: Path) -> None:
    fresh, preserved, recomputed = _reuse_fixture(tmp_path)
    receipt = tmp_path / "ac2.json"

    verification = verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL, receipt_path=receipt)

    assert (verification.recomputed_hashes, verification.hit_hashes) == (2, 2)
    assert verification.misses == ()
    assert verification.ac2_passed
    proof = json.loads(receipt.read_text(encoding="utf-8"))
    assert proof["schema"] == AC2_RECEIPT_SCHEMA
    assert proof["copy"] == str(preserved)

    delete_preserved_copy(preserved, receipt_path=receipt)
    assert not preserved.exists()


def test_deleting_one_preserved_vector_turns_ac2_red_by_exactly_that_hash(tmp_path: Path) -> None:
    """AC2's anti-vacuity condition: removing one preserved vector is the only change,
    and shows up as exactly one miss naming that address."""
    fresh, preserved, recomputed = _reuse_fixture(tmp_path)
    dropped = sorted(recomputed.values())[0]
    with closing(_open(preserved)) as conn:
        conn.execute("DELETE FROM message_embeddings WHERE vector_derivation_hash = ?", (dropped.hex(),))
        conn.commit()

    verification = verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL)

    assert verification.hit_hashes == 1
    assert [(miss.input_hash, miss.reason) for miss in verification.misses] == [
        (dropped.hex(), ReuseMissReason.PRESERVED_VECTOR_ABSENT)
    ]
    assert not verification.ac2_passed


def test_a_restore_that_wrote_no_vector_is_a_miss_not_a_hit(tmp_path: Path) -> None:
    """Red if metadata alone counts as reuse: the embedder would find no vector there."""
    fresh, preserved, recomputed = _reuse_fixture(tmp_path)
    dropped = sorted(recomputed.values())[1]
    with closing(_open(fresh)) as conn:
        conn.execute("DELETE FROM message_embeddings WHERE vector_derivation_hash = ?", (dropped.hex(),))
        conn.commit()

    verification = verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL)

    assert [(miss.input_hash, miss.reason) for miss in verification.misses] == [
        (dropped.hex(), ReuseMissReason.RESTORE_VECTOR_ABSENT)
    ]


def test_content_the_archive_never_embedded_is_its_own_miss_class(tmp_path: Path) -> None:
    fresh, preserved, recomputed = _reuse_fixture(tmp_path)
    recomputed["msg-new"] = _MISSING

    verification = verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL)

    assert verification.recomputed_hashes == 3
    assert [(miss.input_hash, miss.reason) for miss in verification.misses] == [
        (_MISSING.hex(), ReuseMissReason.NOT_PRESERVED)
    ]
    assert verification.as_receipt()["misses_by_reason"] == {"not_preserved": 1}


def test_a_failing_measurement_cannot_authorize_deletion(tmp_path: Path) -> None:
    fresh, preserved, recomputed = _reuse_fixture(tmp_path, restored=())
    receipt = tmp_path / "ac2.json"

    verification = verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL, receipt_path=receipt)

    assert verification.hit_hashes == 0
    assert not verification.ac2_passed
    with pytest.raises(ValueError, match="AC2-passed receipt"):
        delete_preserved_copy(preserved, receipt_path=receipt)
    assert preserved.exists()


def test_an_empty_corpus_proves_no_reuse(tmp_path: Path) -> None:
    """Red without the `recomputed_hashes > 0` guard: a threshold of zero would let a
    corpus that measured nothing authorize discarding the copy."""
    fresh, preserved, _recomputed = _reuse_fixture(tmp_path)

    verification = verify_embedding_reuse(fresh, preserved, {}, model=_MODEL, minimum_hit_rate=0.0)

    assert verification.recomputed_hashes == 0
    assert not verification.ac2_passed


def test_a_preservation_receipt_alone_does_not_authorize_deletion(tmp_path: Path) -> None:
    """Red if the gate accepts any `ac2_passed` flag: only a measured proof counts."""
    _fresh, preserved, _recomputed = _reuse_fixture(tmp_path)
    receipt = json.loads(Path(str(preserved) + ".receipt.json").read_text(encoding="utf-8"))
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(receipt | {"ac2_passed": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="AC2-passed receipt"):
        delete_preserved_copy(preserved, receipt_path=forged)
    assert preserved.exists()


def test_immutable_preservation_leaves_the_source_sidecars_alone(tmp_path: Path) -> None:
    """Red without immutable: a plain read-only open creates the shared-memory file
    beside a source the caller promised only to read."""
    source = tmp_path / "source.db"
    _embeddings_db(source, (_MISSING,))
    with closing(_open(source)) as conn:
        conn.execute("PRAGMA journal_mode=WAL").fetchall()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    for suffix in ("-wal", "-shm"):
        Path(str(source) + suffix).unlink(missing_ok=True)

    preserve_embedding_vectors(source, tmp_path / "preserved.db", immutable=True)

    assert not Path(str(source) + "-shm").exists()


def test_the_reuse_proof_does_not_overwrite_the_preservation_record(tmp_path: Path) -> None:
    """Red if both receipts share a path: proving reuse would erase what was copied."""
    from polylogue.maintenance.embedding_preservation import ac2_receipt_path

    fresh, preserved, recomputed = _reuse_fixture(tmp_path)
    preservation = Path(str(preserved) + ".receipt.json")
    before = preservation.read_text(encoding="utf-8")

    verify_embedding_reuse(fresh, preserved, recomputed, model=_MODEL, receipt_path=ac2_receipt_path(preserved))

    assert preservation.read_text(encoding="utf-8") == before
    assert json.loads(ac2_receipt_path(preserved).read_text(encoding="utf-8"))["schema"] == AC2_RECEIPT_SCHEMA
