from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.maintenance import embedding_preservation
from polylogue.maintenance.embedding_preservation import (
    RestoreMissReason,
    _receipt_path,
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
_VECTOR = b"\x00" * (1024 * 4)
_RECIPE = b"a" * 32
_CONTRACT = b"b" * 32


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error))
    return conn


def _db(path: Path, *, vectors: tuple[bytes, ...] = (_HASH,), metadata_only: tuple[bytes, ...] = ()) -> None:
    """Current-schema embeddings DB holding ``vectors`` plus vector-less metadata rows."""
    initialize_archive_database(path, ArchiveTier.EMBEDDINGS)
    with closing(_open(path)) as conn:
        for value in (*vectors, *metadata_only):
            if value in vectors:
                conn.execute(
                    "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                    (value.hex(), _VECTOR, "test"),
                )
            conn.execute(
                "INSERT INTO message_embeddings_meta "
                "(vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash) "
                "VALUES (?, 'test', 1024, ?, ?)",
                (value, _RECIPE, _CONTRACT),
            )
        conn.commit()


def _legacy_db(
    path: Path,
    *,
    vectors: tuple[bytes, ...] = (_HASH,),
    output_contract_hash: bytes | None = _CONTRACT,
) -> None:
    """Pre-v5 embeddings DB: hashes named ``embedding_input_hash``, identity columns nullable.

    Mirrors the live archive DDL at /realm/state/polylogue/embeddings.db.
    """
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE message_embeddings_meta (
                embedding_input_hash BLOB PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedded_at_ms INTEGER,
                recipe_hash BLOB,
                output_contract_hash BLOB
            );
            CREATE TABLE message_embeddings (
                embedding_input_hash TEXT PRIMARY KEY,
                embedding BLOB,
                model TEXT
            );
            """
        )
        for value in vectors:
            conn.execute("INSERT INTO message_embeddings VALUES (?, ?, 'test')", (value.hex(), _VECTOR))
            conn.execute(
                "INSERT INTO message_embeddings_meta VALUES (?, 'test', 1024, NULL, ?, ?)",
                (value, _RECIPE, output_contract_hash),
            )
        conn.commit()


def _count(path: Path, table: str) -> int:
    with closing(_open(path)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _ac2_proof(receipt: Any, path: Path, **override: Any) -> Path:
    """Write the preservation receipt back as an AC2-passed deletion proof."""
    from dataclasses import asdict

    proof = asdict(receipt) | {"ac2_passed": True} | override
    path.write_text(json.dumps(proof), encoding="utf-8")
    return path


def test_preserve_restore_and_proof_deletion(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(source)
    _db(fresh, vectors=(_OTHER,))

    before = preserve_embedding_vectors(source, preserved)
    assert before.metadata_rows == before.vector_rows == 1
    assert before.table_set_digest

    restored = restore_embedding_vectors(fresh, preserved, {_HASH})
    assert restored.restored_hashes == 1
    assert restored.misses == ()
    with closing(sqlite3.connect(fresh)) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE vector_derivation_hash = ?", (_HASH,)
            ).fetchone()[0]
            == 1
        )

    proof = _ac2_proof(before, preserved.with_suffix(preserved.suffix + ".proof.json"))
    delete_preserved_copy(preserved, receipt_path=proof)
    assert not preserved.exists()
    assert not proof.exists()


def test_missing_preserved_hash_is_enumerated(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(source)
    _db(fresh, vectors=(_OTHER,))
    preserve_embedding_vectors(source, preserved)
    result = restore_embedding_vectors(fresh, preserved, {_HASH, _MISSING})
    assert result.restored_hashes == 1
    assert [(miss.input_hash, miss.reason) for miss in result.misses] == [
        (_MISSING.hex(), RestoreMissReason.METADATA_ABSENT)
    ]


def test_restore_maps_legacy_embedding_input_hash_to_current_identity(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _legacy_db(source)
    _db(fresh, vectors=(_OTHER,))

    preserve_embedding_vectors(source, preserved)
    result = restore_embedding_vectors(fresh, preserved, {_HASH})

    assert result.restored_hashes == 1
    with closing(sqlite3.connect(fresh)) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE vector_derivation_hash = ?", (_HASH,)
            ).fetchone()[0]
            == 1
        )


def test_restore_batches_hashes_below_the_build_variable_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red without batching: one IN list of six hashes exceeds a four-variable build."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    wanted = tuple(bytes([index]) * 32 for index in range(1, 7))
    _db(source, vectors=wanted)
    _db(fresh, vectors=(_OTHER,))
    preserve_embedding_vectors(source, preserved)

    original = embedding_preservation._connect

    def small_limit(path: Any, *, readonly: bool) -> sqlite3.Connection:
        conn = original(path, readonly=readonly)
        if readonly:
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 4)
        return conn

    monkeypatch.setattr(embedding_preservation, "_connect", small_limit)
    result = restore_embedding_vectors(fresh, preserved, set(wanted))

    assert result.restored_hashes == len(wanted)
    assert result.misses == ()


def test_metadata_without_its_vector_is_a_miss_and_writes_nothing(tmp_path: Path) -> None:
    """Red when metadata is written before its vector is found: a metadata row is the
    tier's reuse signal, so one without a vector silently suppresses re-embedding."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(source, vectors=(), metadata_only=(_HASH,))
    _db(fresh, vectors=(_OTHER,))
    preserve_embedding_vectors(source, preserved)

    result = restore_embedding_vectors(fresh, preserved, {_HASH})

    assert result.restored_hashes == 0
    assert [(miss.input_hash, miss.reason) for miss in result.misses] == [
        (_HASH.hex(), RestoreMissReason.VECTOR_ABSENT)
    ]
    with closing(sqlite3.connect(fresh)) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE vector_derivation_hash = ?", (_HASH,)
            ).fetchone()[0]
            == 0
        )


def test_incomplete_legacy_metadata_is_a_typed_miss(tmp_path: Path) -> None:
    """Red when an incomplete row is inserted: the current tier's NOT NULL identity
    contract rejects it and aborts every remaining hash in the restore."""
    source = tmp_path / "legacy.db"
    preserved = tmp_path / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _legacy_db(source, vectors=(_HASH,), output_contract_hash=None)
    _db(fresh, vectors=(_OTHER,))
    preserve_embedding_vectors(source, preserved)

    result = restore_embedding_vectors(fresh, preserved, {_HASH, _MISSING})

    assert result.restored_hashes == 0
    assert [(miss.input_hash, miss.reason, miss.detail) for miss in result.misses] == [
        (_HASH.hex(), RestoreMissReason.METADATA_INCOMPLETE, "output_contract_hash"),
        (_MISSING.hex(), RestoreMissReason.METADATA_ABSENT, ""),
    ]


def test_deletion_refuses_a_receipt_naming_another_copy(tmp_path: Path) -> None:
    """Red when the proof is not bound to the copy: any AC2-passed receipt authorizes
    deleting any file."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    other = tmp_path / "other.db"
    _db(source)
    receipt = preserve_embedding_vectors(source, preserved)
    other.write_bytes(preserved.read_bytes())

    proof = _ac2_proof(receipt, tmp_path / "proof.json", copy=str(other))
    with pytest.raises(ValueError, match="different copy"):
        delete_preserved_copy(preserved, receipt_path=proof)
    assert preserved.exists()


def test_deletion_refuses_a_receipt_whose_digest_is_stale(tmp_path: Path) -> None:
    """Red without a digest re-check: a copy mutated after its receipt still deletes."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    _db(source)
    receipt = preserve_embedding_vectors(source, preserved)
    with closing(_open(preserved)) as conn:
        conn.execute("DELETE FROM message_embeddings WHERE vector_derivation_hash = ?", (_HASH.hex(),))
        conn.commit()

    proof = _ac2_proof(receipt, tmp_path / "proof.json")
    with pytest.raises(ValueError, match="receipt digest"):
        delete_preserved_copy(preserved, receipt_path=proof)
    assert preserved.exists()


# Interrupting a backup from inside this process is impossible: sqlite3 discards
# exceptions raised in a progress callback, so the copy always runs to completion.
# The callback instead crashes the interpreter, which is the failure being guarded.
_CRASH_MID_BACKUP = """
import os, sys
from pathlib import Path
from polylogue.maintenance import embedding_preservation as ep

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
original = ep._connect


def interrupted(path, *, readonly):
    conn = original(path, readonly=readonly)
    if Path(path) != source:
        return conn

    class Crash:
        def __getattr__(self, name):
            return getattr(conn, name)

        def __enter__(self):
            conn.__enter__()
            return self

        def __exit__(self, *exc):
            return conn.__exit__(*exc)

        def backup(self, target, **kwargs):
            conn.backup(target, pages=1, progress=lambda *_: os._exit(9))

    return Crash()


ep._connect = interrupted
ep.preserve_embedding_vectors(source, destination)
"""


def test_a_crash_mid_backup_leaves_no_destination_file(tmp_path: Path) -> None:
    """Red when the backup writes straight to the destination: the crash leaves a
    truncated file there that no later run can tell apart from a whole copy."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    _db(source, vectors=tuple(bytes([index]) * 32 for index in range(1, 25)))

    crash = subprocess.run(
        [sys.executable, "-c", _CRASH_MID_BACKUP, str(source), str(preserved)],
        capture_output=True,
        text=True,
    )

    assert crash.returncode == 9, crash.stderr
    assert not preserved.exists()
    assert not _receipt_path(preserved).exists()


def test_receipt_counts_come_from_the_completed_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red when the receipt is read before the backup: a source written between the two
    reads yields a receipt that describes no file."""
    source = tmp_path / "source.db"
    preserved = tmp_path / "preserved.db"
    _db(source)

    original = embedding_preservation._table_digest
    grew = False

    def growing_source(conn: sqlite3.Connection) -> Any:
        nonlocal grew
        result = original(conn)
        if not grew:
            grew = True
            with closing(_open(source)) as writer:
                writer.execute(
                    "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                    (_OTHER.hex(), _VECTOR, "test"),
                )
                writer.execute(
                    "INSERT INTO message_embeddings_meta "
                    "(vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash) "
                    "VALUES (?, 'test', 1024, ?, ?)",
                    (_OTHER, _RECIPE, _CONTRACT),
                )
                writer.commit()
        return result

    monkeypatch.setattr(embedding_preservation, "_table_digest", growing_source)
    receipt = preserve_embedding_vectors(source, preserved)

    assert receipt.metadata_rows == _count(preserved, "message_embeddings_meta")
    assert receipt.vector_rows == _count(preserved, "message_embeddings")
    with closing(embedding_preservation._connect(preserved, readonly=True)) as conn:
        assert receipt.table_set_digest == original(conn)[0]


def test_preserved_copy_is_self_contained(tmp_path: Path) -> None:
    """Red when the copy keeps the source's WAL mode: the rename moves only the main
    file, leaving the copy beside sidecars that hold its content."""
    source = tmp_path / "source.db"
    _db(source)
    with closing(_open(source)) as conn:
        conn.execute("PRAGMA journal_mode=WAL").fetchall()
    vault = tmp_path / "vault"
    preserved = vault / "preserved.db"
    fresh = tmp_path / "fresh.db"
    _db(fresh, vectors=(_OTHER,))

    preserve_embedding_vectors(source, preserved)

    assert sorted(entry.name for entry in vault.iterdir()) == [
        "preserved.db",
        "preserved.db.receipt.json",
    ]
    assert restore_embedding_vectors(fresh, preserved, {_HASH}).restored_hashes == 1
