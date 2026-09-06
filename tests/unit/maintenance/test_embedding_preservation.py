from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from polylogue.maintenance import embedding_preservation
from polylogue.maintenance.embedding_preservation import (
    RestoreMissReason,
    _receipt_path,
    ac2_receipt_path,
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
_MODEL = "test"


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
    """Write the preservation receipt back in the shape of a passing AC2 proof."""
    from dataclasses import asdict

    from polylogue.maintenance.embedding_preservation import AC2_RECEIPT_SCHEMA

    proof = asdict(receipt) | {"schema": AC2_RECEIPT_SCHEMA, "ac2_passed": True} | override
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


def interrupted(path, *, readonly, immutable=False):
    conn = original(path, readonly=readonly, immutable=immutable)
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


_PROSE = (
    "the preserved corpus prose belonging to the first message",
    "the preserved corpus prose belonging to the second message",
)


def _index_db(path: Path) -> None:
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
            [(f"msg-{index}", value) for index, value in enumerate(_PROSE)],
        )
        conn.commit()


def _outgoing_archive(tmp_path: Path) -> Path:
    """An archive root whose embeddings tier holds the vectors its index will ask for."""
    from polylogue.maintenance.embedding_preservation import recomputed_vector_hashes

    root = tmp_path / "archive"
    root.mkdir()
    _index_db(root / "index.db")
    wanted = tuple(sorted(recomputed_vector_hashes(root / "index.db", model=_MODEL).values()))
    assert len(wanted) == len(_PROSE)
    _db(root / "embeddings.db", vectors=wanted)
    return root


def _wipe_embeddings_tier(root: Path) -> None:
    """Replace the tier with an empty one, as a fresh start does."""
    for suffix in ("", "-wal", "-shm"):
        Path(str(root / "embeddings.db") + suffix).unlink(missing_ok=True)
    _db(root / "embeddings.db", vectors=())


def _maintenance_group() -> Any:
    from polylogue.cli.click_command_registration import OPS_COMMANDS

    for command in OPS_COMMANDS:
        if command.name == "maintenance":
            return command
    raise AssertionError("maintenance is not registered under polylogue ops")


def _run(*args: str) -> Any:
    return CliRunner().invoke(_maintenance_group(), ["embedding-preservation", *args])


def _phase(root: Path, *args: str) -> Any:
    return _run(*args, "--archive-root", str(root))


def test_the_production_route_carries_vectors_across_a_wipe_and_proves_the_reuse(tmp_path: Path) -> None:
    """The four phases run as an operator runs them, against one archive root.

    Red when ``embedding-preservation`` is not reachable under ``ops maintenance``
    -- the state in which this module has no production caller at all -- and red
    when ``verify`` stops writing its proof where ``discard`` reads it, which
    leaves the deletion path gated on a receipt nothing produces.
    """
    root = _outgoing_archive(tmp_path)
    copy = tmp_path / "preserved.db"

    preserve = _phase(root, "preserve", str(copy), "--output-format", "json")
    assert preserve.exit_code == 0, preserve.output
    assert json.loads(preserve.output)["vector_rows"] == len(_PROSE)

    _wipe_embeddings_tier(root)

    restore = _phase(root, "restore", str(copy), "--model", _MODEL, "--output-format", "json")
    assert restore.exit_code == 0, restore.output
    imported = json.loads(restore.output)
    assert (imported["restored_hashes"], imported["misses"]) == (len(_PROSE), [])

    verify = _phase(root, "verify", str(copy), "--model", _MODEL, "--output-format", "json")
    assert verify.exit_code == 0, verify.output
    proof = json.loads(verify.output)
    assert (proof["hit_hashes"], proof["ac2_passed"]) == (len(_PROSE), True)

    assert json.loads(ac2_receipt_path(copy).read_text(encoding="utf-8")) == proof

    discard = _run("discard", str(copy))
    assert discard.exit_code == 0, discard.output
    assert not copy.exists()


def test_the_route_refuses_to_discard_a_copy_whose_reuse_it_could_not_prove(tmp_path: Path) -> None:
    """Red when ``verify`` exits zero below its threshold, or when the proof it wrote
    for a failed measurement still authorizes deletion."""
    root = _outgoing_archive(tmp_path)
    copy = tmp_path / "preserved.db"
    assert _phase(root, "preserve", str(copy), "--output-format", "json").exit_code == 0

    _wipe_embeddings_tier(root)

    verify = _phase(root, "verify", str(copy), "--model", _MODEL, "--output-format", "json")
    assert verify.exit_code == 1, verify.output
    assert json.loads(verify.output)["ac2_passed"] is False

    discard = _run("discard", str(copy))
    assert discard.exit_code != 0
    assert isinstance(discard.exception, ValueError)
    assert copy.exists()
