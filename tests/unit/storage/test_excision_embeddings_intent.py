"""Actual WITHOUT ROWID vec0 input custody and frozen retirement intent."""

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from polylogue.storage.sqlite.reference_seal import KnownTierRowImage, PreparedIndexMutation, ReferenceSealError
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.excision_embeddings import begin_embedding_excision_control, prepare_embedding_excision_source_command


@pytest.mark.parametrize("shared", [False, True])
def test_actual_vec0_intent_uses_logical_key_and_preserves_shared_output(tmp_path: Path, shared: bool) -> None:
    started, args, vector_hash = begin_embedding_excision_control(tmp_path, shared=shared)
    assert started.operation_id is not None
    charges: list[int] = []
    with PreparedIndexMutation.for_excision(
        tmp_path / "index.db", archive_root=tmp_path, input_demand=charges.append
    ) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with seal.original_read_snapshot():
            seal.prepare_excision_embeddings_intent()
            intent = json.loads(b"".join(seal.excision_embeddings_intent_chunks()))
            assert intent["schema_version"] == 1 and intent["present"] is True
            assert intent["outputs"] == [
                {
                    "meta_present": True,
                    "retire": not shared,
                    "vector_derivation_hash": vector_hash.hex(),
                    "vector_present": True,
                }
            ]
            vectors = [row for row in intent["rows"] if row["table"] == "message_embeddings"]
            if shared:
                assert vectors == []
            else:
                assert len(vectors) == 1
                assert vectors[0]["row_address"] == {"vector_derivation_hash": vector_hash.hex()}
                assert vectors[0]["cells"][0]["literal_hex"] == vector_hash.hex().encode().hex()
                assert vectors[0]["cells"][1] == {
                    "byte_length": 4096,
                    "literal_hex": struct.pack("f", 0.25).hex() * 1024,
                    "storage_class": "blob",
                }
                assert sum(charges) >= 4096
                with pytest.raises(ReferenceSealError):
                    seal.retain_tier_row("embeddings", "message_embeddings", 1)
                with pytest.raises(ReferenceSealError):
                    seal.retain_tier_row("embeddings", "message_embeddings", b"wrong-key")
            # The producer captured intent only; no occurrence or output was deleted.
            with seal.original_rows("embeddings", "SELECT count(*) FROM message_embeddings") as rows:
                assert rows.fetchone()[0] == (1 if shared else 2)
            with seal.original_rows("embeddings", "SELECT count(*) FROM message_embedding_refs") as rows:
                assert rows.fetchone()[0] == 2


@pytest.mark.parametrize("shared,last_vector", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("rollback", [False, True])
async def test_actual_embedding_child_guards_deletes_and_physically_settles(
    tmp_path: Path,
    shared: bool,
    last_vector: bool,
    rollback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.operations.mutation_actuators import SessionExcisionActuator
    from polylogue.operations.mutation_transaction import _authorized_removal_apply
    from polylogue.storage.sqlite.connection_profile import native_sql_children
    from polylogue.storage.sqlite.reference_seal import _PreparedExcisionEmbeddingsChild

    class InjectedPostDeleteError(Exception):
        pass

    injected = InjectedPostDeleteError("exact post-delete control")
    entered = []
    deleted = []
    refused = []
    actual_delete = _PreparedExcisionEmbeddingsChild._delete

    def guarded_delete(child: _PreparedExcisionEmbeddingsChild, connection: sqlite3.Connection) -> None:
        entered.append(connection)
        # Parameterized metadata reads are real reads on the paid relations,
        # while no-operand storage mutations and retired setup settings refuse.
        for table in (
            "message_embeddings",
            "message_embedding_refs",
            "message_embeddings_meta",
            "embedding_status",
            "embedding_derivation_state",
            "embedding_failures",
            "excision_embedding_completions",
        ):
            for pragma in ("table_info", "table_xinfo", "index_list", "foreign_key_list"):
                with child._seal._owned_cursor(connection, f'PRAGMA main.{pragma}("{table}")') as rows:
                    tuple(rows)
        for statement in (
            "PRAGMA optimize",
            "PRAGMA wal_checkpoint",
            "PRAGMA incremental_vacuum",
            "PRAGMA synchronous = OFF",
            "PRAGMA query_only = OFF",
            "PRAGMA foreign_keys = ON",
            "PRAGMA user_version = 1",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                with child._seal._owned_cursor(connection, statement):
                    pass
        # A writable relation alone grants no authority outside the exact
        # fixed producer statement. This is actual native authorization.
        with pytest.raises(sqlite3.DatabaseError):
            with child._seal._owned_cursor(connection, "DELETE FROM embedding_status"):
                pass
        for table in ("message_embeddings_vector_chunks00", "message_embeddings_meta"):
            with pytest.raises(sqlite3.DatabaseError):
                with child._seal._owned_cursor(connection, f"DELETE FROM {table}"):
                    pass
            refused.append(table)
        actual_delete(child, connection)
        deleted.append(connection)
        with child._seal._owned_cursor(
            connection, "SELECT count(*) FROM message_embedding_refs WHERE session_id=?", (args_session[0],)
        ) as rows:
            assert rows.fetchone()[0] == 0
        with child._seal._owned_cursor(
            connection, "SELECT count(*) FROM message_embeddings WHERE vector_derivation_hash=?", ("ab" * 32,)
        ) as rows:
            assert rows.fetchone()[0] == int(shared)
        if last_vector:
            with child._seal._owned_cursor(
                connection, "SELECT count(*) FROM message_embeddings_vector_chunks00"
            ) as rows:
                assert rows.fetchone()[0] == 0
        if rollback:
            raise injected

    monkeypatch.setattr(_PreparedExcisionEmbeddingsChild, "_delete", guarded_delete)

    args_session = []

    def apply_child() -> None:
        started, args, vector_hash = begin_embedding_excision_control(
            tmp_path,
            shared=shared,
            include_survivor=not last_vector,
        )
        args_session.append(args.session_id)
        assert started.operation_id is not None
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            with PreparedIndexMutation.for_excision(
                tmp_path / "index.db",
                archive_root=tmp_path,
                input_demand=lambda _size: None,
            ) as seal:
                seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
                with seal.original_read_snapshot():
                    surviving_key = vector_hash if shared else bytes.fromhex("cd" * 32)
                    surviving_vector = (
                        None if last_vector else seal.retain_tier_row("embeddings", "message_embeddings", surviving_key)
                    )
                    selected_vector = seal.retain_tier_row("embeddings", "message_embeddings", vector_hash)
                    assert selected_vector is not None
                    assert last_vector or surviving_vector is not None
                    original_rows: list[KnownTierRowImage | None] = []
                    for table in ("message_embedding_refs", "message_embeddings_meta", "embedding_status"):
                        with seal.original_rows("embeddings", f"SELECT rowid FROM {table} ORDER BY rowid") as rows:
                            addresses = tuple(row[0] for row in rows)
                        original_rows.extend(
                            seal.retain_tier_row("embeddings", table, address) for address in addresses
                        )
                    assert all(image is not None for image in original_rows)
                    seal.prepare_excision_embeddings_intent()
                    prepare_embedding_excision_source_command(seal, started, args)
                child = seal.prepare_excision_embeddings_child()
                original_epoch = seal._original_input_epochs["embeddings"]
                if rollback:
                    with pytest.raises(InjectedPostDeleteError) as caught:
                        child.apply()
                    assert caught.value is injected
                    assert child._completed is False
                    assert seal._original_input_epochs["embeddings"] == original_epoch
                else:
                    child.apply()
                    assert child._completed is True
                assert entered == deleted == [child._connection]
                assert refused == ["message_embeddings_vector_chunks00", "message_embeddings_meta"]
                owner = next(
                    owner for owner in native_sql_children(seal) if owner._connection_identity == id(child._connection)
                )
                assert owner._settled and owner.connection is None
                seal.validate_observers_current()
                with seal.original_read_snapshot():
                    if surviving_vector is not None:
                        assert seal._matches_retained_row(
                            seal.observer("embeddings"),
                            surviving_vector,
                            logical_vector_key=surviving_key,
                        )
                    if rollback:
                        assert seal._matches_retained_row(
                            seal.observer("embeddings"),
                            selected_vector,
                            logical_vector_key=vector_hash,
                        )
                        for image in original_rows:
                            assert image is not None and seal._matches_retained_row(seal.observer("embeddings"), image)
                    with seal.original_rows(
                        "embeddings", "SELECT count(*) FROM excision_embedding_completions"
                    ) as rows:
                        assert rows.fetchone()[0] == int(not rollback)
                    with seal.original_rows("embeddings", "SELECT count(*) FROM message_embedding_refs") as rows:
                        assert rows.fetchone()[0] == (
                            (1 if last_vector else 2) if rollback else (0 if last_vector else 1)
                        )
                    with seal.original_rows("embeddings", "SELECT count(*) FROM message_embeddings") as rows:
                        assert rows.fetchone()[0] == (
                            1 if shared else (1 if last_vector else 2) if rollback else (0 if last_vector else 1)
                        )
                    with seal.original_rows(
                        "embeddings",
                        "SELECT count(*) FROM message_embeddings WHERE vector_derivation_hash=?",
                        (vector_hash.hex(),),
                    ) as rows:
                        assert rows.fetchone()[0] == int(shared or rollback)

    await run_archive_fixture_write(tmp_path, apply_child)
