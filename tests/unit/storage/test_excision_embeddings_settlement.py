"""Actual paid-tier cancellation, native settlement and frozen-ref refusals."""

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.compute_cancel import compute_cancel
from polylogue.operations.mutation_actuators import SessionExcisionActuator
from polylogue.operations.mutation_transaction import _authorized_removal_apply
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    native_sql_children,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.reference_seal import (
    PreparedIndexMutation,
    ReferenceSealError,
    ReferenceSealStaleError,
    _PreparedExcisionEmbeddingsChild,
)
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.excision_embeddings import (
    begin_embedding_excision_control,
    prepare_embedding_excision_source_command,
)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("failure", ["cancellation", "commit"])
async def test_actual_embedding_child_precommit_failure_preserves_original_paid_rows(
    tmp_path: Path,
    shared: bool,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def apply_child() -> None:
        started, args, vector_hash = begin_embedding_excision_control(tmp_path, shared=shared)
        assert started.operation_id is not None
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            with PreparedIndexMutation.for_excision(
                tmp_path / "index.db",
                archive_root=tmp_path,
                input_demand=lambda _size: None,
            ) as seal:
                seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
                with seal.original_read_snapshot():
                    selected_vector = seal.retain_tier_row("embeddings", "message_embeddings", vector_hash)
                    surviving_key = vector_hash if shared else bytes.fromhex("cd" * 32)
                    surviving_vector = seal.retain_tier_row("embeddings", "message_embeddings", surviving_key)
                    assert selected_vector is not None and surviving_vector is not None
                    with seal.original_rows(
                        "embeddings", "SELECT rowid FROM message_embedding_refs ORDER BY rowid"
                    ) as rows:
                        reference_rows = tuple(row[0] for row in rows)
                    references = tuple(
                        seal.retain_tier_row("embeddings", "message_embedding_refs", rowid) for rowid in reference_rows
                    )
                    with seal.original_rows(
                        "embeddings",
                        "SELECT rowid FROM message_embeddings_meta ORDER BY rowid",
                    ) as rows:
                        meta_rows = tuple(row[0] for row in rows)
                    metadata = tuple(
                        seal.retain_tier_row("embeddings", "message_embeddings_meta", rowid) for rowid in meta_rows
                    )
                    assert all(image is not None for image in (*references, *metadata))
                    seal.prepare_excision_embeddings_intent()
                    prepare_embedding_excision_source_command(seal, started, args)
                child = seal.prepare_excision_embeddings_child()
                original_epoch = seal._original_input_epochs["embeddings"]
                cancelled = threading.Event()
                token = compute_cancel.set(cancelled)
                deleted = []
                commits = []
                actual_delete = _PreparedExcisionEmbeddingsChild._delete
                actual_authorize = _PreparedExcisionEmbeddingsChild.authorize_tier_sql

                def delete_then_cancel(
                    selected: _PreparedExcisionEmbeddingsChild, connection: sqlite3.Connection
                ) -> None:
                    actual_delete(selected, connection)
                    deleted.append(connection)
                    if failure == "cancellation":
                        cancelled.set()

                def refuse_native_commit(
                    selected: _PreparedExcisionEmbeddingsChild,
                    connection: sqlite3.Connection,
                    action: int,
                    first: str | None,
                    second: str | None,
                    schema: str | None,
                    trigger: str | None,
                ) -> bool:
                    if selected is child and action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT":
                        commits.append(connection)
                        if failure == "commit":
                            return False
                    return actual_authorize(selected, connection, action, first, second, schema, trigger)

                try:
                    with monkeypatch.context() as patch:
                        patch.setattr(_PreparedExcisionEmbeddingsChild, "_delete", delete_then_cancel)
                        patch.setattr(_PreparedExcisionEmbeddingsChild, "authorize_tier_sql", refuse_native_commit)
                        expected = asyncio.CancelledError if failure == "cancellation" else sqlite3.DatabaseError
                        with pytest.raises(expected):
                            child.apply()
                finally:
                    cancelled.clear()
                    compute_cancel.reset(token)
                assert deleted == [child._connection]
                assert commits == ([] if failure == "cancellation" else [child._connection])
                assert child._completed is False
                assert seal._original_input_epochs["embeddings"] == original_epoch
                owner = next(
                    owner for owner in native_sql_children(seal) if owner._connection_identity == id(child._connection)
                )
                assert owner._settled and owner.connection is None
                seal.validate_observers_current()
                with seal.original_read_snapshot():
                    with seal.original_rows(
                        "embeddings", "SELECT count(*) FROM excision_embedding_completions"
                    ) as rows:
                        assert rows.fetchone()[0] == 0
                    observer = seal.observer("embeddings")
                    assert seal._matches_retained_row(observer, selected_vector, logical_vector_key=vector_hash)
                    assert seal._matches_retained_row(observer, surviving_vector, logical_vector_key=surviving_key)
                    for image in (*references, *metadata):
                        assert image is not None and seal._matches_retained_row(observer, image)
                    with seal.original_rows("embeddings", "SELECT count(*) FROM message_embedding_refs") as rows:
                        assert rows.fetchone()[0] == 2

    await run_archive_fixture_write(tmp_path, apply_child)


@pytest.mark.parametrize("shared", [False, True])
async def test_actual_embedding_child_failed_native_close_keeps_original_custody_and_no_completion(
    tmp_path: Path,
    shared: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def apply_child() -> None:
        started, args, vector_hash = begin_embedding_excision_control(tmp_path, shared=shared)
        assert started.operation_id is not None
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            seal = PreparedIndexMutation.for_excision(
                tmp_path / "index.db",
                archive_root=tmp_path,
                input_demand=lambda _size: None,
            )
            blocked = threading.Event()
            blocked.set()
            captured = []
            try:
                seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
                with seal.original_read_snapshot():
                    surviving_key = vector_hash if shared else bytes.fromhex("cd" * 32)
                    surviving_vector = seal.retain_tier_row("embeddings", "message_embeddings", surviving_key)
                    assert surviving_vector is not None
                    seal.prepare_excision_embeddings_intent()
                    prepare_embedding_excision_source_command(seal, started, args)
                child = seal.prepare_excision_embeddings_child()
                original_epoch = seal._original_input_epochs["embeddings"]
                assert seal._scratch_directory is not None
                witness = Path(seal._scratch_directory.name) / "refs.db"
                actual_postimage = _PreparedExcisionEmbeddingsChild._verify_postimage
                with monkeypatch.context() as patch:

                    def block_exact_native_close(
                        selected: _PreparedExcisionEmbeddingsChild, connection: sqlite3.Connection
                    ) -> None:
                        actual_postimage(selected, connection)
                        owner = next(owner for owner in native_sql_children(seal) if owner.connection is connection)
                        captured.append(owner)
                        actual_close = type(connection).close

                        def close(writer: sqlite3.Connection) -> None:
                            if writer is connection and blocked.is_set():
                                raise OSError("synthetic exact paid-writer native close fault")
                            return actual_close(writer)

                        patch.setattr(type(connection), "close", close)

                    patch.setattr(_PreparedExcisionEmbeddingsChild, "_verify_postimage", block_exact_native_close)
                    with pytest.raises(NativeConnectionSettlementError) as failed:
                        child.apply()
                    assert captured == [failed.value.owner]
                    owner = captured[0]
                    assert owner.connection is child._connection and owner.close_required
                    assert owner._terminal_parent is seal and owner in native_sql_children(seal)
                    assert seal._mutation_custody is not None and witness.exists()
                    assert child._completed is False
                    # Actual commit and observer acceptance happened; neither
                    # proves the original native writer physically closed.
                    assert seal._original_input_epochs["embeddings"] == original_epoch + 1
                    seal.validate_observers_current()
                    with seal.original_read_snapshot():
                        with seal.original_rows(
                            "embeddings",
                            "SELECT operation_id, attempt_id, plan_hash, source_command_sha256 FROM excision_embedding_completions",
                        ) as rows:
                            fact = rows.fetchone()
                            assert seal._begun_excision is not None
                            assert seal._excision_source_command_sha256 is not None
                            assert fact is not None and tuple(fact) == (
                                started.operation_id,
                                seal._begun_excision[1],
                                bytes.fromhex(started.plan.plan_hash),
                                bytes.fromhex(seal._excision_source_command_sha256),
                            )
                            assert rows.fetchone() is None
                        assert seal._matches_retained_row(
                            seal.observer("embeddings"),
                            surviving_vector,
                            logical_vector_key=surviving_key,
                        )
                        with seal.original_rows("embeddings", "SELECT count(*) FROM message_embedding_refs") as rows:
                            assert rows.fetchone()[0] == 1
                        with seal.original_rows(
                            "embeddings",
                            "SELECT count(*) FROM message_embeddings WHERE vector_derivation_hash=?",
                            (vector_hash.hex(),),
                        ) as rows:
                            assert rows.fetchone()[0] == int(shared)
                    with pytest.raises(ReferenceSealError):
                        child.apply()
                    with pytest.raises(NativeConnectionSettlementError) as still_unsettled:
                        seal.close()
                    assert still_unsettled.value.owner is owner
                    assert owner.connection is child._connection and witness.exists()
                    assert seal._mutation_custody is not None
                    assert child._completed is False
                    blocked.clear()
                    seal.close()
                    assert owner.connection is None and owner._settled
                    assert not witness.exists() and seal._mutation_custody is None
                    assert child._completed is False
            finally:
                blocked.clear()
                seal.close()

    await run_archive_fixture_write(tmp_path, apply_child)


@pytest.mark.parametrize("ref_change", ["repointed_selected", "new_selected", "new_outside_after_preflight"])
async def test_actual_embedding_excision_refuses_unfrozen_refs_without_retiring_paid_outputs(
    tmp_path: Path,
    ref_change: str,
) -> None:
    def apply_child() -> None:
        started, args, vector_hash = begin_embedding_excision_control(tmp_path, shared=False)
        assert started.operation_id is not None

        def change_reference() -> None:
            with closing(
                open_isolated_write_connection(
                    tmp_path / "embeddings.db",
                    purpose="test.native-embedding-foreign-ref",
                    archive_root=tmp_path,
                )
            ) as connection:
                if ref_change == "repointed_selected":
                    connection.execute(
                        "UPDATE message_embedding_refs SET session_id=? WHERE session_id=?",
                        ("codex-session:foreign-owner", args.session_id),
                    )
                else:
                    connection.execute(
                        "INSERT INTO message_embedding_refs(message_id,session_id,origin,message_content_hash,vector_derivation_hash,embedded_at_ms) "
                        "SELECT ?,?,origin,message_content_hash,vector_derivation_hash,embedded_at_ms FROM message_embedding_refs WHERE session_id=?",
                        (
                            "synthetic-added-ref",
                            args.session_id if ref_change == "new_selected" else "codex-session:foreign-owner",
                            args.session_id,
                        ),
                    )
                connection.commit()

        if ref_change != "new_outside_after_preflight":
            change_reference()
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            with PreparedIndexMutation.for_excision(
                tmp_path / "index.db",
                archive_root=tmp_path,
                input_demand=lambda _size: None,
            ) as seal:
                seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
                with seal.original_read_snapshot():
                    selected_vector = seal.retain_tier_row("embeddings", "message_embeddings", vector_hash)
                    neighbour_key = bytes.fromhex("cd" * 32)
                    neighbour_vector = seal.retain_tier_row("embeddings", "message_embeddings", neighbour_key)
                    assert selected_vector is not None and neighbour_vector is not None
                    if ref_change != "new_outside_after_preflight":
                        with pytest.raises(ReferenceSealStaleError):
                            seal.prepare_excision_embeddings_intent()
                        assert seal._excision_embeddings_child is None
                    else:
                        seal.prepare_excision_embeddings_intent()
                if ref_change == "new_outside_after_preflight":
                    child = seal.prepare_excision_embeddings_child()
                    change_reference()
                    with pytest.raises(ReferenceSealStaleError):
                        child.apply()
                    assert child._completed is False and child._connection is None
                # Inspect the actual paid rows through the still-original
                # observer; this refuses mutation, not the neighbouring input.
                observer = seal.observer("embeddings")
                assert seal._matches_retained_row(observer, selected_vector, logical_vector_key=vector_hash)
                assert seal._matches_retained_row(observer, neighbour_vector, logical_vector_key=neighbour_key)
                with seal._owned_cursor(observer, "SELECT count(*) FROM message_embedding_refs") as rows:
                    assert rows.fetchone()[0] == (2 if ref_change == "repointed_selected" else 3)

    await run_archive_fixture_write(tmp_path, apply_child)
