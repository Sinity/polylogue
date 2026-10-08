"""Canonical User receipt allocation retains SQLite physical identity."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.operations.mutation_actuators import SessionExcisionActuator
from polylogue.operations.mutation_transaction import _authorized_removal_apply
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.user_write import (
    assertion_upsert_statement,
    prepare_assertion_row,
    upsert_assertion,
)
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation, ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision_execution import begin_excision_control
from tests.infra.storage_records import SessionBuilder


def test_canonical_receipt_allocation_at_max_rowid_preserves_upsert_identity(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with write_lease("test.user-receipt-native-allocation", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(
            open_isolated_write_connection(
                tmp_path / "user.db", purpose="test.user-receipt-native-allocation", archive_root=tmp_path
            )
        ) as user:
            original = prepare_assertion_row(
                user,
                assertion_id="original-max",
                target_ref="user:local",
                kind=AssertionKind.EXCISION_RECORD,
                value={"synthetic": "original"},
                now_ms=1,
            )
            with connection_cursor(user, assertion_upsert_statement(("?",) * 19), ((1 << 63) - 1, *original)):
                pass
            user.commit()
            upsert_assertion(
                user,
                assertion_id="receipt-new",
                target_ref="user:local",
                kind=AssertionKind.EXCISION_RECORD,
                value={"synthetic": "first"},
                now_ms=2,
            )
            with connection_cursor(
                user, "SELECT rowid,created_at_ms FROM assertions WHERE assertion_id=?", ("receipt-new",)
            ) as cursor:
                inserted = cursor.fetchone()
            assert inserted is not None and 0 < inserted[0] < (1 << 63) - 1
            assert inserted[1] == 2
            upsert_assertion(
                user,
                assertion_id="receipt-new",
                target_ref="user:local",
                kind=AssertionKind.EXCISION_RECORD,
                value={"synthetic": "updated"},
                now_ms=3,
            )
            with connection_cursor(
                user, "SELECT rowid,created_at_ms,updated_at_ms FROM assertions WHERE assertion_id=?", ("receipt-new",)
            ) as cursor:
                updated = cursor.fetchone()
            assert updated is not None and tuple(updated) == (inserted[0], 2, 3)
            with connection_cursor(user, "SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as cursor:
                assert cursor.fetchone()[0] == 3
            with connection_cursor(
                user, "SELECT rowid,value_json FROM assertions WHERE assertion_id=?", ("original-max",)
            ) as cursor:
                surviving = cursor.fetchone()
            assert surviving is not None and tuple(surviving) == ((1 << 63) - 1, '{"synthetic":"original"}')


@pytest.mark.parametrize("existing_receipt", [False, True])
@pytest.mark.parametrize("maximum_original", [False, True])
@pytest.mark.parametrize("defer_insert_after", [False, True])
def test_begun_user_native_tape_replays_exact_allocation_and_frame(
    tmp_path: Path,
    existing_receipt: bool,
    maximum_original: bool,
    defer_insert_after: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with write_lease("test.user-tape-setup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        builder = SessionBuilder(tmp_path / "index.db", "user-tape").provider("codex").add_message(text="Neutral")
        builder.save()
        session_id = builder.native_session_id()
        if maximum_original:
            with closing(
                open_isolated_write_connection(
                    tmp_path / "user.db",
                    purpose="test.user-tape-maximum",
                    archive_root=tmp_path,
                )
            ) as user:
                maximum = prepare_assertion_row(
                    user,
                    assertion_id="unrelated-maximum",
                    target_ref="user:local",
                    kind=AssertionKind.EXCISION_RECORD,
                    value={"synthetic": "survives"},
                    now_ms=1,
                )
                with connection_cursor(user, assertion_upsert_statement(("?",) * 19), ((1 << 63) - 1, *maximum)):
                    pass
                user.commit()
        if existing_receipt:
            with closing(
                open_isolated_write_connection(
                    tmp_path / "user.db",
                    purpose="test.user-tape-original",
                    archive_root=tmp_path,
                )
            ) as user:
                upsert_assertion(
                    user,
                    assertion_id="user-tape-receipt",
                    target_ref=f"session:{session_id}",
                    kind=AssertionKind.EXCISION_RECORD,
                    value={"synthetic": "before"},
                    now_ms=1,
                )
                user.commit()
    started, args = begin_excision_control(tmp_path, session_id, reason="synthetic")
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (session_id,))
        pending_stage: tuple[str, str, str, int | None, int | None] | None = None
        pending_live: tuple[sqlite3.Connection, str, str, str, int | None, int | None] | None = None
        capture = PreparedIndexMutation._capture_source_transition
        consume = KnownTierMutationPermit._consume_native_effect

        def reordered_capture(
            owner: PreparedIndexMutation,
            table: str,
            phase: str,
            operation: str,
            old_rowid: int | None,
            new_rowid: int | None,
        ) -> int:
            nonlocal pending_stage
            if table == "assertions" and phase == "AFTER" and operation == "INSERT":
                assert pending_stage is None
                pending_stage = (table, phase, operation, old_rowid, new_rowid)
                return 1
            result = capture(owner, table, phase, operation, old_rowid, new_rowid)
            if table == "query_unit_frame_state" and phase == "AFTER" and pending_stage is not None:
                deferred = pending_stage
                pending_stage = None
                capture(owner, *deferred)
            return result

        def reordered_consume(
            permit: KnownTierMutationPermit,
            connection: sqlite3.Connection,
            table: str,
            phase: str,
            operation: str,
            old_rowid: int | None,
            new_rowid: int | None,
        ) -> int:
            nonlocal pending_live
            if permit.tier == "user" and table == "assertions" and phase == "AFTER" and operation == "INSERT":
                assert pending_live is None
                pending_live = (connection, table, phase, operation, old_rowid, new_rowid)
                return 1
            result = consume(permit, connection, table, phase, operation, old_rowid, new_rowid)
            if table == "query_unit_frame_state" and phase == "AFTER" and pending_live is not None:
                deferred = pending_live
                pending_live = None
                consume(permit, *deferred)
            return result

        if defer_insert_after:
            # Change only delivery order of actual native callbacks. Canonical
            # SQL/triggers, real images, allocation, and guards remain intact.
            monkeypatch.setattr(PreparedIndexMutation, "_capture_source_transition", reordered_capture)
            monkeypatch.setattr(KnownTierMutationPermit, "_consume_native_effect", reordered_consume)
        with seal.original_read_snapshot(), seal.user_producer():
            with seal.original_rows(
                "user", "SELECT rowid FROM assertions WHERE assertion_id=?", ("user-tape-receipt",)
            ) as cursor:
                selected = cursor.fetchone()
            old_rowid = None if selected is None else selected[0]
            if old_rowid is not None:
                old = seal.retain_tier_row("user", "assertions", old_rowid)
                assert old is not None
                seal.load_user_row(old)
            canonical = prepare_assertion_row(
                seal.observer("user"),
                assertion_id="user-tape-receipt",
                target_ref=f"session:{session_id}",
                kind=AssertionKind.EXCISION_RECORD,
                value={"synthetic": "after"},
                now_ms=2,
            )
            columns, _keys = seal._known_tier_table_shape("user", "assertions")
            cells = {
                column: seal.retain_literal_scalar(value) for column, value in zip(columns, canonical, strict=True)
            }
            expressions = ["?"]
            parameters: tuple[object, ...] = (None,)
            for column in columns:
                expression, operands = seal.source_literal_expression(cells[column])
                expressions.append(expression)
                parameters += operands
            with seal.user_statement(
                assertion_upsert_statement(tuple(expressions)),
                parameters,
                table="assertions",
                writable_targets=(("assertions", (cells["assertion_id"],)),),
                prepared_cells=cells,
                allocation_parameter=0,
            ):
                pass
            with seal.user_rows(
                "SELECT rowid,created_at_ms FROM assertions WHERE assertion_id=?", ("user-tape-receipt",)
            ) as cursor:
                staged = tuple(cursor.fetchone())
            with seal.user_rows("SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as cursor:
                staged_epoch = cursor.fetchone()[0]
            if old_rowid is not None:
                assert staged == (old_rowid, 1)
            else:
                assert type(staged[0]) is int and staged[1] == 2
            with seal.user_rows(
                "SELECT table_name,canonical_trigger FROM temp.known_tier_effects WHERE tier='user' ORDER BY ordinal"
            ) as cursor:
                assert [tuple(row) for row in cursor] == [
                    ("assertions", None),
                    (
                        "query_unit_frame_state",
                        "query_unit_frame_assertions_update"
                        if existing_receipt
                        else "query_unit_frame_assertions_insert",
                    ),
                ]
            with pytest.raises(ReferenceSealError):
                with seal.user_statement(
                    "UPDATE query_unit_frame_state SET epoch=epoch+1",
                    table="query_unit_frame_state",
                    writable_targets=(),
                ):
                    pass
        permit = seal.prepare_user_mutation()
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            with permit.hold_authority(), permit.mutation_connection() as user:
                with connection_cursor(user, "BEGIN IMMEDIATE"):
                    pass
                permit.apply_user_statements(user)
                permit.allow_commit(user)
                user.commit()
                seal.accept_known_tier_commit(permit.committed())
                with connection_cursor(
                    user, "SELECT rowid,created_at_ms FROM assertions WHERE assertion_id=?", ("user-tape-receipt",)
                ) as cursor:
                    assert tuple(cursor.fetchone()) == staged
                with connection_cursor(user, "SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as cursor:
                    assert cursor.fetchone()[0] == staged_epoch
                if maximum_original:
                    with connection_cursor(
                        user, "SELECT rowid,value_json FROM assertions WHERE assertion_id='unrelated-maximum'"
                    ) as cursor:
                        assert tuple(cursor.fetchone()) == ((1 << 63) - 1, '{"synthetic":"survives"}')
        assert seal._pending_tier_permits == {}
        assert pending_stage is None and pending_live is None


@pytest.mark.parametrize("bad_attempt", [False, True])
def test_canonical_excision_user_cleanup_uses_original_targets_and_attempt_receipt(
    tmp_path: Path,
    bad_attempt: bool,
) -> None:
    from polylogue.security.excision import ExcisionReceipt, _stage_excision_user_receipt, excision_target_from_replay

    with write_lease("test.user-canonical-cleanup-setup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        builder = (
            SessionBuilder(tmp_path / "index.db", "user-canonical-cleanup")
            .provider("codex")
            .add_message(text="Neutral")
        )
        builder.save()
        session_id = builder.native_session_id()
        with closing(
            open_isolated_write_connection(
                tmp_path / "user.db",
                purpose="test.user-canonical-cleanup-setup",
                archive_root=tmp_path,
            )
        ) as user:
            for assertion_id, target_ref, kind in (
                ("owned-content", f"session:{session_id}", AssertionKind.ANNOTATION),
                ("marker-owned", f"session:{session_id}", AssertionKind.ANNOTATION),
                ("retained-lifecycle", f"session:{session_id}", AssertionKind.EXCISION_RECORD),
                ("unrelated-content", "user:local", AssertionKind.ANNOTATION),
            ):
                upsert_assertion(
                    user,
                    assertion_id=assertion_id,
                    target_ref=target_ref,
                    kind=kind,
                    value={"synthetic": "before"},
                    body_text="Neutral original content",
                    now_ms=1,
                )
            user.commit()
    started, args = begin_excision_control(tmp_path, session_id, reason="synthetic")
    assert started.operation_id is not None
    targets = started.plan.context["targets"]
    assert isinstance(targets, list) and len(targets) == 1
    target = excision_target_from_replay(targets[0])
    original = ExcisionReceipt(
        session_id=session_id,
        found=True,
        reason="synthetic",
        actor="user:local",
        excised_at_ms=2,
        removed_blob_hashes=("a" * 64,),
        counts={"index_sessions": 1},
    )
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        attempt_id = seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (session_id,))
        if bad_attempt:
            with seal.original_read_snapshot(), seal.user_producer():
                with pytest.raises(ReferenceSealError):
                    _stage_excision_user_receipt(
                        seal,
                        target,
                        original,
                        operation_id=started.operation_id,
                        attempt_id="attempt:foreign",
                        plan_hash=started.plan.plan_hash,
                    )
                with seal.user_rows("SELECT count(*) FROM assertions") as rows:
                    assert rows.fetchone()[0] == 0
            with closing(sqlite3.connect(tmp_path / "user.db")) as user:
                with closing(user.execute("SELECT count(*) FROM assertions")) as rows:
                    assert rows.fetchone()[0] == 4
            return
        with seal.original_read_snapshot(), seal.user_producer():
            staged = _stage_excision_user_receipt(
                seal,
                target,
                original,
                operation_id=started.operation_id,
                attempt_id=attempt_id,
                plan_hash=started.plan.plan_hash,
            )
        assert staged.counts["user_assertions_removed"] == 1
        assert staged.counts["user_assertions_tombstoned"] == 1
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            permit = seal.prepare_user_mutation()
            with permit.hold_authority(), permit.mutation_connection() as user:
                with connection_cursor(user, "BEGIN IMMEDIATE"):
                    pass
                permit.apply_user_statements(user)
                permit.allow_commit(user)
                user.commit()
                seal.accept_known_tier_commit(permit.committed())
        with seal.original_read_snapshot():
            with seal.original_rows(
                "user",
                "SELECT assertion_id,target_ref,value_json,body_text,status FROM assertions ORDER BY assertion_id",
            ) as rows:
                actual = {row[0]: tuple(row[1:]) for row in rows}
            assert "owned-content" not in actual
            assert actual["marker-owned"] == ("assertion:marker-owned", "{}", None, "deleted")
            assert actual["retained-lifecycle"][0] == f"session:{session_id}"
            assert actual["unrelated-content"][0:3] == (
                "user:local",
                '{"synthetic":"before"}',
                "Neutral original content",
            )
            assert staged.receipt_assertion_id in actual
            receipt_value = json.loads(actual[staged.receipt_assertion_id][1])
            assert receipt_value["attempt_id"] == attempt_id
            assert receipt_value["operation_id"] == started.operation_id
            assert receipt_value["plan_hash"] == started.plan.plan_hash
            assert receipt_value["counts"] == staged.counts
            with seal.original_rows("user", "SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as rows:
                assert rows.fetchone()[0] == 7
