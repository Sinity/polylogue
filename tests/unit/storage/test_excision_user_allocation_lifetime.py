"""User native allocation and failure settlement under actual begun custody."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.operations.mutation_actuators import SessionExcisionActuator
from polylogue.operations.mutation_transaction import _authorized_removal_apply
from polylogue.storage.io_phase_metrics import _MeasuredCursor, connection_cursor, live_connection_cursors
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    native_sql_children,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError
from polylogue.storage.sqlite.write_lease import current_sql_custody, write_lease
from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor, control_archive_connections
from tests.infra.user_allocation_probe import MAXIMUM, apply, begin, load, stage_receipt


def test_user_original_only_random_collision_retries_actual_loaded_allocator(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.infra.user_allocation_probe", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate()
        assert process.returncode == 0, stderr
        result = json.loads(stdout)
        assert result["sqlite"] == sqlite3.sqlite_version
        assert result["attempts"] >= 2
        assert result["allocated"] != result["occupied"]
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


@pytest.mark.parametrize("delete_maximum", [False, True])
@pytest.mark.parametrize("existing_receipt", [False, True])
def test_user_receipt_allocation_obeys_post_delete_maximum_and_upsert_identity(
    tmp_path: Path,
    delete_maximum: bool,
    existing_receipt: bool,
) -> None:
    originals = (
        (MAXIMUM, "maximum-original"),
        (17, "allocation-receipt" if existing_receipt else "surviving-original"),
    )
    started, args = begin(tmp_path, originals, delete_maximum=delete_maximum)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with seal.original_read_snapshot(), seal.user_producer():
            load(seal, MAXIMUM)
            load(seal, 17)
            if delete_maximum:
                key = seal.retain_literal_scalar("maximum-original")
                expression, parameters = seal.source_literal_expression(key)
                with seal.user_statement(
                    f"DELETE FROM assertions WHERE assertion_id={expression}",
                    parameters,
                    table="assertions",
                    writable_targets=(("assertions", (key,)),),
                ):
                    pass
            allocated = stage_receipt(seal, session_id=args.session_id)
            if existing_receipt:
                assert allocated == 17
            elif delete_maximum:
                assert allocated == 18
            else:
                assert 0 < allocated < MAXIMUM and allocated != 17
            # A fully executed DML statement with no matching row leaves no
            # assertion transition and must not advance the canonical frame.
            with seal.user_rows("SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as rows:
                before = rows.fetchone()[0]
            absent = seal.retain_literal_scalar("missing-original")
            expression, parameters = seal.source_literal_expression(absent)
            with seal.user_statement(
                f"DELETE FROM assertions WHERE assertion_id={expression}",
                parameters,
                table="assertions",
                writable_targets=(),
            ):
                pass
            with seal.user_rows("SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as rows:
                assert rows.fetchone()[0] == before
        apply(seal, seal.prepare_user_mutation(), started, args, tmp_path)
        with seal.original_read_snapshot():
            with seal.original_rows(
                "user",
                "SELECT rowid,created_at_ms,updated_at_ms FROM assertions WHERE assertion_id='allocation-receipt'",
            ) as rows:
                assert tuple(rows.fetchone()) == (allocated, 1 if existing_receipt else 2, 2)
            with seal.original_rows(
                "user", "SELECT count(*) FROM assertions WHERE assertion_id='maximum-original'"
            ) as rows:
                assert rows.fetchone()[0] == int(not delete_maximum)
            with seal.original_rows("user", "SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as rows:
                assert rows.fetchone()[0] == 3 + int(delete_maximum)


def test_changed_user_refuses_adoption_before_original_effects(tmp_path: Path) -> None:
    started, args = begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with seal.original_read_snapshot(), seal.user_producer():
            stage_receipt(seal, session_id=args.session_id)
        with write_lease("test.user-allocation-stale", archive_root=tmp_path):
            with closing(
                open_isolated_write_connection(
                    tmp_path / "user.db", purpose="test.user-allocation-stale", archive_root=tmp_path
                )
            ) as user:
                upsert_assertion(
                    user,
                    assertion_id="concurrent-original",
                    target_ref="user:local",
                    kind=AssertionKind.EXCISION_RECORD,
                    value={"synthetic": "new"},
                    now_ms=3,
                )
                user.commit()
        with pytest.raises(ReferenceSealStaleError):
            seal.prepare_user_mutation()
        assert seal._pending_tier_permits == {}
    with closing(sqlite3.connect(tmp_path / "user.db")) as user:
        with closing(user.execute("SELECT assertion_id FROM assertions")) as rows:
            assert [row[0] for row in rows] == ["concurrent-original"]


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("fault", ["cursor", "rollback"])
def test_user_failure_retains_physical_custody_until_original_native_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
    fault: str,
) -> None:
    started, args = begin(tmp_path)
    assert started.operation_id is not None
    seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
    primary = asyncio.CancelledError("synthetic User cancellation") if cancelled else OSError("synthetic User failure")
    blocked: list[ControlledCursor] = []
    writer: sqlite3.Connection | None = None
    custody = None

    class FailingCursor(ControlledCursor, _MeasuredCursor):
        def execute(self, sql: str, parameters: Any = (), /) -> FailingCursor:
            super().execute(sql, parameters)
            if sql.lstrip().startswith("INSERT INTO assertions"):
                self.allow_cleanup.clear()
                blocked.append(self)
                raise primary
            return self

    try:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with seal.original_read_snapshot(), seal.user_producer():
            stage_receipt(seal, session_id=args.session_id)
        permit = seal.prepare_user_mutation()
        if fault == "rollback":
            control_archive_connections(monkeypatch, tmp_path / "user.db")
        with pytest.raises(NativeConnectionSettlementError) as failure:
            with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
                custody = current_sql_custody()
                with permit.hold_authority(), permit.mutation_connection() as user:
                    writer = user
                    with connection_cursor(user, "BEGIN IMMEDIATE"):
                        pass
                    if fault == "cursor":
                        original_cursor = user.cursor
                        monkeypatch.setattr(user, "cursor", lambda: original_cursor(factory=FailingCursor))
                        permit.apply_user_statements(user)
                    else:
                        permit.apply_user_statements(user)
                        assert isinstance(user, ControlledConnection)
                        user.rollback_failure = OSError("synthetic actual rollback fault")
                        user.close_failure = OSError("synthetic actual close fault")
                        raise primary
        pending: list[BaseException] = [failure.value]
        seen: set[int] = set()
        retained_primary = False
        while pending:
            error = pending.pop()
            if error is primary:
                retained_primary = True
            if id(error) in seen:
                continue
            seen.add(id(error))
            if isinstance(error, NativeConnectionSettlementError):
                pending.append(error.failure)
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)
            for linked in (error.__cause__, error.__context__):
                if linked is not None:
                    pending.append(linked)
        assert retained_primary
        assert writer is not None and custody is not None
        owner = next(child for child in native_sql_children(seal) if child.connection is writer)
        assert owner.close_required and not owner._settled
        assert seal._mutation_custody is custody
        assert custody._fd >= 0
        if fault == "cursor":
            assert blocked and blocked[0] in live_connection_cursors(writer)
        assert seal._scratch_directory is not None
        artifact = Path(seal._scratch_directory.name) / "refs.db"
        assert artifact.exists()
        with pytest.raises(NativeConnectionSettlementError):
            seal.close()
        assert artifact.exists() and custody._fd >= 0
        for cursor in blocked:
            cursor.allow_cleanup.set()
        if isinstance(writer, ControlledConnection):
            writer.rollback_failure = None
            writer.close_failure = None
        seal.close()
        assert not artifact.exists() and custody._fd == -1
        with closing(sqlite3.connect(tmp_path / "user.db")) as user:
            with closing(user.execute("SELECT count(*) FROM assertions")) as rows:
                assert rows.fetchone()[0] == 0
            with closing(user.execute("SELECT epoch FROM query_unit_frame_state WHERE singleton=1")) as rows:
                assert rows.fetchone()[0] == 0
    finally:
        for cursor in blocked:
            cursor.allow_cleanup.set()
        if isinstance(writer, ControlledConnection):
            writer.rollback_failure = None
            writer.close_failure = None
        seal.close()
