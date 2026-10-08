"""Actual User allocation controls, including child-only SQLite PRNG replay."""

from __future__ import annotations

import _sqlite3
import ctypes
import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import patch

from polylogue.core.enums import AssertionKind
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import StartedBoundMutation, _authorized_removal_apply
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.source_write import PreparedParserSingletonWitness
from polylogue.storage.sqlite.archive_tiers.user_write import assertion_upsert_statement, prepare_assertion_row
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.managed_connection import sqlite_connection
from polylogue.storage.sqlite.reference_seal import KnownTierCell, KnownTierMutationPermit, PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision_execution import begin_excision_control
from tests.infra.storage_records import SessionBuilder

MAXIMUM = (1 << 63) - 1


def begin(
    root: Path, originals: tuple[tuple[int, str], ...] = (), *, delete_maximum: bool = False
) -> tuple[StartedBoundMutation, SessionExcisionArgs]:
    with write_lease("test.user-allocation-setup", archive_root=root):
        bootstrap_archive_root(root)
        builder = SessionBuilder(root / "index.db", "user-allocation").provider("codex").add_message(text="Neutral")
        builder.save()
        with closing(
            open_isolated_write_connection(root / "user.db", purpose="test.user-allocation-setup", archive_root=root)
        ) as user:
            for rowid, assertion_id in originals:
                values = prepare_assertion_row(
                    user,
                    assertion_id=assertion_id,
                    target_ref=f"session:{builder.native_session_id()}"
                    if assertion_id == "allocation-receipt" or delete_maximum and rowid == MAXIMUM
                    else "user:local",
                    kind=AssertionKind.ANNOTATION
                    if delete_maximum and rowid == MAXIMUM
                    else AssertionKind.EXCISION_RECORD,
                    value={"synthetic": "before"},
                    now_ms=1,
                )
                with connection_cursor(user, assertion_upsert_statement(("?",) * 19), (rowid, *values)):
                    pass
            user.commit()
        return begin_excision_control(root, builder.native_session_id(), reason="synthetic")


def load(seal: PreparedIndexMutation, rowid: int) -> None:
    image = seal.retain_tier_row("user", "assertions", rowid)
    assert image is not None
    seal.load_user_row(image)


def stage_receipt(seal: PreparedIndexMutation, *, session_id: str, assertion_id: str = "allocation-receipt") -> int:
    values = prepare_assertion_row(
        seal.observer("user"),
        assertion_id=assertion_id,
        target_ref=f"session:{session_id}",
        kind=AssertionKind.EXCISION_RECORD,
        value={"synthetic": "after"},
        now_ms=2,
    )
    columns, _keys = seal._known_tier_table_shape("user", "assertions")
    cells = {column: seal.retain_literal_scalar(value) for column, value in zip(columns, values, strict=True)}
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
    with seal.user_rows("SELECT rowid FROM assertions WHERE assertion_id=?", (assertion_id,)) as rows:
        return int(rows.fetchone()[0])


def apply(
    seal: PreparedIndexMutation,
    permit: KnownTierMutationPermit,
    started: StartedBoundMutation,
    args: SessionExcisionArgs,
    root: Path,
) -> None:
    with _authorized_removal_apply(started.plan, root, SessionExcisionActuator(), args):
        with permit.hold_authority(), permit.mutation_connection() as user:
            with connection_cursor(user, "BEGIN IMMEDIATE"):
                pass
            permit.apply_user_statements(user)
            permit.allow_commit(user)
            user.commit()
            seal.accept_known_tier_commit(permit.committed())


def run(root: Path) -> dict[str, object]:
    # TESTCTRL 5/6 replay the actual extension's loaded SQLite PRNG. They are
    # process-global, so this function is invoked only by an isolated child.
    native = ctypes.CDLL(_sqlite3.__file__)
    native.sqlite3_libversion.restype = ctypes.c_char_p
    assert native.sqlite3_libversion().decode("ascii") == sqlite3.sqlite_version
    control = native.sqlite3_test_control
    control.argtypes = [ctypes.c_int]
    control.restype = ctypes.c_int
    saved = False
    try:
        with sqlite_connection(":memory:") as probe:
            with connection_cursor(probe, "CREATE TABLE allocation_probe(value INTEGER)"):
                pass
            with connection_cursor(probe, "INSERT INTO allocation_probe(rowid,value) VALUES(?,0)", (MAXIMUM,)):
                pass
            assert control(5) == 0
            saved = True
            with connection_cursor(probe, "INSERT INTO allocation_probe VALUES(1)") as cursor:
                occupied = cursor.lastrowid
            with connection_cursor(probe, "DELETE FROM allocation_probe WHERE rowid=?", (occupied,)):
                pass
            assert control(6) == 0
            with connection_cursor(probe, "INSERT INTO allocation_probe VALUES(1)") as cursor:
                assert cursor.lastrowid == occupied, "native allocator PRNG replay is unsupported"
        assert isinstance(occupied, int) and occupied != MAXIMUM
        started, args = begin(root, ((MAXIMUM, "maximum-original"), (occupied, "occupied-original")))
        assert started.operation_id is not None
        with PreparedIndexMutation(root / "index.db", archive_root=root) as seal:
            seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
            with seal.original_read_snapshot(), seal.user_producer():
                load(seal, MAXIMUM)
                prior = seal.retain_literal_scalar("retained before collision")
                attempts = 0
                original_attempt = seal._source_statement_attempt

                @contextmanager
                def replay_first(
                    sql: str,
                    parameters: tuple[object, ...],
                    *,
                    table: str,
                    columns: tuple[str, ...],
                    keys: tuple[int, ...],
                    prepared_cells: dict[str, KnownTierCell] | None,
                    allocation_parameter: int | None,
                    generated_primary_key: bool,
                    parser_singleton_witness: PreparedParserSingletonWitness | None,
                    binding_cells: tuple[KnownTierCell, ...],
                ) -> Iterator[sqlite3.Cursor]:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        assert control(6) == 0
                    with original_attempt(
                        sql,
                        parameters,
                        table=table,
                        columns=columns,
                        keys=keys,
                        prepared_cells=prepared_cells,
                        allocation_parameter=allocation_parameter,
                        generated_primary_key=generated_primary_key,
                        parser_singleton_witness=parser_singleton_witness,
                        binding_cells=binding_cells,
                    ) as cursor:
                        yield cursor

                with patch.object(seal, "_source_statement_attempt", replay_first):
                    allocated = stage_receipt(seal, session_id=args.session_id)
                assert attempts >= 2 and allocated != occupied
                assert seal._literal_scalar_equal(prior, "retained before collision")
                with seal.user_rows("SELECT rowid FROM assertions WHERE assertion_id='occupied-original'") as rows:
                    assert rows.fetchone()[0] == occupied
            apply(seal, seal.prepare_user_mutation(), started, args, root)
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "user", "SELECT assertion_id,rowid FROM assertions ORDER BY assertion_id"
                ) as rows:
                    actual = dict(rows)
                assert actual == {
                    "maximum-original": MAXIMUM,
                    "occupied-original": occupied,
                    "allocation-receipt": allocated,
                }
        return {"sqlite": sqlite3.sqlite_version, "attempts": attempts, "occupied": occupied, "allocated": allocated}
    finally:
        if saved:
            assert control(6) == 0


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]))))
