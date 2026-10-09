"""Canonical Source statements retain allocation, order and their original owner."""

import asyncio
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Never

import pytest

from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
from polylogue.storage.sqlite.literal_cells import SQLiteLiteralCell
from polylogue.storage.sqlite.reference_seal import (
    KnownTierCell,
    KnownTierRowImage,
    PreparedIndexMutation,
    ReferenceSealError,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.fixture
def source_statement_root(tmp_path: Path) -> Iterator[Path]:
    with write_lease("test.source-statement-schedule", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        yield tmp_path


def _seed_original_raw(root: Path, *, rowid: int = 1) -> None:
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
        with closing(
            source.execute(
                "INSERT INTO raw_sessions(rowid,raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                "VALUES(?, 'original-raw', 'unknown-export', 'synthetic/source', ?, 1, 1)",
                (rowid, b"o" * 32),
            )
        ):
            pass
        source.commit()


def _stage_raw(seal: PreparedIndexMutation, raw_id: str, *, allocation: bool = True) -> int:
    values = tuple(
        seal.retain_literal_scalar(value)
        for value in (
            raw_id,
            "unknown-export",
            "synthetic/" + raw_id,
            b"n" * 32,
        )
    )
    expressions = []
    parameters: tuple[object, ...] = (None,)
    for cell in values:
        expression, operands = seal.source_literal_expression(cell)
        expressions.append(expression)
        parameters += operands
    sql = (
        "INSERT INTO raw_sessions(rowid,raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
        f"VALUES(?,{','.join(expressions)},1,1) ON CONFLICT(raw_id) DO NOTHING"
    )
    with seal.source_statement(
        sql,
        parameters,
        table="raw_sessions",
        writable_targets=(("raw_sessions", (values[0],)),),
        # raw_sessions feeds the frontier journal, which keys its insert on
        # the exact prepared primary key cell.
        prepared_cells=dict(zip(("raw_id", "origin", "source_path", "blob_hash"), values, strict=True)),
        allocation_parameter=0 if allocation else None,
    ) as cursor:
        rowid = cursor.lastrowid
        assert isinstance(rowid, int)
        return rowid


def _publish_source(seal: PreparedIndexMutation) -> None:
    permit = seal.prepare_source_mutation()
    with permit.hold_authority(), permit.mutation_connection() as source:
        with closing(source.execute("BEGIN IMMEDIATE")):
            pass
        permit.apply_source_statements(source)
        permit.allow_commit(source)
        source.commit()
        seal.accept_known_tier_commit(permit.committed())


def test_source_only_ordered_allocations_publish_the_actual_staged_rowids(source_statement_root: Path) -> None:
    root = source_statement_root
    _seed_original_raw(root, rowid=41)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            first = _stage_raw(seal, "first-new")
            second = _stage_raw(seal, "second-new")
            assert (first, second) == (42, 43)
        _publish_source(seal)
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT rowid,raw_id FROM raw_sessions ORDER BY rowid") as rows:
                assert [tuple(row) for row in rows] == [(41, "original-raw"), (42, "first-new"), (43, "second-new")]


def test_no_effect_source_statement_still_requires_schedule_consumption_and_refuses_reapply(
    source_statement_root: Path,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None
            seal.load_source_row(image)
            _stage_raw(seal, "original-raw")
        permit = seal.prepare_source_mutation()
        with permit.hold_authority(), permit.mutation_connection() as source:
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            with pytest.raises(ReferenceSealError):
                permit.allow_commit(source)
            permit.apply_source_statements(source)
            with pytest.raises(ReferenceSealError):
                permit.apply_source_statements(source)
            permit.allow_commit(source)
            source.commit()
            seal.accept_known_tier_commit(permit.committed())


def test_parser_census_reuses_selected_inputs_before_retaining_cells(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.revision_governance import _load_parser_census_source_inputs

    root = source_statement_root
    _seed_original_raw(root)
    charges: list[int] = []
    retained: list[tuple[str, str, int | bytes]] = []
    with PreparedIndexMutation(root / "index.db", archive_root=root) as seal:
        retain = seal.retain_tier_row

        def counted_retain(tier: str, table: str, rowid: int | bytes) -> KnownTierRowImage | None:
            retained.append((tier, table, rowid))
            return retain(tier, table, rowid)

        monkeypatch.setattr(seal, "retain_tier_row", counted_retain)
        with seal.original_read_snapshot(input_demand=charges.append), seal.source_producer():
            assert not seal.source_row_is_loaded("raw_sessions", 1)
            _load_parser_census_source_inputs(seal, "original-raw")
            first_charges = tuple(charges)
            assert first_charges and all(charge > 0 for charge in first_charges)
            first_retained = tuple(retained)
            for _ in range(99):
                _load_parser_census_source_inputs(seal, "original-raw")
            assert tuple(retained) == first_retained
            assert retained.count(("source", "raw_sessions", 1)) == 1
            assert tuple(charges) == first_charges
            assert seal.source_row_is_loaded("raw_sessions", 1)
            with seal.source_rows("SELECT raw_id,blob_size FROM raw_sessions WHERE rowid=1") as rows:
                row = rows.fetchone()
                assert row is not None and tuple(row) == ("original-raw", 1)

            key = seal.retain_literal_scalar("original-raw")
            with seal.source_statement(
                "UPDATE raw_sessions SET validation_status='failed' WHERE rowid=1",
                table="raw_sessions",
                writable_targets=(("raw_sessions", (key,)),),
            ):
                pass
            retained_after_update = tuple(retained)
            _load_parser_census_source_inputs(seal, "original-raw")
            assert tuple(retained) == retained_after_update
            with seal.source_rows("SELECT validation_status FROM raw_sessions WHERE rowid=1") as rows:
                row = rows.fetchone()
                assert row is not None and tuple(row) == ("failed",)
            new_rowid = _stage_raw(seal, "new-selected")
            assert seal.source_row_is_loaded("raw_sessions", new_rowid)
            with seal.source_statement(
                "DELETE FROM raw_sessions WHERE rowid=1",
                table="raw_sessions",
                writable_targets=(("raw_sessions", (key,)),),
            ):
                pass
            retained_after_delete = tuple(retained)
            _load_parser_census_source_inputs(seal, "original-raw")
            assert tuple(retained) == retained_after_delete
            assert seal.source_row_is_loaded("raw_sessions", 1)
            with seal.source_rows("SELECT 1 FROM raw_sessions WHERE rowid=1") as rows:
                assert rows.fetchone() is None


@pytest.mark.parametrize("load_state", [0, 1])
def test_source_input_reuse_refuses_unfinished_loading(source_statement_root: Path, load_state: int) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None
            assert seal.load_source_row(image)
            with seal._owned_cursor(
                seal._scratch,
                "UPDATE temp.polylogue_source_stage_rows SET load_state=? "
                "WHERE table_name='raw_sessions' AND physical_rowid=1",
                (load_state,),
            ):
                pass
            with pytest.raises(ReferenceSealError, match="completed original load"):
                seal.source_row_is_loaded("raw_sessions", 1)
            with seal._owned_cursor(
                seal._scratch,
                "UPDATE temp.polylogue_source_stage_rows SET load_state=2 "
                "WHERE table_name='raw_sessions' AND physical_rowid=1",
            ):
                pass


def test_source_input_reuse_keeps_failed_first_load_and_cancellation_visible(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.storage.sqlite.archive_tiers.revision_governance import _load_parser_census_source_inputs

    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            retain = seal.retain_tier_row

            def failed_retain(tier: str, table: str, rowid: int | bytes) -> Never:
                raise RuntimeError("synthetic first retention failure")

            monkeypatch.setattr(seal, "retain_tier_row", failed_retain)
            with pytest.raises(RuntimeError, match="first retention failure"):
                _load_parser_census_source_inputs(seal, "original-raw")
            assert not seal.source_row_is_loaded("raw_sessions", 1)
            monkeypatch.setattr(seal, "retain_tier_row", retain)
            _load_parser_census_source_inputs(seal, "original-raw")
            assert seal.source_row_is_loaded("raw_sessions", 1)
            cancelled = threading.Event()
            token = compute_cancel.set(cancelled)
            try:
                cancelled.set()
                with pytest.raises(asyncio.CancelledError):
                    seal.source_row_is_loaded("raw_sessions", 1)
            finally:
                cancelled.clear()
                compute_cancel.reset(token)


def test_source_rollback_publishes_no_captured_row_or_commit_receipt(source_statement_root: Path) -> None:
    root = source_statement_root
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "rolled-back")
        permit = seal.prepare_source_mutation()
        with permit.hold_authority(), permit.mutation_connection() as source:
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            permit.apply_source_statements(source)
            permit.allow_commit(source)
            source.rollback()
            with pytest.raises(ReferenceSealError):
                permit.committed()
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT 1 FROM raw_sessions WHERE raw_id='rolled-back'") as rows:
                assert rows.fetchone() is None
        assert seal._pending_tier_receipts == {}


def test_source_acceptance_settles_after_commit_even_when_cancellation_is_pending(
    source_statement_root: Path,
) -> None:
    import threading

    from polylogue.core.compute_cancel import compute_cancel

    root = source_statement_root
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "committed-before-cancel")
        permit = seal.prepare_source_mutation()
        with permit.hold_authority(), permit.mutation_connection() as source:
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            permit.apply_source_statements(source)
            permit.allow_commit(source)
            source.commit()
            cancelled = threading.Event()
            token = compute_cancel.set(cancelled)
            try:
                cancelled.set()
                seal.accept_known_tier_commit(permit.committed())
            finally:
                cancelled.clear()
                compute_cancel.reset(token)
        with seal.original_read_snapshot():
            with seal.original_rows(
                "source", "SELECT raw_id FROM raw_sessions WHERE raw_id=?", ("committed-before-cancel",)
            ) as rows:
                assert tuple(rows.fetchone()) == ("committed-before-cancel",)
        seal.validate_observers_current()


@pytest.mark.parametrize("boundary", ["before_begin", "after_commit"])
def test_source_capture_refuses_foreign_commit_on_either_side_of_publication(
    source_statement_root: Path,
    boundary: str,
) -> None:
    import sqlite3

    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "captured-by-original")
        permit = seal.prepare_source_mutation()
        original_version = seal.observer_version("source")
        with permit.hold_authority(), permit.mutation_connection() as source:
            if boundary == "before_begin":
                with closing(sqlite3.connect(root / "source.db")) as foreign:
                    with closing(
                        foreign.execute("UPDATE raw_sessions SET file_mtime_ms=2 WHERE raw_id='original-raw'")
                    ):
                        pass
                    foreign.commit()
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            if boundary == "before_begin":
                with pytest.raises(ReferenceSealStaleError):
                    permit.apply_source_statements(source)
                source.rollback()
            else:
                permit.apply_source_statements(source)
                permit.allow_commit(source)
                source.commit()
                receipt = permit.committed()
                with closing(sqlite3.connect(root / "source.db")) as foreign:
                    with closing(
                        foreign.execute("UPDATE raw_sessions SET file_mtime_ms=3 WHERE raw_id='original-raw'")
                    ):
                        pass
                    foreign.commit()
                with pytest.raises(ReferenceSealStaleError):
                    seal.accept_known_tier_commit(receipt)
        assert seal.observer_version("source") == original_version
        with closing(sqlite3.connect(root / "source.db")) as observed:
            with closing(
                observed.execute("SELECT raw_id FROM raw_sessions WHERE raw_id=?", ("captured-by-original",))
            ) as rows:
                captured = rows.fetchone()
            if boundary == "before_begin":
                assert captured is None
            else:
                assert captured == ("captured-by-original",)


def test_source_captured_statement_rejects_an_undeclared_row_effect(source_statement_root: Path) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "declared-only")
        permit = seal.prepare_source_mutation()
        with permit.hold_authority(), permit.mutation_connection() as source:
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            permit.apply_source_statements(source)
            with pytest.raises(sqlite3.DatabaseError):
                source.execute("UPDATE raw_sessions SET file_mtime_ms=99 WHERE raw_id='original-raw'")
            source.rollback()
        with seal.original_read_snapshot():
            with seal.original_rows(
                "source", "SELECT file_mtime_ms FROM raw_sessions WHERE raw_id='original-raw'"
            ) as rows:
                assert tuple(rows.fetchone()) == (None,)
            with seal.original_rows("source", "SELECT 1 FROM raw_sessions WHERE raw_id='declared-only'") as rows:
                assert rows.fetchone() is None


@pytest.mark.parametrize("route", ["connection", "custom_cursor", "executemany"])
def test_source_writer_rejects_uncaptured_row_effect_through_each_sql_entrypoint(
    source_statement_root: Path,
    route: str,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "only-captured-insert")
        permit = seal.prepare_source_mutation()
        with permit.hold_authority(), permit.mutation_connection() as source:
            with closing(source.execute("BEGIN IMMEDIATE")):
                pass
            sql = "UPDATE raw_sessions SET file_mtime_ms=99 WHERE raw_id=?"
            with pytest.raises(sqlite3.DatabaseError):
                if route == "custom_cursor":
                    with closing(source.cursor(factory=sqlite3.Cursor)) as cursor:
                        cursor.execute(sql, ("original-raw",))
                elif route == "executemany":
                    source.executemany(sql, [("original-raw",)])
                else:
                    source.execute(sql, ("original-raw",))
            source.rollback()
        with seal.original_read_snapshot():
            with seal.original_rows(
                "source", "SELECT file_mtime_ms FROM raw_sessions WHERE raw_id='original-raw'"
            ) as rows:
                assert tuple(rows.fetchone()) == (None,)


def test_source_commit_reserves_original_observer_until_acceptance(
    source_statement_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "acceptance-reserved")
        permit = seal.prepare_source_mutation()
        attempted = False
        with closing(sqlite3.connect(root / "source.db", timeout=0)) as foreign:
            original_unpinned = seal._require_unpinned_observer

            def observe_reservation(tier: str) -> sqlite3.Connection:
                nonlocal attempted
                if tier == "source" and permit._acceptance_reservation:
                    attempted = True
                    with pytest.raises(sqlite3.OperationalError) as refusal:
                        foreign.execute("UPDATE raw_sessions SET file_mtime_ms=9 WHERE raw_id='original-raw'")
                    assert refusal.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
                    foreign.rollback()
                return original_unpinned(tier)

            monkeypatch.setattr(seal, "_require_unpinned_observer", observe_reservation)
            with permit.hold_authority(), permit.mutation_connection() as source:
                with closing(source.execute("BEGIN IMMEDIATE")):
                    pass
                permit.apply_source_statements(source)
                permit.allow_commit(source)
                source.commit()
                seal.accept_known_tier_commit(permit.committed())
        assert attempted
        seal.validate_observers_current()


@pytest.mark.parametrize(
    "field", ["native_id", "source_path", "blob_hash", "source_revision", "origin", "file_mtime_ms"]
)
def test_source_only_existing_raw_identity_changes_refuse_before_publication(
    source_statement_root: Path,
    field: str,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    if field in {"origin", "file_mtime_ms"}:
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
            with closing(source.execute("UPDATE raw_sessions SET origin='claude-code', file_mtime_ms=9")):
                pass
            source.commit()
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with pytest.raises(ReferenceSealError):
            with seal.original_read_snapshot(), seal.source_producer():
                image = seal.retain_tier_row("source", "raw_sessions", 1)
                assert image is not None
                seal.load_source_row(image)
                columns = dict(zip(image.columns, image.cells, strict=True))
                value = b"x" * 32 if field == "blob_hash" else 10 if field == "file_mtime_ms" else "changed"
                expression, operands = seal.source_literal_expression(seal.retain_literal_scalar(value))
                with seal.source_statement(
                    f"UPDATE raw_sessions SET {field}={expression} WHERE rowid=1",
                    operands,
                    table="raw_sessions",
                    writable_targets=(("raw_sessions", (columns["raw_id"],)),),
                ):
                    pass


def test_source_only_raw_origin_refinement_and_null_mtime_backfill_publish(source_statement_root: Path) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None
            seal.load_source_row(image)
            columns = dict(zip(image.columns, image.cells, strict=True))
            expression, operands = seal.source_literal_expression(seal.retain_literal_scalar("claude-code-session"))
            with seal.source_statement(
                f"UPDATE raw_sessions SET origin={expression},file_mtime_ms=9 WHERE rowid=1",
                operands,
                table="raw_sessions",
                writable_targets=(("raw_sessions", (columns["raw_id"],)),),
            ):
                pass
        _publish_source(seal)


@pytest.mark.parametrize("binding", ["unretained text", b"unretained bytes"])
def test_source_statement_refuses_variable_adapter_before_stage_effects(
    source_statement_root: Path,
    binding: str | bytes,
) -> None:
    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            with pytest.raises(ReferenceSealError):
                with seal.source_statement(
                    "DELETE FROM raw_sessions WHERE raw_id=?",
                    (binding,),
                    table="raw_sessions",
                    writable_targets=(),
                ):
                    pass
            with seal.source_rows("SELECT count(*) FROM raw_sessions") as rows:
                assert rows.fetchone()[0] == 0


@pytest.mark.parametrize("restore", [False, True])
def test_source_only_blob_reference_transient_delete_requires_exact_original_key_restoration(
    source_statement_root: Path,
    restore: bool,
) -> None:
    root = source_statement_root
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
        with closing(
            source.execute(
                "INSERT INTO blob_refs(blob_hash,ref_id,ref_type,size_bytes,acquired_at_ms) "
                "VALUES(?, 'original-ref', 'raw_payload', 1, 1)",
                (b"b" * 32,),
            )
        ):
            pass
        source.commit()
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            image = seal.retain_tier_row("source", "blob_refs", 1)
            assert image is not None
            seal.load_source_row(image)
            values = dict(zip(image.columns, image.cells, strict=True))
            key = tuple(values[name] for name in ("blob_hash", "ref_type", "ref_id", "source_path"))
            with seal.source_statement(
                "DELETE FROM blob_refs WHERE rowid=1",
                table="blob_refs",
                writable_targets=(("blob_refs", key),),
            ):
                pass
            if restore:
                expressions = []
                parameters: tuple[object, ...] = (None,)
                for name in ("blob_hash", "ref_id", "ref_type"):
                    expression, operands = seal.source_literal_expression(values[name])
                    expressions.append(expression)
                    parameters += operands
                with seal.source_statement(
                    "INSERT INTO blob_refs(rowid,blob_hash,ref_id,ref_type,size_bytes,acquired_at_ms) "
                    f"VALUES(?,{','.join(expressions)},1,2)",
                    parameters,
                    table="blob_refs",
                    writable_targets=(("blob_refs", key),),
                    prepared_cells=dict(zip(("blob_hash", "ref_type", "ref_id", "source_path"), key, strict=True)),
                    allocation_parameter=0,
                ):
                    pass
        if restore:
            _publish_source(seal)
        else:
            with pytest.raises(ReferenceSealError):
                seal.prepare_source_mutation()


def test_source_only_original_raw_delete_refuses_even_with_explicit_writable_role(source_statement_root: Path) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with pytest.raises(ReferenceSealError):
            with seal.original_read_snapshot(), seal.source_producer():
                image = seal.retain_tier_row("source", "raw_sessions", 1)
                assert image is not None
                seal.load_source_row(image)
                values = dict(zip(image.columns, image.cells, strict=True))
                with seal.source_statement(
                    "DELETE FROM raw_sessions WHERE rowid=1",
                    table="raw_sessions",
                    writable_targets=(("raw_sessions", (values["raw_id"],)),),
                ):
                    pass


def test_actual_native_random_allocation_collision_retries_only_in_original_owner(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-m", "tests.infra.source_allocation_probe", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate()
        assert process.returncode == 0, stderr
        result = json.loads(stdout)
        assert result["attempts"] >= 2
        assert result["allocated"] != result["occupied"]
        assert result["sqlite"] == sqlite3.sqlite_version
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def test_source_only_non_session_artifact_retarget_and_empty_parser_census_keep_old_raw(
    source_statement_root: Path,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
        with closing(
            source.execute(
                "INSERT INTO raw_artifacts(artifact_id,raw_id,origin,source_path,source_index,artifact_kind,"
                "support_status,classification_reason,first_observed_at_ms,last_observed_at_ms) "
                "VALUES('artifact','original-raw','unknown-export','synthetic/artifact',0,"
                "'terminal_unknown_export_no_session','unsupported','synthetic',1,1)"
            )
        ):
            pass
        source.commit()
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "new-non-session")
            artifact = seal.retain_tier_row("source", "raw_artifacts", 1)
            assert artifact is not None
            seal.load_source_row(artifact)
            original_values = dict(zip(artifact.columns, artifact.cells, strict=True))
            new_raw = seal.retain_literal_scalar("new-non-session")
            raw_expression, raw_operands = seal.source_literal_expression(new_raw)
            artifact_key = seal.retain_literal_scalar("artifact")
            artifact_expression, artifact_operands = seal.source_literal_expression(artifact_key)
            with seal.source_statement(
                "INSERT INTO raw_artifacts(rowid,artifact_id,raw_id,origin,source_path,source_index,artifact_kind,"
                "support_status,classification_reason,first_observed_at_ms,last_observed_at_ms) "
                f"VALUES(?,{artifact_expression},{raw_expression},'unknown-export','synthetic/artifact',0,"
                "'terminal_unknown_export_no_session','unsupported','synthetic',1,2) "
                "ON CONFLICT(artifact_id) DO UPDATE SET raw_id=excluded.raw_id,last_observed_at_ms=excluded.last_observed_at_ms",
                (None, *artifact_operands, *raw_operands),
                table="raw_artifacts",
                writable_targets=(("raw_artifacts", (original_values["artifact_id"],)),),
                prepared_cells={"artifact_id": artifact_key},
                allocation_parameter=0,
            ):
                pass
            empty = seal.retain_literal_stream("text", 2, iter((b"[]",)))
            empty_expression, empty_operands = seal.source_literal_expression(empty)
            with seal.source_statement(
                "INSERT INTO raw_authority_parser_census(rowid,raw_id,parser_fingerprint,status,logical_keys_json) "
                f"VALUES(?,{raw_expression},'synthetic-parser','complete',{empty_expression}) "
                "ON CONFLICT(raw_id) DO UPDATE SET logical_keys_json=excluded.logical_keys_json",
                (None, *raw_operands, *empty_operands),
                table="raw_authority_parser_census",
                writable_targets=(("raw_authority_parser_census", (new_raw,)),),
                prepared_cells={"raw_id": new_raw, "logical_keys_json": empty},
                allocation_parameter=0,
            ):
                pass
        _publish_source(seal)
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT raw_id FROM raw_artifacts") as rows:
                assert rows.fetchone()[0] == "new-non-session"
            with seal.original_rows("source", "SELECT logical_keys_json FROM raw_authority_parser_census") as rows:
                assert rows.fetchone()[0] == "[]"
            with seal.original_rows("source", "SELECT count(*) FROM raw_sessions WHERE raw_id='original-raw'") as rows:
                assert rows.fetchone()[0] == 1


def test_same_witness_source_stage_reuses_only_successfully_accepted_original_postimages(
    source_statement_root: Path,
) -> None:
    root = source_statement_root
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        prior = seal.retain_literal_scalar("same original literal owner")
        with seal.original_read_snapshot(), seal.source_producer():
            first = _stage_raw(seal, "first-stage")
        _publish_source(seal)
        with seal.original_read_snapshot(), seal.source_producer():
            assert seal._literal_scalar_equal(prior, "same original literal owner")
            second = _stage_raw(seal, "second-stage")
            assert second == first + 1
        _publish_source(seal)
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT raw_id FROM raw_sessions ORDER BY rowid") as rows:
                assert [row[0] for row in rows] == ["first-stage", "second-stage"]


def test_same_witness_source_stage_cannot_reset_unapplied_original_tape(source_statement_root: Path) -> None:
    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "pending-original")
        seal.prepare_source_mutation()
        with seal.original_read_snapshot():
            with pytest.raises(ReferenceSealError):
                with seal.source_producer():
                    pytest.fail("an unaccepted tape was discarded")


def test_source_statement_failed_physical_cursor_close_retains_original_witness_until_creator_retry(
    source_statement_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.io_phase_metrics import _MeasuredCursor, live_connection_cursors
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, native_sql_children
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    root = source_statement_root
    seal = PreparedIndexMutation.source_only(archive_root=root)
    assert seal._scratch_directory is not None
    artifact = Path(seal._scratch_directory.name) / "refs.db"
    original_cursor = seal._scratch.cursor
    blocked: list[ControlledCursor] = []

    class SourceCursor(ControlledCursor, _MeasuredCursor):
        def execute(self, sql: str, parameters: object = (), /) -> "SourceCursor":
            super().execute(sql, parameters)
            if sql.startswith("INSERT INTO raw_sessions(rowid"):
                self.allow_cleanup.clear()
                blocked.append(self)
            return self

    def source_cursor() -> sqlite3.Cursor:
        return original_cursor(factory=SourceCursor)

    monkeypatch.setattr(seal._scratch, "cursor", source_cursor)
    try:
        with pytest.raises(NativeConnectionSettlementError):
            with seal.original_read_snapshot(), seal.source_producer():
                _stage_raw(seal, "unsettled-stage")
        assert blocked and artifact.exists()
        original_owner = next(child for child in native_sql_children(seal) if child.connection is seal._scratch)
        assert original_owner.connection is seal._scratch
        assert original_owner.close_required and blocked[0] in live_connection_cursors(seal._scratch)
        with pytest.raises(ReferenceSealError):
            seal.prepare_source_mutation()
        with pytest.raises(NativeConnectionSettlementError):
            seal.close()
        assert artifact.exists()
        for cursor in blocked:
            cursor.allow_cleanup.set()
        seal.close()
        assert seal._closed and not artifact.exists()
    finally:
        for cursor in blocked:
            cursor.allow_cleanup.set()
        seal.close()


@pytest.mark.parametrize("failure", [None, "dependent", "missing_census", "unknown_fingerprint"])
def test_source_only_original_full_normalization_requires_complete_current_census_and_no_dependents(
    source_statement_root: Path,
    failure: str | None,
) -> None:
    from polylogue.archive.revision_authority import raw_authority_parser_fingerprint

    root = source_statement_root
    _seed_original_raw(root)
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
        with closing(
            source.execute(
                "UPDATE raw_sessions SET logical_source_key='unknown-export:original',revision_kind='full',"
                "source_revision='original-raw',acquisition_generation=0"
            )
        ):
            pass
        if failure == "dependent":
            with closing(
                source.execute(
                    "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms,"
                    "revision_kind,predecessor_raw_id) VALUES('dependent','unknown-export','synthetic/dependent',?,1,1,'append','original-raw')",
                    (b"d" * 32,),
                )
            ):
                pass
        source.commit()
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None
            seal.load_source_row(image)
            values = dict(zip(image.columns, image.cells, strict=True))
            with seal.source_statement(
                "UPDATE raw_sessions SET logical_source_key=NULL,revision_kind='unknown',source_revision=NULL,"
                "predecessor_source_revision=NULL,predecessor_raw_id=NULL,baseline_raw_id=NULL,append_start_offset=NULL,"
                "append_end_offset=NULL,acquisition_generation=NULL,revision_authority='quarantined' WHERE rowid=1",
                table="raw_sessions",
                writable_targets=(("raw_sessions", (values["raw_id"],)),),
            ):
                pass
            if failure != "missing_census":
                fingerprint = seal.retain_literal_scalar(
                    "unrecognized" if failure == "unknown_fingerprint" else raw_authority_parser_fingerprint()
                )
                fingerprint_sql, fingerprint_parameters = seal.source_literal_expression(fingerprint)
                raw_sql, raw_parameters = seal.source_literal_expression(values["raw_id"])
                with seal.source_statement(
                    "INSERT INTO raw_membership_census(rowid,raw_id,parser_fingerprint,status,member_count,censused_at_ms) "
                    f"VALUES(?,{raw_sql},{fingerprint_sql},'non_session',0,1)",
                    (None, *raw_parameters, *fingerprint_parameters),
                    table="raw_membership_census",
                    writable_targets=(("raw_membership_census", (values["raw_id"],)),),
                    prepared_cells={"raw_id": values["raw_id"], "parser_fingerprint": fingerprint},
                    allocation_parameter=0,
                ):
                    pass
                with seal.source_statement(
                    "INSERT INTO raw_authority_parser_census(rowid,raw_id,parser_fingerprint,status,logical_keys_json) "
                    f"VALUES(?,{raw_sql},{fingerprint_sql},'complete','[]')",
                    (None, *raw_parameters, *fingerprint_parameters),
                    table="raw_authority_parser_census",
                    writable_targets=(("raw_authority_parser_census", (values["raw_id"],)),),
                    prepared_cells={"raw_id": values["raw_id"]},
                    allocation_parameter=0,
                ):
                    pass
        if failure is not None:
            with pytest.raises(ReferenceSealError):
                seal.prepare_source_mutation()
        else:
            _publish_source(seal)
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "source",
                    "SELECT raw_id,source_path,blob_hash,blob_size,revision_kind,logical_source_key FROM raw_sessions WHERE rowid=1",
                ) as rows:
                    assert tuple(rows.fetchone()) == ("original-raw", "synthetic/source", b"o" * 32, 1, "unknown", None)


def test_publisher_reservation_pages_rebase_only_after_original_receipt_and_close(source_statement_root: Path) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        for payload in (b"first synthetic page", b"second synthetic page"):
            prepared = publisher.prepare_from_bytes(payload)
            claim = publisher.prepare_claim(prepared)
            publisher.queue_prepared(prepared, claim=claim)
            publisher.prepare_flush(reference_seal=seal)
            assert not publisher._store.blob_path(claim.receipt.blob_hash).exists()
            assert publisher.flush(reference_seal=seal) == (claim.receipt,)
            assert publisher._store.blob_path(claim.receipt.blob_hash).read_bytes() == payload
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "after-reservations")
        _publish_source(seal)
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT count(*) FROM blob_publication_reservations") as rows:
                assert rows.fetchone()[0] == 2


def test_publisher_reservation_cannot_commit_a_preceding_raw_schedule(source_statement_root: Path) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    prepared = publisher.prepare_from_bytes(b"synthetic first-unit refusal")
    claim = publisher.prepare_claim(prepared)
    publisher.queue_prepared(prepared, claim=claim)
    try:
        with PreparedIndexMutation.source_only(archive_root=root) as seal:
            with seal.original_read_snapshot(), seal.source_producer():
                _stage_raw(seal, "not-yet-published")
            with pytest.raises(ReferenceSealError):
                publisher.prepare_flush(reference_seal=seal)
            assert not publisher._store.blob_path(claim.receipt.blob_hash).exists()
    finally:
        publisher.discard_pending()


def test_publisher_changed_prepared_batch_never_exposes_paths(source_statement_root: Path) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    try:
        with PreparedIndexMutation.source_only(archive_root=root) as seal:
            first = publisher.prepare_from_bytes(b"synthetic first claim")
            claim = publisher.prepare_claim(first)
            publisher.queue_prepared(first, claim=claim)
            publisher.prepare_flush(reference_seal=seal)
            second = publisher.prepare_from_bytes(b"synthetic later claim")
            publisher.queue_prepared(second, claim=publisher.prepare_claim(second))
            with pytest.raises(ValueError):
                publisher.flush(reference_seal=seal)
            assert not publisher._store.blob_path(claim.receipt.blob_hash).exists()
            seal.validate_observers_current()
    finally:
        publisher.discard_pending()


@pytest.mark.parametrize("foreign_change", [False, True])
def test_accepted_reservation_close_retry_never_reapplies_or_exposes_changed_authority(
    source_statement_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_change: bool,
) -> None:
    from typing import Any

    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, sqlite_factory_targets_database

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    prepared = publisher.prepare_from_bytes(b"synthetic accepted unsettled reservation")
    claim = publisher.prepare_claim(prepared)
    publisher.queue_prepared(prepared, claim=claim)
    commits: list[sqlite3.Connection] = []
    from polylogue.storage.io_phase_metrics import connect_measured as original_factory

    class ReservationConnection(ControlledConnection):
        def commit(self) -> None:
            super().commit()
            commits.append(self)
            self.close_failure = OSError("synthetic committed reservation close remains unsettled")

    def controlled(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if sqlite_factory_targets_database(database, (root / "source.db",)):
            return sqlite3.connect(database, *args, factory=ReservationConnection, **kwargs)
        return original_factory(database, *args, **kwargs)

    seal = PreparedIndexMutation.source_only(archive_root=root)
    try:
        publisher.prepare_flush(reference_seal=seal)
        monkeypatch.setattr(connection_profile, "connect_measured", controlled)
        with pytest.raises(NativeConnectionSettlementError):
            publisher.flush(reference_seal=seal)
        assert len(commits) == 1
        owner = publisher._reservation_native_owner
        assert owner is not None and owner.connection is commits[0]
        assert publisher._reservation_accepted
        assert not publisher._store.blob_path(claim.receipt.blob_hash).exists()
        with pytest.raises(NativeConnectionSettlementError):
            publisher.discard_pending()
        assert prepared.temporary_path.exists()
        assert publisher._reservation_permit is not None and publisher._reservation_accepted
        with pytest.raises(NativeConnectionSettlementError):
            publisher.settle_prepared_flush(reference_seal=seal)
        with pytest.raises(NativeConnectionSettlementError):
            publisher.discard_pending_receipt(claim.receipt.publication_id)
        assert prepared.temporary_path.exists()
        assert seal._mutation_custody is not None
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT count(*) FROM blob_publication_reservations") as rows:
                assert rows.fetchone()[0] == 1
        connection = commits[0]
        assert isinstance(connection, ReservationConnection)
        connection.close_failure = None
        publisher.settle_prepared_flush(reference_seal=seal)
        assert owner.connection is None and seal._mutation_custody is None
        if foreign_change:
            with closing(sqlite3.connect(root / "source.db")) as foreign:
                with closing(foreign.execute("UPDATE blob_publication_reservations SET publisher_id='changed'")):
                    pass
                foreign.commit()
            with pytest.raises(ReferenceSealError):
                publisher.flush(reference_seal=seal)
            assert not publisher._store.blob_path(claim.receipt.blob_hash).exists()
        else:
            assert publisher.flush(reference_seal=seal) == (claim.receipt,)
            assert (
                publisher._store.blob_path(claim.receipt.blob_hash).read_bytes()
                == b"synthetic accepted unsettled reservation"
            )
        assert len(commits) == 1
        assert owner.connection is None
    finally:
        for connection in commits:
            assert isinstance(connection, ReservationConnection)
            connection.close_failure = None
        seal.close()
        publisher.discard_pending()


@pytest.mark.parametrize(
    "pragma",
    [
        "table_info(raw_sessions)",
        "table_xinfo(raw_sessions)",
        "foreign_key_list(raw_sessions)",
        "index_list(raw_sessions)",
        "index_info(idx_raw_sessions_predecessor_raw_id)",
        "index_xinfo(idx_raw_sessions_predecessor_raw_id)",
        "main.table_list",
    ],
)
def test_original_source_schema_metadata_reads_settle_without_widening_writer_authority(
    source_statement_root: Path,
    pragma: str,
) -> None:
    from polylogue.storage.io_phase_metrics import live_connection_cursors

    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:
        with seal.original_read_snapshot():
            with seal.original_rows("source", "PRAGMA " + pragma) as rows:
                for _row in rows:
                    pass
            assert live_connection_cursors(seal.observer("source")) == ()
            for sql in (
                "PRAGMA foreign_keys=OFF",
                "PRAGMA recursive_triggers=OFF",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA journal_mode=DELETE",
                "CREATE TABLE forbidden_metadata_ddl(value INTEGER)",
                "DELETE FROM raw_sessions",
            ):
                with pytest.raises(sqlite3.DatabaseError):
                    with seal.original_rows("source", sql):
                        pytest.fail("schema read permission widened into writer authority")
            assert live_connection_cursors(seal.observer("source")) == ()
        seal.validate_observers_current()


@pytest.mark.parametrize("foreign_change", [False, True])
def test_accepted_reservation_placement_retry_retains_exact_batch_without_reapplication(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch, foreign_change: bool
) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import PreparedBlob
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    prepared = [
        publisher.prepare_from_bytes(value) for value in (b"first partial placement", b"second partial placement")
    ]
    claims = [publisher.prepare_claim(item) for item in prepared]
    for item, claim in zip(prepared, claims, strict=True):
        publisher.queue_prepared(item, claim=claim)
    applications: list[KnownTierMutationPermit] = []
    original_apply = KnownTierMutationPermit.apply_source_statements
    original_place = publisher._store._place_prepared
    placements = 0

    def apply(permit: KnownTierMutationPermit, connection: sqlite3.Connection) -> None:
        applications.append(permit)
        original_apply(permit, connection)

    def place(item: PreparedBlob) -> tuple[tuple[str, int], Path | None, bool]:
        nonlocal placements
        placements += 1
        if placements == 2:
            raise OSError("synthetic second placement failure")
        return original_place(item)

    monkeypatch.setattr(KnownTierMutationPermit, "apply_source_statements", apply)
    monkeypatch.setattr(publisher._store, "_place_prepared", place)
    try:
        with PreparedIndexMutation.source_only(archive_root=root) as seal:
            publisher.prepare_flush(reference_seal=seal)
            with pytest.raises(OSError):
                publisher.flush(reference_seal=seal)
            assert publisher._reservation_accepted
            assert publisher._reservation_permit is applications[0]
            assert publisher._reservation_receipts == tuple(claim.receipt for claim in claims)
            assert publisher._store.blob_path(claims[0].receipt.blob_hash).read_bytes() == b"first partial placement"
            assert not publisher._store.blob_path(claims[1].receipt.blob_hash).exists()
            monkeypatch.setattr(publisher._store, "_place_prepared", original_place)
            if foreign_change:
                with closing(sqlite3.connect(root / "source.db")) as foreign:
                    with closing(foreign.execute("UPDATE blob_publication_reservations SET publisher_id='changed'")):
                        pass
                    foreign.commit()
                with pytest.raises(ReferenceSealError):
                    publisher.flush(reference_seal=seal)
                assert not publisher._store.blob_path(claims[1].receipt.blob_hash).exists()
            else:
                assert publisher.flush(reference_seal=seal) == tuple(claim.receipt for claim in claims)
                assert publisher._reservation_permit is None
                assert (
                    publisher._store.blob_path(claims[1].receipt.blob_hash).read_bytes() == b"second partial placement"
                )
            assert len(applications) == 1
    finally:
        publisher.discard_pending()


def test_live_source_execute_failure_retains_actual_cursor_and_primary_until_creator_retry(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from builtins import BaseExceptionGroup
    from typing import Any

    from polylogue.storage.io_phase_metrics import _MeasuredCursor, live_connection_cursors
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, native_sql_children
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    seal = PreparedIndexMutation.source_only(archive_root=source_statement_root)
    primary = OSError("synthetic failure after actual live Source SQL")
    blocked: list[ControlledCursor] = []
    writer: sqlite3.Connection | None = None

    class FailingCursor(ControlledCursor, _MeasuredCursor):
        def execute(self, sql: str, parameters: Any = (), /) -> "FailingCursor":
            super().execute(sql, parameters)
            if sql.startswith("INSERT INTO raw_sessions(rowid"):
                self.allow_cleanup.clear()
                blocked.append(self)
                raise primary
            return self

    def contains(error: BaseException) -> bool:
        pending = [error]
        visited: set[int] = set()
        while pending:
            selected = pending.pop()
            if selected is primary:
                return True
            if id(selected) in visited:
                continue
            visited.add(id(selected))
            if isinstance(selected, NativeConnectionSettlementError):
                pending.append(selected.failure)
            if isinstance(selected, BaseExceptionGroup):
                pending.extend(selected.exceptions)
            if selected.__cause__ is not None:
                pending.append(selected.__cause__)
            if selected.__context__ is not None:
                pending.append(selected.__context__)
        return False

    try:
        with seal.original_read_snapshot(), seal.source_producer():
            _stage_raw(seal, "failed-live-application")
        permit = seal.prepare_source_mutation()
        with pytest.raises(NativeConnectionSettlementError) as failure:
            with permit.hold_authority(), permit.mutation_connection() as connection:
                writer = connection
                with seal._owned_cursor(connection, "BEGIN IMMEDIATE"):
                    pass
                original_cursor = connection.cursor
                monkeypatch.setattr(connection, "cursor", lambda: original_cursor(factory=FailingCursor))
                permit.apply_source_statements(connection)
        assert contains(failure.value)
        assert writer is not None and blocked[0] in live_connection_cursors(writer)
        owner = next(child for child in native_sql_children(seal) if child.connection is writer)
        assert owner.close_required
        assert seal._scratch_directory is not None
        artifact = Path(seal._scratch_directory.name) / "refs.db"
        assert artifact.exists()
        with pytest.raises(NativeConnectionSettlementError):
            seal.close()
        assert artifact.exists()
        for cursor in blocked:
            cursor.allow_cleanup.set()
        seal.close()
        assert not artifact.exists()
        with closing(sqlite3.connect(source_statement_root / "source.db")) as source:
            with closing(source.execute("SELECT count(*) FROM raw_sessions")) as rows:
                assert rows.fetchone()[0] == 0
    finally:
        for cursor in blocked:
            cursor.allow_cleanup.set()
        seal.close()


def test_same_witness_accepted_source_units_release_only_their_claim_across_separate_gates(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread
    from polylogue.storage.sqlite.managed_connection import sqlite_connection
    from polylogue.storage.sqlite.write_lease import current_sql_custody

    with write_lease("test.source-unit-bootstrap", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
        for raw_id in ("first-separate-gate", "second-separate-gate"):
            with seal.original_read_snapshot(), seal.source_producer():
                _stage_raw(seal, raw_id)
            with write_lease("test.source-unit-publication", archive_root=tmp_path):
                custody = current_sql_custody()
                assert custody is not None
                with sqlite_connection(":memory:") as unrelated:
                    unrelated_owner = next(
                        owner
                        for owner in retained_native_sql_owners_on_current_thread()
                        if owner.connection is unrelated
                    )
                    _publish_source(seal)
                    assert seal._mutation_custody is None
                    assert unrelated_owner.custody is custody
                    assert unrelated_owner in custody.retained_sql_owners_on_current_thread()
                    assert custody._fd >= 0
                assert unrelated_owner.connection is None
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT count(*) FROM raw_sessions") as rows:
                assert rows.fetchone()[0] == 2


@pytest.mark.parametrize("accepted", [False, True])
def test_original_reservation_partial_cancel_preserves_remaining_claim_authority(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import PreparedBlob
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    prepared = [
        publisher.prepare_from_bytes(value) for value in (b"canceled original claim", b"retained original claim")
    ]
    claims = [publisher.prepare_claim(item) for item in prepared]
    for item, claim in zip(prepared, claims, strict=True):
        publisher.queue_prepared(item, claim=claim)
    applications: list[KnownTierMutationPermit] = []
    original_apply = KnownTierMutationPermit.apply_source_statements
    original_place = publisher._store._place_prepared

    def apply(permit: KnownTierMutationPermit, connection: sqlite3.Connection) -> None:
        applications.append(permit)
        original_apply(permit, connection)

    def refuse_placement(item: PreparedBlob) -> tuple[tuple[str, int], Path | None, bool]:
        raise OSError("synthetic placement failure before exposure")

    monkeypatch.setattr(KnownTierMutationPermit, "apply_source_statements", apply)
    try:
        with PreparedIndexMutation.source_only(archive_root=root) as seal:
            publisher.prepare_flush(reference_seal=seal)
            if accepted:
                monkeypatch.setattr(publisher._store, "_place_prepared", refuse_placement)
                with pytest.raises(OSError):
                    publisher.flush(reference_seal=seal)
                monkeypatch.setattr(publisher._store, "_place_prepared", original_place)
                assert publisher.discard_pending_receipt(claims[0].receipt.publication_id)
                assert not prepared[0].temporary_path.exists()
                assert publisher._reservation_receipts == tuple(claim.receipt for claim in claims)
                assert publisher.flush(reference_seal=seal) == (claims[1].receipt,)
                assert publisher._reservation_permit is None
                assert (
                    publisher._store.blob_path(claims[1].receipt.blob_hash).read_bytes() == b"retained original claim"
                )
                assert not publisher._store.blob_path(claims[0].receipt.blob_hash).exists()
                assert len(applications) == 1
            else:
                with pytest.raises(ValueError):
                    publisher.settle_prepared_flush(reference_seal=seal)
                with pytest.raises(ValueError):
                    publisher.discard_pending_receipt(claims[0].receipt.publication_id)
                assert prepared[0].temporary_path.exists() and prepared[1].temporary_path.exists()
                publisher.discard_pending()
                assert publisher._reservation_permit is None
                assert not prepared[0].temporary_path.exists() and not prepared[1].temporary_path.exists()
                assert applications == []
    finally:
        publisher.discard_pending()


def test_original_input_demand_precedes_copy_and_reuses_only_same_epoch(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        original_copy = seal._retain_variable_cell

        def measured_copy(metadata: SQLiteLiteralCell, chunks: Iterable[bytes]) -> KnownTierCell:
            assert charges, "original payload copied before creator input demand amendment"
            return original_copy(metadata, chunks)

        monkeypatch.setattr(seal, "_retain_variable_cell", measured_copy)
        with seal.original_read_snapshot(input_demand=charges.append):
            assert seal.retain_tier_row("source", "raw_sessions", 999) is None
            assert charges == []
            first = seal.retain_tier_row("source", "raw_sessions", 1)
            again = seal.retain_tier_row("source", "raw_sessions", 1)
            assert first is not None and again == first
            assert len(charges) == 1 and charges[0] > 0
        with seal.original_read_snapshot(input_demand=charges.append):
            assert seal.retain_tier_row("source", "raw_sessions", 1) == first
            assert len(charges) == 1
        with seal.original_read_snapshot(), seal.source_producer():
            seal.load_source_row(first)
            _stage_raw(seal, "original-raw")
        _publish_source(seal)
        with seal.original_read_snapshot(input_demand=charges.append):
            advanced = seal.retain_tier_row("source", "raw_sessions", 1)
            assert advanced is not None and advanced.cells != first.cells
            assert charges == [charges[0], charges[0]]


def test_original_input_demand_is_not_recharged_after_cell_copy_failure(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_original_raw(source_statement_root)
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:
        original_copy = seal._retain_variable_cell
        failure = OSError("synthetic selected literal read failure")
        fail_once = True

        def copy(metadata: SQLiteLiteralCell, chunks: Iterable[bytes]) -> KnownTierCell:
            nonlocal fail_once
            if fail_once:
                fail_once = False

                def interrupted() -> Iterator[bytes]:
                    for chunk in chunks:
                        yield chunk
                        raise failure

                return original_copy(metadata, interrupted())
            return original_copy(metadata, chunks)

        monkeypatch.setattr(seal, "_retain_variable_cell", copy)
        with seal.original_read_snapshot(input_demand=charges.append):
            with seal._owned_cursor(seal._scratch, "SELECT count(*) FROM known_tier_literal_cells") as rows:
                before = rows.fetchone()[0]
            with pytest.raises(OSError) as caught:
                seal.retain_tier_row("source", "raw_sessions", 1)
            assert caught.value is failure
            with seal._owned_cursor(seal._scratch, "SELECT count(*) FROM known_tier_literal_cells") as rows:
                assert rows.fetchone()[0] == before
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None and len(charges) == 1


def test_prepared_cas_input_charges_actual_claim_before_read_and_reuses_original_alias(
    source_statement_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.core.storage_faults import ArchiveStorageFaultError
    from polylogue.storage.blob_publication import (
        ArchiveBlobPublisher,
        BlobPublicationSourceRead,
        PreparedBlobPublicationClaim,
        consume_blob_publication_receipt,
    )

    root = source_statement_root
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    payload = b"synthetic current-phase raw sidecar"
    prepared = publisher.prepare_from_bytes(payload)
    claim = publisher.prepare_claim(prepared)
    publisher.queue_prepared(prepared, claim=claim)
    alias_prepared = publisher.prepare_from_bytes(payload)
    alias_claim = publisher.prepare_claim(alias_prepared)
    publisher.queue_prepared(alias_prepared, claim=alias_claim)
    charges: list[int] = []
    try:
        with PreparedIndexMutation.source_only(archive_root=root, input_demand=charges.append) as seal:
            with seal.original_read_snapshot():
                assert seal.retain_prepared_blob_input(claim) == (bytes.fromhex(claim.receipt.blob_hash), len(payload))
                assert charges == [len(payload)]
                assert claim.prepared_path.read_bytes() == payload
                seal.retain_prepared_blob_input(claim)
                seal.retain_prepared_blob_input(alias_claim)
                assert charges == [len(payload)]
            publisher.prepare_flush(reference_seal=seal)
            publisher.flush(reference_seal=seal)
            assert not claim.prepared_path.exists()
            with seal.original_read_snapshot():
                # Final-path enrollment must use the original accepted
                # reservation; an already exposed file alone is insufficient.
                original_validate = publisher.validate_published_claim
                checked_fields: list[dict[str, int]] = []

                def validate_after_field_charge(
                    source: BlobPublicationSourceRead,
                    selected_claim: PreparedBlobPublicationClaim,
                    *,
                    source_path: str,
                ) -> tuple[str, int]:
                    with seal._owned_cursor(
                        seal.observer("source"),
                        "SELECT rowid FROM blob_publication_reservations WHERE publication_id=?",
                        (selected_claim.receipt.publication_id,),
                    ) as cursor:
                        physical_rowid = cursor.fetchone()[0]
                    with seal._owned_cursor(
                        seal._scratch,
                        "SELECT column_name,byte_length FROM temp.original_input_fields "
                        "WHERE tier='source' AND epoch=? AND table_name='blob_publication_reservations' "
                        "AND row_address=? AND column_name IN ('blob_hash','size_bytes','publisher_id')",
                        (seal._original_input_epochs["source"], physical_rowid),
                    ) as cursor:
                        fields = dict(cursor)
                    assert fields == {
                        "blob_hash": 32,
                        "size_bytes": 8,
                        "publisher_id": len(selected_claim.receipt.publisher_id.encode()),
                    }
                    checked_fields.append(fields)
                    return original_validate(source, selected_claim, source_path=source_path)

                with monkeypatch.context() as scope:
                    scope.setattr(publisher, "validate_published_claim", validate_after_field_charge)
                    seal.retain_prepared_blob_input(claim)
                    accepted_charges = tuple(charges)
                    seal.retain_prepared_blob_input(claim)
                    assert tuple(charges) == accepted_charges
                assert len(checked_fields) == 2
                reservation_fields = sum(checked_fields[0].values())
        fresh: list[int] = []
        with PreparedIndexMutation.source_only(archive_root=root, input_demand=fresh.append) as later:
            with later.original_read_snapshot():
                later.retain_prepared_blob_input(claim)
                assert fresh == [reservation_fields, len(payload)]
        with pytest.raises(ReferenceSealError):
            later.retain_prepared_blob_input(claim)
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
            with closing(
                source.execute(
                    "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                    "VALUES('published-original','unknown-export','synthetic/published',?,?,1)",
                    (bytes.fromhex(claim.receipt.blob_hash), len(payload)),
                )
            ):
                pass
            consume_blob_publication_receipt(
                source, claim.receipt.publication_id, bytes.fromhex(claim.receipt.blob_hash)
            )
            source.commit()
        unpaid: list[int] = []
        with PreparedIndexMutation.source_only(archive_root=root, input_demand=unpaid.append) as missing:
            with pytest.raises(ArchiveStorageFaultError):
                with missing.original_read_snapshot():
                    missing.retain_prepared_blob_input(claim)
        assert unpaid == []
        assert publisher._store.blob_path(claim.receipt.blob_hash).read_bytes() == payload
    finally:
        publisher.discard_pending()


@pytest.mark.parametrize("refusal", ["foreign", "stale", "missing-reservation", "demand"])
def test_prepared_cas_input_refuses_invalid_or_cancelled_provenance_before_payload_read(
    source_statement_root: Path,
    refusal: str,
) -> None:
    from polylogue.core.storage_faults import ArchiveStorageFaultError
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = source_statement_root
    publisher = ArchiveBlobPublisher(
        root / ("foreign-source.db" if refusal == "foreign" else "source.db"), root / "blob"
    )
    prepared = publisher.prepare_from_bytes(b"synthetic refused raw sidecar")
    claim = publisher.prepare_claim(prepared)
    publisher.queue_prepared(prepared, claim=claim)
    charges: list[int] = []
    cancelled = OSError("synthetic current-phase demand cancellation")

    def amend(byte_length: int) -> None:
        if refusal == "demand":
            raise cancelled
        charges.append(byte_length)

    try:
        if refusal == "stale":
            claim.prepared_path.unlink()
            claim.prepared_path.write_bytes(b"changed")
        elif refusal == "missing-reservation":
            # A final file without this exact accepted receipt is not input
            # provenance, even when its hash and size are otherwise correct.
            publisher._store.publish_many((prepared,))
        with PreparedIndexMutation.source_only(archive_root=root, input_demand=amend) as seal:
            expected = (
                ArchiveStorageFaultError
                if refusal == "missing-reservation"
                else OSError
                if refusal == "demand"
                else (ReferenceSealError, ValueError)
            )
            with pytest.raises(expected) as caught:
                with seal.original_read_snapshot():
                    seal.retain_prepared_blob_input(claim)
            assert charges == []
            if refusal == "demand":
                assert caught.value is cancelled
                with pytest.raises(ReferenceSealError):
                    with seal.original_read_snapshot():
                        pass
    finally:
        publisher.discard_pending()


def test_original_input_demand_refusal_prevents_native_payload_copy(
    source_statement_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_original_raw(source_statement_root)
    refusal = RuntimeError("synthetic creator amendment refusal")
    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:

        def refuse(_byte_length: int) -> None:
            raise refusal

        def forbidden_copy(_metadata: SQLiteLiteralCell, _chunks: Iterable[bytes]) -> KnownTierCell:
            pytest.fail("payload copied after failed demand amendment")

        monkeypatch.setattr(seal, "_retain_variable_cell", forbidden_copy)
        with pytest.raises(RuntimeError) as caught:
            with seal.original_read_snapshot(input_demand=refuse):
                seal.retain_tier_row("source", "raw_sessions", 1)
        assert caught.value is refusal
        with pytest.raises(ReferenceSealError):
            with seal.original_read_snapshot(input_demand=refuse):
                pass


def test_original_cas_demand_precedes_initial_and_late_reads_and_deduplicates_aliases(
    source_statement_root: Path,
) -> None:
    root = source_statement_root
    _seed_original_raw(root)
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
        with closing(
            source.execute(
                "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                "VALUES('alias-raw','unknown-export','synthetic/alias',?,1,1),"
                "('late-raw','unknown-export','synthetic/late',?,9,1)",
                (b"o" * 32, b"l" * 32),
            )
        ):
            pass
        source.commit()
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        with seal.original_read_snapshot(input_demand=charges.append):
            assert seal.retain_original_blob_input("original-raw") == (b"o" * 32, 1)
            assert seal.retain_original_blob_input("alias-raw") == (b"o" * 32, 1)
            assert charges == [40, 1, 40]
            assert seal.retain_original_blob_input("late-raw") == (b"l" * 32, 9)
            assert seal.retain_original_blob_input("late-raw") == (b"l" * 32, 9)
            assert charges == [40, 1, 40, 40, 9]
        with seal.original_read_snapshot(input_demand=charges.append):
            seal.retain_original_blob_input("late-raw")
            assert charges == [40, 1, 40, 40, 9]


def test_original_cas_missing_descriptor_refuses_before_payload_hydration(source_statement_root: Path) -> None:
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=source_statement_root) as seal:
        with seal.original_read_snapshot(input_demand=charges.append):
            with pytest.raises(ReferenceSealError):
                seal.retain_original_blob_input("missing-original-input")
        assert charges == []


def test_prior_original_row_charge_does_not_prepay_cas_payload(source_statement_root: Path) -> None:
    _seed_original_raw(source_statement_root)
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=source_statement_root, input_demand=charges.append) as seal:
        with seal.original_read_snapshot():
            assert seal.retain_tier_row("source", "raw_sessions", 1) is not None
            row_charge = tuple(charges)
            assert seal.retain_original_blob_input("original-raw") == (b"o" * 32, 1)
            assert tuple(charges) == (*row_charge, 1)
            seal.retain_original_blob_input("original-raw")
            assert tuple(charges) == (*row_charge, 1)


def test_append_prepaid_cas_requires_matching_original_acquisition(source_statement_root: Path) -> None:
    _seed_original_raw(source_statement_root)
    charges: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=source_statement_root, input_demand=charges.append) as seal:
        with seal.original_read_snapshot(prepaid_blob_inputs=(("original-raw", b"o" * 32, 1),)):
            assert seal.retain_original_blob_input("original-raw") == (b"o" * 32, 1)
        assert charges == [40]
        with pytest.raises(ReferenceSealError):
            with seal.original_read_snapshot(prepaid_blob_inputs=(("original-raw", b"x" * 32, 1),)):
                pytest.fail("changed accepted append input admitted payload hydration")
        assert charges == [40]


def test_omitted_original_windows_keep_constructor_demand_and_override_restores_it(source_statement_root: Path) -> None:
    _seed_original_raw(source_statement_root)
    original: list[int] = []
    override: list[int] = []
    with PreparedIndexMutation.source_only(archive_root=source_statement_root, input_demand=original.append) as seal:
        with seal.original_read_snapshot(input_demand=override.append):
            seal.retain_original_blob_input("original-raw")
        assert override == [40, 1] and original == []
        with seal.original_read_snapshot():
            image = seal.retain_tier_row("source", "raw_sessions", 1)
            assert image is not None
        assert original and override == [40, 1]
        with pytest.raises(ReferenceSealError):
            seal.before_index_input("sessions", ("session_id",), "SELECT rowid FROM sessions", ())
