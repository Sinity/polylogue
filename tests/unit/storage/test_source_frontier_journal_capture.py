"""Actual Source INSERT parents survive either native AFTER callback order."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.fixture
def journal_root(tmp_path: Path) -> Iterator[Path]:
    with write_lease("test.source-frontier-journal-capture", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        yield tmp_path


def _stage_raw(seal: PreparedIndexMutation, *, mtime: int) -> int:
    cells = {
        name: seal.retain_literal_scalar(value)
        for name, value in (
            ("raw_id", "journal-raw"),
            ("origin", "unknown-export"),
            ("source_path", "synthetic/journal"),
            ("blob_hash", b"j" * 32),
        )
    }
    expressions = []
    parameters: tuple[object, ...] = (None,)
    for cell in cells.values():
        expression, operands = seal.source_literal_expression(cell)
        expressions.append(expression)
        parameters += operands
    with seal.source_statement(
        "INSERT INTO raw_sessions(rowid,raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms,file_mtime_ms) "
        f"VALUES(?,{','.join(expressions)},1,1,{mtime}) "
        "ON CONFLICT(raw_id) DO UPDATE SET file_mtime_ms=excluded.file_mtime_ms",
        parameters,
        table="raw_sessions",
        writable_targets=(("raw_sessions", (cells["raw_id"],)),),
        prepared_cells=cells,
        allocation_parameter=0,
    ) as cursor:
        assert isinstance(cursor.lastrowid, int)
        return cursor.lastrowid


def _publish(seal: PreparedIndexMutation) -> None:
    permit = seal.prepare_source_mutation()
    with permit.hold_authority(), permit.mutation_connection() as connection:
        with closing(connection.execute("BEGIN IMMEDIATE")):
            pass
        permit.apply_source_statements(connection)
        permit.allow_commit(connection)
        connection.commit()
        seal.accept_known_tier_commit(permit.committed())


@pytest.mark.parametrize("journal_first", [False, True])
def test_exact_insert_parent_and_upsert_publish_in_both_callback_orders(
    journal_root: Path, monkeypatch: pytest.MonkeyPatch, journal_first: bool
) -> None:
    root = journal_root
    capture_pending: list[tuple[str, str, str, int | None, int | None]] = []
    publication_pending: list[
        tuple[KnownTierMutationPermit, sqlite3.Connection, str, str, str, int | None, int | None]
    ] = []
    observed: list[str] = []
    with PreparedIndexMutation.source_only(archive_root=root) as seal:
        capture = seal._capture_source_transition

        def capture_order(table: str, phase: str, operation: str, old_rowid: int | None, new_rowid: int | None) -> int:
            if journal_first and table == "raw_sessions" and phase == "AFTER" and operation == "INSERT":
                assert not capture_pending
                capture_pending.append((table, phase, operation, old_rowid, new_rowid))
                return 1
            result = capture(table, phase, operation, old_rowid, new_rowid)
            if table == "raw_existence_changes" and phase == "AFTER" and capture_pending:
                observed.append("prepared-journal-before-root-after")
                result = capture(*capture_pending.pop())
            return result

        monkeypatch.setattr(seal, "_capture_source_transition", capture_order)
        consume = KnownTierMutationPermit._consume_native_effect

        def publication_order(
            permit: KnownTierMutationPermit,
            connection: sqlite3.Connection,
            table: str,
            phase: str,
            operation: str,
            old_rowid: int | None,
            new_rowid: int | None,
        ) -> int:
            if (
                journal_first
                and permit._seal is seal
                and table == "raw_sessions"
                and phase == "AFTER"
                and operation == "INSERT"
            ):
                assert not publication_pending
                publication_pending.append((permit, connection, table, phase, operation, old_rowid, new_rowid))
                return 1
            result = consume(permit, connection, table, phase, operation, old_rowid, new_rowid)
            if permit._seal is seal and table == "raw_existence_changes" and phase == "AFTER" and publication_pending:
                observed.append("published-journal-before-root-after")
                result = consume(*publication_pending.pop())
            return result

        monkeypatch.setattr(KnownTierMutationPermit, "_consume_native_effect", publication_order)
        with seal.original_read_snapshot(), seal.source_producer():
            rowid = _stage_raw(seal, mtime=7)
            _stage_raw(seal, mtime=9)
            assert not capture_pending
            with seal._owned_cursor(
                seal._scratch,
                "SELECT old_image IS NULL,new_image IS NOT NULL,parent_effect_id,canonical_trigger "
                "FROM temp.known_tier_effects WHERE tier='source' AND table_name='raw_sessions' ORDER BY ordinal",
            ) as cursor:
                # Exactly one allocated INSERT, then the genuine UPSERT UPDATE.
                assert [tuple(row) for row in cursor] == [(1, 1, None, None), (0, 1, None, None)]
        _publish(seal)
        assert not publication_pending
        assert observed == (
            ["prepared-journal-before-root-after", "published-journal-before-root-after"] if journal_first else []
        )
    with readonly_connection_context(root / "source.db") as connection:
        with closing(connection.execute("SELECT rowid,raw_id,file_mtime_ms FROM raw_sessions")) as cursor:
            assert [tuple(row) for row in cursor] == [(rowid, "journal-raw", 9)]
        with closing(connection.execute("SELECT raw_id FROM raw_existence_changes ORDER BY sequence")) as cursor:
            assert [tuple(row) for row in cursor] == [("journal-raw",), ("journal-raw",)]
