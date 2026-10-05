from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_archive_tier,
    initialize_runtime_tier_probe,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.managed_connection import sqlite_connection


def _execute(conn: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()) -> None:
    conn.execute(sql, parameters).close()


def _row(conn: sqlite3.Connection, sql: str) -> tuple[object, ...]:
    cursor = conn.execute(sql)
    try:
        row = cursor.fetchone()
        assert row is not None
        return tuple(row)
    finally:
        cursor.close()


@contextmanager
def _source_schema(runtime: bool) -> Iterator[sqlite3.Connection]:
    with sqlite_connection(":memory:") as conn:
        _execute(conn, "PRAGMA foreign_keys = ON")
        if runtime:
            initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
        else:
            initialize_archive_tier(conn, ArchiveTier.SOURCE)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] if runtime else 1
        assert _row(conn, "PRAGMA user_version") == (expected_version,)
        _execute(
            conn,
            "INSERT INTO source_generations "
            "(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms) "
            "VALUES ('generation', ?, 'whole_member', 1, 0)",
            ("a" * 64,),
        )
        conn.commit()
        yield conn


def _item(conn: sqlite3.Connection, diagnostic: str | None) -> None:
    _execute(
        conn,
        "INSERT INTO source_items (source_generation_id, source_item_id, logical_coordinate, "
        "addressing_mode, disposition, outcome_code, stage, diagnostic, observed_at_ms, updated_at_ms) "
        "VALUES ('generation', 'item', 'coordinate', 'whole_member', 'unsupported', "
        "'unsupported_shape', 'acquisition', ?, 0, 0)",
        (diagnostic,),
    )


def _material(conn: sqlite3.Connection, diagnostic: str | None, *, default: bool = False) -> None:
    columns = "material_id, referrer_ref, source_uri, acquisition_state, retryable, custody, "
    columns += "privacy_classification, acquired_at_ms, created_at_ms"
    values = "'material', 'reference', 'synthetic:material', 'unavailable', 1, 'claimed', 'synthetic', 0, 0"
    parameters: tuple[object, ...] = ()
    if not default:
        columns += ", diagnostic"
        values += ", ?"
        parameters = (diagnostic,)
    _execute(conn, f"INSERT INTO material_observations ({columns}) VALUES ({values})", parameters)


def _member(conn: sqlite3.Connection, diagnostic: str | None, *, default: bool = False) -> None:
    columns = "source_generation_id, source_item_id, entry_ordinal, member_name, disposition, observed_at_ms"
    values = "'generation', 'item', 0, 'member.json', 'refused', 0"
    parameters: tuple[object, ...] = ()
    if not default:
        columns += ", diagnostic"
        values += ", ?"
        parameters = (diagnostic,)
    _execute(conn, f"INSERT INTO source_item_member_dispositions ({columns}) VALUES ({values})", parameters)


@pytest.mark.parametrize("runtime", [False, True], ids=["fresh-v1", "runtime-train"])
@pytest.mark.parametrize("diagnostic", ["x" * 20_000, "\U0001f600é\n" * 7_000])
def test_source_diagnostics_retain_complete_text_through_canonical_schema(runtime: bool, diagnostic: str) -> None:
    """A length cap in baseline or replay DDL refuses this valid evidence."""
    with _source_schema(runtime) as conn:
        _item(conn, diagnostic)
        _material(conn, diagnostic)
        _member(conn, diagnostic)
        conn.commit()
        for table in ("source_items", "material_observations", "source_item_member_dispositions"):
            assert _row(conn, f"SELECT diagnostic, typeof(diagnostic) FROM {table}") == (diagnostic, "text")
        cursor = conn.execute("PRAGMA foreign_key_check")
        try:
            assert cursor.fetchone() is None
        finally:
            cursor.close()


@pytest.mark.parametrize("runtime", [False, True], ids=["fresh-v1", "runtime-train"])
def test_source_diagnostic_nullability_and_defaults_remain_distinct(runtime: bool) -> None:
    with _source_schema(runtime) as conn:
        _item(conn, None)
        assert _row(conn, "SELECT diagnostic, typeof(diagnostic) FROM source_items") == (None, "null")
        for insert in (_material, _member):
            with pytest.raises(sqlite3.IntegrityError):
                insert(conn, None)
            insert(conn, None, default=True)
        assert _row(conn, "SELECT diagnostic FROM material_observations") == ("",)
        assert _row(conn, "SELECT diagnostic FROM source_item_member_dispositions") == ("",)


@pytest.mark.parametrize("runtime", [False, True], ids=["fresh-v1", "runtime-train"])
def test_complete_source_diagnostics_preserve_rollback_and_foreign_key_custody(runtime: bool) -> None:
    diagnostic = "diagnostic" * 3_000
    with _source_schema(runtime) as conn:
        _item(conn, diagnostic)
        _material(conn, diagnostic)
        _member(conn, diagnostic)
        conn.rollback()
        for table in ("source_items", "material_observations", "source_item_member_dispositions"):
            assert _row(conn, f"SELECT count(*) FROM {table}") == (0,)
        with pytest.raises(sqlite3.IntegrityError):
            _member(conn, diagnostic)
        _item(conn, diagnostic)
        _member(conn, diagnostic)
        _material(conn, diagnostic)
        _execute(
            conn,
            "INSERT INTO material_evidence_links "
            "(material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic) "
            "VALUES ('material', 'evidence', 'supports', 'unknown', 1.0, 0, ?)",
            (diagnostic,),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            _execute(conn, "UPDATE source_items SET source_generation_id = 'missing'")
        with pytest.raises(sqlite3.IntegrityError):
            _execute(conn, "UPDATE material_observations SET supersedes_material_id = 'missing'")
        _execute(conn, "DELETE FROM source_generations WHERE source_generation_id = 'generation'")
        assert _row(conn, "SELECT count(*) FROM source_items") == (0,)
        assert _row(conn, "SELECT count(*) FROM source_item_member_dispositions") == (0,)
        _execute(conn, "DELETE FROM material_observations WHERE material_id = 'material'")
        assert _row(conn, "SELECT count(*) FROM material_evidence_links") == (0,)
