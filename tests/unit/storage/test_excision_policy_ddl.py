"""``excision_policy_projections`` belongs to current source DDL.

Fresh source tiers create the projection before any manifest write. Ordinary
publication must not create schema, and the reader can treat an absent row as
an unbound generation without probing whether the table exists.

The test that matters here is ``test_writer_creates_no_durable_shape``: an
authorizer denies every DDL verb for the duration of one ordinary write, so a
restored write-time ``CREATE`` fails with ``not authorized`` even though it is
an ``IF NOT EXISTS`` no-op against a table canonical DDL already made. A test
that only compared ``sqlite_schema`` before and after would stay green under
exactly that mutation, which is why it is not the test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.security.excision_policy import (
    ExcisionPolicySnapshot,
    read_excision_policy_projection,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

PROJECTION_TABLE = "excision_policy_projections"


_DDL_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
    }
)


def _deny_ddl(action: int, _a: object, _b: object, _c: object, _d: object) -> int:
    """Refuse every schema-creating verb; SQLite consults this before IF NOT EXISTS."""
    return sqlite3.SQLITE_DENY if action in _DDL_ACTIONS else sqlite3.SQLITE_OK


def _snapshot(generation_id: str) -> ExcisionPolicySnapshot:
    return ExcisionPolicySnapshot(
        removed_hashes=(bytes(range(32)),),
        assertion_refs=("assertion:excision-policy-ddl",),
        user_generation=7,
        audit_generation=9,
        audit_head="b" * 64,
        source_generation_id=generation_id,
    )


def _publish(conn: sqlite3.Connection, generation_id: str) -> None:
    publish_source_generation(
        conn,
        source_generation_id=generation_id,
        manifest_digest="c" * 64,
        addressing_mode="path",
        coordinates=("fixture/one.jsonl",),
        observed_at_ms=1_780_000_000_000,
        policy_snapshot=_snapshot(generation_id),
    )


@pytest.fixture
def fresh_source(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A source tier built by the production bootstrap route and nothing else."""
    path = tmp_path / "source.db"
    initialize_archive_database(path, ArchiveTier.SOURCE)
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def test_canonical_ddl_owns_the_projection(fresh_source: sqlite3.Connection) -> None:
    """A freshly bootstrapped source tier has the table before anything is written.

    Anti-vacuity: removing the table from ``SOURCE_DDL`` leaves a bootstrapped
    tier without it, and this assertion is the only thing between that and a
    reader whose query raises ``no such table``.
    """
    assert (
        fresh_source.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?", (PROJECTION_TABLE,)
        ).fetchone()
        is not None
    )


def test_writer_creates_no_durable_shape(fresh_source: sqlite3.Connection) -> None:
    """An ordinary manifest write must not execute a single DDL statement.

    Anti-vacuity: restoring the write-time ``CREATE TABLE IF NOT EXISTS`` in
    ``publish_source_generation`` turns this red with ``sqlite3.DatabaseError:
    not authorized``. SQLite runs the authorizer while preparing the statement,
    before it discovers the table already exists, so the mutation cannot hide
    behind ``IF NOT EXISTS`` being a runtime no-op.
    """
    fresh_source.set_authorizer(_deny_ddl)
    try:
        _publish(fresh_source, "authorized-generation")
    finally:
        fresh_source.set_authorizer(None)

    projection = read_excision_policy_projection(fresh_source, "authorized-generation")
    assert projection is not None
    assert projection["policy_digest"] == _snapshot("authorized-generation").digest


def test_reader_needs_no_schema_probe(fresh_source: sqlite3.Connection) -> None:
    """An absent row means "no binding", never "this archive lacks the shape".

    Anti-vacuity: a reader that still short-circuits on a ``sqlite_schema``
    probe answers ``None`` for both rows below, so the published binding is
    indistinguishable from the unpublished one and the first assertion fails.
    """
    _publish(fresh_source, "bound-generation")

    assert read_excision_policy_projection(fresh_source, "bound-generation") is not None
    assert read_excision_policy_projection(fresh_source, "unbound-generation") is None
