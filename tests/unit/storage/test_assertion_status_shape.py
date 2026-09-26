"""``assertions.status`` is NOT NULL in canonical DDL.

``status TEXT DEFAULT 'active'`` was nullable. No production writer could put a
NULL there -- ``upsert_assertion`` and ``mark_assertion_status`` both route
through ``_normalize_assertion_status``, which resolves an absent status to
``ASSERTION_DEFAULT_STATUS`` -- so the nullability was reachable only by a raw
SQL write, and readers compensated for it with ``COALESCE(status, 'active')``.
Fresh archives create the current canonical shape directly.

The vocabulary half stays where repo policy puts it. ``devtools gate
durable-enum-checks`` refuses any durable-tier membership list whose member set
equals a reachable enum's values, so an ``AssertionStatus``-generated CHECK
cannot live in ``USER_DDL``; ``test_durable_ddl_pins_no_status_vocabulary``
pins that refutation so a later lane does not re-add one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools import verify_durable_enum_checks
from polylogue.core.enums import AssertionStatus
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from polylogue.storage.sqlite.archive_tiers.user_write import (
    mark_assertion_status,
    read_assertion_envelope,
    upsert_assertion,
)

_ASSERTION_COLUMN_COUNT = 18


@pytest.fixture
def fresh_user(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A user tier built by the production bootstrap route and nothing else."""
    path = tmp_path / "user.db"
    initialize_archive_database(path, ArchiveTier.USER)
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def test_fresh_schema_refuses_a_null_status(fresh_user: sqlite3.Connection) -> None:
    """Canonical DDL makes an absent status unrepresentable, not merely unusual.

    Anti-vacuity: restoring ``status TEXT DEFAULT 'active'`` in ``USER_DDL``
    makes this INSERT succeed and the ``pytest.raises`` block fail. A test that
    only read the column back through the ordinary writer would stay green
    under that mutation, because the writer never produces the NULL in the
    first place -- which is why this goes through raw SQL.
    """
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL constraint failed: assertions.status"):
        fresh_user.execute(
            "INSERT INTO assertions VALUES (" + ",".join("?" * _ASSERTION_COLUMN_COUNT) + ")",
            (
                "raw-null",
                None,
                "session:x",
                None,
                "note",
                None,
                None,
                "user:local",
                "user",
                "[]",
                None,
                "private",
                None,
                None,
                '{"inject":false}',
                "[]",
                1,
                2,
            ),
        )


def test_ordinary_write_resolves_an_absent_status(fresh_user: sqlite3.Connection) -> None:
    """The opposite direction: NOT NULL must not turn an ordinary write into a refusal.

    Anti-vacuity: a blanket rejection of the column -- a CHECK that admits
    nothing, or a writer that stopped resolving ``status=None`` -- fails here
    while leaving :func:`test_fresh_schema_refuses_a_null_status` green, so the
    pair pins the constraint rather than a refusal.
    """
    envelope = upsert_assertion(
        fresh_user,
        assertion_id="ordinary-write",
        target_ref="session:ordinary",
        kind="note",
        body_text="no status supplied",
        author_ref="user:test",
        author_kind="user",
        now_ms=1_780_000_000_000,
    )
    assert envelope.status is AssertionStatus.ACTIVE
    assert fresh_user.execute(
        "SELECT status FROM assertions WHERE assertion_id = ?", ("ordinary-write",)
    ).fetchone() == (AssertionStatus.ACTIVE.value,)

    assert mark_assertion_status(fresh_user, "ordinary-write", AssertionStatus.SUPERSEDED) is True
    reread = read_assertion_envelope(fresh_user, "ordinary-write")
    assert reread is not None
    assert reread.status is AssertionStatus.SUPERSEDED


def test_durable_ddl_pins_no_status_vocabulary() -> None:
    """``assertions.status`` carries no membership CHECK, generated or otherwise.

    ``polylogue-lbk1`` was scoped to generate the CHECK from ``AssertionStatus``
    via ``check()``/``nullable_check()``. At this head that is refused by
    ``devtools gate durable-enum-checks``: a durable membership list whose
    member set equals a reachable enum's values pins that enum's evolution into
    durable DDL, so every later token would become a durable migration. The
    vocabulary is owned at the write boundary instead.

    Anti-vacuity: adding ``CHECK ({check('status', AssertionStatus)})`` to the
    ``assertions`` DDL makes the second assertion report an
    ``EnumCheckViolation`` for ``status``/``AssertionStatus`` and turns this
    red -- the same finding ``gate durable-enum-checks`` reports.
    """
    status_declaration = next(line for line in USER_DDL.splitlines() if line.strip().startswith("status "))
    assert "CHECK" not in status_declaration.upper()
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(USER_DDL, tier="user") == []
