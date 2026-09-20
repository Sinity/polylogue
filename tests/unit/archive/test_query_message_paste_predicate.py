"""Message-grain ``has_paste`` predicate in the terminal unit grammar.

polylogue-q54dt: ``/api/paste-browser`` needs paste-bearing MESSAGES and the
declared route could only express paste-bearing SESSIONS
(``sessions.paste_count > 0``, ``storage/sqlite/queries/filter_builder.py``),
so the handler hand-rolled a full-archive walk. ``messages.has_paste`` is a
declared column; this file pins the predicate that finally reads it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import MessageQueryRowPayload, QueryUnitEnvelope
from tests.infra.storage_records import SessionBuilder


def _seed_lagging_aggregate(index_db: Path) -> None:
    """Seed one paste-bearing message whose session aggregate has not caught up.

    This is the exact shape the paste browser was written against: the
    message row carries the evidence while ``sessions.paste_count`` still
    reads zero, so a session-scoped predicate cannot see it.
    """

    (
        SessionBuilder(index_db, "paste-session")
        .provider("claude-code")
        .add_message("typed", role="user", text="typed by hand")
        .add_message("pasted", role="user", text="@@ -1 +1 @@\n-old\n+new")
        .save()
    )
    with sqlite3.connect(index_db) as conn:
        conn.execute("UPDATE messages SET has_paste = 1 WHERE native_id = 'pasted'")
        assert conn.execute("SELECT paste_count FROM sessions").fetchone()[0] == 0


def _native_ids(envelope: QueryUnitEnvelope) -> list[str]:
    rows = [cast(MessageQueryRowPayload, row) for row in envelope.items]
    return [row.message_id.rsplit(":", 1)[-1] for row in rows]


def test_message_has_paste_predicate_reads_the_message_row(workspace_env: dict[str, Path]) -> None:
    """``messages where has_paste:true`` reads ``messages.has_paste``.

    Anti-vacuity: lowering the message-unit ``has_paste`` to the session
    aggregate returns zero rows here (``sessions.paste_count`` is 0), and
    dropping the predicate entirely returns both messages. Either makes the
    single-row assertion red.
    """
    index_db = workspace_env["archive_root"] / "index.db"
    _seed_lagging_aggregate(index_db)

    source = parse_unit_source_expression("messages where has_paste:true")
    assert source is not None
    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="paste", limit=20)

    assert isinstance(envelope, QueryUnitEnvelope)
    assert _native_ids(envelope) == ["pasted"]


def test_message_has_paste_false_selects_the_complement(workspace_env: dict[str, Path]) -> None:
    """``has_paste:false`` is the complement, not an ignored token.

    Anti-vacuity: a lowering that treats every value as truthy (or ignores
    the value and emits ``has_paste = 1``) returns the pasted message and
    this assertion is red.
    """
    index_db = workspace_env["archive_root"] / "index.db"
    _seed_lagging_aggregate(index_db)

    source = parse_unit_source_expression("messages where has_paste:false")
    assert source is not None
    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="typed", limit=20)

    assert isinstance(envelope, QueryUnitEnvelope)
    assert _native_ids(envelope) == ["typed"]


def test_has_paste_is_declared_only_where_the_column_exists() -> None:
    """The field is declared for the message unit only.

    ``blocks`` carries no ``has_paste`` column, so accepting the token there
    would lower to invalid SQL at execution time instead of failing as a
    typed compile error.

    Anti-vacuity: declaring ``has_paste`` on every unit's structural field
    set makes this red.
    """
    with pytest.raises(ExpressionCompileError):
        parse_unit_source_expression("blocks where has_paste:true")
