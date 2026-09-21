"""Thread reads serve the profile the producer wrote, and re-derive nothing.

polylogue-ga6ib / polylogue-6kur AC4. ``session_profiles`` carried
``parent_id``/``is_continuation``; the thread loader used to fill ``parent_id``
back in from ``sessions.parent_session_id`` whenever the stored row was empty,
which is the same authority the profile producer itself reads. A producer that
stopped writing the field would therefore never be noticed -- and the patch had
to guess ``is_continuation=True``, which is wrong for every sidechain that has
a parent.

The condition it papered over already has an owner: ``parent_session_id`` is in
the session-row projection the profile's value-complete staleness binding
digests (``derived/session/input_binding.py``), so a profile whose lineage
moved reads ``stale`` and the ordinary derivation converger re-materializes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derived.session.derivation import inspect_session_profiles
from polylogue.storage.derived.session.refresh import refresh_session_insights_for_session_async
from polylogue.storage.derived.session.threads import load_thread_profile_records_by_root_sync
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection

_CHILD_ID = "codex-session:child"
_PARENT_ID = "codex-session:parent"


def _msg(provider_message_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,
        text=text,
        position=position,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _subagent_child() -> ParsedSession:
    """A spawned subagent: it names a parent but shares no prefix with it."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.SIDECHAIN,
        messages=[_msg("c0", Role.USER, "delegated work", 0), _msg("c1", Role.ASSISTANT, "done", 1)],
    )


def _parent() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[_msg("p0", Role.USER, "p0", 0), _msg("p1", Role.ASSISTANT, "p1", 1)],
    )


def _stored_profile_lineage(index_db: Path) -> tuple[str | None, bool | None]:
    with open_connection(index_db) as conn:
        row = conn.execute(
            "SELECT evidence_payload_json FROM session_profiles WHERE session_id = ?",
            (_CHILD_ID,),
        ).fetchone()
    assert row is not None, "the child's profile row should exist"
    payload = json.loads(row[0])
    return payload.get("parent_id"), payload.get("is_continuation")


async def _materialize_child(index_db: Path) -> None:
    backend = SQLiteBackend(db_path=index_db)
    async with backend.connection() as conn:
        await refresh_session_insights_for_session_async(conn, _CHILD_ID, transaction_depth=1)
        await conn.commit()


def _thread_lineage(index_db: Path, root_id: str) -> list[tuple[str, str | None, bool]]:
    with open_connection(index_db) as conn:
        grouped = load_thread_profile_records_by_root_sync(conn, [root_id])
    return [
        (str(record.session_id), record.evidence_payload.parent_id, record.evidence_payload.is_continuation)
        for record in grouped.get(root_id, [])
    ]


@pytest.mark.asyncio
async def test_thread_read_reports_a_stale_profile_as_written_not_as_recovered(tmp_path: Path) -> None:
    """A profile materialized before its parent arrived must read as it is stored.

    The setup is an ordinary ingest order, not a fabricated row: the subagent
    is written and materialized while its parent is still absent, then the
    parent lands and resolves the edge. ``sessions.parent_session_id`` moves;
    the stored profile does not, and the thread read must say so.

    Anti-vacuity: restore ``_repair_profile_parent_ids`` in
    ``derived/session/threads.py`` and the middle assertion goes red -- the
    read returns ``('codex-session:parent', True)`` for a row that stores
    ``(None, False)``, re-labelling a sidechain as a continuation on the way
    out.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    index_db = archive_root / "index.db"

    with open_connection(index_db) as conn:
        write_parsed_session_to_archive(conn, _subagent_child())
        conn.commit()
    await _materialize_child(index_db)
    assert _stored_profile_lineage(index_db) == (None, False)

    with open_connection(index_db) as conn:
        write_parsed_session_to_archive(conn, _parent())
        conn.commit()
        assert (
            conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (_CHILD_ID,)).fetchone()[0]
            == _PARENT_ID
        )
        root_id = str(
            conn.execute("SELECT root_session_id FROM sessions WHERE session_id = ?", (_CHILD_ID,)).fetchone()[0]
        )

    # The producer has not re-run. The read must not pretend otherwise.
    assert _stored_profile_lineage(index_db) == (None, False)
    assert _thread_lineage(index_db, root_id) == [(_CHILD_ID, None, False)]

    # ... and the drift is not lost: it is the converger's declared condition.
    with open_connection(index_db) as conn:
        assert inspect_session_profiles(
            conn, [_CHILD_ID], materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION
        ) == {_CHILD_ID: "stale"}


@pytest.mark.asyncio
async def test_reconvergence_is_what_makes_the_profile_lineage_current(tmp_path: Path) -> None:
    """The producer's own answer, and it is not the one the read patch guessed.

    Re-materializing through the ordinary route writes ``parent_id`` and keeps
    ``is_continuation`` false, because a spawned sidechain that names a parent
    is not a continuation of it.

    Anti-vacuity: drop ``parent_session_id`` from
    ``SESSION_ROW_PROJECTION_COLUMNS`` and the profile never reads stale, so
    this re-materialization never happens on the daemon route and the first
    assertion here is the only thing that would have caught it.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    index_db = archive_root / "index.db"

    with open_connection(index_db) as conn:
        write_parsed_session_to_archive(conn, _subagent_child())
        conn.commit()
    await _materialize_child(index_db)
    with open_connection(index_db) as conn:
        write_parsed_session_to_archive(conn, _parent())
        conn.commit()
        root_id = str(
            conn.execute("SELECT root_session_id FROM sessions WHERE session_id = ?", (_CHILD_ID,)).fetchone()[0]
        )

    await _materialize_child(index_db)

    assert _stored_profile_lineage(index_db) == (_PARENT_ID, False)
    assert _thread_lineage(index_db, root_id) == [(_CHILD_ID, _PARENT_ID, False)]
    with open_connection(index_db) as conn:
        assert inspect_session_profiles(
            conn, [_CHILD_ID], materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION
        ) == {_CHILD_ID: "valid"}
