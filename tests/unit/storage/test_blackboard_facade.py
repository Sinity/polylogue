"""Blackboard facade round-trip against a real archive user.db (#1697).

Posts go through the async ``Polylogue`` facade, which writes the
``blackboard_notes`` row in ``user.db``; reads come back through the same
facade with structured decoding and filters applied.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, list_assertions_for_target
from tests.infra.storage_records import db_setup


@pytest.mark.asyncio
async def test_post_then_list_round_trips_through_user_db(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        posted = await poly.post_blackboard_note(
            kind="finding",
            title="FTS unicode61 only",
            content="No porter stemmer compiled in this build.",
            scope_repo="polylogue",
        )
        assert posted.kind == "finding"
        assert posted.scope_repo == "polylogue"
        assert posted.note_id

        notes = await poly.list_blackboard_notes()
        assert [n.note_id for n in notes] == [posted.note_id]
        assert notes[0].title == "FTS unicode61 only"
        assert notes[0].content == "No porter stemmer compiled in this build."


@pytest.mark.asyncio
async def test_post_mirrors_agent_metadata_into_assertion(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        await poly.post_blackboard_note(
            kind="handoff",
            title="resume here",
            content="Merge the green PR.",
            scope_session="codex-session:abc",
            author_ref="agent:codex-session:abc",
            author_kind="agent",
            evidence_refs=("message:m1", "block:m1:2"),
            staleness={"expires_after_days": 7},
            context_policy={"inject": False},
        )

    conn = sqlite3.connect(workspace_env["archive_root"] / "user.db")
    try:
        conn.row_factory = sqlite3.Row
        assertions = list_assertions_for_target(conn, "session:codex-session:abc", kind=AssertionKind.NOTE)
    finally:
        conn.close()

    assert len(assertions) == 1
    assertion = assertions[0]
    assert assertion.author_ref == "agent:codex-session:abc"
    assert assertion.author_kind == "agent"
    assert assertion.evidence_refs == ["message:m1", "block:m1:2"]
    assert assertion.staleness == {"expires_after_days": 7}
    assert assertion.context_policy == {"inject": False}


@pytest.mark.asyncio
async def test_list_filters_by_kind_scope_and_unresolved(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        await poly.post_blackboard_note(kind="finding", title="f", content="c", scope_repo="polylogue")
        await poly.post_blackboard_note(kind="blocker", title="b", content="c", scope_repo="sinex")
        await poly.post_blackboard_note(kind="question", title="q", content="c", scope_repo="polylogue")

        by_kind = await poly.list_blackboard_notes(kind="blocker")
        assert [n.kind for n in by_kind] == ["blocker"]

        by_scope = await poly.list_blackboard_notes(scope_repo="polylogue")
        assert {n.kind for n in by_scope} == {"finding", "question"}

        unresolved = await poly.list_blackboard_notes(unresolved=True)
        assert {n.kind for n in unresolved} == {"blocker", "question"}


@pytest.mark.asyncio
async def test_post_rejects_unknown_kind(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        with pytest.raises(ValueError, match="kind must be one of"):
            await poly.post_blackboard_note(kind="bogus", title="t", content="c")


@pytest.mark.asyncio
async def test_list_on_fresh_archive_is_empty(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
        assert await poly.list_blackboard_notes() == []
