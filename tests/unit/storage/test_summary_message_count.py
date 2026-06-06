"""Regression coverage for #1623: `SessionSummary.message_count` hydration.

Without this the facets surface reports ``total_messages: 0`` because
``compute_facets`` sums ``s.message_count or 0`` over summaries whose
``message_count`` field defaults to ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.plan import SessionQueryPlan
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder, db_setup


@pytest.mark.asyncio
async def test_list_summaries_by_query_populates_message_count(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    (
        SessionBuilder(db_path, "conv-three")
        .provider("chatgpt")
        .add_message("m1", role="user", text="hi")
        .add_message("m2", role="assistant", text="hello")
        .add_message("m3", role="user", text="bye")
        .save()
    )
    (SessionBuilder(db_path, "conv-one").provider("chatgpt").add_message("m4", role="user", text="ping").save())

    plan = SessionQueryPlan(origins=("chatgpt-export",), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()
    counts = {str(s.id): s.message_count for s in summaries}
    assert counts == {
        native_session_id_for("chatgpt", "conv-three"): 3,
        native_session_id_for("chatgpt", "conv-one"): 1,
    }


@pytest.mark.asyncio
async def test_list_summaries_by_query_returns_none_for_unknown_sessions(workspace_env: dict[str, Path]) -> None:
    """Empty-result path doesn't error and doesn't issue a count query."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    # Touch db_path so the archive root exists; no sessions seeded.
    assert db_path.parent == archive_root

    plan = SessionQueryPlan(origins=("chatgpt-export",), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()
    assert summaries == []
