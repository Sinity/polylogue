"""Execution-guard tests for ``near:id:<ref>`` session-seeded similarity (#1842).

The compiler accepts ``near:id:<ref>`` and threads it into
``SessionQuerySpec.similar_session_id`` / ``SessionQueryPlan.similar_session_id``
so surfaces and completion already know the field. Executing it requires a
storage primitive that does not exist yet (a ``VectorProvider`` lookup that
vector-searches a stored session's embeddings rather than re-embedding text).

Until that primitive lands, execution must fail *typed* rather than silently
broaden to an unfiltered listing — no execution branch inspects
``similar_session_id``, so without the guard the plan would degrade into "list
everything". These tests pin that guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.archive_execution import _reject_unexecutable_session_seed
from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.search_hits import search_hits_for_plan
from polylogue.config import Config


def test_guard_passes_through_plain_plan() -> None:
    # No session seed → no-op.
    _reject_unexecutable_session_seed(SessionQueryPlan())
    _reject_unexecutable_session_seed(SessionQueryPlan(similar_text="hello"))


def test_guard_rejects_session_seed_typed() -> None:
    plan = SessionQueryPlan(similar_session_id="abc123")
    with pytest.raises(ExpressionCompileError) as excinfo:
        _reject_unexecutable_session_seed(plan)
    assert excinfo.value.field == "near"
    assert "not executable yet" in str(excinfo.value)


def test_compiled_near_id_plan_is_rejected_at_execution() -> None:
    # End-to-end: a compiled `near:id:` query reaches the guard with the seed set.
    plan = compile_expression("near:id:abc123").to_plan()
    assert plan.similar_session_id == "abc123"
    with pytest.raises(ExpressionCompileError):
        _reject_unexecutable_session_seed(plan)


async def test_search_hits_for_plan_rejects_session_seed_not_silently_empty(tmp_path: Path) -> None:
    """search_hits_for_plan must reject near:id: typed, not return [] (Codex #1899).

    This path checks plan_has_search_hit_evidence() first; a session-seeded plan
    has no FTS/text evidence, so without the guard it silently returned no hits.
    """
    plan = compile_expression("near:id:abc123").to_plan()
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )
    with pytest.raises(ExpressionCompileError):
        await search_hits_for_plan(plan, config)
