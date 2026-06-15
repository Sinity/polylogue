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

import pytest

from polylogue.archive.query.archive_execution import _reject_unexecutable_session_seed
from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.plan import SessionQueryPlan


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
