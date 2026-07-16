"""Archive execution adapter filter coverage."""

from __future__ import annotations

from polylogue.archive.query.archive_execution import _plan_filter_kwargs
from polylogue.archive.query.expression import compile_expression


def test_archive_filter_kwargs_include_session_id() -> None:
    plan = compile_expression("id:abc123").to_plan()

    assert _plan_filter_kwargs(plan)["session_id"] == "abc123"
