"""Archive execution adapter filter coverage."""

from __future__ import annotations

from polylogue.api.archive import _archive_query_kwargs
from polylogue.archive.query.archive_execution import _plan_filter_kwargs
from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.spec import SessionQuerySpec


def test_archive_filter_kwargs_include_session_id() -> None:
    plan = compile_expression("id:abc123").to_plan()

    assert _plan_filter_kwargs(plan)["session_id"] == "abc123"


def test_alternate_query_kwargs_preserve_canonical_structural_filters() -> None:
    spec = SessionQuerySpec.from_params(
        {
            "project": "project-ref",
            "conv_id": "codex-session:abc",
            "root": False,
            "exclude_text": ("secret",),
        },
        strict=True,
    )

    kwargs = _archive_query_kwargs(spec, default_limit=50)

    assert kwargs["project_refs"] == ("project-ref",)
    assert kwargs["session_id"] == "codex-session:abc"
    assert kwargs["root"] is False


def test_alternate_query_kwargs_resolve_implicit_root_filter() -> None:
    spec = SessionQuerySpec.from_params({}, strict=True)

    assert _archive_query_kwargs(spec, default_limit=50)["root"] is True
