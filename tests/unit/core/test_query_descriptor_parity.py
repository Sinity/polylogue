"""Query descriptor parity tests for the shared query substrate.

Proves that query-field names are consistent across CLI, MCP, and API surfaces.
"""

from __future__ import annotations

from dataclasses import fields

from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS
from polylogue.mcp.query_contracts import MCPSessionQueryRequest


def test_descriptor_mcp_names_match_mcp_query_request_fields() -> None:
    """Every query-bearing MCPSessionQueryRequest field has descriptor metadata."""
    mcp_fields = {field.name for field in fields(MCPSessionQueryRequest)}
    declared: set[str] = set()
    for d in QUERY_FIELD_DESCRIPTORS:
        declared.update(d.mcp_names)
    control_fields = {"limit", "offset", "sort", "include_affordances"}
    query_fields = mcp_fields - control_fields
    assert query_fields - declared == set()


def test_all_descriptors_have_mcp_or_api_name() -> None:
    """Public selection filters declare at least one surface name."""
    missing = []
    for d in QUERY_FIELD_DESCRIPTORS:
        if d.selection_filter and not d.mcp_names and not d.api_names:
            missing.append(d.name)
    internal_plan_only = {
        "parent_id",
        "continuation",
        "sidechain",
        "root",
        "has_branches",
        "predicates",
    }
    actionable = [m for m in missing if m not in internal_plan_only]
    if actionable:
        raise AssertionError(f"Selection filters without surface names: {actionable}")


def test_session_query_spec_rejects_path_traversal() -> None:
    """cwd_prefix with ../ or absolute path must raise QuerySpecError."""
    from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

    dangerous = ["../escape", "foo/../../bar", ".", ""]
    for value in dangerous:
        try:
            SessionQuerySpec.from_params({"cwd_prefix": value}, strict=True)
            # Must not succeed for dangerous values
            if value not in {".", ""}:
                raise AssertionError(f"cwd_prefix={value!r} should have raised QuerySpecError")
        except QuerySpecError:
            pass


def test_session_query_spec_accepts_safe_paths() -> None:
    """Normal relative paths in cwd_prefix should pass validation."""
    from polylogue.archive.query.spec import SessionQuerySpec

    safe = ["realm/project/polylogue", "home/user", None]
    for value in safe:
        spec = SessionQuerySpec.from_params(
            {"cwd_prefix": value} if value is not None else {},
            strict=True,
        )
        assert spec.cwd_prefix == value


def test_session_query_spec_rejects_unknown_params_in_strict_mode() -> None:
    """Unknown parameter names must raise QuerySpecError in strict mode."""
    from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

    try:
        SessionQuerySpec.from_params({"providr": "chatgpt"}, strict=True)
        raise AssertionError("typo 'providr' should have been rejected")
    except QuerySpecError:
        pass


def test_boundary_alternatives_can_be_consumed_by_the_requested_spec() -> None:
    from polylogue.archive.query.fields import query_boundary_alternatives
    from polylogue.archive.query.spec import SessionQuerySpec

    for supplied, accepted, spec in [
        ("origin", "origins", SessionQuerySpec(origins=("claude-code-session",))),
        ("conv_id", "session_id", SessionQuerySpec(session_id="neutral")),
        ("query", "query_terms", SessionQuerySpec(query_terms=("neutral",))),
    ]:
        alternatives = query_boundary_alternatives(supplied, "spec")
        assert accepted in alternatives
        assert supplied not in alternatives
        for name in alternatives:
            assert name in {field.name for field in fields(SessionQuerySpec)}
            assert getattr(spec, name)
    assert query_boundary_alternatives("unknown_operand", "spec") == ()
    assert "origin" in query_boundary_alternatives("origins", "dsl")


def test_dsl_boundary_names_are_executable_grammar_names() -> None:
    from polylogue.archive.query.expression import compile_expression
    from polylogue.archive.query.fields import query_boundary_names

    names = query_boundary_names("dsl")
    assert {"has", "id", "repo", "date"} <= names
    assert not {"parent_id", "predicates", "repo_names"} & names
    for expression in ["has:tools", "id:neutral", "repo:neutral"]:
        compile_expression(expression)


def test_strict_parameter_error_suggests_an_executable_parameter() -> None:
    import pytest

    from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

    with pytest.raises(QuerySpecError) as error:
        SessionQuerySpec.from_params({"origins": "claude-code-session"}, strict=True)
    assert "accepted alternatives: origin" in str(error.value)
    assert SessionQuerySpec.from_params({"origin": "claude-code-session"}, strict=True).origins == (
        "claude-code-session",
    )
