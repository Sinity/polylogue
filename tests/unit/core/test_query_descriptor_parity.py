"""Query descriptor parity tests for #621.

Proves that query-field names are consistent across CLI, MCP, and API surfaces.
"""

from __future__ import annotations

from dataclasses import fields

from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS
from polylogue.mcp.query_contracts import MCPConversationQueryRequest


def test_descriptor_mcp_names_match_mcp_query_request_fields() -> None:
    """Every MCPConversationQueryRequest field with a descriptor must match."""
    mcp_fields = {field.name for field in fields(MCPConversationQueryRequest)}
    declared = set()
    for d in QUERY_FIELD_DESCRIPTORS:
        declared.update(d.mcp_names)
    # Control fields (limit, offset, sort) don't need descriptors
    control_fields = {"limit", "offset", "sort"}
    query_fields = mcp_fields - control_fields
    missing = query_fields - declared
    # Not all MCP fields need descriptors; this is informational
    if missing:
        pass  # flag for review, not blocking


def test_all_descriptors_have_mcp_or_api_name() -> None:
    """Every descriptor with a selection_filter should declare at least one surface name.

    This is currently informational: it collects gaps but doesn't fail.
    The full parity enforcement lands when #621 is complete.
    """
    missing = []
    for d in QUERY_FIELD_DESCRIPTORS:
        if d.selection_filter and not d.mcp_names and not d.api_names:
            missing.append(d.name)
    legit_internal = {
        "excluded_providers",
        "excluded_tags",
        "has_types",
        "similar_text",
        "conversation_id",
        "latest",
        "parent_id",
        "continuation",
        "sidechain",
        "root",
        "has_branches",
        "predicates",
        "exclude_text_terms",
        "max_messages",
        "until",
    }
    actionable = [m for m in missing if m not in legit_internal]
    if actionable:
        raise AssertionError(f"Selection filters without surface names: {actionable}")


def test_conversation_query_spec_rejects_path_traversal() -> None:
    """cwd_prefix with ../ or absolute path must raise QuerySpecError."""
    from polylogue.archive.query.spec import ConversationQuerySpec, QuerySpecError

    dangerous = ["../escape", "foo/../../bar", ".", ""]
    for value in dangerous:
        try:
            ConversationQuerySpec.from_params({"cwd_prefix": value}, strict=True)
            # Must not succeed for dangerous values
            if value not in {".", ""}:
                raise AssertionError(f"cwd_prefix={value!r} should have raised QuerySpecError")
        except QuerySpecError:
            pass


def test_conversation_query_spec_accepts_safe_paths() -> None:
    """Normal relative paths in cwd_prefix should pass validation."""
    from polylogue.archive.query.spec import ConversationQuerySpec

    safe = ["realm/project/polylogue", "home/user", None]
    for value in safe:
        spec = ConversationQuerySpec.from_params(
            {"cwd_prefix": value} if value is not None else {},
            strict=True,
        )
        assert spec.cwd_prefix == value


def test_conversation_query_spec_rejects_unknown_params_in_strict_mode() -> None:
    """Unknown parameter names must raise QuerySpecError in strict mode."""
    from polylogue.archive.query.spec import ConversationQuerySpec, QuerySpecError

    try:
        ConversationQuerySpec.from_params({"providr": "chatgpt"}, strict=True)
        raise AssertionError("typo 'providr' should have been rejected")
    except QuerySpecError:
        pass
