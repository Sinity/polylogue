"""Per-tool contract coverage for the MCP mutation-tools surface (#1294).

Pins three durable contracts for every ``@mcp.tool()`` registered by
``polylogue.mcp.server_mutation_tools.register_mutation_tools``:

1. **Argument validation** — each tool has a registered handler, the
   required parameters are mandatory (omitting them raises ``TypeError``),
   and at least one input rejection or happy path returns a structured
   JSON envelope (no raw tracebacks).
2. **Authorization** — every mutation tool is absent from a ``read``-role
   server. ``server_tools.register_tools`` gates the whole mutation
   register on ``role_allows(role, "write")``; this test pins that gate
   per tool so a future regression that promoted any tool to
   read-role registration fails loudly.
3. **Idempotency** — create-twice, delete-missing, update-with-same
   payload return the documented ``status``/``outcome`` shape rather
   than crashing.

Tool names are discovered from the admin-role FastMCP server (not a
hard-coded list) so adding a new ``@mcp.tool()`` without coverage
fails the discovery test until it is added to ``TOOL_MATRIX``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.surfaces.payloads import (
    BulkTagMutationResult,
    DeleteSessionResult,
    MetadataMutationResult,
    TagMutationResult,
)
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface,
    make_polylogue_mock,
)

# ---------------------------------------------------------------------------
# Discovery — tools owned by register_mutation_tools.
# ---------------------------------------------------------------------------


def _discover_mutation_tool_names() -> frozenset[str]:
    """Discover every ``@mcp.tool()`` registered by
    ``register_mutation_tools``.

    Uses set difference between an admin-role server (everything) and a
    server with mutation registration skipped. This dynamic derivation
    means a new mutation tool is automatically picked up.
    """
    from polylogue.mcp.server import build_server

    admin = cast(MCPServerUnderTest, build_server(role="admin"))
    read = cast(MCPServerUnderTest, build_server(role="read"))
    admin_tools = set(admin._tool_manager._tools.keys())
    read_tools = set(read._tool_manager._tools.keys())
    # Mutation tools are in admin but not in read. Maintenance tools are
    # admin-only too, but they live in server_maintenance_tools and are
    # excluded by importing the registered names from the mutation module.
    from polylogue.mcp import server_mutation_tools

    src = inspect.getsource(server_mutation_tools)
    declared: set[str] = set()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("async def ") and "(" in stripped:
            name = stripped[len("async def ") :].split("(", 1)[0]
            if name in admin_tools and name not in read_tools:
                declared.add(name)
    return frozenset(declared)


MUTATION_TOOL_NAMES: frozenset[str] = _discover_mutation_tool_names()


# ---------------------------------------------------------------------------
# Per-tool argument matrix.
#
# Each row: tool name -> dict with
#   "required": set[str] of required kwargs (no defaults)
#   "happy":   minimal happy-path kwargs
#   "wrong_type": kwargs that violate a typed contract (e.g. empty string
#                 where a non-empty string is required, or invalid enum)
# ---------------------------------------------------------------------------

_CONV_ID = "test:conv-mutation"
_LONG_STRING = "x" * 10_000

TOOL_MATRIX: dict[str, dict[str, Any]] = {
    "add_tag": {
        "required": {"session_id", "tag"},
        "happy": {"session_id": _CONV_ID, "tag": "review"},
    },
    "remove_tag": {
        "required": {"session_id", "tag"},
        "happy": {"session_id": _CONV_ID, "tag": "review"},
    },
    "bulk_tag_sessions": {
        "required": {"session_ids", "tags"},
        "happy": {"session_ids": [_CONV_ID], "tags": ["review"]},
    },
    "list_tags": {
        "required": set(),
        "happy": {},
    },
    "list_marks": {
        "required": set(),
        "happy": {},
    },
    "add_mark": {
        "required": {"session_id", "mark_type"},
        "happy": {"session_id": _CONV_ID, "mark_type": "star"},
        "invalid_value": {"session_id": _CONV_ID, "mark_type": "bogus"},
    },
    "remove_mark": {
        "required": {"session_id", "mark_type"},
        "happy": {"session_id": _CONV_ID, "mark_type": "star"},
        "invalid_value": {"session_id": _CONV_ID, "mark_type": "bogus"},
    },
    "list_annotations": {
        "required": set(),
        "happy": {},
    },
    "save_annotation": {
        "required": {"annotation_id", "session_id", "note_text"},
        "happy": {"annotation_id": "ann-1", "session_id": _CONV_ID, "note_text": "hello"},
        "empty_string": {"annotation_id": "", "session_id": _CONV_ID, "note_text": "hello"},
    },
    "delete_annotation": {
        "required": {"annotation_id"},
        "happy": {"annotation_id": "ann-1"},
    },
    "list_saved_views": {
        "required": set(),
        "happy": {},
    },
    "save_saved_view": {
        "required": {"name", "query_json"},
        "happy": {"name": "view-1", "query_json": "{}"},
        "invalid_value": {"name": "view-1", "query_json": "not-json"},
    },
    "delete_saved_view": {
        "required": {"view_id"},
        "happy": {"view_id": "view-1"},
    },
    "list_recall_packs": {
        "required": set(),
        "happy": {},
    },
    "save_recall_pack": {
        "required": {"pack_id", "label"},
        "happy": {"pack_id": "pack-1", "label": "lbl", "payload_json": '{"items":[]}'},
        "invalid_value": {"pack_id": "pack-1", "label": "lbl", "payload_json": "not-json"},
    },
    "delete_recall_pack": {
        "required": {"pack_id"},
        "happy": {"pack_id": "pack-1"},
    },
    "list_workspaces": {
        "required": set(),
        "happy": {},
    },
    "save_workspace": {
        "required": {"workspace_id", "name"},
        "happy": {"workspace_id": "ws-1", "name": "ws"},
        "invalid_value": {"workspace_id": "ws-1", "name": "ws", "mode": "bogus"},
    },
    "delete_workspace": {
        "required": {"workspace_id"},
        "happy": {"workspace_id": "ws-1"},
    },
    "get_metadata": {
        "required": {"session_id"},
        "happy": {"session_id": _CONV_ID},
    },
    "set_metadata": {
        "required": {"session_id", "key", "value"},
        "happy": {"session_id": _CONV_ID, "key": "note", "value": "v"},
        "invalid_value": {"session_id": _CONV_ID, "key": "", "value": "v"},
    },
    "delete_metadata": {
        "required": {"session_id", "key"},
        "happy": {"session_id": _CONV_ID, "key": "note"},
        "invalid_value": {"session_id": _CONV_ID, "key": ""},
    },
    "delete_session": {
        "required": {"session_id"},
        "happy": {"session_id": _CONV_ID, "confirm": True},
        "missing_confirm": {"session_id": _CONV_ID},
    },
    "record_correction": {
        "required": {"session_id", "kind", "payload"},
        "happy": {
            "session_id": _CONV_ID,
            "kind": "summary_override",
            "payload": {"summary": "Operator-authored summary."},
        },
        "invalid_value": {
            "session_id": _CONV_ID,
            "kind": "bogus_kind",
            "payload": {},
        },
    },
    "list_corrections": {
        "required": set(),
        "happy": {},
    },
    "clear_corrections": {
        "required": {"session_id"},
        "happy": {"session_id": _CONV_ID},
    },
    "blackboard_post": {
        "required": {"kind", "title", "content"},
        "happy": {"kind": "finding", "title": "t", "content": "c"},
        "invalid_value": {"kind": "bogus", "title": "t", "content": "c"},
    },
}


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_server() -> MCPServerUnderTest:
    """Admin-role server — every mutation tool is registered."""
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server(role="admin"))


@pytest.fixture
def read_server() -> MCPServerUnderTest:
    """Read-role server — mutation tools must NOT be registered."""
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server(role="read"))


@pytest.fixture
def patched_mutation_seam() -> Iterator[Any]:
    """Patch the polylogue facade and query store seams used by every
    mutation tool. Yields the mock polylogue facade so individual tests
    can override per-method return values.
    """
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_poly:
        poly = make_polylogue_mock(resolved_id=_CONV_ID)
        mock_get_poly.return_value = poly
        yield poly


# ---------------------------------------------------------------------------
# 1. Discovery / matrix coverage.
# ---------------------------------------------------------------------------


class TestMutationToolDiscovery:
    """Every registered mutation tool must appear in TOOL_MATRIX and
    vice versa.
    """

    def test_every_mutation_tool_is_in_matrix(self) -> None:
        missing = MUTATION_TOOL_NAMES - set(TOOL_MATRIX.keys())
        assert not missing, (
            f"Mutation tools registered but missing from TOOL_MATRIX: "
            f"{sorted(missing)}. Add a row covering required params, a "
            f"happy-path payload, and any invalid-value rejection so this "
            f"contract test pins their argument shape."
        )

    def test_no_stale_matrix_entries(self) -> None:
        stale = set(TOOL_MATRIX.keys()) - MUTATION_TOOL_NAMES
        assert not stale, (
            f"TOOL_MATRIX references tools not registered by "
            f"register_mutation_tools: {sorted(stale)}. Remove them or "
            f"restore the tool."
        )

    def test_mutation_tool_count_nonzero(self) -> None:
        # Sanity floor — there are at least 20 mutation tools today
        # (marks, annotations, saved views, recall packs, workspaces,
        # tags, metadata, delete_session, learning corrections).
        assert len(MUTATION_TOOL_NAMES) >= 20, sorted(MUTATION_TOOL_NAMES)


# ---------------------------------------------------------------------------
# 2. Authorization — read role denies every mutation tool.
# ---------------------------------------------------------------------------


class TestMutationToolAuthorization:
    """Pin the per-tool authorization boundary: a ``read``-role server
    must NOT register any mutation tool. ``server_tools.register_tools``
    skips ``register_mutation_tools`` for the read role; this test
    pins that gate per tool so promoting any single tool to read-role
    registration fails loudly.
    """

    @pytest.mark.parametrize("tool_name", sorted(MUTATION_TOOL_NAMES))
    def test_read_role_omits_mutation_tool(self, read_server: MCPServerUnderTest, tool_name: str) -> None:
        assert tool_name not in read_server._tool_manager._tools, (
            f"Tool {tool_name!r} appears in a read-role MCP server. Mutation "
            f"tools must be gated behind the write role. Check "
            f"polylogue.mcp.server_tools.register_tools and the role check."
        )

    @pytest.mark.parametrize("tool_name", sorted(MUTATION_TOOL_NAMES))
    def test_admin_role_registers_mutation_tool(self, admin_server: MCPServerUnderTest, tool_name: str) -> None:
        assert tool_name in admin_server._tool_manager._tools


# ---------------------------------------------------------------------------
# 3. Argument validation matrix.
# ---------------------------------------------------------------------------


class TestMutationToolArgumentContracts:
    """For every mutation tool: required kwargs are mandatory, the
    happy path produces a structured JSON envelope, and at least one
    invalid value produces a structured error envelope (no traceback).
    """

    @pytest.mark.parametrize("tool_name", sorted(MUTATION_TOOL_NAMES))
    def test_tool_is_registered_and_discoverable(self, admin_server: MCPServerUnderTest, tool_name: str) -> None:
        tool = admin_server._tool_manager._tools.get(tool_name)
        assert tool is not None, f"{tool_name} missing from admin-role server"
        assert callable(tool.fn)

    @pytest.mark.parametrize(
        "tool_name",
        sorted(name for name in MUTATION_TOOL_NAMES if TOOL_MATRIX[name]["required"]),
    )
    def test_required_kwargs_are_mandatory(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
        tool_name: str,
    ) -> None:
        """Calling the tool with no kwargs at all must raise TypeError
        for required parameters (FastMCP wraps the handler as a plain
        coroutine; required params have no defaults).
        """
        del patched_mutation_seam  # ensure fixture is applied
        fn = admin_server._tool_manager._tools[tool_name].fn
        with pytest.raises(TypeError):
            invoke_surface(fn)

    @pytest.mark.parametrize("tool_name", sorted(MUTATION_TOOL_NAMES))
    def test_happy_path_returns_json_envelope(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
        tool_name: str,
    ) -> None:
        """Happy path returns a JSON-parseable string. We do not pin
        the full payload shape here — :mod:`test_envelope_contracts`
        owns the registry-wide envelope rule. This pins that the tool
        does not raise on a minimal valid input.
        """
        poly = patched_mutation_seam
        _arrange_poly_for(poly, tool_name)
        fn = admin_server._tool_manager._tools[tool_name].fn
        kwargs = TOOL_MATRIX[tool_name]["happy"]
        result = invoke_surface(fn, **kwargs)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, (dict, list))

    @pytest.mark.parametrize(
        "tool_name",
        sorted(
            name
            for name in MUTATION_TOOL_NAMES
            if "invalid_value" in TOOL_MATRIX[name] or "empty_string" in TOOL_MATRIX[name]
        ),
    )
    def test_invalid_value_returns_structured_error(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
        tool_name: str,
    ) -> None:
        """Invalid value -> structured ``MCPErrorPayload`` (``error``,
        ``is_error: true``). No raw traceback bubbling out."""
        poly = patched_mutation_seam
        _arrange_poly_for(poly, tool_name)
        # record_correction's "bogus_kind" rejection happens in the
        # operations layer (``UnknownCorrectionKindError``). Wire that
        # into the mock so the MCP tool's error path is exercised.
        if tool_name == "record_correction":
            from polylogue.insights.feedback import UnknownCorrectionKindError

            poly.record_correction = AsyncMock(side_effect=UnknownCorrectionKindError("bogus_kind"))
        # blackboard_post's kind validation lives in the facade
        # (``post_blackboard_note`` raises ``ValueError``); wire it into the
        # mock so the MCP tool's error path is exercised.
        if tool_name == "blackboard_post":
            poly.post_blackboard_note = AsyncMock(side_effect=ValueError("kind must be one of [...]"))
        fn = admin_server._tool_manager._tools[tool_name].fn
        kwargs = TOOL_MATRIX[tool_name].get("invalid_value") or TOOL_MATRIX[tool_name]["empty_string"]
        # The contract is: invalid input becomes a structured error
        # envelope OR a typed ``PolylogueError`` (FastMCP marshals the
        # latter into ``isError=True`` for the MCP client). Either way,
        # no raw traceback bubbles out.
        from polylogue.errors import PolylogueError

        try:
            result = invoke_surface(fn, **kwargs)
        except PolylogueError:
            return
        body = json.loads(result)
        assert body.get("is_error") is True, f"{tool_name}: missing is_error in {body}"
        assert "message" in body, f"{tool_name}: missing error in {body}"


# ---------------------------------------------------------------------------
# 4. Idempotency contract.
#
# Create-twice, delete-missing, and update-with-same-payload paths should
# return a documented status/outcome envelope, not crash.
# ---------------------------------------------------------------------------


_IDEMPOTENT_CASES = [
    # (tool_name, kwargs, mock_attr, first_return, second_return,
    #  first_expected_outcome, second_expected_outcome)
    pytest.param(
        "add_tag",
        {"session_id": _CONV_ID, "tag": "review"},
        "add_tag",
        TagMutationResult(outcome="added", detail=None),
        TagMutationResult(outcome="no_op", detail="already_present"),
        "added",
        "no_op",
        id="add_tag_twice",
    ),
    pytest.param(
        "remove_tag",
        {"session_id": _CONV_ID, "tag": "missing"},
        "remove_tag",
        TagMutationResult(outcome="not_present", detail="tag_absent"),
        TagMutationResult(outcome="not_present", detail="tag_absent"),
        "not_present",
        "not_present",
        id="remove_missing_tag",
    ),
    pytest.param(
        "add_mark",
        {"session_id": _CONV_ID, "mark_type": "star"},
        "add_mark",
        True,
        False,
        "added",
        "no_op",
        id="add_mark_twice",
    ),
    pytest.param(
        "remove_mark",
        {"session_id": _CONV_ID, "mark_type": "star"},
        "remove_mark",
        False,
        False,
        "not_present",
        "not_present",
        id="remove_missing_mark",
    ),
    pytest.param(
        "save_annotation",
        {
            "annotation_id": "ann-1",
            "session_id": _CONV_ID,
            "note_text": "n",
        },
        "save_annotation",
        True,
        False,
        "added",
        "updated",
        id="save_annotation_twice",
    ),
    pytest.param(
        "delete_annotation",
        {"annotation_id": "missing-ann"},
        "delete_annotation",
        False,
        False,
        None,
        None,
        id="delete_missing_annotation",
    ),
    pytest.param(
        "delete_saved_view",
        {"view_id": "missing-view"},
        "delete_view",
        False,
        False,
        None,
        None,
        id="delete_missing_saved_view",
    ),
    pytest.param(
        "delete_recall_pack",
        {"pack_id": "missing-pack"},
        "delete_recall_pack",
        False,
        False,
        None,
        None,
        id="delete_missing_recall_pack",
    ),
    pytest.param(
        "delete_workspace",
        {"workspace_id": "missing-ws"},
        "delete_workspace",
        False,
        False,
        None,
        None,
        id="delete_missing_workspace",
    ),
]


class TestMutationToolIdempotency:
    """Pin create-twice, delete-missing, and update-with-same-payload
    behaviors so every mutation tool surfaces a documented outcome
    rather than crashing or returning an inconsistent shape.
    """

    @pytest.mark.parametrize(
        "tool_name,kwargs,mock_attr,first_return,second_return,first_outcome,second_outcome",
        _IDEMPOTENT_CASES,
    )
    def test_idempotent_outcomes(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
        tool_name: str,
        kwargs: dict[str, Any],
        mock_attr: str,
        first_return: Any,
        second_return: Any,
        first_outcome: str | None,
        second_outcome: str | None,
    ) -> None:
        poly = patched_mutation_seam
        # Need separate calls returning different values — patch via
        # side_effect so the second invocation hits the second case.
        method = AsyncMock(side_effect=[first_return, second_return])
        setattr(poly, mock_attr, method)
        # delete_recall_pack/delete_saved_view tool use poly.delete_view /
        # poly.delete_recall_pack — already aliased above. For
        # delete_annotation the tool calls poly.delete_annotation directly.

        fn = admin_server._tool_manager._tools[tool_name].fn

        first_result = json.loads(invoke_surface(fn, **kwargs))
        second_result = json.loads(invoke_surface(fn, **kwargs))

        assert "is_error" not in first_result, first_result
        assert "is_error" not in second_result, second_result

        # When an outcome name is declared, pin it on the relevant call.
        if first_outcome is not None:
            assert first_result.get("outcome") == first_outcome, first_result
        if second_outcome is not None:
            assert second_result.get("outcome") == second_outcome, second_result

        # Both calls must surface a status field (documented envelope).
        assert "status" in first_result
        assert "status" in second_result

    def test_delete_session_requires_confirm(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
    ) -> None:
        """``delete_session`` without ``confirm=true`` must return a
        structured error envelope (safety guard). Confirmed delete must
        succeed and surface the documented outcome.
        """
        poly = patched_mutation_seam
        poly.delete_session_safe = AsyncMock(return_value=DeleteSessionResult(outcome="deleted", session_id=_CONV_ID))
        fn = admin_server._tool_manager._tools["delete_session"].fn

        # Without confirm.
        unconfirmed = json.loads(invoke_surface(fn, session_id=_CONV_ID))
        assert unconfirmed.get("is_error") is True
        assert "message" in unconfirmed

        # With confirm.
        confirmed = json.loads(invoke_surface(fn, session_id=_CONV_ID, confirm=True))
        assert "is_error" not in confirmed
        assert confirmed.get("status") == "deleted"

    def test_set_metadata_update_with_same_payload(
        self,
        admin_server: MCPServerUnderTest,
        patched_mutation_seam: Any,
    ) -> None:
        """``set_metadata`` called twice with the same payload returns a
        documented outcome rather than raising. The first call returns
        ``outcome=set``, the second returns ``outcome=unchanged``.
        """
        poly = patched_mutation_seam
        poly.set_metadata = AsyncMock(
            side_effect=[
                MetadataMutationResult(outcome="set", session_id=_CONV_ID, key="k"),
                MetadataMutationResult(outcome="unchanged", session_id=_CONV_ID, key="k", detail="no_change"),
            ]
        )
        fn = admin_server._tool_manager._tools["set_metadata"].fn

        first = json.loads(invoke_surface(fn, session_id=_CONV_ID, key="k", value="v"))
        second = json.loads(invoke_surface(fn, session_id=_CONV_ID, key="k", value="v"))

        assert first.get("status") == "ok"
        assert second.get("status") == "unchanged"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _arrange_poly_for(poly: Any, tool_name: str) -> None:
    """Configure the polylogue facade mock with a sensible default
    return value for the given tool so the happy-path test does not
    blow up on missing attribute behaviour.
    """
    # Tools that are missing from the shared make_polylogue_mock seam.
    # Wire each as an AsyncMock so ``await poly.method(...)`` succeeds.
    poly.list_recall_packs = AsyncMock(return_value=[])
    poly.create_recall_pack = AsyncMock(return_value=True)
    poly.delete_recall_pack = AsyncMock(return_value=False)
    poly.list_workspaces = AsyncMock(return_value=[])
    poly.save_workspace = AsyncMock(return_value=True)
    poly.delete_workspace = AsyncMock(return_value=False)
    poly.record_correction = AsyncMock()
    poly.list_corrections = AsyncMock(return_value=[])
    poly.clear_corrections = AsyncMock(return_value=0)
    poly.delete_correction = AsyncMock(return_value=False)
    if tool_name == "add_tag":
        poly.add_tag = AsyncMock(return_value=TagMutationResult(outcome="added"))
    elif tool_name == "remove_tag":
        poly.remove_tag = AsyncMock(return_value=TagMutationResult(outcome="removed"))
    elif tool_name == "bulk_tag_sessions":
        poly.bulk_tag_sessions = AsyncMock(
            return_value=BulkTagMutationResult(session_count=1, tag_count=1, affected_count=1, skipped_count=0)
        )
    elif tool_name == "set_metadata":
        poly.set_metadata = AsyncMock(
            return_value=MetadataMutationResult(outcome="set", session_id=_CONV_ID, key="note")
        )
    elif tool_name == "delete_metadata":
        poly.delete_metadata = AsyncMock(
            return_value=MetadataMutationResult(outcome="deleted", session_id=_CONV_ID, key="note")
        )
    elif tool_name == "delete_session":
        poly.delete_session_safe = AsyncMock(return_value=DeleteSessionResult(outcome="deleted", session_id=_CONV_ID))
    elif tool_name == "record_correction":
        from polylogue.insights.feedback import CorrectionKind, LearningCorrection
        from tests.infra.frozen_clock import fixed_now

        poly.record_correction = AsyncMock(
            return_value=LearningCorrection(
                session_id=_CONV_ID,
                kind=CorrectionKind.SUMMARY_OVERRIDE,
                payload={"summary": "Operator-authored summary."},
                note=None,
                created_at=fixed_now(),
            )
        )
    elif tool_name == "list_corrections":
        poly.list_corrections = AsyncMock(return_value=[])
    elif tool_name == "clear_corrections":
        poly.clear_corrections = AsyncMock(return_value=0)
    elif tool_name == "save_workspace":
        poly.save_workspace = AsyncMock(return_value=True)
    elif tool_name == "save_recall_pack":
        poly.create_recall_pack = AsyncMock(return_value=True)
    elif tool_name == "save_saved_view":
        poly.save_view = AsyncMock(return_value=True)
    elif tool_name == "save_annotation":
        poly.save_annotation = AsyncMock(return_value=True)
    elif tool_name == "add_mark":
        poly.add_mark = AsyncMock(return_value=True)
    elif tool_name == "blackboard_post":
        from polylogue.archive.blackboard import BlackboardNote

        poly.post_blackboard_note = AsyncMock(
            return_value=BlackboardNote(
                note_id="note-1",
                kind="finding",
                title="t",
                content="c",
                scope_repo=None,
                target_type=None,
                target_id=None,
                created_at_ms=1,
                updated_at_ms=1,
            )
        )
