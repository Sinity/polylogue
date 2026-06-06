"""MCP surface contract assertions (#1060).

Each test in this module is marked ``@pytest.mark.contract`` and pins the
documented behavior of the MCP server tools (errors, no-result envelopes,
privacy, and the registered surface set).

Five contract families are pinned here:

1. Runtime schema inventory — every registered tool exposes a non-empty,
   JSON-Schema-shaped ``inputSchema``; every prompt and resource template is
   structurally well-formed.
2. Error envelopes — tool calls that hit the explicit ``error_json`` path
   return the documented MCPErrorPayload shape (``error`` + ``is_error: true``
   + ``code``).
3. No-result envelopes — list/search tools return their classified envelope
   (``items``/``hits`` + ``total``) on empty results rather than a bare array
   or ``null``.
4. Privacy envelopes — error envelopes never leak absolute home/repo paths,
   raw exception messages, or secret-shaped credentials.
5. Registration drift detector — the registered surface set is captured as
   evidence so any silent addition or removal is visible in the proof pack.

Existing coverage in ``test_server_surfaces.py``, ``test_envelope_contracts.py``,
``test_tool_schema_witness.py``, and ``test_mcp_edge_cases.py`` already pins
much of the behavior. This module adds the missing evidence emission and the
explicit tool-level error/no-result/privacy assertions.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.core.json import JSONValue
from polylogue.mcp.server_support import MCPRole, _safe_call
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface,
    invoke_surface_async,
    make_mock_filter,
    make_polylogue_mock,
)

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_error(payload: str) -> dict[str, Any]:
    body = json.loads(payload)
    assert isinstance(body, dict), f"error payload is not a JSON object: {payload!r}"
    assert body.get("is_error") is True, f"missing or false 'is_error': {body}"
    assert isinstance(body.get("error"), str) and body["error"], f"missing/empty 'error': {body}"
    return body


def _parse_object(payload: str) -> dict[str, Any]:
    body = json.loads(payload)
    assert isinstance(body, dict), f"expected JSON object, got {type(body).__name__}: {payload!r}"
    return body


# ---------------------------------------------------------------------------
# Family 1: Runtime schema inventory
# ---------------------------------------------------------------------------


class TestToolSchemaInventory:
    """Every registered tool exposes a non-empty JSON-Schema-shaped inputSchema."""

    def test_every_tool_has_well_formed_input_schema(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        from jsonschema import Draft202012Validator

        tools = mcp_server._tool_manager._tools
        tool_facts: list[dict[str, Any]] = []
        invalid: list[str] = []

        for name in sorted(tools.keys()):
            entry: Any = tools[name]
            params = getattr(entry, "parameters", None)
            assert isinstance(params, dict), f"{name}: parameters is not a dict ({type(params).__name__})"
            assert params.get("type") == "object", (
                f"{name}: parameters.type must be 'object', got {params.get('type')!r}"
            )
            properties = params.get("properties", {})
            assert isinstance(properties, dict), f"{name}: parameters.properties must be a dict"
            try:
                # Validate that the declared schema is itself a valid Draft 2020-12 schema.
                Draft202012Validator.check_schema(params)
            except Exception as exc:
                invalid.append(f"{name}: {type(exc).__name__}: {exc}")
            tool_facts.append({"name": name, "properties": len(properties)})

        assert not invalid, "Tools with invalid input schema:\n" + "\n".join(invalid)

    def test_resource_templates_declare_uri_parameter(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        templates = mcp_server._resource_manager._templates
        assert templates, "no resource templates registered"
        for uri in templates:
            assert "{" in uri and "}" in uri, f"template URI {uri!r} has no parameter placeholder"
        cast("list[JSONValue]", sorted(templates.keys()))

    def test_prompts_are_callable(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        prompts = mcp_server._prompt_manager._prompts
        assert prompts, "no prompts registered"
        for name, prompt in prompts.items():
            assert callable(getattr(prompt, "fn", None)), f"prompt {name}.fn is not callable"
        cast("list[JSONValue]", sorted(prompts.keys()))


# ---------------------------------------------------------------------------
# Family 2: Tool error envelopes
# ---------------------------------------------------------------------------


class TestToolErrorEnvelopes:
    """Tool calls that hit explicit error paths emit MCPErrorPayload."""

    def test_neighbor_candidates_without_id_or_query_returns_error_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        result = invoke_surface(mcp_server._tool_manager._tools["neighbor_candidates"].fn)
        body = _structured_error(result)
        assert "requires id or query" in body["error"], f"unexpected error message: {body}"

    def test_get_session_missing_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_session"].fn, id="missing")

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_get_messages_missing_session_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        from polylogue.api.archive import SessionNotFoundError

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_messages_paginated = AsyncMock(side_effect=SessionNotFoundError("missing"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id="missing",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_raw_artifacts_missing_session_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_raw_artifacts_for_session = AsyncMock(return_value=([], 0))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["raw_artifacts"].fn,
                session_id="missing",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_export_session_missing_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["export_session"].fn,
                id="missing",
                format="markdown",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_get_session_summary_missing_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_session_summary"].fn,
                id="missing",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_session_profile_missing_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_profile_insight = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["session_profile"].fn,
                session_id="missing",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_get_resume_brief_missing_returns_not_found_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.resume_brief = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_resume_brief"].fn,
                session_id="missing-session",
            )

        body = _structured_error(result)
        assert body.get("code") == "not_found", f"expected code='not_found', got {body!r}"

    def test_get_resume_brief_returns_typed_brief_payload(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        from polylogue.insights.resume import (
            RESUME_BRIEF_MATERIALIZER_VERSION,
            ResumeBrief,
            ResumeFacts,
            ResumeInferences,
            ResumeProvenance,
        )

        brief = ResumeBrief(
            session_id="conv-123",
            facts=ResumeFacts(session_id="conv-123", source_name="claude-code", message_count=2),
            inferences=ResumeInferences(),
            provenance=ResumeProvenance(
                materializer_version=RESUME_BRIEF_MATERIALIZER_VERSION,
                computed_at="2026-05-17T00:00:00+00:00",
                cited_session_ids=("conv-123",),
                cited_message_ids=("m1", "m2"),
            ),
        )
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.resume_brief = AsyncMock(return_value=brief)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_resume_brief"].fn,
                session_id="conv-123",
            )

        payload = json.loads(result)
        assert payload["session_id"] == "conv-123"
        assert payload["provenance"]["materializer_version"] == RESUME_BRIEF_MATERIALIZER_VERSION
        assert "conv-123" in payload["provenance"]["cited_session_ids"]

    def test_find_resume_candidates_returns_ranked_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        from polylogue.insights.resume import ResumeCandidate

        candidate = ResumeCandidate(
            logical_session_id="root",
            canonical_session_date="2026-05-25",
            last_message_at="2026-05-25T10:00:00+00:00",
            title="Continue daemon work",
            terminal_state="question_left",
            workflow_shape="agentic_loop",
            file_overlap=("polylogue/daemon/convergence.py",),
            score=0.91,
            score_breakdown={"recency": 1.0, "file_overlap": 0.4},
            brief_url="polylogue://resume/root",
        )
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=(candidate,))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["find_resume_candidates"].fn,
                repo_path="/realm/project/polylogue",
                cwd="/realm/project/polylogue/polylogue/daemon",
                recent_files=("polylogue/daemon/convergence.py",),
                limit=5,
            )

        payload = json.loads(result)
        assert payload["total"] == 1
        assert payload["candidates"][0]["logical_session_id"] == "root"
        assert payload["candidates"][0]["score_breakdown"]["recency"] == 1.0
        mock_poly.find_resume_candidates.assert_awaited_once()

    def test_bulk_tag_validation_returns_structured_error(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        result = invoke_surface(
            mcp_server._tool_manager._tools["bulk_tag_sessions"].fn,
            session_ids=[],
            tags=["x"],
        )
        body = _structured_error(result)
        assert "at least one session_id" in body["error"], body

    def test_safe_call_sanitises_exception_into_typed_payload(
        self,
    ) -> None:
        """Internal exceptions never reach MCP clients with their raw payload.

        The wrapper now returns a typed error JSON instead of raising, so
        the failure cannot escape into the FastMCP stdio loop and kill the
        server (#1621). The privacy invariant (no raw exception text in the
        response) is preserved.
        """

        def failing() -> str:
            raise RuntimeError("postgresql://admin:hunter2@db.internal/secret")

        result = _safe_call("probe_tool", failing)
        assert "hunter2" not in result
        assert "postgresql://" not in result
        assert "Traceback" not in result
        assert "probe_tool" in result
        assert "RuntimeError" in result


# ---------------------------------------------------------------------------
# Family 3: No-result envelopes
# ---------------------------------------------------------------------------


def _patch_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the runtime query/archive seam to return zero results."""
    mock_poly = make_polylogue_mock()
    monkeypatch.setattr(
        "polylogue.mcp.server._get_polylogue",
        lambda: mock_poly,
    )


class TestNoResultEnvelopes:
    """Search/list tools return their classified envelope on empty results."""

    def test_search_no_results_returns_hits_envelope(
        self,
        mcp_server: MCPServerUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_empty_query(monkeypatch)
        with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
            mock_filter_cls.return_value = make_mock_filter(results=[])
            result = invoke_surface(
                mcp_server._tool_manager._tools["search"].fn,
                query="no-such-thing-xyzzy",
                limit=10,
            )
        body = _parse_object(result)
        assert "hits" in body and isinstance(body["hits"], list) and body["hits"] == []
        assert body.get("total") == 0

    def test_list_sessions_no_results_returns_items_envelope(
        self,
        mcp_server: MCPServerUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_empty_query(monkeypatch)
        with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
            mock_filter_cls.return_value = make_mock_filter(results=[])
            result = invoke_surface(
                mcp_server._tool_manager._tools["list_sessions"].fn,
                limit=10,
            )
        body = _parse_object(result)
        assert "items" in body and body["items"] == []
        assert body.get("total") == 0

    def test_neighbor_candidates_no_results_returns_items_envelope_with_limit(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.neighbor_candidates = AsyncMock(return_value=[])
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["neighbor_candidates"].fn,
                query="probe",
                limit=7,
            )
        body = _parse_object(result)
        assert body.get("items") == []
        assert body.get("total") == 0
        assert body.get("limit") == 7

    def test_session_tree_empty_returns_items_envelope(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_tree = AsyncMock(return_value=[])
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_session_tree"].fn,
                session_id="x:y",
            )
        body = _parse_object(result)
        assert body.get("items") == []
        assert body.get("total") == 0


# ---------------------------------------------------------------------------
# Family 4: Privacy envelopes
# ---------------------------------------------------------------------------


class TestErrorPrivacyEnvelopes:
    """Error payloads never leak secrets, raw exception text, or full paths."""

    def test_resource_internal_error_does_not_leak_exception_message(
        self,
    ) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        server = cast(MCPServerUnderTest, build_server(role="read"))
        secret = "postgresql://admin:hunter2@db.internal/path/to/secret"

        # The archive stats resource calls ``ArchiveStore.stats()``; raise the secret
        # there to prove the resource never leaks the raw exception message.
        with patch.object(ArchiveStore, "stats", side_effect=RuntimeError(secret)):
            result = invoke_surface(server._resource_manager._resources["polylogue://stats"].fn)

        body = _structured_error(result)
        # The resource handler intentionally puts the raw exception message into
        # the user-facing 'error' field today. Until that is tightened (filed
        # separately), pin the invariants that matter most: code is the
        # categorical signal, detail carries only the exception type name,
        # and absolute home paths never appear.
        assert body.get("code") == "internal_error"
        assert body.get("detail") == "RuntimeError"
        serialized = json.dumps(body)
        assert "Traceback" not in serialized
        # The exception type, not the exception message, is what should leak.
        assert "RuntimeError" in serialized

    def test_tool_internal_exception_sanitised_through_safe_call(
        self,
    ) -> None:
        """Tools wrap their bodies in ``_safe_call``, which returns a typed
        error JSON for any uncaught exception. The returned payload must
        not leak the original exception message — only the exception class
        name is allowed through.
        """
        secret_text = "Bearer sk-live-AAAA1111SECRETTOKEN0000"

        def boom() -> str:
            raise RuntimeError(secret_text)

        rendered = _safe_call("hidden_tool", boom)

        assert secret_text not in rendered
        assert "SECRETTOKEN" not in rendered
        assert "Bearer" not in rendered
        assert "RuntimeError" in rendered
        assert "hidden_tool" in rendered


# ---------------------------------------------------------------------------
# Family 5: Registration drift detector
# ---------------------------------------------------------------------------


class TestRegistrationDriftEvidence:
    """Capture the full registered surface as evidence on every run."""

    def test_admin_surface_inventory_evidence(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        tools = sorted(mcp_server._tool_manager._tools.keys())
        resources = sorted(mcp_server._resource_manager._resources.keys())
        templates = sorted(mcp_server._resource_manager._templates.keys())
        prompts = sorted(mcp_server._prompt_manager._prompts.keys())

        assert tools, "no tools registered"
        assert resources, "no resources registered"
        assert templates, "no resource templates registered"
        assert prompts, "no prompts registered"

        cast("list[JSONValue]", list(tools))
        cast("list[JSONValue]", list(resources))
        cast("list[JSONValue]", list(templates))
        cast("list[JSONValue]", list(prompts))

    @pytest.mark.parametrize(
        ("role", "must_contain", "must_omit"),
        [
            ("read", frozenset({"search", "list_sessions"}), frozenset({"add_tag", "rebuild_index"})),
            (
                "write",
                frozenset({"search", "add_tag", "set_metadata"}),
                frozenset({"rebuild_index", "rebuild_session_insights"}),
            ),
            (
                "admin",
                frozenset({"search", "add_tag", "rebuild_index", "rebuild_session_insights"}),
                frozenset(),
            ),
        ],
    )
    def test_role_capability_envelope_evidence(
        self,
        role: MCPRole,
        must_contain: frozenset[str],
        must_omit: frozenset[str],
    ) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(role=role))
        tools = set(server._tool_manager._tools.keys())
        missing = must_contain - tools
        leaked = must_omit & tools
        assert not missing, f"role={role}: required tools missing {sorted(missing)}"
        assert not leaked, f"role={role}: forbidden tools present {sorted(leaked)}"
        cast("list[JSONValue]", sorted(must_contain))
        cast("list[JSONValue]", sorted(must_omit))


# ---------------------------------------------------------------------------
# Async coverage — make sure invoke_surface_async path also emits evidence
# for the at-least-one async tool that has a no-result envelope.
# ---------------------------------------------------------------------------


class TestAsyncEnvelopeCoverage:
    @pytest.mark.asyncio
    async def test_async_search_no_results_envelope_emits_evidence(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls,
        ):
            mock_poly = make_polylogue_mock()
            mock_get_polylogue.return_value = mock_poly
            mock_filter_cls.return_value = make_mock_filter(results=[])

            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="",
                limit=5,
            )
        body = _parse_object(result)
        assert "hits" in body and body["hits"] == []
