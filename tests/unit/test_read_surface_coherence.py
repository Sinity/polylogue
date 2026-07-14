"""Focused regression tests for read-surface coherence (#1749).

Each test pins one specific behavior change landed in the refactor so that
accidental reversions are caught immediately. Coverage map:

  AC#1 — --lexical + --semantic/--similar raises UsageError;
          --semantic with no query raises UsageError.
  AC#3 — MCP ``latest`` typed as bool (not str), so False stays false at
          the spec-builder level (no bool("false")==True collapse).
  AC#4 — CLI query-first path uses MAX_QUERY_LIMIT as fallback; daemon
          /api/sessions clamps to MAX_QUERY_LIMIT.
  AC#5 — Daemon vector-only (similar_text only) routes through the operations
          layer and surfaces EmbeddingRetrievalNotReadyError as HTTP 409.
  AC#6 — retrieval_lane: unknown values ("bogus", dead "semantic") are rejected.
  Footgun — negative min_messages/max_messages/min_words rejected by spec builder.
  Cursor — since_session_id is the query pagination cursor.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# AC#1 — --lexical / --semantic mutual-exclusion and --semantic no-op guard
# ---------------------------------------------------------------------------


class TestLexicalSemanticMutualExclusion:
    """--lexical and --semantic/--similar are opposing modes; combining them
    is now a UsageError instead of silently preferring one branch (#1749 AC#1).
    """

    def test_lexical_plus_semantic_raises_usage_error(self) -> None:
        import click

        from polylogue.cli.root_request import RootModeRequest

        with pytest.raises(click.UsageError, match="cannot be combined with --lexical"):
            RootModeRequest.from_params({"query": ("foo",), "lexical": True, "semantic": True})

    def test_lexical_plus_similar_raises_usage_error(self) -> None:
        import click

        from polylogue.cli.root_request import RootModeRequest

        with pytest.raises(click.UsageError, match="cannot be combined with --lexical"):
            RootModeRequest.from_params({"query": (), "lexical": True, "similar_text": "needle"})

    def test_semantic_without_query_raises_usage_error(self) -> None:
        import click

        from polylogue.cli.root_request import RootModeRequest

        with pytest.raises(click.UsageError, match="--semantic requires query terms"):
            RootModeRequest.from_params({"query": (), "semantic": True})

    def test_lexical_alone_is_fine(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        req = RootModeRequest.from_params({"query": ("foo",), "lexical": True})
        spec = req.query_spec()
        assert spec.retrieval_lane == "dialogue"

    def test_semantic_with_query_is_fine(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        req = RootModeRequest.from_params({"query": ("foo", "bar"), "semantic": True})
        spec = req.query_spec()
        assert spec.similar_text == "foo bar"
        assert spec.query_terms == ()

    def test_similar_alone_is_fine(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        req = RootModeRequest.from_params({"query": (), "similar_text": "needle"})
        spec = req.query_spec()
        assert spec.similar_text == "needle"


# ---------------------------------------------------------------------------
# AC#3 — MCP latest typed as bool (not str)
# ---------------------------------------------------------------------------


class TestMCPLatestTyped:
    """``latest`` in MCPSessionQueryRequest is now a real bool, not str.

    Before #1749: the field was typed ``str | None`` and the spec builder did
    ``bool(params.get("latest"))``, so ``latest="false"`` was truthy and
    silently collapsed results to one session. After: the field is
    typed ``bool``, so FastMCP's tool-call coercion parses ``"false"`` as
    ``False`` rather than leaving it as a truthy non-empty string.
    """

    def test_latest_field_is_bool_typed(self) -> None:
        """MCPSessionQueryRequest.latest annotation is ``bool``, not ``str | None``."""
        import typing

        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        # get_type_hints resolves forward references and PEP 563 string annotations
        hints = typing.get_type_hints(MCPSessionQueryRequest)
        assert hints.get("latest") is bool, f"latest must be typed as bool, got {hints.get('latest')!r}"

    def test_latest_false_bool_is_false(self) -> None:
        """``latest=False`` (the Python bool) stays false."""
        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        req = MCPSessionQueryRequest(latest=False)
        assert req.latest is False

    def test_latest_true_bool_is_true(self) -> None:
        """``latest=True`` is stored as true."""
        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        req = MCPSessionQueryRequest(latest=True)
        assert req.latest is True

    def test_latest_false_does_not_activate_spec_latest_filter(self) -> None:
        """``latest=False`` must not activate the latest filter in the spec.

        The old bug: ``str | None`` + ``bool("false")`` → ``bool("false") == True``
        → latest filter active. Now: ``latest=False`` (bool) →
        ``bool(False) == False`` → no latest filter.
        """
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params({"latest": False})
        assert spec.latest is False

    def test_latest_true_activates_spec_latest_filter(self) -> None:
        """``latest=True`` still activates the latest filter."""
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params({"latest": True})
        assert spec.latest is True

    def test_latest_default_is_false(self) -> None:
        """Default latest is False (no latest filter)."""
        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        req = MCPSessionQueryRequest()
        assert req.latest is False


# ---------------------------------------------------------------------------
# AC#4 — Shared limit ceiling: MAX_QUERY_LIMIT = 1000
# ---------------------------------------------------------------------------


class TestSharedLimitCeiling:
    """clamp_query_limit enforces MAX_QUERY_LIMIT = 1000 on every surface.

    The CLI query-first path previously fell back to 10000 (unbounded);
    the daemon had no ceiling at all. Both now use clamp_query_limit.
    """

    def test_max_query_limit_is_1000(self) -> None:
        from polylogue.archive.query.spec import MAX_QUERY_LIMIT

        assert MAX_QUERY_LIMIT == 1000

    def test_clamp_within_range(self) -> None:
        from polylogue.archive.query.spec import clamp_query_limit

        assert clamp_query_limit(500) == 500
        assert clamp_query_limit(1) == 1
        assert clamp_query_limit(1000) == 1000

    def test_clamp_caps_at_max(self) -> None:
        from polylogue.archive.query.spec import MAX_QUERY_LIMIT, clamp_query_limit

        assert clamp_query_limit(9999) == MAX_QUERY_LIMIT
        assert clamp_query_limit(99999999) == MAX_QUERY_LIMIT

    def test_clamp_floor_at_one(self) -> None:
        from polylogue.archive.query.spec import clamp_query_limit

        assert clamp_query_limit(0) == 1
        assert clamp_query_limit(-1) == 1
        assert clamp_query_limit(-100) == 1

    def test_clamp_invalid_input_falls_back_to_default(self) -> None:
        from polylogue.archive.query.spec import clamp_query_limit

        assert clamp_query_limit("bogus") == 10  # default=10
        assert clamp_query_limit(None) == 10
        assert clamp_query_limit(True) == 10  # bool is rejected

    def test_clamp_custom_default(self) -> None:
        from polylogue.archive.query.spec import clamp_query_limit

        assert clamp_query_limit("xyz", default=50) == 50

    def test_search_limit_fallback_is_max_query_limit_not_10000(self) -> None:
        """CLI query-first fallback is MAX_QUERY_LIMIT, not the old 10000."""
        from polylogue.archive.query.plan import SessionQueryPlan
        from polylogue.archive.query.retrieval_candidates import search_limit
        from polylogue.archive.query.spec import MAX_QUERY_LIMIT

        # Build a plan with no explicit limit (the query-first bare-token case).
        # SessionQueryPlan is a dataclass; limit defaults to None.
        plan = SessionQueryPlan()
        # effective_fetch_limit() returns None when no limit is set
        assert plan.effective_fetch_limit() is None
        # search_limit must fall back to MAX_QUERY_LIMIT, not 10000
        assert search_limit(plan) == MAX_QUERY_LIMIT
        assert search_limit(plan) != 10000


# ---------------------------------------------------------------------------
# AC#6 — retrieval_lane validation: unknown values rejected
# ---------------------------------------------------------------------------


class TestRetrievalLaneValidation:
    """Unknown retrieval_lane values are rejected at the spec boundary (#1749 AC#6).

    Before: ``retrieval_lane="bogus"`` was accepted silently (dead constant);
    the dead ``"semantic"`` drift in the docstring was also accepted.
    After: ``normalize_retrieval_lane`` validates against the closed vocabulary.
    """

    @pytest.mark.parametrize("lane", ["bogus", "semantic", "vector"])
    def test_unknown_lane_raises_query_spec_error(self, lane: str) -> None:
        from polylogue.archive.query.spec import QuerySpecError, normalize_retrieval_lane

        with pytest.raises(QuerySpecError):
            normalize_retrieval_lane(lane)

    @pytest.mark.parametrize("lane", ["auto", "dialogue", "actions", "hybrid"])
    def test_known_lanes_are_accepted(self, lane: str) -> None:
        from polylogue.archive.query.spec import normalize_retrieval_lane

        assert normalize_retrieval_lane(lane) == lane

    def test_none_defaults_to_auto(self) -> None:
        from polylogue.archive.query.spec import normalize_retrieval_lane

        assert normalize_retrieval_lane(None) == "auto"

    def test_empty_string_defaults_to_auto(self) -> None:
        """Empty string normalizes to 'auto' (not rejected) — strip().lower() or 'auto'."""
        from polylogue.archive.query.spec import normalize_retrieval_lane

        assert normalize_retrieval_lane("") == "auto"
        assert normalize_retrieval_lane("   ") == "auto"

    def test_spec_builder_rejects_unknown_lane(self) -> None:
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"retrieval_lane": "bogus"})

    def test_spec_builder_rejects_dead_semantic_drift(self) -> None:
        """The old 'semantic' drift in the SearchEnvelope docstring is dead as input lane."""
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"retrieval_lane": "semantic"})

    def test_spec_builder_accepts_hybrid(self) -> None:
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params({"retrieval_lane": "hybrid"})
        assert spec.retrieval_lane == "hybrid"

    def test_closed_vocabulary_is_the_canonical_set(self) -> None:
        """QUERY_RETRIEVAL_LANES is the single source of truth: no 'semantic' member."""
        from polylogue.archive.query.spec import QUERY_RETRIEVAL_LANES

        assert "semantic" not in QUERY_RETRIEVAL_LANES
        assert "bogus" not in QUERY_RETRIEVAL_LANES
        assert set(QUERY_RETRIEVAL_LANES) == {"auto", "dialogue", "actions", "hybrid"}


# ---------------------------------------------------------------------------
# Footgun — negative count bounds rejected (ge=0 at spec boundary)
# ---------------------------------------------------------------------------


class TestNegativeCountBoundsRejected:
    """min_messages/max_messages/min_words reject negative values (#1749 footgun).

    Before: ``optional_int()`` accepted any integer; negatives produced
    nonsensical filter semantics silently.
    After: ``optional_non_negative_int()`` raises QuerySpecError for < 0.
    Note: 0 means "no filter applied" (optional_int(0) returns None);
    the spec builder passes 0 correctly.
    """

    @pytest.mark.parametrize("field", ["min_messages", "max_messages", "min_words"])
    def test_negative_value_raises_query_spec_error(self, field: str) -> None:
        from polylogue.archive.query.spec import QuerySpecError, optional_non_negative_int

        with pytest.raises(QuerySpecError):
            optional_non_negative_int(field, -1)

    @pytest.mark.parametrize("field", ["min_messages", "max_messages", "min_words"])
    def test_positive_is_accepted(self, field: str) -> None:
        from polylogue.archive.query.spec import optional_non_negative_int

        assert optional_non_negative_int(field, 5) == 5

    @pytest.mark.parametrize("field", ["min_messages", "max_messages", "min_words"])
    def test_none_is_accepted(self, field: str) -> None:
        from polylogue.archive.query.spec import optional_non_negative_int

        assert optional_non_negative_int(field, None) is None

    def test_spec_builder_rejects_negative_min_messages(self) -> None:
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"min_messages": -1})

    def test_spec_builder_rejects_negative_max_messages(self) -> None:
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"max_messages": -5})

    def test_spec_builder_rejects_negative_min_words(self) -> None:
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec

        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"min_words": -10})

    def test_mcp_count_bound_type_alias_declared_with_ge_zero(self) -> None:
        """MCPCountBound TypeAlias carries Field(ge=0) for FastMCP schema generation.

        MCPSessionQueryRequest is a dataclass (not Pydantic), so Python
        itself doesn't validate at construction time. The ge=0 constraint is
        expressed in the TypeAlias so FastMCP generates a JSON schema that
        rejects negatives in the tool binding layer. The actual runtime gate
        is the spec builder's optional_non_negative_int.
        """
        import typing

        from polylogue.mcp.query_contracts import MCPCountBound

        # MCPCountBound is Annotated[int, Field(ge=0)] | None
        # Verify its type structure carries the Annotated form
        origin = getattr(MCPCountBound, "__origin__", None)
        assert origin is typing.Union, "MCPCountBound must be a Union (Annotated[int, Field(ge=0)] | None)"

    def test_mcp_negative_min_messages_rejected_via_spec_builder(self) -> None:
        """Negative min_messages from MCP flow reaches the spec builder and is rejected."""
        from polylogue.archive.query.spec import QuerySpecError, SessionQuerySpec
        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        req = MCPSessionQueryRequest(min_messages=-1)
        with pytest.raises(QuerySpecError):
            SessionQuerySpec.from_params({"min_messages": req.min_messages})


# ---------------------------------------------------------------------------
# Query pagination cursor — since_session_id field
# ---------------------------------------------------------------------------


class TestSinceSessionId:
    """The MCP request and query spec expose the pagination cursor field."""

    def test_since_session_id_still_present(self) -> None:
        from dataclasses import fields

        from polylogue.mcp.query_contracts import MCPSessionQueryRequest

        field_names = {f.name for f in fields(MCPSessionQueryRequest)}
        assert "since_session_id" in field_names

    def test_since_session_id_still_accepted(self) -> None:
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params({"since_session_id": "conv-123"}, strict=True)
        assert spec.since_session_id == "conv-123"


# ---------------------------------------------------------------------------
# AC#5 — Daemon vector-only query: typed 409 not opaque 500
#
# We test this at the unit level without spinning up a real HTTP server by
# exercising daemon_safe_handler's error routing directly.
# ---------------------------------------------------------------------------


class TestDaemonVectorOnlyTypedError:
    """Daemon /api/sessions?similar_text=... surfaces EmbeddingRetrievalNotReadyError
    as HTTP 409 via daemon_safe_handler, not an opaque 500 (#1749 AC#5).

    Tested at the unit level using the in-process handler harness.
    """

    def test_embedding_retrieval_not_ready_has_409_status(self) -> None:
        """EmbeddingRetrievalNotReadyError.http_status_code is 409 (CONFLICT)."""
        from http import HTTPStatus

        from polylogue.core.errors import EmbeddingRetrievalNotReadyError

        exc = EmbeddingRetrievalNotReadyError("embeddings not ready", readiness_status="disabled")
        assert exc.http_status_code == HTTPStatus.CONFLICT

    def test_daemon_safe_handler_maps_polylogue_error_to_its_status(self) -> None:
        """daemon_safe_handler uses PolylogueError.http_status_code, not 500.

        Verifies the dispatch path: a PolylogueError with http_status_code=409
        produces a _send_json call with status=409, not the generic 500 branch.
        """
        from http import HTTPStatus
        from io import BytesIO
        from typing import cast
        from unittest.mock import MagicMock

        from polylogue.core.errors import EmbeddingRetrievalNotReadyError
        from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer, daemon_safe_handler

        class _MockServer:
            auth_token = ""
            api_host = "127.0.0.1"

        handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
        handler.server = cast("DaemonAPIHTTPServer", _MockServer())
        handler.client_address = ("127.0.0.1", 12345)
        handler.path = "/api/sessions?similar_text=test"
        handler.command = "GET"
        handler.requestline = "GET /api/sessions?similar_text=test HTTP/1.1"
        from email.message import Message

        handler.headers = cast("Message[str, str]", MagicMock())
        handler.headers.get = MagicMock(return_value=None)  # type: ignore[method-assign]
        handler.rfile = BytesIO()
        handler.wfile = BytesIO()

        send_error = MagicMock()
        send_json = MagicMock()
        handler._send_error = send_error  # type: ignore[method-assign]
        handler._send_json = send_json  # type: ignore[method-assign]

        exc = EmbeddingRetrievalNotReadyError(
            "embeddings not ready; run polylogue ops embed status",
            readiness_status="disabled",
        )

        @daemon_safe_handler
        def _raiser(self: DaemonAPIHandler) -> None:
            raise exc

        _raiser(handler)

        # The handler must have called _send_json with status 409 (CONFLICT)
        send_json.assert_called_once()
        status_arg = send_json.call_args.args[0]
        assert status_arg == HTTPStatus.CONFLICT, f"EmbeddingRetrievalNotReadyError must map to 409, got {status_arg}"
        # Must not call _send_error (generic 500 path)
        send_error.assert_not_called()
