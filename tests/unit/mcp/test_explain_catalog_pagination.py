"""Regression coverage for polylogue-3k30.

``explain(subject="result")``/``explain(subject="recovery")`` serialize the
full ``QUERY_DISCOVERY_EXAMPLES`` catalog (106 examples, ~82.5KB) inside one
``MCPRootPayload`` -- comfortably over ``MCP_RESPONSE_BUDGET_BYTES`` (25000).

Before the fix, two independent defects compounded into an unrecoverable
loop:

1. ``_bounded_item_page`` only searched for a *single* attribute named
   ``items``/``messages``/``hits`` on the payload. ``MCPRootPayload`` is a
   pydantic ``RootModel`` wrapping a plain dict, so none of those attributes
   ever resolved (``getattr(payload, "items", None)`` is always ``None``),
   and the dict in question additionally carries multiple list-valued keys
   (``result_semantics``, ``examples``, ``read_views``) even if attribute
   lookup had worked. The trimmer therefore always returned ``page=None``.
2. The synthesized continuation for a budget-exceeded response came from
   ``_fallback_response_arguments(fn_name, session_id)``, which only knows
   ``session_id`` -- for a non-session-scoped tool call like
   ``explain(subject="result")`` it returns ``{}``, silently dropping
   ``subject`` entirely. Retrying ``continuation.tool``/
   ``continuation.arguments`` therefore replayed the identical oversized,
   argument-losing call forever.

This module exercises the real, wired MCP ``explain`` tool (not a
hand-rolled replica) end-to-end through as many continuation hops as it
takes, and asserts every example in the catalog is retrieved exactly once.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.infra.mcp import MCPServerUnderTest, invoke_surface


def _explain_examples_via_continuation(
    mcp_server: MCPServerUnderTest, *, subject: str, max_steps: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive ``explain(subject=...)`` through continuations to exhaustion.

    Returns ``(examples, steps)`` where ``steps`` records each hop's raw
    envelope body for assertions on progress/argument preservation.
    """
    fn = mcp_server._tool_manager._tools["explain"].fn
    examples: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    call_kwargs: dict[str, Any] = {"subject": subject}

    for _ in range(max_steps):
        raw = invoke_surface(fn, **call_kwargs)
        body = json.loads(raw)
        steps.append(body)
        if body.get("budget_exceeded"):
            page = body["page"]
            assert page is not None, (
                f"budget-exceeded response returned no bounded page at all "
                f"(the original zero-progress defect): {body!r}"
            )
            examples.extend(page["examples"])
            continuation = body["continuation"]
            assert continuation is not None, f"budget-exceeded response has no continuation: {body!r}"
            assert continuation["tool"] == "explain", f"continuation routes to the wrong tool: {continuation!r}"
            call_kwargs = dict(continuation["arguments"])
            continue
        examples.extend(body["examples"])
        return examples, steps

    pytest.fail(f"explain(subject={subject!r}) did not terminate within {max_steps} continuation hops")


class TestExplainCatalogPagination:
    """``explain(subject="result"|"recovery")`` pages through continuation."""

    @pytest.mark.parametrize("subject", ["result", "recovery"])
    def test_full_catalog_is_retrieved_exactly_once(self, mcp_server: MCPServerUnderTest, subject: str) -> None:
        from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES

        examples, steps = _explain_examples_via_continuation(mcp_server, subject=subject)

        # More than one hop proves the response really did exceed the byte
        # budget (this is not exercising some already-small payload).
        assert len(steps) > 1, "expected the oversized catalog to require multiple continuation hops"

        keys = [example["key"] for example in examples]
        assert len(keys) == len(QUERY_DISCOVERY_EXAMPLES), (
            f"expected every one of {len(QUERY_DISCOVERY_EXAMPLES)} examples, got {len(keys)}"
        )
        assert len(set(keys)) == len(keys), "continuation retrieved a duplicate example"
        assert set(keys) == {example.key for example in QUERY_DISCOVERY_EXAMPLES}, (
            "continuation retrieved a different example set than the declared catalog (skip/mismatch)"
        )

    def test_continuation_preserves_subject_and_advances_offset(self, mcp_server: MCPServerUnderTest) -> None:
        """The defect's second half: continuation must not drop ``subject``."""
        fn = mcp_server._tool_manager._tools["explain"].fn
        raw = invoke_surface(fn, subject="result")
        body = json.loads(raw)

        assert body["budget_exceeded"] is True, "expected the full catalog to exceed the MCP response budget"
        continuation = body["continuation"]
        assert continuation is not None

        arguments = continuation["arguments"]
        assert arguments.get("subject") == "result", (
            f"continuation dropped the original 'subject' argument -- retrying would repeat the identical "
            f"oversized call forever: {arguments!r}"
        )
        offset = arguments.get("offset")
        assert isinstance(offset, int) and offset > 0, f"continuation did not advance an offset: {arguments!r}"

        returned = len(body["page"]["examples"])
        assert offset == returned, (
            f"offset should equal items consumed on this page: offset={offset} returned={returned}"
        )

    def test_non_progressing_call_never_repeats_the_oversized_response(self, mcp_server: MCPServerUnderTest) -> None:
        """Following the continuation chain must not revisit offset=0."""
        fn = mcp_server._tool_manager._tools["explain"].fn
        seen_offsets: list[int] = []
        call_kwargs: dict[str, Any] = {"subject": "result"}
        for _ in range(20):
            raw = invoke_surface(fn, **call_kwargs)
            body = json.loads(raw)
            seen_offsets.append(call_kwargs.get("offset", 0))
            if not body.get("budget_exceeded"):
                break
            call_kwargs = dict(body["continuation"]["arguments"])
        assert seen_offsets == sorted(set(seen_offsets)), (
            f"continuation chain revisited an earlier offset (non-progressing loop): {seen_offsets!r}"
        )
        assert len(seen_offsets) > 1

    def test_response_budget_exceeded_page_fits_the_transport_budget(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.mcp.server_support import MCP_RESPONSE_BUDGET_BYTES

        fn = mcp_server._tool_manager._tools["explain"].fn
        raw = invoke_surface(fn, subject="result")
        assert len(raw.encode("utf-8")) <= MCP_RESPONSE_BUDGET_BYTES
