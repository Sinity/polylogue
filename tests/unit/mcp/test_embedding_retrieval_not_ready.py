"""MCP surface for ``EmbeddingRetrievalNotReadyError`` (#1503 AC4).

Pins the contract that:

1. The operations layer raises the typed
   ``EmbeddingRetrievalNotReadyError`` (not generic ``DatabaseError``)
   when a semantic/hybrid query asks for vectors that aren't ready.
2. The MCP exception-to-error-JSON translator forwards the operator-
   actionable message verbatim under ``code="embedding_retrieval_not_ready"``
   and exposes the readiness status enum on the typed payload — rather
   than swallowing the message to ``"search: DatabaseError"`` like the
   generic-PolylogueError path does.
3. The error message names the same operator next step the CLI gives
   (``polylogue embed status`` → ``polylogue embed backfill`` /
   ``polylogue embed enable``).
"""

from __future__ import annotations

import json

import pytest

from polylogue.errors import (
    DatabaseError,
    EmbeddingRetrievalNotReadyError,
    PolylogueError,
)
from polylogue.mcp.server_support import _exception_to_error_json


def test_error_class_is_database_error_subclass() -> None:
    """``EmbeddingRetrievalNotReadyError`` IS-A ``DatabaseError`` so existing
    handlers that catch ``DatabaseError`` still cover the case."""
    err = EmbeddingRetrievalNotReadyError("msg", readiness_status="none")
    assert isinstance(err, DatabaseError)
    assert isinstance(err, PolylogueError)
    assert err.readiness_status == "none"


def test_mcp_error_json_carries_actionable_message_verbatim() -> None:
    """The MCP error JSON must forward the operator message instead of
    redacting it to the class name (#1503 AC4)."""
    err = EmbeddingRetrievalNotReadyError(
        "Semantic or hybrid retrieval requires retrieval-ready embeddings "
        "(current status: none). Run `polylogue embed status`, then "
        "`polylogue embed backfill` or let polylogued converge after enabling embeddings.",
        readiness_status="none",
    )
    payload = json.loads(_exception_to_error_json("search", err))

    assert payload["code"] == "embedding_retrieval_not_ready"
    assert payload["detail"] == "EmbeddingRetrievalNotReadyError"
    assert payload["tool"] == "search"
    assert payload["readiness_status"] == "none"
    assert "polylogue embed status" in payload["message"]
    assert "polylogue embed backfill" in payload["message"]


def test_mcp_error_json_distinct_from_generic_polylogue_error() -> None:
    """Generic ``DatabaseError`` still redacts to the class name; only
    ``EmbeddingRetrievalNotReadyError`` is the verbatim-forwarding subclass."""
    generic = DatabaseError("opaque internal SQL error mentioning a path")
    payload = json.loads(_exception_to_error_json("search", generic))
    assert payload["code"] == "polylogue_error"
    assert payload["message"] == "search: DatabaseError"
    # The raw message is intentionally not echoed.
    assert "opaque internal SQL error" not in payload["message"]


def test_mcp_error_json_includes_readiness_status_for_other_states() -> None:
    """The status enum surfaces every state, not just ``none``."""
    for status in ("none", "partial", "pending", "disabled"):
        err = EmbeddingRetrievalNotReadyError(f"current status: {status}", readiness_status=status)
        payload = json.loads(_exception_to_error_json("search", err))
        assert payload["readiness_status"] == status


@pytest.mark.parametrize(
    "command_hint",
    [
        "polylogue embed status",
        "polylogue embed backfill",
        "polylogue embed enable",
    ],
)
def test_error_messages_name_the_canonical_operator_commands(command_hint: str) -> None:
    """The error message references the actual subcommands operators run.

    Refactoring the embed CLI must keep these in sync; the test fails
    if the readiness-not-ready message stops pointing at the canonical
    next step.
    """
    backfill_msg = EmbeddingRetrievalNotReadyError(
        "Semantic or hybrid retrieval requires retrieval-ready embeddings "
        "(current status: none). Run `polylogue embed status`, then "
        "`polylogue embed backfill` or let polylogued converge after enabling embeddings.",
        readiness_status="none",
    )
    enable_msg = EmbeddingRetrievalNotReadyError(
        "Semantic or hybrid retrieval requires vector search support, but vector provider initialization "
        "failed or embeddings are disabled. Run `polylogue embed status`, then `polylogue embed enable` "
        "if needed.",
        readiness_status="disabled",
    )
    union = str(backfill_msg) + " " + str(enable_msg)
    assert command_hint in union
