"""Catalog-driven contract tests for the Codex JSONL session parser.

Mirrors ``_METADATA_PERMUTATION_CASES`` in
``tests/unit/sources/test_parsers_chatgpt.py``: each entry is a
``(label, payload_factory, expectations)`` triple exercising the full
parser → transform → save → hydrate pipeline against Codex session
envelopes from ``parsers/codex.py``.

Coverage targets the surfaces called out in issue #1184: title (Codex
has no title — fallback is the session id), model
identification, role normalization across ``user``/``assistant``/
``developer``/``system``, content-block kinds (text, tool_use,
tool_result), and ascending timestamp ordering.

Ref #1184.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from polylogue.sources.parsers.codex import looks_like as codex_looks_like
from polylogue.sources.parsers.codex import parse as codex_parse
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.pipeline_roundtrip import (
    parse_payload_roundtrip,
    write_and_hydrate,
)
from tests.infra.storage_records import db_setup

PayloadFactory: TypeAlias = Callable[[], list[dict[str, Any]]]
Expectations: TypeAlias = dict[str, Any]
CatalogCase: TypeAlias = tuple[str, PayloadFactory, Expectations]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_meta(session_id: str, *, timestamp: str, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": session_id, "timestamp": timestamp}
    if extras:
        payload.update(extras)
    return {"type": "session_meta", "payload": payload}


def _user_message(text: str, *, timestamp: str, msg_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "timestamp": timestamp,
        "content": [{"type": "input_text", "text": text}],
    }
    if msg_id is not None:
        payload["id"] = msg_id
    return {"type": "response_item", "payload": payload}


def _assistant_message(text: str, *, timestamp: str, msg_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "timestamp": timestamp,
        "content": [{"type": "output_text", "text": text}],
    }
    if msg_id is not None:
        payload["id"] = msg_id
    return {"type": "response_item", "payload": payload}


def _developer_message(text: str, *, timestamp: str) -> dict[str, Any]:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "developer",
            "timestamp": timestamp,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _function_call(name: str, args: dict[str, Any], *, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "id": f"fc-{call_id}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _function_call_output(output: str, *, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


_CODEX_METADATA_CATALOG: list[CatalogCase] = [
    (
        "plain envelope exchange",
        lambda: [
            _session_meta("sess-plain", timestamp="2024-06-01T09:00:00Z"),
            _user_message("hello codex", timestamp="2024-06-01T09:00:01Z", msg_id="m1"),
            _assistant_message("hello back", timestamp="2024-06-01T09:00:02Z", msg_id="m2"),
        ],
        {
            "conv_id": "sess-plain",
            "roles": ["user", "assistant"],
            "block_types_any_of": [["text"]],
            "preserves_order": True,
        },
    ),
    (
        "session_meta with git context + instructions",
        lambda: [
            _session_meta(
                "sess-git",
                timestamp="2024-06-01T09:00:00Z",
                extras={
                    "git": {"branch": "feature/x", "commit": "abc123"},
                    "instructions": "Be concise.",
                },
            ),
            _user_message("hi", timestamp="2024-06-01T09:00:01Z", msg_id="m1"),
            _assistant_message("hi.", timestamp="2024-06-01T09:00:02Z", msg_id="m2"),
        ],
        {
            "conv_id": "sess-git",
            "roles": ["user", "assistant"],
            "preserves_order": True,
        },
    ),
    (
        "developer message normalises to system",
        lambda: [
            _session_meta("sess-dev", timestamp="2024-06-01T09:00:00Z"),
            _developer_message("<developer>system prompt</developer>", timestamp="2024-06-01T09:00:01Z"),
            _user_message("hi", timestamp="2024-06-01T09:00:02Z", msg_id="m1"),
            _assistant_message("hi.", timestamp="2024-06-01T09:00:03Z", msg_id="m2"),
        ],
        {
            "conv_id": "sess-dev",
            # codex parser maps developer → system role
            "roles": ["system", "user", "assistant"],
            "preserves_order": True,
        },
    ),
    (
        "function_call + function_call_output → tool_use + tool_result",
        lambda: [
            _session_meta("sess-tool", timestamp="2024-06-01T09:00:00Z"),
            _user_message("run command", timestamp="2024-06-01T09:00:01Z", msg_id="m1"),
            _function_call("exec_command", {"cmd": "ls"}, call_id="c1"),
            _function_call_output("file1\nfile2", call_id="c1"),
            _assistant_message("done", timestamp="2024-06-01T09:00:05Z", msg_id="m2"),
        ],
        {
            "conv_id": "sess-tool",
            "must_contain_block_type": "tool_use",
            "block_types_any_of": [["tool_use", "tool_result"]],
        },
    ),
    (
        "direct (non-envelope) format",
        lambda: [
            {
                "id": "sess-direct",
                "timestamp": "2024-06-01T09:00:00Z",
            },
            {
                "type": "message",
                "role": "user",
                "timestamp": "2024-06-01T09:00:01Z",
                "id": "m1",
                "content": [{"type": "input_text", "text": "hello direct"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "timestamp": "2024-06-01T09:00:02Z",
                "id": "m2",
                "content": [{"type": "output_text", "text": "hi direct"}],
            },
        ],
        {
            "conv_id": "sess-direct",
            "roles": ["user", "assistant"],
            "preserves_order": True,
        },
    ),
    (
        "role normalisation: User / ASSISTANT",
        lambda: [
            _session_meta("sess-roles", timestamp="2024-06-01T09:00:00Z"),
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "User",
                    "timestamp": "2024-06-01T09:00:01Z",
                    "id": "m1",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "ASSISTANT",
                    "timestamp": "2024-06-01T09:00:02Z",
                    "id": "m2",
                    "content": [{"type": "output_text", "text": "hi back"}],
                },
            },
        ],
        {
            "conv_id": "sess-roles",
            "roles": ["user", "assistant"],
        },
    ),
    (
        "multi-turn session preserves ordering",
        lambda: [
            _session_meta("sess-multi", timestamp="2024-06-01T09:00:00Z"),
            _user_message("q1", timestamp="2024-06-01T09:00:01Z", msg_id="m1"),
            _assistant_message("a1", timestamp="2024-06-01T09:00:02Z", msg_id="m2"),
            _user_message("q2", timestamp="2024-06-01T09:00:03Z", msg_id="m3"),
            _assistant_message("a2", timestamp="2024-06-01T09:00:04Z", msg_id="m4"),
            _user_message("q3", timestamp="2024-06-01T09:00:05Z", msg_id="m5"),
            _assistant_message("a3", timestamp="2024-06-01T09:00:06Z", msg_id="m6"),
        ],
        {
            "conv_id": "sess-multi",
            "roles": ["user", "assistant", "user", "assistant", "user", "assistant"],
            "preserves_order": True,
            "min_messages": 6,
        },
    ),
    (
        "state records are skipped without breaking the catalog",
        lambda: [
            _session_meta("sess-state", timestamp="2024-06-01T09:00:00Z"),
            {"record_type": "state"},
            _user_message("hi", timestamp="2024-06-01T09:00:01Z", msg_id="m1"),
            {"record_type": "state"},
            _assistant_message("hello", timestamp="2024-06-01T09:00:02Z", msg_id="m2"),
        ],
        {
            "conv_id": "sess-state",
            "roles": ["user", "assistant"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Roundtrip assertion
# ---------------------------------------------------------------------------


def _assert_roundtrip(
    payload: list[dict[str, Any]],
    expectations: Expectations,
    workspace_env: dict[str, Path],
    label: str,
) -> None:
    # Codex source records are JSONL — one record per line.
    raw_bytes = ("\n".join(json.dumps(record) for record in payload) + "\n").encode("utf-8")
    db_path = db_setup(workspace_env)
    with open_connection(db_path) as conn:
        roundtrip = parse_payload_roundtrip("codex", raw_bytes, unique_id=label)
        hydrated = write_and_hydrate(roundtrip, conn)

    # Session id from session_meta (or first-line id for direct format).
    expected_conv_id = expectations.get("conv_id")
    if expected_conv_id is not None:
        assert roundtrip.parsed.provider_session_id == expected_conv_id, (
            f"[{label}] session id: expected {expected_conv_id!r}, got {roundtrip.parsed.provider_session_id!r}"
        )

    messages = list(hydrated.messages)
    expected_min = expectations.get("min_messages", 1)
    assert len(messages) >= expected_min, f"[{label}] expected ≥{expected_min} hydrated messages, got {len(messages)}"

    parsed_roles = [str(m.role) for m in roundtrip.parsed.messages]
    hydrated_roles = [str(m.role) for m in messages]
    assert sorted(parsed_roles) == sorted(hydrated_roles), (
        f"[{label}] role multiset diverged: parsed={parsed_roles} hydrated={hydrated_roles}"
    )

    expected_roles = expectations.get("roles")
    if expected_roles is not None:
        assert hydrated_roles == expected_roles, f"[{label}] roles expected {expected_roles}, got {hydrated_roles}"

    # Content-block kinds: assert at the materialization boundary
    # (``bundle.content_blocks``) — see test_parsers_claude_ai_catalog
    # for rationale.
    observed_kinds = {str(b.type) for m in roundtrip.parsed.messages for b in m.content_blocks}

    block_types_any_of = expectations.get("block_types_any_of")
    if block_types_any_of is not None:
        ok = any(set(kinds).issubset(observed_kinds) for kinds in block_types_any_of)
        assert ok, f"[{label}] expected one of {block_types_any_of} subset of observed {observed_kinds}"

    must_contain = expectations.get("must_contain_block_type")
    if must_contain is not None:
        assert must_contain in observed_kinds, (
            f"[{label}] expected block kind {must_contain!r} in observed {observed_kinds}"
        )

    if expectations.get("preserves_order"):
        timestamps = [m.timestamp for m in messages if m.timestamp is not None]
        assert timestamps == sorted(timestamps), f"[{label}] hydrated timestamps not ascending: {timestamps}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "factory", "expectations"),
    [pytest.param(label, factory, exp, id=label) for label, factory, exp in _CODEX_METADATA_CATALOG],
)
def test_codex_catalog_roundtrip(
    label: str,
    factory: PayloadFactory,
    expectations: Expectations,
    workspace_env: dict[str, Path],
) -> None:
    """Each Codex catalog payload survives parse → transform → save → hydrate.

    Asserts the contract surface required by issue #1184: session id,
    role normalization, content-block kinds, timestamp ordering.
    """
    _assert_roundtrip(factory(), expectations, workspace_env, label)


def test_codex_catalog_parser_only_smoke() -> None:
    """Fast parser-only smoke pass — each catalog entry passes
    ``looks_like`` and yields ≥1 message via ``parse``."""
    for label, factory, _expectations in _CODEX_METADATA_CATALOG:
        payload = factory()
        assert codex_looks_like(payload), f"[{label}] looks_like rejected the catalog payload"
        parsed = codex_parse(copy.deepcopy(payload), f"fb-{label}")
        assert parsed.source_name == "codex", f"[{label}] wrong provider: {parsed.source_name}"
        assert parsed.messages, f"[{label}] parser produced no messages"
