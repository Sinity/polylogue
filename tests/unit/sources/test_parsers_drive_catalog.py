"""Catalog-driven contract tests for the Drive/Gemini chunked-prompt parser.

Mirrors ``_METADATA_PERMUTATION_CASES`` in
``tests/unit/sources/test_parsers_chatgpt.py``. Each entry is a
``(label, payload_factory, expectations)`` triple exercising the
full parser → transform → save → hydrate pipeline against Gemini's
``chunkedPrompt.chunks`` shape from ``parsers/drive.py``.

Coverage targets the surfaces called out in issue #1184: text chunks,
image/inline-file chunks, role variants (``user`` ↔ ``model``), and
metadata that should survive — ``displayName`` → title, ``createTime``
ordering, thinking blocks, code-execution results, Drive document
attachments.

Ref #1184.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.pipeline_roundtrip import (
    parse_payload_roundtrip,
    write_and_hydrate,
)
from tests.infra.storage_records import db_setup

PayloadFactory: TypeAlias = Callable[[], dict[str, Any]]
Expectations: TypeAlias = dict[str, Any]
CatalogCase: TypeAlias = tuple[str, PayloadFactory, Expectations]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    role: str,
    *,
    text: str | None = None,
    create_time: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {"id": chunk_id, "role": role}
    if text is not None:
        chunk["text"] = text
    if create_time is not None:
        chunk["createTime"] = create_time
    if extras:
        chunk.update(extras)
    return chunk


def _payload(
    *,
    conv_id: str,
    display_name: str | None,
    chunks: list[dict[str, Any]],
    create_time: str = "2024-04-10T08:00:00Z",
    update_time: str = "2024-04-10T08:30:00Z",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": conv_id,
        "createTime": create_time,
        "updateTime": update_time,
        "chunkedPrompt": {"chunks": chunks},
    }
    if display_name is not None:
        payload["displayName"] = display_name
    return payload


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


_GEMINI_METADATA_CATALOG: list[CatalogCase] = [
    (
        "plain text exchange",
        lambda: _payload(
            conv_id="gem-plain",
            display_name="Plain Gemini",
            chunks=[
                _chunk("c1", "user", text="hello", create_time="2024-04-10T08:00:00Z"),
                _chunk("c2", "model", text="hi back", create_time="2024-04-10T08:00:05Z"),
            ],
        ),
        {
            "title": "Plain Gemini",
            "roles": ["user", "assistant"],
            "block_types_any_of": [["text"]],
            "preserves_order": True,
        },
    ),
    (
        "displayName missing → fallback title",
        lambda: _payload(
            conv_id="gem-fallback",
            display_name=None,
            chunks=[
                _chunk("c1", "user", text="q", create_time="2024-04-10T08:00:00Z"),
                _chunk("c2", "model", text="a", create_time="2024-04-10T08:00:01Z"),
            ],
        ),
        {
            # ``parse_chunked_prompt`` falls back to the fallback_id when no
            # displayName is supplied.
            "fallback_title": True,
            "roles": ["user", "assistant"],
        },
    ),
    (
        "role variants: model normalised to assistant",
        lambda: _payload(
            conv_id="gem-roles",
            display_name="Roles",
            chunks=[
                _chunk("c1", "user", text="ping", create_time="2024-04-10T08:00:00Z"),
                _chunk("c2", "model", text="pong", create_time="2024-04-10T08:00:01Z"),
                _chunk("c3", "user", text="again", create_time="2024-04-10T08:00:02Z"),
                _chunk("c4", "model", text="sure", create_time="2024-04-10T08:00:03Z"),
            ],
        ),
        {
            "title": "Roles",
            "roles": ["user", "assistant", "user", "assistant"],
            "preserves_order": True,
        },
    ),
    (
        "thinking chunk (isThought + thinkingBudget)",
        lambda: _payload(
            conv_id="gem-think",
            display_name="Thinking",
            chunks=[
                _chunk("c1", "user", text="reason", create_time="2024-04-10T08:00:00Z"),
                _chunk(
                    "c2",
                    "model",
                    text="reasoning trace",
                    create_time="2024-04-10T08:00:01Z",
                    extras={"isThought": True, "thinkingBudget": 64},
                ),
            ],
        ),
        {
            "title": "Thinking",
            "roles": ["user", "assistant"],
            "must_contain_block_type": "thinking",
        },
    ),
    (
        "executable code + code execution result",
        lambda: _payload(
            conv_id="gem-code",
            display_name="Code Exec",
            chunks=[
                _chunk("c1", "user", text="run code", create_time="2024-04-10T08:00:00Z"),
                _chunk(
                    "c2",
                    "model",
                    text="here you go",
                    create_time="2024-04-10T08:00:01Z",
                    extras={
                        "executableCode": {"language": "python", "code": "print('hi')"},
                        "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "hi"},
                    },
                ),
            ],
        ),
        {
            "title": "Code Exec",
            "roles": ["user", "assistant"],
            "must_contain_block_type": "code",
            "block_types_any_of": [["code", "tool_result"]],
        },
    ),
    (
        "Drive document attachment",
        lambda: _payload(
            conv_id="gem-doc",
            display_name="Doc Attached",
            chunks=[
                _chunk(
                    "c1",
                    "user",
                    text="see attached",
                    create_time="2024-04-10T08:00:00Z",
                    extras={
                        "driveDocument": {
                            "id": "doc-1",
                            "name": "spec.pdf",
                            "mimeType": "application/pdf",
                            "sizeBytes": "2048",
                        }
                    },
                ),
                _chunk("c2", "model", text="ok", create_time="2024-04-10T08:00:01Z"),
            ],
        ),
        {
            "title": "Doc Attached",
            "roles": ["user", "assistant"],
            "must_contain_block_type": "document",
            "min_attachments": 1,
        },
    ),
    (
        "inline image attachment",
        lambda: _payload(
            conv_id="gem-img",
            display_name="Inline Image",
            chunks=[
                _chunk(
                    "c1",
                    "user",
                    create_time="2024-04-10T08:00:00Z",
                    extras={
                        "inlineFile": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgo=",
                        }
                    },
                ),
                _chunk("c2", "model", text="image received", create_time="2024-04-10T08:00:01Z"),
            ],
        ),
        {
            "title": "Inline Image",
            "roles": ["user", "assistant"],
            "min_attachments": 1,
        },
    ),
    (
        "timestamp ordering across out-of-order chunks",
        lambda: _payload(
            conv_id="gem-order",
            display_name="Ordering",
            # Chunks declared in author order; parser preserves declared order.
            chunks=[
                _chunk("c1", "user", text="first", create_time="2024-04-10T08:00:00Z"),
                _chunk("c2", "model", text="second", create_time="2024-04-10T08:00:01Z"),
                _chunk("c3", "user", text="third", create_time="2024-04-10T08:00:02Z"),
            ],
        ),
        {
            "title": "Ordering",
            "roles": ["user", "assistant", "user"],
            "preserves_order": True,
        },
    ),
]


# ---------------------------------------------------------------------------
# Roundtrip assertion
# ---------------------------------------------------------------------------


def _assert_roundtrip(
    payload: dict[str, Any],
    expectations: Expectations,
    workspace_env: dict[str, Path],
    label: str,
) -> None:
    raw_bytes = json.dumps(payload).encode("utf-8")
    db_path = db_setup(workspace_env)
    with open_connection(db_path) as conn:
        roundtrip = parse_payload_roundtrip("gemini", raw_bytes, unique_id=label)
        hydrated = write_and_hydrate(roundtrip, conn)

    # Title.
    expected_title = expectations.get("title")
    if expected_title is not None:
        assert hydrated.title == expected_title, f"[{label}] title: expected {expected_title!r}, got {hydrated.title!r}"
    elif expectations.get("fallback_title"):
        assert hydrated.title and hydrated.title.startswith("rt-"), (
            f"[{label}] expected fallback title (parser hydrates fallback_id); got {hydrated.title!r}"
        )

    messages = list(hydrated.messages)
    assert messages, f"[{label}] no hydrated messages"

    parsed_roles = [str(m.role) for m in roundtrip.parsed.messages]
    hydrated_roles = [str(m.role) for m in messages]
    assert sorted(parsed_roles) == sorted(hydrated_roles), (
        f"[{label}] role multiset diverged: parsed={parsed_roles} hydrated={hydrated_roles}"
    )

    expected_roles = expectations.get("roles")
    if expected_roles is not None:
        assert hydrated_roles == expected_roles, f"[{label}] roles expected {expected_roles}, got {hydrated_roles}"

    # Content-block kinds are the parser's contract: assert on the parsed
    # messages the parser emitted.
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

    # Timestamp ordering.
    if expectations.get("preserves_order"):
        timestamps = [m.timestamp for m in messages if m.timestamp is not None]
        assert timestamps == sorted(timestamps), f"[{label}] hydrated timestamps not ascending: {timestamps}"

    # Attachments.
    expected_min_attachments = expectations.get("min_attachments")
    if expected_min_attachments is not None:
        attachments = list(roundtrip.parsed.attachments)
        assert len(attachments) >= expected_min_attachments, (
            f"[{label}] expected ≥{expected_min_attachments} attachments, got {len(attachments)}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "factory", "expectations"),
    [pytest.param(label, factory, exp, id=label) for label, factory, exp in _GEMINI_METADATA_CATALOG],
)
def test_gemini_catalog_roundtrip(
    label: str,
    factory: PayloadFactory,
    expectations: Expectations,
    workspace_env: dict[str, Path],
) -> None:
    """Each Gemini catalog payload survives parse → transform → save → hydrate.

    Asserts the contract surface required by issue #1184: title, role
    normalization, content-block kinds, timestamp ordering, attachments.
    """
    _assert_roundtrip(factory(), expectations, workspace_env, label)


def test_gemini_catalog_parser_only_smoke() -> None:
    """Fast parser-only smoke pass — each catalog entry must parse as
    Gemini and produce at least one message without raising."""
    for label, factory, _expectations in _GEMINI_METADATA_CATALOG:
        payload = factory()
        parsed = parse_chunked_prompt("gemini", copy.deepcopy(payload), f"fb-{label}")
        assert parsed.source_name == "gemini", f"[{label}] wrong provider: {parsed.source_name}"
        assert parsed.messages, f"[{label}] parser produced no messages"
