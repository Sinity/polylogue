"""Catalog-driven contract tests for the Claude AI (web export) parser.

Mirrors the ``_METADATA_PERMUTATION_CASES`` pattern in
``tests/unit/sources/test_parsers_chatgpt.py``: each case is a
``(label, payload_factory, expectations)`` triple and runs the full
parser → transform → save → hydrate pipeline using the shared
``parse_payload_roundtrip`` / ``write_and_hydrate``
helpers. The catalog covers the ``chat_messages`` shape from
``polylogue/sources/parsers/claude/ai_parser.py`` — model variants,
tool-use blocks, thinking blocks, attachments, citations, and a
system-prompt prelude.

Codex/Claude-Code paths are intentionally out of scope (different
shapes, see ``test_parsers_codex_catalog.py`` and
``test_parsers_claude_code_artifacts.py``).

Ref #1184.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from polylogue.sources.parsers.claude import looks_like_ai, parse_ai
from polylogue.sources.parsers.claude.ai_parser import CLAUDE_DESIGN_CHAT_INGEST_FLAG, CLAUDE_TEMPORARY_CHAT_INGEST_FLAG
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


def _text_segment(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _thinking_segment(text: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text}


def _tool_use_segment(name: str, *, tool_id: str = "tu-1", inp: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": inp or {"query": "x"},
    }


def _tool_result_segment(text: str, *, tool_use_id: str = "tu-1") -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
    }


def _chat_message(
    uuid: str,
    sender: str,
    content: list[dict[str, Any]],
    *,
    created_at: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # ``ClaudeAISession`` validation in ``parsers/claude/ai_parser.py``
    # is strict; payloads that don't fit the pydantic model fall through to
    # the loose extractor. We exercise the loose path here so the catalog
    # stays focused on parser behavior independent of validator drift.
    text = " ".join(seg.get("text", "") for seg in content if seg.get("type") == "text").strip() or None
    msg: dict[str, Any] = {
        "uuid": uuid,
        "sender": sender,
        "content": content,
    }
    if text is not None:
        msg["text"] = text
    if created_at is not None:
        msg["created_at"] = created_at
    if attachments is not None:
        msg["attachments"] = attachments
    return msg


def _payload(
    *,
    title: str,
    conv_id: str,
    chat_messages: list[dict[str, Any]],
    model: str | None = None,
    created_at: str = "2024-05-01T10:00:00Z",
    updated_at: str = "2024-05-01T10:30:00Z",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "uuid": conv_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "chat_messages": chat_messages,
    }
    if model is not None:
        payload["model"] = model
    return payload


def test_claude_design_chat_shape_is_parsed_as_session() -> None:
    payload = {
        "uuid": "design-1",
        "title": "Design system",
        "project": {"uuid": "project-1", "name": "Polylogue"},
        "created_at": "2026-06-01T00:00:00Z",
        "updated_at": "2026-06-01T00:01:00Z",
        "messages": [
            {
                "uuid": "m1",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": "Create a design system.",
                    "attachments": [
                        {
                            "id": "att-1",
                            "name": "brief.md",
                            "type": "text/markdown",
                            "content": "# brief",
                        }
                    ],
                    "timestamp": "2026-06-01T00:00:01Z",
                },
            }
        ],
    }

    assert looks_like_ai(payload)
    session = parse_ai(payload, "fallback")

    assert session.provider_session_id == "design-1"
    assert session.title == "Design system"
    assert [message.text for message in session.messages] == ["Create a design system."]
    design_constructs = session.messages[0].blocks[0].web_constructs
    assert len(design_constructs) == 1
    assert design_constructs[0].construct_type.value == "canvas"
    assert design_constructs[0].provider_key == "claude_design_chat"
    assert design_constructs[0].title == "Polylogue"
    assert session.attachments[0].provider_attachment_id == "att-1"
    assert CLAUDE_DESIGN_CHAT_INGEST_FLAG in session.ingest_flags


def test_claude_rich_segments_and_attachment_fields_are_preserved() -> None:
    payload = {
        "uuid": "claude-1",
        "name": "Claude rich",
        "is_temporary": True,
        "chat_messages": [
            {
                "uuid": "m1",
                "sender": "assistant",
                "text": "answer\nremaining",
                "created_at": "2026-06-01T00:00:01Z",
                "updated_at": "2026-06-01T00:00:02Z",
                "content": [
                    {
                        "type": "text",
                        "text": "answer",
                        "citations": [{"url": "https://example.test"}],
                        "start_timestamp": "2026-06-01T00:00:01Z",
                    },
                    {"type": "token_budget", "remaining": 1234},
                ],
                "attachments": [
                    {
                        "file_name": "report.pdf",
                        "file_size": 42,
                        "file_type": "application/pdf",
                        "extracted_content": "report text",
                    }
                ],
                "files": [{"file_uuid": "file-1", "file_name": "source.py"}],
            }
        ],
    }

    session = parse_ai(payload, "fallback")

    assert CLAUDE_TEMPORARY_CHAT_INGEST_FLAG in session.ingest_flags
    assert session.messages[0].blocks[0].metadata is None
    token_budget_constructs = session.messages[0].blocks[1].web_constructs
    assert len(token_budget_constructs) == 1
    assert token_budget_constructs[0].construct_type.value == "token_budget"
    assert token_budget_constructs[0].text == "1234"
    assert session.attachments[0].name == "report.pdf"
    assert session.attachments[0].size_bytes == 42
    assert session.attachments[0].mime_type == "application/pdf"
    assert session.attachments[1].provider_attachment_id == "file-1"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


_CLAUDE_AI_METADATA_CATALOG: list[CatalogCase] = [
    (
        "plain text exchange",
        lambda: _payload(
            title="Plain Session",
            conv_id="conv-plain",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("Hi")], created_at="2024-05-01T10:00:00Z"),
                _chat_message("m2", "assistant", [_text_segment("Hello!")], created_at="2024-05-01T10:00:05Z"),
            ],
        ),
        {
            "title": "Plain Session",
            "roles": ["user", "assistant"],
            "min_messages": 2,
            "block_types_any_of": [["text"]],
        },
    ),
    (
        "model variant: claude-3-opus",
        lambda: _payload(
            title="Opus Session",
            conv_id="conv-opus",
            model="claude-3-opus-20240229",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("Hello opus")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [_text_segment("Greetings.")],
                    created_at="2024-05-01T10:00:01Z",
                ),
            ],
        ),
        {
            "title": "Opus Session",
            "roles": ["user", "assistant"],
            "min_messages": 2,
        },
    ),
    (
        "model variant: claude-3-5-sonnet",
        lambda: _payload(
            title="Sonnet Session",
            conv_id="conv-sonnet",
            model="claude-3-5-sonnet-20241022",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("Hi sonnet")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [_text_segment("Hi.")],
                    created_at="2024-05-01T10:00:01Z",
                ),
            ],
        ),
        {
            "title": "Sonnet Session",
            "roles": ["user", "assistant"],
            "min_messages": 2,
        },
    ),
    (
        "thinking block on assistant",
        lambda: _payload(
            title="Thinking",
            conv_id="conv-think",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("Reason about 2+2")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [
                        _thinking_segment("Let me consider arithmetic."),
                        _text_segment("It's 4."),
                    ],
                    created_at="2024-05-01T10:00:02Z",
                ),
            ],
        ),
        {
            "title": "Thinking",
            "roles": ["user", "assistant"],
            "block_types_any_of": [["thinking", "text"]],
        },
    ),
    (
        "tool_use + tool_result pair",
        lambda: _payload(
            title="Tool Use",
            conv_id="conv-tool",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("search docs")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [
                        _text_segment("calling search"),
                        _tool_use_segment("search", tool_id="tu-A", inp={"q": "polylogue"}),
                    ],
                    created_at="2024-05-01T10:00:02Z",
                ),
                # Anthropic protocol: tool_result is carried under role: user
                _chat_message(
                    "m3",
                    "user",
                    [_tool_result_segment("found 3 hits", tool_use_id="tu-A")],
                    created_at="2024-05-01T10:00:03Z",
                ),
            ],
        ),
        {
            "title": "Tool Use",
            # m3 is reclassified user→tool by reclassify_tool_result_envelope.
            "roles": ["user", "assistant", "tool"],
            "must_contain_block_type": "tool_use",
        },
    ),
    (
        "attachments on user message",
        lambda: _payload(
            title="Attachments",
            conv_id="conv-att",
            chat_messages=[
                _chat_message(
                    "m1",
                    "human",
                    [_text_segment("see file")],
                    created_at="2024-05-01T10:00:00Z",
                    attachments=[
                        {
                            "id": "att-1",
                            "file_name": "notes.pdf",
                            "file_type": "application/pdf",
                            "file_size": 4096,
                        }
                    ],
                ),
                _chat_message("m2", "assistant", [_text_segment("ok")], created_at="2024-05-01T10:00:01Z"),
            ],
        ),
        {
            "title": "Attachments",
            "roles": ["user", "assistant"],
            "min_attachments": 1,
        },
    ),
    (
        "citations metadata in content block",
        lambda: _payload(
            title="Citations",
            conv_id="conv-cite",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("source?")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [
                        {
                            "type": "text",
                            "text": "see [1]",
                            "citations": [{"title": "Ref", "url": "https://example.com"}],
                        }
                    ],
                    created_at="2024-05-01T10:00:01Z",
                ),
            ],
        ),
        {
            "title": "Citations",
            "roles": ["user", "assistant"],
            "block_types_any_of": [["text"]],
        },
    ),
    (
        "system prompt prelude + ordering",
        lambda: _payload(
            title="System Prompt",
            conv_id="conv-sys",
            chat_messages=[
                _chat_message(
                    "m0",
                    "system",
                    [_text_segment("You are a helpful assistant.")],
                    created_at="2024-05-01T09:59:59Z",
                ),
                _chat_message("m1", "human", [_text_segment("hi")], created_at="2024-05-01T10:00:00Z"),
                _chat_message("m2", "assistant", [_text_segment("hello")], created_at="2024-05-01T10:00:01Z"),
            ],
        ),
        {
            "title": "System Prompt",
            "roles": ["system", "user", "assistant"],
            "preserves_order": True,
        },
    ),
    (
        "mixed thinking + tool_use + text in one assistant turn",
        lambda: _payload(
            title="Rich Turn",
            conv_id="conv-rich",
            chat_messages=[
                _chat_message("m1", "human", [_text_segment("research X")], created_at="2024-05-01T10:00:00Z"),
                _chat_message(
                    "m2",
                    "assistant",
                    [
                        _thinking_segment("planning"),
                        _tool_use_segment("search", tool_id="tu-rich"),
                        _text_segment("starting search..."),
                    ],
                    created_at="2024-05-01T10:00:01Z",
                ),
            ],
        ),
        {
            "title": "Rich Turn",
            "roles": ["user", "assistant"],
            "must_contain_block_type": "tool_use",
            "block_types_any_of": [["thinking", "tool_use", "text"]],
        },
    ),
]


# ---------------------------------------------------------------------------
# Shared roundtrip assertion
# ---------------------------------------------------------------------------


def _assert_roundtrip(
    source_name: str,
    payload: dict[str, Any],
    expectations: Expectations,
    workspace_env: dict[str, Path],
    label: str,
) -> None:
    """Run payload → parse → archive-write → hydrate and assert contract
    expectations.

    Title, role normalization, and timestamp ordering are asserted at the
    hydrated ``Session`` level. Content-block kinds and attachment counts are
    asserted against the parser output (``roundtrip.parsed``): these are
    parser catalog tests, so the parser's emitted blocks are the contract
    under test.
    """
    raw_bytes = json.dumps(payload).encode("utf-8")
    db_path = db_setup(workspace_env)
    with open_connection(db_path) as conn:
        roundtrip = parse_payload_roundtrip(source_name, raw_bytes, unique_id=label)
        hydrated = write_and_hydrate(roundtrip, conn)

    # Title survives.
    expected_title = expectations.get("title")
    if expected_title is not None:
        assert hydrated.title == expected_title, f"[{label}] title: expected {expected_title!r}, got {hydrated.title!r}"

    messages = list(hydrated.messages)

    expected_min = expectations.get("min_messages", 1)
    assert len(messages) >= expected_min, (
        f"[{label}] expected at least {expected_min} hydrated messages, got {len(messages)}"
    )

    # Role multiset is preserved between parsed and hydrated (sanity).
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
    observed_kinds = {str(b.type) for m in roundtrip.parsed.messages for b in m.blocks}

    block_types_any_of = expectations.get("block_types_any_of")
    if block_types_any_of is not None:
        ok = any(set(kinds).issubset(observed_kinds) for kinds in block_types_any_of)
        assert ok, f"[{label}] expected at least one of {block_types_any_of} subset of observed kinds {observed_kinds}"

    must_contain = expectations.get("must_contain_block_type")
    if must_contain is not None:
        assert must_contain in observed_kinds, (
            f"[{label}] expected block type {must_contain!r} in observed kinds {observed_kinds}"
        )

    # Timestamp ordering survives. Hydrated messages must be sorted by
    # ascending timestamp where timestamps are present.
    if expectations.get("preserves_order"):
        timestamps = [m.timestamp for m in messages if m.timestamp is not None]
        assert timestamps == sorted(timestamps), f"[{label}] hydrated timestamps not ascending: {timestamps}"

    # Attachment counts (loose lower bound only — provider may bind some
    # attachments at the session level).
    expected_min_attachments = expectations.get("min_attachments")
    if expected_min_attachments is not None:
        attachments = list(roundtrip.parsed.attachments)
        assert len(attachments) >= expected_min_attachments, (
            f"[{label}] expected at least {expected_min_attachments} attachments, got {len(attachments)}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "factory", "expectations"),
    [pytest.param(label, factory, exp, id=label) for label, factory, exp in _CLAUDE_AI_METADATA_CATALOG],
)
def test_claude_ai_catalog_roundtrip(
    label: str,
    factory: PayloadFactory,
    expectations: Expectations,
    workspace_env: dict[str, Path],
) -> None:
    """Each catalog payload survives parse → transform → save → hydrate.

    Asserts the contract surface required by issue #1184: title, role
    normalization, content-block kinds, and timestamp ordering.
    """
    payload = factory()
    assert looks_like_ai(payload), f"[{label}] payload must look like a Claude AI export"
    _assert_roundtrip("claude-ai", payload, expectations, workspace_env, label)


def test_claude_ai_catalog_parser_only_smoke() -> None:
    """Parser-only smoke test: every catalog entry parses to >=1 message
    without raising and reports ``claude-ai`` as the provider name.

    This is the fast feedback signal — failure here means the parser
    itself rejects the payload before the roundtrip helper even runs.
    """
    for label, factory, _expectations in _CLAUDE_AI_METADATA_CATALOG:
        payload = factory()
        # Deep-copy guards against accidental mutation by the parser.
        parsed = parse_ai(copy.deepcopy(payload), fallback_id=f"fb-{label}")
        assert parsed.source_name == "claude-ai", f"[{label}] wrong provider: {parsed.source_name}"
        assert parsed.messages, f"[{label}] parser produced no messages"
