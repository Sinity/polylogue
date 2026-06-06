"""Gemini ``chunkedPrompt`` contract tests.

These tests pin the on-disk semantics of Gemini AI Studio's ``chunkedPrompt``
export shape -- what the persisted Google Drive JSON looks like once a
multi-turn session has been realized. They are explicitly contract-shaped:
if Google changes the chunk schema, executableCode / codeExecutionResult key
tokens, or per-chunk role/timestamp semantics, these tests should fail with a
precise reason naming the broken field.

Catalog payloads live under ``tests/data/gemini_chunked_prompt/`` and cover
three realistic session shapes:

- ``text_only_prompt.json`` -- alternating user/model text turns with token
  counts and finish reasons.
- ``code_execution_prompt.json`` -- model thought, executableCode tool call,
  codeExecutionResult, and a final summarising message.
- ``multi_turn_prompt.json`` -- driveDocument attachment, model thought,
  safety ratings, and a follow-up turn pair.

Ref #1297, Ref #1184, Ref #1186.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.drive import looks_like as _looks_like_impl
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.types import Provider

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gemini_chunked_prompt"
CATALOG_FIXTURES = (
    "text_only_prompt.json",
    "code_execution_prompt.json",
    "multi_turn_prompt.json",
)


# ---------------------------------------------------------------------------
# Catalog access helpers
# ---------------------------------------------------------------------------


def _load_catalog(name: str) -> JSONDocument:
    path = CATALOG_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{name} must be a JSON object"
    return cast(JSONDocument, payload)


def _parse(payload: JSONDocument, fallback_id: str = "fallback") -> ParsedSession:
    return parse_chunked_prompt(Provider.GEMINI, payload, fallback_id)


def _looks_like(payload: JSONDocument) -> bool:
    return _looks_like_impl(payload)


def _meta_blocks(message_provider_meta: dict[str, object] | None) -> list[dict[str, object]]:
    assert isinstance(message_provider_meta, dict), "message is missing provider_meta"
    blocks = message_provider_meta.get("content_blocks")
    assert isinstance(blocks, list), "provider_meta.content_blocks must be a list"
    typed: list[dict[str, object]] = []
    for block in blocks:
        assert isinstance(block, dict), "content_blocks entries must be dicts"
        typed.append(block)
    return typed


# ---------------------------------------------------------------------------
# Catalog smoke -- every shape must be detected and parsed without raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", CATALOG_FIXTURES)
def test_catalog_payloads_are_detected_as_gemini(fixture: str) -> None:
    payload = _load_catalog(fixture)
    assert _looks_like(payload), f"{fixture} must satisfy drive.looks_like()"


@pytest.mark.parametrize("fixture", CATALOG_FIXTURES)
def test_catalog_payloads_parse_without_raising(fixture: str) -> None:
    payload = _load_catalog(fixture)
    session = _parse(payload, fixture)
    assert session.source_name == Provider.GEMINI
    assert session.messages, f"{fixture} produced zero messages"


# ---------------------------------------------------------------------------
# chunkedPrompt envelope contract -- chunks decomposition into messages
# ---------------------------------------------------------------------------


class TestChunkedPromptEnvelopeShape:
    """Pin the ``chunkedPrompt.chunks`` -> messages decomposition contract.

    Gemini stores every realised turn in ``chunkedPrompt.chunks``; the parser
    must walk that list, assign one message per chunk that carries either
    text, attachments, or a structured payload, and drop chunks that have
    nothing usable. These tests pin both halves of that contract.
    """

    def test_each_text_chunk_becomes_its_own_message(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        assert len(session.messages) == 4
        assert [m.role for m in session.messages] == [
            Role.USER,
            Role.ASSISTANT,
            Role.USER,
            Role.ASSISTANT,
        ]
        assert [m.text for m in session.messages] == [
            "What is the capital of France?",
            "The capital of France is Paris.",
            "And of Germany?",
            "Berlin.",
        ]

    def test_chunks_without_role_are_skipped(self) -> None:
        """Role is required -- a chunk missing both ``role`` and ``author``
        must be dropped silently. This pins the existing parser invariant so a
        future change to "raise on missing role" or "default to user" fails
        loudly."""
        payload: JSONDocument = {
            "id": "no-role",
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "kept"},
                    {"text": "dropped -- no role"},
                ]
            },
        }
        session = _parse(payload, "no-role")
        assert [m.text for m in session.messages] == ["kept"]

    def test_empty_chunk_without_text_attachments_or_blocks_is_dropped(self) -> None:
        payload: JSONDocument = {
            "id": "empty",
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": ""},
                    {"role": "model", "text": "real reply"},
                ]
            },
        }
        session = _parse(payload, "empty")
        assert [m.text for m in session.messages] == ["real reply"]

    def test_chunks_top_level_fallback_is_honoured(self) -> None:
        """When ``chunkedPrompt`` is absent but a top-level ``chunks`` array
        exists, the parser must walk that list instead. This is the
        compatibility branch for older exports."""
        payload: JSONDocument = {
            "id": "top-level",
            "chunks": [
                {"role": "user", "text": "top-level chunks"},
                {"role": "model", "text": "fallback path"},
            ],
        }
        session = _parse(payload, "top-level")
        assert [m.text for m in session.messages] == [
            "top-level chunks",
            "fallback path",
        ]

    def test_string_chunk_with_role_promotes_to_text_message(self) -> None:
        """``str`` chunks become ``{"text": chunk}`` -- but without a role,
        they must still be dropped by the role guard."""
        payload: JSONDocument = {
            "id": "string-chunks",
            "chunkedPrompt": {"chunks": ["bare string", {"role": "user", "text": "kept"}]},
        }
        session = _parse(payload, "string-chunks")
        # String chunks have no role -> dropped; only the typed user chunk survives.
        assert [m.text for m in session.messages] == ["kept"]


# ---------------------------------------------------------------------------
# Per-chunk content_blocks: thought / executableCode / codeExecutionResult
# ---------------------------------------------------------------------------


class TestPerChunkContentBlocks:
    """Pin how a chunk's typed payload realises as ``content_blocks``.

    ``provider_meta.content_blocks`` is the structured representation that
    downstream surfaces (renderers, FTS indexing, MCP) consume. The contract:
    a thought chunk emits a ``thinking`` block; ``executableCode`` emits a
    ``code`` block carrying the source; ``codeExecutionResult`` emits a
    ``tool_result`` block carrying the output.
    """

    def test_thought_chunk_emits_thinking_block(self) -> None:
        payload = _load_catalog("code_execution_prompt.json")
        session = _parse(payload, "code_execution_prompt")
        thought_msgs = [m for m in session.messages if (m.provider_meta or {}).get("isThought")]
        assert len(thought_msgs) == 1
        blocks = _meta_blocks(thought_msgs[0].provider_meta)
        assert [b["type"] for b in blocks] == ["thinking"]
        assert blocks[0]["text"] == "I should use code execution."

    def test_executable_code_chunk_emits_code_block_with_source(self) -> None:
        payload = _load_catalog("code_execution_prompt.json")
        session = _parse(payload, "code_execution_prompt")
        code_msgs = [m for m in session.messages if any(b.get("type") == "code" for b in _meta_blocks(m.provider_meta))]
        assert len(code_msgs) == 1
        code_block = next(b for b in _meta_blocks(code_msgs[0].provider_meta) if b["type"] == "code")
        assert code_block["text"] == "print(2+2)"

    def test_code_execution_result_chunk_emits_tool_result_block(self) -> None:
        payload = _load_catalog("code_execution_prompt.json")
        session = _parse(payload, "code_execution_prompt")
        result_blocks = [
            block
            for m in session.messages
            for block in _meta_blocks(m.provider_meta)
            if block.get("type") == "tool_result"
        ]
        assert len(result_blocks) == 1
        # The on-disk output ("4\n") is surfaced verbatim, not the outcome label.
        result_text = result_blocks[0]["text"]
        assert isinstance(result_text, str)
        assert result_text.strip() == "4"

    def test_text_chunk_emits_single_text_block(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        for message in session.messages:
            blocks = _meta_blocks(message.provider_meta)
            assert [b["type"] for b in blocks] == ["text"]
            assert blocks[0]["text"] == message.text


# ---------------------------------------------------------------------------
# Metadata roundtrip -- provider_meta carries the raw chunk + typed fields
# ---------------------------------------------------------------------------


class TestMetadataRoundtrip:
    """Pin the per-message ``provider_meta`` roundtrip contract.

    ``provider_meta`` is the bridge between provider-native JSON and the
    archive's typed representation. Two invariants matter:

    1. The original chunk dictionary is preserved verbatim under ``raw`` so
       lossless reconstruction is possible.
    2. Typed Gemini fields (``isThought``, ``tokenCount``, ``finishReason``,
       ``thinkingBudget``, ``safetyRatings``, ``executableCode``,
       ``codeExecutionResult``) appear at the top level when present so
       downstream consumers do not have to re-parse ``raw``.
    """

    def test_raw_chunk_is_preserved_verbatim_under_provider_meta_raw(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        original_chunks = copy.deepcopy(cast(dict[str, Any], payload["chunkedPrompt"])["chunks"])
        session = _parse(payload, "text_only_prompt")
        observed_raw = [(m.provider_meta or {}).get("raw") for m in session.messages]
        assert observed_raw == original_chunks

    def test_token_count_and_finish_reason_surface_at_top_level(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        assistant_msgs = [m for m in session.messages if m.role == Role.ASSISTANT]
        assert assistant_msgs
        first_meta = assistant_msgs[0].provider_meta or {}
        assert first_meta.get("tokenCount") == 8
        assert first_meta.get("finishReason") == "STOP"

    def test_thinking_budget_and_is_thought_surface_at_top_level(self) -> None:
        payload = _load_catalog("code_execution_prompt.json")
        session = _parse(payload, "code_execution_prompt")
        thought_msgs = [m for m in session.messages if (m.provider_meta or {}).get("isThought")]
        assert len(thought_msgs) == 1
        meta = thought_msgs[0].provider_meta or {}
        assert meta.get("isThought") is True
        assert meta.get("thinkingBudget") == 256

    def test_safety_ratings_round_trip_into_provider_meta(self) -> None:
        payload = _load_catalog("multi_turn_prompt.json")
        session = _parse(payload, "multi_turn_prompt")
        rated = [m for m in session.messages if isinstance((m.provider_meta or {}).get("safetyRatings"), list)]
        assert len(rated) == 1
        ratings = (rated[0].provider_meta or {})["safetyRatings"]
        assert isinstance(ratings, list)
        assert ratings == [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}]

    def test_executable_code_and_result_round_trip_into_provider_meta(self) -> None:
        payload = _load_catalog("code_execution_prompt.json")
        session = _parse(payload, "code_execution_prompt")
        exec_meta = next(
            (m.provider_meta or {}) for m in session.messages if (m.provider_meta or {}).get("executableCode")
        )
        assert exec_meta["executableCode"] == {"language": "PYTHON", "code": "print(2+2)"}
        result_meta = next(
            (m.provider_meta or {}) for m in session.messages if (m.provider_meta or {}).get("codeExecutionResult")
        )
        assert result_meta["codeExecutionResult"] == {"outcome": "OUTCOME_OK", "output": "4\n"}

    def test_reparse_after_provider_meta_serialisation_yields_identical_messages(self) -> None:
        """The roundtrip end-to-end: parse -> serialise provider_meta.raw back
        into a synthetic payload -> reparse -> compare message shape.

        This is the strongest possible "metadata round-trips" claim: the raw
        chunk surfaced under provider_meta is sufficient to reconstruct the
        same parser output. If a future change to the parser drops a field
        from ``raw``, this test catches it.
        """
        payload = _load_catalog("multi_turn_prompt.json")
        original = _parse(payload, "multi_turn_prompt")
        reconstructed_chunks = [(m.provider_meta or {}).get("raw") for m in original.messages]
        # No chunk should be missing its raw projection.
        assert all(isinstance(chunk, dict) for chunk in reconstructed_chunks)
        synthetic_payload: dict[str, Any] = {
            "id": payload["id"],
            "displayName": payload.get("displayName"),
            "createTime": payload.get("createTime"),
            "updateTime": payload.get("updateTime"),
            "chunkedPrompt": {"chunks": reconstructed_chunks},
        }
        replay = _parse(cast(JSONDocument, synthetic_payload), "multi_turn_prompt")
        assert [m.role for m in replay.messages] == [m.role for m in original.messages]
        assert [m.text for m in replay.messages] == [m.text for m in original.messages]
        assert [m.timestamp for m in replay.messages] == [m.timestamp for m in original.messages]
        assert [(m.provider_meta or {}).get("tokenCount") for m in replay.messages] == [
            (m.provider_meta or {}).get("tokenCount") for m in original.messages
        ]
        assert [(m.provider_meta or {}).get("isThought") for m in replay.messages] == [
            (m.provider_meta or {}).get("isThought") for m in original.messages
        ]


# ---------------------------------------------------------------------------
# Session-level metadata: title, timestamps, fallback id
# ---------------------------------------------------------------------------


class TestSessionLevelMetadata:
    """Pin the session envelope contract built around ``chunkedPrompt``."""

    def test_title_prefers_title_field_with_imported_source_marker(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        assert session.title == "Capital trivia"
        assert (session.provider_meta or {}).get("title_source") == "imported:title"

    def test_title_falls_back_to_display_name_when_title_absent(self) -> None:
        payload = _load_catalog("multi_turn_prompt.json")
        session = _parse(payload, "multi_turn_prompt")
        assert session.title == "Drive doc summary"
        assert (session.provider_meta or {}).get("title_source") == "imported:displayName"

    def test_title_falls_back_to_id_when_no_title_or_display_name(self) -> None:
        payload: JSONDocument = {
            "id": "no-title",
            "chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]},
        }
        session = _parse(payload, "no-title")
        assert session.title == "no-title"
        assert (session.provider_meta or {}).get("title_source") == "fallback:id"

    def test_provider_session_id_uses_id_field_then_fallback(self) -> None:
        payload: JSONDocument = {
            "title": "Untitled",
            "chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]},
        }
        session = _parse(payload, "the-fallback-id")
        assert session.provider_session_id == "the-fallback-id"

    def test_create_and_update_time_come_from_envelope_when_present(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        assert session.created_at == "2025-02-01T08:00:00Z"
        assert session.updated_at == "2025-02-01T08:00:30Z"

    def test_create_and_update_time_fall_back_to_chunk_timestamps(self) -> None:
        """Without envelope-level timestamps, the parser must derive
        ``created_at`` from the earliest chunk timestamp and ``updated_at``
        from the latest, sorted by parsed instant."""
        payload: JSONDocument = {
            "id": "no-envelope-ts",
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "first", "createTime": "2025-03-01T11:00:00Z"},
                    {"role": "model", "text": "middle", "createTime": "2025-03-01T11:00:05Z"},
                    {"role": "user", "text": "last", "createTime": "2025-03-01T11:00:30Z"},
                ]
            },
        }
        session = _parse(payload, "no-envelope-ts")
        assert session.created_at == "2025-03-01T11:00:00Z"
        assert session.updated_at == "2025-03-01T11:00:30Z"


# ---------------------------------------------------------------------------
# Per-chunk timestamp pinning
# ---------------------------------------------------------------------------


class TestPerChunkTimestamps:
    """Pin the per-message timestamp contract.

    A chunk's own ``createTime`` (or ``timestamp`` / ``updateTime``) drives
    the message timestamp; when absent the parser falls back to the
    envelope-level ``createTime``. This guards against silent loss of
    per-turn temporal information when round-tripping.
    """

    def test_per_chunk_create_time_propagates_to_message_timestamp(self) -> None:
        payload = _load_catalog("text_only_prompt.json")
        session = _parse(payload, "text_only_prompt")
        assert [m.timestamp for m in session.messages] == [
            "2025-02-01T08:00:00Z",
            "2025-02-01T08:00:05Z",
            "2025-02-01T08:00:20Z",
            "2025-02-01T08:00:30Z",
        ]

    def test_missing_chunk_timestamp_falls_back_to_envelope_create_time(self) -> None:
        payload: JSONDocument = {
            "id": "ts-fallback",
            "createTime": "2025-04-01T12:00:00Z",
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "no-ts"},
                    {"role": "model", "text": "explicit-ts", "createTime": "2025-04-01T12:30:00Z"},
                ]
            },
        }
        session = _parse(payload, "ts-fallback")
        assert [m.timestamp for m in session.messages] == [
            "2025-04-01T12:00:00Z",
            "2025-04-01T12:30:00Z",
        ]
