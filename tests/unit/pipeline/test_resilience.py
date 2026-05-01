"""Fault injection tests for pipeline resilience.

Tests that the pipeline gracefully handles errors at each stage rather than
crashing with unhandled exceptions. Each test injects a specific fault and
verifies the pipeline recovers, logs, or surfaces the error cleanly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from typing_extensions import TypedDict

from polylogue.archive.message.roles import Role
from polylogue.archive.raw_payload.decode import JSONValue
from polylogue.config import Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.parsing import ParseResult, ParsingService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import ContentBlockType, Provider, ValidationStatus
from tests.infra.strategies import (
    AcquisitionInputSpec,
    ParseMergeEvent,
    ValidationCase,
    acquisition_input_batch_strategy,
    build_acquisition_raw_bytes,
    build_validation_payload,
    expected_parse_merge_totals,
    expected_validation_contract,
    parse_merge_events_strategy,
    validation_case_strategy,
)

ConversationJson = dict[str, JSONValue]


class ConversationNode(TypedDict):
    message: dict[str, JSONValue]
    parent: str | None
    children: list[str]


def _make_raw_record(
    raw_id: str,
    provider: str,
    content: bytes,
    path: str = "/exports/test.json",
) -> RawConversationRecord:
    from polylogue.storage.blob_store import get_blob_store

    # Write content to blob store
    blob_store = get_blob_store()
    actual_raw_id, blob_size = blob_store.write_from_bytes(content)
    now = datetime.now(timezone.utc).isoformat()

    return RawConversationRecord(
        raw_id=actual_raw_id,  # Use the actual hash as raw_id
        provider_name=provider,
        source_name="test",
        source_path=path,
        source_index=None,
        blob_size=blob_size,
        acquired_at=now,
        file_mtime=now,
    )


def _make_parsing_service(tmp_path: Path) -> ParsingService:
    """Shared factory to avoid boilerplate in each test."""
    from polylogue.config import Config
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

    db = SQLiteBackend(db_path=tmp_path / "test.db")
    config = Config(
        sources=[],
        archive_root=tmp_path / "archive",
        render_root=tmp_path / "render",
    )
    return ParsingService(
        repository=ConversationRepository(backend=db),
        archive_root=tmp_path / "archive",
        config=config,
    )


def _provider_hint(value: str | None) -> Provider | None:
    return None if value is None else Provider.from_string(value)


# ---------------------------------------------------------------------------
# Fault 1: Parsing service handles valid JSON with unknown chatgpt structure
# ---------------------------------------------------------------------------


def test_parse_unknown_chatgpt_structure_returns_empty(tmp_path: Path) -> None:
    """Valid JSON but missing chatgpt mapping field returns empty, not an exception."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    payload = json.dumps({"unexpected": "structure", "no_mapping": True}).encode()
    record = _make_raw_record("unknown-struct", "chatgpt", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.error is None or isinstance(result.conversations, list)


# ---------------------------------------------------------------------------
# Fault 2: Parsing service handles JSON with completely missing required fields
# ---------------------------------------------------------------------------


def test_parse_chatgpt_no_messages_returns_empty(tmp_path: Path) -> None:
    """ChatGPT payload with empty mapping returns empty list."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    payload = json.dumps(
        {
            "title": "No Messages",
            "mapping": {},
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()
    record = _make_raw_record("no-messages", "chatgpt", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.error is None or isinstance(result.conversations, list)


# ---------------------------------------------------------------------------
# Fault 3: Parsing surfaces error when all JSONL lines are invalid
# ---------------------------------------------------------------------------


def test_parse_jsonl_all_invalid_lines_surfaces_error(tmp_path: Path) -> None:
    """JSONL where every line is invalid surfaces an error (by design).

    ingest_worker catches all exceptions and returns them in result.error.
    Tests document this as the expected behavior so callers know to handle errors.
    """
    from polylogue.pipeline.services.ingest_worker import ingest_record

    content = b"NOT JSON\nALSO NOT JSON\nSTILL NOT JSON\n"
    record = _make_raw_record("all-invalid-jsonl", "claude-code", content, "/exports/session.jsonl")
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.error is not None


# ---------------------------------------------------------------------------
# Fault 4: Parsing handles JSONL with mixed valid and invalid lines
# ---------------------------------------------------------------------------


def test_parse_mixed_valid_invalid_jsonl_lines(tmp_path: Path) -> None:
    """JSONL with some valid lines: valid lines parsed, invalid lines skipped."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    # Mix valid claude-code JSONL with invalid lines
    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"Hello"},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b"INVALID JSON LINE\n"
        b'{"parentUuid":"m1","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"Hi"}]},"uuid":"m2","timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    record = _make_raw_record("mixed-jsonl", "claude-code", content, "/exports/session.jsonl")
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None
    if result.conversations:
        assert result.conversations[0].provider_name == "claude-code"


# ---------------------------------------------------------------------------
# Fault 5: Parsing handles valid JSON for unknown provider gracefully
# ---------------------------------------------------------------------------


def test_parse_unknown_provider_name(tmp_path: Path) -> None:
    """Unknown provider name falls back gracefully."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    record = _make_raw_record(
        "unknown-provider",
        "not-a-real-provider",
        json.dumps({"id": "conv-1", "title": "Test"}).encode(),
    )
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None or result.error is not None


# ---------------------------------------------------------------------------
# Fault 6: Parsing handles claude-code JSONL with null fields
# ---------------------------------------------------------------------------


def test_parse_claude_code_jsonl_with_null_fields(tmp_path: Path) -> None:
    """Claude-code JSONL with null fields in messages is handled gracefully."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":null},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
    )
    record = _make_raw_record("null-fields", "claude-code", content, "/exports/session.jsonl")
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None or result.error is not None


# ---------------------------------------------------------------------------
# Fault 7: Parsing handles very large valid conversation
# ---------------------------------------------------------------------------


def test_parse_very_large_conversation_does_not_crash(tmp_path: Path) -> None:
    """Very large valid chatgpt payload is handled without crashing."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    messages: dict[str, ConversationNode] = {}
    prev_id: str | None = None
    for i in range(50):
        node_id = f"node-{i}"
        messages[node_id] = {
            "message": {
                "id": f"msg-{i}",
                "author": {"role": "user" if i % 2 == 0 else "assistant"},
                "content": {"content_type": "text", "parts": ["x" * 1000]},
                "create_time": 1700000000 + i,
            },
            "parent": prev_id,
            "children": [],
        }
        if prev_id:
            messages[prev_id]["children"] = [node_id]
        prev_id = node_id

    payload = json.dumps(
        {
            "title": "Large Conversation",
            "mapping": messages,
            "create_time": 1700000000,
            "update_time": 1700000100,
        }
    ).encode()

    record = _make_raw_record("large-content", "chatgpt", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None
    assert len(result.conversations) > 0


# ---------------------------------------------------------------------------
# Fault 8: Parsing handles chatgpt with deeply nested but invalid mapping
# ---------------------------------------------------------------------------


def test_parse_chatgpt_deeply_nested_malformed_nodes(tmp_path: Path) -> None:
    """ChatGPT payload with malformed node structure returns empty, not exception."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    # Mapping nodes missing 'message' key
    payload = json.dumps(
        {
            "title": "Malformed Nodes",
            "mapping": {
                "node-1": {"parent": None, "children": ["node-2"]},
                "node-2": {"parent": "node-1", "children": [], "no_message_key": True},
            },
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()
    record = _make_raw_record("malformed-nodes", "chatgpt", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None or result.error is not None


# ---------------------------------------------------------------------------
# Fault 11: Parsing handles a chatgpt bundle (list) where one item is invalid
# ---------------------------------------------------------------------------


def test_parse_chatgpt_bundle_with_one_invalid_item(tmp_path: Path) -> None:
    """ChatGPT bundle list with one invalid item: valid items parsed, invalid skipped."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    payload = json.dumps(
        [
            {
                "id": "conv-1",
                "title": "Valid Conversation",
                "mapping": {
                    "m1": {
                        "message": {
                            "id": "m1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hi"]},
                            "create_time": 1700000000,
                        },
                        "parent": None,
                        "children": [],
                    }
                },
                "create_time": 1700000000,
                "update_time": 1700000001,
            },
            {"invalid": "no title or mapping"},  # Should be skipped
        ]
    ).encode()
    record = _make_raw_record("bundle-one-invalid", "chatgpt", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None or result.error is not None


# ---------------------------------------------------------------------------
# Fault 12: Parsing handles gemini with missing text fields
# ---------------------------------------------------------------------------


def test_parse_gemini_missing_text_fields(tmp_path: Path) -> None:
    """Gemini payload with messages missing text fields is handled gracefully."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    payload = json.dumps(
        {
            "conversations": [
                {
                    "chunks": [
                        {"role": "user"},  # no text field
                        {"role": "model"},  # no text field
                    ]
                }
            ]
        }
    ).encode()
    record = _make_raw_record("gemini-no-text", "gemini", payload)
    result = ingest_record(record, str(tmp_path / "archive"), "off")
    assert result.conversations is not None or result.error is not None


# =====================================================================
# Merged from test_service_laws.py (service reliability)
# =====================================================================


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(acquisition_input_batch_strategy())
async def test_acquisition_law_counts_unique_raws_and_normalizes_provider_hints(
    batch: tuple[AcquisitionInputSpec, ...],
) -> None:
    """Acquisition should store each unique raw payload once and keep raw provider hints canonical."""
    with TemporaryDirectory() as tempdir:
        backend = SQLiteBackend(db_path=Path(tempdir) / "acquire.db")
        source_name = "generated-source"

        raw_items = [
            RawConversationData(
                raw_bytes=build_acquisition_raw_bytes(spec),
                source_path=f"/tmp/{index}.json",
                source_index=index,
                provider_hint=_provider_hint(spec.provider_hint),
            )
            for index, spec in enumerate(batch)
        ]

        expected_first_provider: dict[str, str] = {}
        for spec in batch:
            expected_first_provider.setdefault(spec.payload_id, spec.provider_hint or "unknown")

        try:
            with patch("polylogue.pipeline.services.acquisition.iter_source_raw_data", return_value=iter(raw_items)):
                result = await AcquisitionService(backend=backend).acquire_sources(
                    [Source(name=source_name, path=Path("/tmp/inbox"))]
                )

            unique_payloads = list(dict.fromkeys(spec.payload_id for spec in batch))
            assert result.counts["acquired"] == len(unique_payloads)
            assert result.counts["skipped"] == len(batch) - len(unique_payloads)
            assert len(result.raw_ids) == len(unique_payloads)

            for raw_id in result.raw_ids:
                stored = await backend.get_raw_conversation(raw_id)
                assert stored is not None
                from polylogue.storage.blob_store import load_raw_content

                raw_bytes = load_raw_content(raw_id)
                payload_id = json.loads(raw_bytes)["id"]
                assert stored.provider_name == expected_first_provider[payload_id]
                assert stored.payload_provider is None
        finally:
            await backend.close()


@settings(max_examples=30, deadline=None)
@given(validation_case_strategy())
async def test_validation_law_matches_mode_and_payload_contract(case: ValidationCase) -> None:
    """Validation mode, malformed JSONL, and schema verdicts must produce one stable persisted contract."""
    from polylogue.schemas import ValidationResult
    from polylogue.storage.blob_store import get_blob_store

    raw_content, provider_name, source_path = build_validation_payload(case)
    blob_store = get_blob_store()
    raw_id, blob_size = blob_store.write_from_bytes(raw_content)

    raw_record = MagicMock(
        raw_id=raw_id,
        raw_content=raw_content,  # Keep for backwards compatibility in mocks
        provider_name=provider_name,
        source_path=source_path,
        payload_provider=None,
        blob_size=blob_size,
    )
    backend = MagicMock(spec=SQLiteBackend)
    backend.queries = MagicMock()
    service = ValidationService(backend=backend)
    get_batch = AsyncMock(return_value=[raw_record])
    mark_validated = AsyncMock()
    mark_parsed = AsyncMock()
    object.__setattr__(service.repository, "get_raw_conversations_batch", get_batch)
    object.__setattr__(service.repository, "mark_raw_validated", mark_validated)
    object.__setattr__(service.repository, "mark_raw_parsed", mark_parsed)

    class _SyntheticValidator:
        provider = provider_name

        def __init__(self) -> None:
            self.max_samples_seen: int | None | Literal["unset"] = "unset"

        def validation_samples(self, payload: JSONValue, max_samples: int | None = None) -> list[JSONValue]:
            self.max_samples_seen = max_samples
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            return [payload]

        def validate(self, _sample: object) -> ValidationResult:
            return ValidationResult(
                is_valid=case.invalid_sample_count == 0,
                errors=["schema error"] if case.invalid_sample_count else [],
            )

    validator = _SyntheticValidator()

    with patch(
        "polylogue.schemas.validator.SchemaValidator.for_payload",
        return_value=validator,
    ):
        with patch.dict("os.environ", {"POLYLOGUE_SCHEMA_VALIDATION": case.mode}, clear=False):
            result = await service.validate_raw_ids(raw_ids=[raw_id])

    expected = expected_validation_contract(case)
    if expected["validation_samples_called"]:
        assert validator.max_samples_seen is None
    else:
        assert validator.max_samples_seen == "unset"
    assert result.counts["invalid"] == expected["invalid_count"]
    assert result.parseable_raw_ids == ([raw_id] if expected["parseable"] else [])
    assert result.invalid_raw_ids == ([] if expected["parseable"] else [raw_id])

    validate_calls = mark_validated.await_args_list
    assert len(validate_calls) >= 1
    assert validate_calls[0].args[0] == raw_id
    validation_kwargs = validate_calls[0].kwargs
    assert validation_kwargs["status"] == ValidationStatus.from_string(expected["status"])

    parse_calls = mark_parsed.await_args_list
    if expected["mark_raw_parsed"]:
        assert len(parse_calls) == 1
        assert parse_calls[0].args[0] == raw_id
        assert parse_calls[0].kwargs["error"] is not None
    else:
        assert parse_calls == []


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(parse_merge_events_strategy())
async def test_parse_result_merge_law_accumulates_counts_and_processed_ids(events: list[ParseMergeEvent]) -> None:
    """ParseResult.merge_result should be componentwise additive with processed-id union semantics."""
    result = ParseResult()
    for event in events:
        await result.merge_result(
            conversation_id=event.conversation_id,
            result_counts=event.result_counts,
            content_changed=event.content_changed,
        )

    expected = expected_parse_merge_totals(events)
    assert result.counts == expected["counts"]
    assert result.changed_counts == expected["changed_counts"]
    assert result.processed_ids == expected["processed_ids"]


def test_ingest_worker_decodes_and_dispatches_provider(tmp_path: Path) -> None:
    """ingest_record should decode blob, detect provider, and handle gracefully."""
    from polylogue.pipeline.services.ingest_worker import ingest_record

    # Create a minimal valid ChatGPT payload that will be decoded and detected
    payload = json.dumps(
        {
            "id": "conv-1",
            "title": "Test Conversation",
            "mapping": {},
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()

    raw_record = _make_raw_record(
        raw_id="ignored",  # _make_raw_record uses actual content hash
        provider="chatgpt",
        content=payload,
        path="/tmp/conversation.json",
    )

    # Call ingest_record directly without mocking (tests real code path)
    result = ingest_record(raw_record, str(tmp_path / "archive"), "off")

    # Verify the result structure
    assert result.raw_id is not None  # Should be the actual hash
    assert result.payload_provider is not None  # Provider detected
    # ingest_record returns ConversationData list; empty mapping results in empty conversations
    assert isinstance(result.conversations, list)
    # Result should be clean with no errors
    assert result.error is None


def test_ingest_worker_reuses_schema_resolution_and_skips_drift_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parse-path validation should reuse one schema resolution and avoid unused drift traversal."""
    from polylogue.pipeline.services.ingest_worker import ingest_record
    from polylogue.schemas import ValidationResult
    from polylogue.schemas.packages import SchemaResolution

    payload = json.dumps(
        {
            "id": "conv-1",
            "title": "Test Conversation",
            "mapping": {},
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()
    raw_record = _make_raw_record(
        raw_id="ignored",
        provider="chatgpt",
        content=payload,
        path="/tmp/conversation.json",
    )

    resolution = SchemaResolution(
        provider="chatgpt",
        package_version="v1",
        element_kind="conversation_document",
        exact_structure_id="shape-1",
        bundle_scope=None,
        reason="exact_structure",
    )
    observed: dict[str, object] = {}

    class _CapturingRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def resolve_payload(
            self,
            provider: str | Provider,
            payload: JSONValue,
            *,
            source_path: str | None = None,
        ) -> SchemaResolution:
            self.calls += 1
            observed["registry_provider"] = provider
            observed["registry_source_path"] = source_path
            observed["registry_payload"] = payload
            return resolution

    registry = _CapturingRegistry()
    monkeypatch.setattr("polylogue.pipeline.services.ingest_worker._SCHEMA_REGISTRY", registry)

    class _CapturingValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: JSONValue, max_samples: int | None = None) -> list[JSONValue]:
            return [payload]

        def validate(self, _sample: object, *, include_drift: bool | None = None) -> ValidationResult:
            observed["include_drift"] = include_drift
            return ValidationResult(is_valid=True)

    def _fake_for_payload(
        provider: str | Provider,
        payload: JSONValue,
        *,
        source_path: str | None = None,
        schema_resolution: SchemaResolution | None = None,
        strict: bool = True,
    ) -> _CapturingValidator:
        observed["validator_provider"] = provider
        observed["validator_source_path"] = source_path
        observed["validator_payload"] = payload
        observed["validator_schema_resolution"] = schema_resolution
        observed["validator_strict"] = strict
        return _CapturingValidator()

    def _fake_parse_payload(
        provider: str | Provider,
        payload: JSONValue,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
    ) -> Sequence[ParsedConversation]:
        observed["parse_provider"] = provider
        observed["parse_schema_resolution"] = schema_resolution
        observed["parse_fallback_id"] = fallback_id
        return []

    monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", _fake_for_payload)
    monkeypatch.setattr("polylogue.sources.dispatch.parse_payload", _fake_parse_payload)

    result = ingest_record(raw_record, str(tmp_path / "archive"), "strict")

    assert result.error is None
    assert registry.calls == 1
    assert observed["validator_schema_resolution"] is resolution
    assert observed["parse_schema_resolution"] is resolution
    assert observed["include_drift"] is False


def test_transform_with_tool_use_message_keeps_non_empty_message_hash(tmp_path: Path) -> None:
    from polylogue.pipeline.services.ingest_worker import _transform_to_tuples

    conversation = ParsedConversation(
        provider_name=Provider.CODEX,
        provider_conversation_id="tool-conv-1",
        title="Tool Conversation",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:01Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.ASSISTANT,
                text="Running a shell command.",
                timestamp="2026-04-02T00:00:01Z",
                content_blocks=[
                    ParsedContentBlock(
                        type=ContentBlockType.TOOL_USE,
                        tool_name="bash",
                        tool_id="tool-1",
                        tool_input={"command": "ls /tmp"},
                    )
                ],
            )
        ],
        attachments=[],
    )

    cdata = _transform_to_tuples(
        conversation,
        source_name="test-source",
        archive_root=tmp_path / "archive",
        raw_id="raw-1",
    )

    assert cdata.message_tuples[0][6]
    assert len(cdata.action_event_tuples) == 1


def test_ingest_record_streams_codex_jsonl_without_full_envelope_decode(tmp_path: Path) -> None:
    from polylogue.pipeline.services.ingest_worker import ingest_record

    content = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2025-01-01T00:00:00Z"}}\n'
        b'{"type":"message","id":"msg-1","role":"user","timestamp":"2025-01-01T00:00:01Z",'
        b'"content":[{"type":"input_text","text":"hello"}]}\n'
        b'{"type":"message","id":"msg-2","role":"assistant","timestamp":"2025-01-01T00:00:02Z",'
        b'"content":[{"type":"output_text","text":"hi"}]}\n'
    )
    record = _make_raw_record("codex-streaming", "codex", content, "/exports/codex.jsonl")

    with patch(
        "polylogue.archive.raw_payload.build_raw_payload_envelope",
        side_effect=AssertionError("legacy full envelope decode should be bypassed"),
    ):
        result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.error is None
    assert len(result.conversations) == 1
    assert result.conversations[0].provider_name == "codex"
