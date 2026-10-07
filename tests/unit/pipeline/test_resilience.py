"""Current decoder/parser edge cases and pipeline service laws."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from typing_extensions import TypedDict

from polylogue.archive.message.roles import Role
from polylogue.archive.raw_payload.decode import JSONValue, RawPayloadEnvelope
from polylogue.config import Source
from polylogue.core.enums import BlockType, Provider, ValidationStatus
from polylogue.core.json import JSONDecodeError
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.parsing import ParseResult
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    RawSessionData,
)
from polylogue.sources.retained_acquisition import SourceInputRecord
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session
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

pytestmark = pytest.mark.uses_real_clock(
    "Resilience harness sets acquired_at to now() as opaque metadata; no production timing comparison."
)


class SessionNode(TypedDict):
    message: dict[str, JSONValue]
    parent: str | None
    children: list[str]


def _provider_hint(value: str | None) -> Provider | None:
    return None if value is None else Provider.from_string(value)


# Parser robustness laws exercise the current decoder and dispatch boundary.
# The removed ingest_worker was a private writer mechanism; retained publication
# and refusal outcomes are owned by test_raw_observation_derivation and
# test_retained_schema_drift_route.


def _decode_and_parse(
    content: bytes, provider: Provider, source_path: str, fallback_id: str
) -> tuple[RawPayloadEnvelope, list[ParsedSession]]:
    from polylogue.archive.raw_payload.decode import build_raw_payload_envelope
    from polylogue.sources.dispatch import parse_payload

    envelope = build_raw_payload_envelope(
        content,
        source_path=source_path,
        fallback_provider=provider,
        jsonl_dict_only=source_path.endswith((".jsonl", ".jsonl.txt.json")),
    )
    return envelope, parse_payload(envelope.provider, envelope.payload, fallback_id, source_path=source_path)


def test_chatgpt_unknown_and_empty_documents_produce_no_sessions() -> None:
    for payload in (
        {"unexpected": "structure", "no_mapping": True},
        {"title": "No Messages", "mapping": {}, "create_time": 1700000000, "update_time": 1700000001},
    ):
        envelope, sessions = _decode_and_parse(
            json.dumps(payload).encode(), Provider.CHATGPT, "session.json", "fallback"
        )
        assert envelope.wire_format == "json"
        assert sessions == []


def test_all_invalid_jsonl_is_a_typed_decode_refusal() -> None:
    from polylogue.archive.raw_payload.decode import build_raw_payload_envelope

    with pytest.raises(JSONDecodeError):
        build_raw_payload_envelope(
            b"NOT JSON\nALSO NOT JSON\nSTILL NOT JSON\n",
            source_path="session.jsonl",
            fallback_provider=Provider.CLAUDE_CODE,
            jsonl_dict_only=True,
        )


def test_mixed_valid_invalid_claude_jsonl_keeps_valid_messages() -> None:
    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"Hello"},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b"INVALID JSON LINE\n"
        b'{"parentUuid":"m1","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"Hi"}]},"uuid":"m2","timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    envelope, sessions = _decode_and_parse(content, Provider.CLAUDE_CODE, "session.jsonl", "fallback")
    assert envelope.malformed_jsonl_lines == 1
    assert len(sessions) == 1
    assert sessions[0].source_name is Provider.CLAUDE_CODE
    assert [message.text for message in sessions[0].messages if message.text] == ["Hello", "Hi"]


def test_jsonl_shape_precedes_drive_cache_json_suffix() -> None:
    content = (
        b'{"type":"summary","summary":"continued session","leafUuid":"leaf-1"}\n'
        b'{"parentUuid":null,"isSidechain":false,"userType":"external","cwd":"/fixture",'
        b'"sessionId":"session-from-drive","version":"1.0.6","type":"user",'
        b'"timestamp":"2025-06-06T12:55:19.000Z","uuid":"m1",'
        b'"message":{"role":"user","content":[{"type":"text","text":"hello from drive cache"}]}}\n'
        b'{"parentUuid":"m1","isSidechain":false,"userType":"external","cwd":"/fixture",'
        b'"sessionId":"session-from-drive","version":"1.0.6","type":"assistant",'
        b'"timestamp":"2025-06-06T12:55:20.000Z","uuid":"m2",'
        b'"message":{"role":"assistant","content":[{"type":"text","text":"parsed as stream"}]}}\n'
    )
    envelope, sessions = _decode_and_parse(
        content, Provider.GEMINI, "/drive-cache/gemini/session-from-drive.jsonl.txt.json", "fallback"
    )
    assert envelope.wire_format == "jsonl"
    assert envelope.provider is Provider.CLAUDE_CODE
    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "session-from-drive"
    assert [message.text for message in sessions[0].messages if message.text] == [
        "continued session",
        "hello from drive cache",
        "parsed as stream",
    ]


def test_claude_jsonl_null_content_is_not_materialized() -> None:
    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":null},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
    )
    _envelope, sessions = _decode_and_parse(content, Provider.CLAUDE_CODE, "session.jsonl", "fallback")
    assert sessions == []


def test_large_chatgpt_session_parses_all_messages() -> None:
    mapping: dict[str, SessionNode] = {}
    previous: str | None = None
    for index in range(50):
        node_id = f"node-{index}"
        mapping[node_id] = {
            "message": {
                "id": f"msg-{index}",
                "author": {"role": "user" if index % 2 == 0 else "assistant"},
                "content": {"content_type": "text", "parts": ["x" * 1000]},
                "create_time": 1700000000 + index,
            },
            "parent": previous,
            "children": [],
        }
        if previous is not None:
            mapping[previous]["children"] = [node_id]
        previous = node_id
    payload = {
        "title": "Large Session",
        "mapping": mapping,
        "create_time": 1700000000,
        "update_time": 1700000100,
    }
    _envelope, sessions = _decode_and_parse(json.dumps(payload).encode(), Provider.CHATGPT, "session.json", "fallback")
    assert len(sessions) == 1
    assert len(sessions[0].messages) == 50
    assert all(len(message.text or "") == 1000 for message in sessions[0].messages)


def test_malformed_chatgpt_nodes_are_skipped_without_losing_valid_sibling() -> None:
    payload = {
        "title": "Mixed Nodes",
        "mapping": {
            "bad": {"parent": None, "children": [], "no_message_key": True},
            "good": {
                "parent": None,
                "children": [],
                "message": {
                    "id": "good-message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["kept"]},
                    "create_time": 1700000000,
                },
            },
        },
        "create_time": 1700000000,
        "update_time": 1700000001,
    }
    _envelope, sessions = _decode_and_parse(json.dumps(payload).encode(), Provider.CHATGPT, "session.json", "fallback")
    assert len(sessions) == 1
    assert [message.text for message in sessions[0].messages if message.text] == ["kept"]


def test_chatgpt_bundle_keeps_valid_item_and_skips_invalid_item() -> None:
    valid = {
        "id": "conv-1",
        "title": "Valid Session",
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
    }
    _envelope, sessions = _decode_and_parse(
        json.dumps([valid, {"invalid": "no title or mapping"}]).encode(), Provider.CHATGPT, "bundle.json", "fallback"
    )
    assert [session.provider_session_id for session in sessions] == ["conv-1"]
    assert [message.text for message in sessions[0].messages if message.text] == ["Hi"]


def test_gemini_missing_text_fields_produce_no_sessions() -> None:
    payload = {"sessions": [{"chunks": [{"role": "user"}, {"role": "model"}]}]}
    _envelope, sessions = _decode_and_parse(json.dumps(payload).encode(), Provider.GEMINI, "session.json", "fallback")
    assert sessions == []


# =====================================================================
# Merged from test_service_laws.py (service reliability)
# =====================================================================


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(acquisition_input_batch_strategy(max_items=4))
async def test_acquisition_law_preserves_coordinates_deduplicates_blobs_and_normalizes_provider_hints(
    batch: tuple[AcquisitionInputSpec, ...],
) -> None:
    """Acquisition preserves observations while identical payloads share one blob."""
    from polylogue.daemon.drive_catchup import DriveCatchupExecution
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.live_ingest import prepared_live_convergence_owner

    with TemporaryDirectory() as tempdir:
        archive_root = Path(tempdir)
        run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
        backend = SQLiteBackend(db_path=archive_root / "index.db")
        source_name = "generated-source"

        raw_items = [
            RawSessionData(
                raw_bytes=build_acquisition_raw_bytes(spec),
                source_path=f"/tmp/{index}.json",
                # Every file-backed raw carries the canonical path acquisition froze.
                canonical_source_path=f"/tmp/{index}.json",
                source_index=index,
                provider_hint=_provider_hint(spec.provider_hint),
            )
            for index, spec in enumerate(batch)
        ]

        try:
            with patch(
                "polylogue.pipeline.services.acquisition.iter_source_acquisition_records",
                return_value=iter(SourceInputRecord('["physical-file-v1",0]', item) for item in raw_items),
            ):
                # Acquisition publishes through the daemon's admitted writer.
                async with prepared_live_convergence_owner(archive_root) as owner:
                    execution = DriveCatchupExecution(owner._write_coordinator, compute_adapter=owner._compute_adapter)
                    result = await AcquisitionService(backend=backend, execution=execution).acquire_sources(
                        [Source(name=source_name, path=Path("/tmp/inbox"))]
                    )

            assert result.counts["acquired"] == len(batch)
            assert result.counts["skipped"] == 0
            assert len(result.raw_ids) == len(batch)
            assert len(set(result.raw_ids)) == len(batch)

            blob_hashes: set[str] = set()
            for raw_id, spec in zip(result.raw_ids, batch, strict=True):
                stored = await backend.get_raw_session(raw_id)
                assert stored is not None
                assert stored.blob_hash is not None
                blob_hashes.add(stored.blob_hash)
                raw_bytes = BlobStore(archive_root / "blob").read_all(stored.blob_hash)
                payload_id = json.loads(raw_bytes)["id"]
                assert payload_id == spec.payload_id
                expected_provider = spec.provider_hint or "unknown"
                assert stored.source_name == expected_provider
                # #1743: raw_sessions stores a single origin column; payload_provider
                # projects from it on read (no separate nullable hint column), so the
                # normalized hint surfaces through both source_name and payload_provider.
                assert stored.payload_provider is not None
                assert stored.payload_provider.value == expected_provider
            assert len(blob_hashes) == len({spec.payload_id for spec in batch})
        finally:
            await backend.close()


@settings(max_examples=30, deadline=None)
@given(validation_case_strategy())
async def test_validation_law_matches_mode_and_payload_contract(case: ValidationCase) -> None:
    """Validation mode, malformed JSONL, and schema verdicts must produce one stable persisted contract."""
    from polylogue.schemas import ValidationResult
    from polylogue.schemas.validator import PayloadValidation, SchemaValidator
    from polylogue.storage.blob_store import get_blob_store

    raw_content, source_name, source_path = build_validation_payload(case)
    blob_store = get_blob_store()
    raw_id, blob_size = blob_store.write_from_bytes(raw_content)

    raw_record = MagicMock(
        raw_id=raw_id,
        raw_content=raw_content,  # Keep for backwards compatibility in mocks
        source_name=source_name,
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
    object.__setattr__(service.repository, "get_raw_sessions_batch", get_batch)
    object.__setattr__(service.repository, "mark_raw_validated", mark_validated)
    object.__setattr__(service.repository, "mark_raw_parsed", mark_parsed)

    class _SyntheticValidator:
        provider = source_name

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

    def _fake_validate_payload(provider: object, payload: JSONValue, **kwargs: object) -> PayloadValidation:
        del provider, kwargs
        samples = tuple(validator.validation_samples(payload))
        results = tuple(validator.validate(sample) for sample in samples)
        return PayloadValidation(
            validator=cast(SchemaValidator, validator),
            samples=samples,
            results=results,
            schema_resolution=None,
            schema_resolution_is_explicit=True,
        )

    with patch(
        "polylogue.schemas.validator.SchemaValidator.validate_payload",
        side_effect=_fake_validate_payload,
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
            session_id=event.session_id,
            result_counts=event.result_counts,
            content_changed=event.content_changed,
        )

    expected = expected_parse_merge_totals(events)
    assert result.counts == expected["counts"]
    assert result.changed_counts == expected["changed_counts"]
    assert result.processed_ids == expected["processed_ids"]


# The old ingest_worker outcome/schema hooks no longer exist. Current typed
# refusal, non-session and drift outcomes are asserted at their retained owner
# in test_raw_observation_derivation and test_retained_schema_drift_route.


def _open_index_archive(tmp_path: Path) -> sqlite3.Connection:
    """Bootstrap the archive and open its Index through a measured creator.

    The fixture writer captures Index mutations against the connection's
    original physical creator, which a bare ``sqlite3.connect`` lacks.
    """
    archive_root = bootstrap_archive_root(tmp_path / "archive")
    conn = connect_measured(archive_root / "index.db")
    conn.row_factory = sqlite3.Row
    return conn


def test_transform_with_tool_use_message_keeps_non_empty_message_hash(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="tool-conv-1",
        title="Tool Session",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:01Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.ASSISTANT,
                text="Running a shell command.",
                timestamp="2026-04-02T00:00:01Z",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="bash",
                        tool_id="tool-1",
                        tool_input={"command": "ls /tmp"},
                    )
                ],
            )
        ],
        attachments=[],
    )

    conn = _open_index_archive(tmp_path)
    try:
        session_id = write_fixture_index_session(
            conn,
            session,
            content_hash=hashlib.sha256(b"tool-conv-1").hexdigest(),
            raw_id="raw-1",
        )
        message_hashes = conn.execute(
            "SELECT content_hash FROM messages WHERE session_id = ?", (session_id,)
        ).fetchall()
        action_count = conn.execute("SELECT COUNT(*) FROM actions WHERE session_id = ?", (session_id,)).fetchone()[0]
    finally:
        close_fixture_index_connection(conn)

    assert len(message_hashes) == 1
    assert message_hashes[0]["content_hash"]
    assert action_count == 1


def test_transform_deduplicates_materialized_message_rows_by_primary_key(tmp_path: Path) -> None:
    """Duplicate provider-native ids must not collapse distinct archive rows.

    Provider message ids are only trustworthy while unique within a session.
    Large Codex streams can reuse them, so the writer stores duplicate native ids
    as NULL and falls back to position/variant archive ids.
    """
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="duplicate-message-conv",
        title="Duplicate Message Session",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:02Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="older text",
                timestamp="2026-04-02T00:00:01Z",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="older text")],
            ),
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.USER,
                text="newer text",
                timestamp="2026-04-02T00:00:02Z",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="newer text")],
            ),
        ],
        attachments=[],
    )

    conn = _open_index_archive(tmp_path)
    try:
        session_id = write_fixture_index_session(
            conn,
            session,
            content_hash=hashlib.sha256(b"duplicate-message-conv").hexdigest(),
            raw_id="raw-1",
        )
        message_rows = conn.execute(
            "SELECT message_id, native_id, position FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        block_rows = conn.execute(
            "SELECT message_id, text FROM blocks WHERE session_id = ? ORDER BY message_id",
            (session_id,),
        ).fetchall()
        session_message_count = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        close_fixture_index_connection(conn)

    # Id-less messages, so both ids come from the content fallback.
    ordered_ids = [str(row["message_id"]) for row in message_rows]
    assert [(row["native_id"], row["position"]) for row in message_rows] == [(None, 0), (None, 1)]
    assert all(value.startswith(f"{session_id}:c:") for value in ordered_ids)
    assert len(set(ordered_ids)) == 2
    assert {str(row["message_id"]): row["text"] for row in block_rows} == {
        ordered_ids[0]: "older text",
        ordered_ids[1]: "newer text",
    }
    assert session_message_count == 2
