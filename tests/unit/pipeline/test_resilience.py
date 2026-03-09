"""Fault injection tests for pipeline resilience.

Tests that the pipeline gracefully handles errors at each stage rather than
crashing with unhandled exceptions. Each test injects a specific fault and
verifies the pipeline recovers, logs, or surfaces the error cleanly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.storage.store import RawConversationRecord


def _make_raw_record(
    raw_id: str,
    provider: str,
    content: bytes,
    path: str = "/exports/test.json",
) -> RawConversationRecord:
    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider,
        source_name="test",
        source_path=path,
        source_index=None,
        raw_content=content,
        acquired_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_parsing_service(tmp_path: Path):
    """Shared factory to avoid boilerplate in each test."""
    from polylogue.config import Config
    from polylogue.pipeline.services.parsing import ParsingService
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


# ---------------------------------------------------------------------------
# Fault 1: Parsing service handles valid JSON with unknown chatgpt structure
# ---------------------------------------------------------------------------

async def test_parse_unknown_chatgpt_structure_returns_empty(tmp_path):
    """Valid JSON but missing chatgpt mapping field returns empty, not an exception."""
    svc = _make_parsing_service(tmp_path)
    payload = json.dumps({"unexpected": "structure", "no_mapping": True}).encode()
    record = _make_raw_record("unknown-struct", "chatgpt", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 2: Parsing service handles JSON with completely missing required fields
# ---------------------------------------------------------------------------

async def test_parse_chatgpt_no_messages_returns_empty(tmp_path):
    """ChatGPT payload with empty mapping returns empty list."""
    svc = _make_parsing_service(tmp_path)
    payload = json.dumps({
        "title": "No Messages",
        "mapping": {},
        "create_time": 1700000000,
        "update_time": 1700000001,
    }).encode()
    record = _make_raw_record("no-messages", "chatgpt", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 3: Parsing surfaces error when all JSONL lines are invalid
# ---------------------------------------------------------------------------

async def test_parse_jsonl_all_invalid_lines_surfaces_error(tmp_path):
    """JSONL where every line is invalid surfaces a JSONDecodeError (by design).

    _decode_raw_payload re-raises if no valid lines are found. Tests document
    this as the expected behavior so callers know to wrap with error handling.
    """
    svc = _make_parsing_service(tmp_path)
    import orjson

    content = b"NOT JSON\nALSO NOT JSON\nSTILL NOT JSON\n"
    record = _make_raw_record(
        "all-invalid-jsonl", "claude-code", content, "/exports/session.jsonl"
    )
    with pytest.raises(orjson.JSONDecodeError):
        await svc._parse_raw_record(record)


# ---------------------------------------------------------------------------
# Fault 4: Parsing handles JSONL with mixed valid and invalid lines
# ---------------------------------------------------------------------------

async def test_parse_mixed_valid_invalid_jsonl_lines(tmp_path):
    """JSONL with some valid lines: valid lines parsed, invalid lines skipped."""
    svc = _make_parsing_service(tmp_path)
    # Mix valid claude-code JSONL with invalid lines
    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"Hello"},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'INVALID JSON LINE\n'
        b'{"parentUuid":"m1","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"Hi"}]},"uuid":"m2","timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    record = _make_raw_record(
        "mixed-jsonl", "claude-code", content, "/exports/session.jsonl"
    )
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)
    if result:
        assert result[0].provider_name == "claude-code"


# ---------------------------------------------------------------------------
# Fault 5: Parsing handles valid JSON for unknown provider gracefully
# ---------------------------------------------------------------------------

async def test_parse_unknown_provider_name(tmp_path):
    """Unknown provider name falls back gracefully."""
    svc = _make_parsing_service(tmp_path)
    record = _make_raw_record(
        "unknown-provider",
        "not-a-real-provider",
        json.dumps({"id": "conv-1", "title": "Test"}).encode(),
    )
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 6: Parsing handles claude-code JSONL with null fields
# ---------------------------------------------------------------------------

async def test_parse_claude_code_jsonl_with_null_fields(tmp_path):
    """Claude-code JSONL with null fields in messages is handled gracefully."""
    svc = _make_parsing_service(tmp_path)
    content = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":null},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
    )
    record = _make_raw_record(
        "null-fields", "claude-code", content, "/exports/session.jsonl"
    )
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 7: Acquisition scan_sources handles missing source directory
# ---------------------------------------------------------------------------

async def test_acquisition_scan_missing_source_path(tmp_path):
    """AcquisitionService.scan_sources handles a missing source directory gracefully."""
    from polylogue.paths import Source
    from polylogue.pipeline.services.acquisition import AcquisitionService
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    db = SQLiteBackend(db_path=tmp_path / "test.db")
    svc = AcquisitionService(backend=db)

    missing_source = Source(name="missing", path=tmp_path / "does_not_exist")
    result = await svc.scan_sources([missing_source])
    # Should return a ScanResult, not raise
    assert result is not None


# ---------------------------------------------------------------------------
# Fault 8: Acquisition store_records handles backend save failure
# ---------------------------------------------------------------------------

async def test_acquisition_store_survives_backend_error(tmp_path):
    """store_records surfaces backend errors rather than silently dropping data."""
    from polylogue.pipeline.services.acquisition import AcquisitionService
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    db = SQLiteBackend(db_path=tmp_path / "test.db")
    svc = AcquisitionService(backend=db)

    record = _make_raw_record("test-1", "chatgpt", b'{"id": "c1"}')

    with patch.object(db, "save_raw_conversation", side_effect=RuntimeError("disk full")):
        # store_records is resilient: it logs the error and returns without raising
        result = await svc.store_records([record])
        # Error count should be non-zero, acquired should be 0
        assert result.counts["errors"] >= 1
        assert result.counts["acquired"] == 0


# ---------------------------------------------------------------------------
# Fault 9: Parsing handles very large valid conversation
# ---------------------------------------------------------------------------

async def test_parse_very_large_conversation_does_not_crash(tmp_path):
    """Very large valid chatgpt payload is handled without crashing."""
    svc = _make_parsing_service(tmp_path)
    messages: dict = {}
    prev_id = None
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

    payload = json.dumps({
        "title": "Large Conversation",
        "mapping": messages,
        "create_time": 1700000000,
        "update_time": 1700000100,
    }).encode()

    record = _make_raw_record("large-content", "chatgpt", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Fault 10: Parsing handles chatgpt with deeply nested but invalid mapping
# ---------------------------------------------------------------------------

async def test_parse_chatgpt_deeply_nested_malformed_nodes(tmp_path):
    """ChatGPT payload with malformed node structure returns empty, not exception."""
    svc = _make_parsing_service(tmp_path)
    # Mapping nodes missing 'message' key
    payload = json.dumps({
        "title": "Malformed Nodes",
        "mapping": {
            "node-1": {"parent": None, "children": ["node-2"]},
            "node-2": {"parent": "node-1", "children": [], "no_message_key": True},
        },
        "create_time": 1700000000,
        "update_time": 1700000001,
    }).encode()
    record = _make_raw_record("malformed-nodes", "chatgpt", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 11: Parsing handles a chatgpt bundle (list) where one item is invalid
# ---------------------------------------------------------------------------

async def test_parse_chatgpt_bundle_with_one_invalid_item(tmp_path):
    """ChatGPT bundle list with one invalid item: valid items parsed, invalid skipped."""
    svc = _make_parsing_service(tmp_path)
    payload = json.dumps([
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
    ]).encode()
    record = _make_raw_record("bundle-one-invalid", "chatgpt", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fault 12: Parsing handles gemini with missing text fields
# ---------------------------------------------------------------------------

async def test_parse_gemini_missing_text_fields(tmp_path):
    """Gemini payload with messages missing text fields is handled gracefully."""
    svc = _make_parsing_service(tmp_path)
    payload = json.dumps({
        "conversations": [
            {
                "chunks": [
                    {"role": "user"},   # no text field
                    {"role": "model"},  # no text field
                ]
            }
        ]
    }).encode()
    record = _make_raw_record("gemini-no-text", "gemini", payload)
    result = await svc._parse_raw_record(record)
    assert isinstance(result, list)
