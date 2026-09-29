"""Parser-supported shapes reach their parser through the production ingest worker.

Artifact classification decides ``parse_as_session`` before any parser runs;
a shape the parser supports but the classifier refuses is skipped as a
non-session artifact. These tests drive ``ingest_record`` on synthetic raw
blobs so the classifier, not a direct parser call, is what they exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.archive.artifact_taxonomy import classify_artifact
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, ingest_record
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord


def _ingest(tmp_path: Path, provider: Provider, source_path: str, content: bytes) -> IngestRecordResult:
    blob_store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size = blob_store.write_from_bytes(content)
    record = RawSessionRecord(
        raw_id=f"raw-{blob_hash[:12]}",
        source_name=provider.value,
        payload_provider=provider,
        source_path=source_path,
        source_index=0,
        blob_size=blob_size,
        blob_hash=blob_hash,
        acquired_at="2026-01-01T00:00:00+00:00",
    )
    return ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))


def test_claude_project_export_reaches_parse_project_through_ingest(tmp_path: Path) -> None:
    """Anti-vacuity: without the project signature the classifier calls this
    payload an unrecognized or metadata document and no session is produced."""
    payload: JSONValue = {"uuid": "p1", "docs": [{"uuid": "d1", "content": "x"}], "prompt_template": "t"}

    assert classify_artifact(payload, provider=Provider.CLAUDE_AI).parse_as_session

    result = _ingest(tmp_path, Provider.CLAUDE_AI, "projects/p1.json", json.dumps(payload).encode())

    assert result.error is None
    assert [session.parsed_session.provider_session_id for session in result.sessions] == ["project:p1"]


def test_claude_account_memory_export_reaches_its_parser_through_ingest(tmp_path: Path) -> None:
    """The sibling claude.ai shape with no message list is admitted the same way."""
    payload: JSONValue = [{"account_uuid": "acct-1", "conversations_memory": "Prefers short answers."}]

    assert classify_artifact(payload, provider=Provider.CLAUDE_AI).parse_as_session

    result = _ingest(tmp_path, Provider.CLAUDE_AI, "memories.json", json.dumps(payload).encode())

    assert result.error is None
    assert [session.parsed_session.provider_session_id for session in result.sessions] == ["account-memory:acct-1"]


def test_project_signature_alone_does_not_admit_a_bare_uuid_document() -> None:
    """``uuid`` plus ``docs`` without a project-only key stays unrecognized."""
    assert not classify_artifact({"uuid": "p1", "docs": []}, provider=Provider.CLAUDE_AI).parse_as_session


def _codex_rollout(*extra: JSONValue) -> list[JSONValue]:
    return [
        {
            "type": "session_meta",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"id": "rollout-usage", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/workspace"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-01T00:00:01Z",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        },
        *extra,
    ]


def test_codex_rollout_with_bare_token_usage_record_ingests_its_usage(tmp_path: Path) -> None:
    """Anti-vacuity: without the bare-record admission the whole rollout is
    classified as unsupported and the usage extraction never runs."""
    records = _codex_rollout({"type": "token_usage_record", "input_tokens": 10})
    content = "\n".join(json.dumps(record) for record in records).encode() + b"\n"

    assert classify_artifact(records, provider=Provider.CODEX).parse_as_session

    result = _ingest(tmp_path, Provider.CODEX, "sessions/2026/01/01/rollout-usage.jsonl", content)

    assert result.error is None
    [session] = [payload.parsed_session for payload in result.sessions]
    usage_events = [event for event in session.session_events if event.event_type == "token_usage_record"]
    assert [event.payload.get("usage") for event in usage_events] == [{"input_tokens": 10}]


def test_codex_bare_token_usage_record_needs_a_known_counter() -> None:
    """The type name alone is not evidence; the stream stays refused."""
    assert not classify_artifact(
        _codex_rollout({"type": "token_usage_record", "note": "no counters"}), provider=Provider.CODEX
    ).parse_as_session
