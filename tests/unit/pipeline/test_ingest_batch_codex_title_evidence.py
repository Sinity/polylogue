"""Codex title evidence is carried by acquisition, never discovered by replay."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.enums import Provider, TitleSource
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord


def _codex_stream(session_id: str, text: str) -> bytes:
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        ),
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _codex_runtime_root(tmp_path: Path, session_id: str, content: bytes) -> Path:
    rollout = tmp_path / ".codex" / "sessions" / "2026" / f"rollout-{session_id}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(content)
    return rollout


def _record(
    store: BlobStore,
    content: bytes,
    *,
    source_path: str,
    sidecar_snapshot: dict[str, object] | None = None,
) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=source_path,
        payload_provider=Provider.CODEX,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        sidecar_snapshot=sidecar_snapshot,
    )


def _ingest_title(record: RawSessionRecord, tmp_path: Path, store: BlobStore) -> tuple[str | None, str | None]:
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(store.root))
    assert result.error is None, result.error
    assert result.sessions, "expected one materializable session"
    parsed = result.sessions[0].parsed_session
    title_source = str(parsed.title_source) if parsed.title_source is not None else None
    return parsed.title, title_source


def test_ingest_does_not_rediscover_missing_acquisition_evidence(tmp_path: Path) -> None:
    """A live sidecar cannot affect a raw record with no carried snapshot."""
    session_id = "aaaa1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "opening prompt from acquired bytes")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "AMBIENT title must be ignored"}) + "\n",
        encoding="utf-8",
    )
    store = BlobStore(tmp_path / "blobs")

    title, title_source = _ingest_title(_record(store, content, source_path=str(rollout)), tmp_path, store)

    assert title == "opening prompt from acquired bytes"
    assert title_source == TitleSource.HEURISTIC.value


def test_ingest_uses_acquisition_carried_evidence(tmp_path: Path) -> None:
    """The ordinary worker route uses the acquisition snapshot when present."""
    session_id = "bbbb1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "opening prompt from acquired bytes")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "AMBIENT title must lose"}) + "\n",
        encoding="utf-8",
    )
    store = BlobStore(tmp_path / "blobs")
    record = _record(
        store,
        content,
        source_path=str(rollout),
        sidecar_snapshot={"history_titles": {session_id: "Acquired title"}},
    )

    title, title_source = _ingest_title(record, tmp_path, store)

    assert title == "Acquired title"
    assert title_source == TitleSource.ORIGIN.value
