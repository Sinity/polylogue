"""Canonical raw-record ingest applies provider assembly enrichment (polylogue-ih67).

The daemon's raw-record worker historically bypassed the assembly layer that
direct ingest runs, so every daemon-ingested Codex session kept its native
UUID as title. These tests pin the parity contract: removing the
``_enrich_parsed_sessions`` call from ``_run_parse_plan`` (the named
production mutation) fails ``test_canonical_ingest_applies_thread_name``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, TitleSource
from polylogue.pipeline.services import ingest_worker as ingest_worker_module
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    reset_blob_store()
    yield store
    reset_blob_store()


def _codex_stream(session_id: str, *texts: str) -> bytes:
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}}),
    ]
    for text in texts:
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _codex_runtime_root(tmp_path: Path, session_id: str, content: bytes) -> Path:
    codex_dir = tmp_path / ".codex"
    rollout = codex_dir / "sessions" / "2026" / f"rollout-{session_id}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(content)
    return rollout


def _record(store: BlobStore, content: bytes, *, source_path: str) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=source_path,
        payload_provider=Provider.CODEX,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


def _ingest_title(record: RawSessionRecord, tmp_path: Path, store: BlobStore) -> tuple[str | None, str | None]:
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(store.root))
    assert result.error is None, result.error
    assert result.sessions, "expected one materializable session"
    parsed = result.sessions[0].parsed_session
    source = parsed.title_source
    return parsed.title, str(source) if source is not None else None


def test_canonical_ingest_applies_thread_name(blob_store: BlobStore, tmp_path: Path) -> None:
    """The daemon worker resolves the provider thread name, like direct ingest."""
    session_id = "aaaa1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "please fix the ingest bug")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "session_index.jsonl").write_text(
        json.dumps({"id": session_id, "thread_name": "Ingest bug hunt"}) + "\n",
        encoding="utf-8",
    )

    record = _record(blob_store, content, source_path=str(rollout))
    title, title_source = _ingest_title(record, tmp_path, blob_store)

    assert title == "Ingest bug hunt"
    assert title_source == TitleSource.ORIGIN.value


def test_canonical_ingest_uses_history_title(blob_store: BlobStore, tmp_path: Path) -> None:
    """Without a thread name, the authored history entry becomes the title."""
    session_id = "bbbb1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "opening prompt typed by the operator")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "Wire the Hermes bridge"}) + "\n",
        encoding="utf-8",
    )

    record = _record(blob_store, content, source_path=str(rollout))
    title, title_source = _ingest_title(record, tmp_path, blob_store)

    assert title == "Wire the Hermes bridge"
    assert title_source == TitleSource.ORIGIN.value


def test_canonical_ingest_message_fallback_without_sidecars(blob_store: BlobStore, tmp_path: Path) -> None:
    """A missing runtime root leaves only the human-authored message fallback."""
    session_id = "cccc1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "refactor the daemon status loop")

    record = _record(
        blob_store,
        content,
        source_path=str(tmp_path / "gone" / "sessions" / f"rollout-{session_id}.jsonl"),
    )
    title, title_source = _ingest_title(record, tmp_path, blob_store)

    assert title == "refactor the daemon status loop"
    assert title_source == TitleSource.HEURISTIC.value


def test_runtime_schema_registry_singleton_is_race_safe_under_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first access must construct exactly one SchemaRegistry (polylogue-xikl.1).

    ``_runtime_schema_registry()``'s check-then-set on the module-global
    ``_SCHEMA_REGISTRY`` used to be unguarded, and this is called once per
    parsed record on the parse path that phase 2 plans to run on a
    ThreadPoolExecutor. A short delay in the registry constructor forces the
    interleaving window open: before the fix this reliably produced multiple
    distinct registries (last writer wins); with the lock guarding the whole
    check-then-construct section, exactly one thread ever constructs one.
    """
    original = ingest_worker_module._SCHEMA_REGISTRY
    ingest_worker_module._SCHEMA_REGISTRY = None

    from polylogue.schemas.runtime_registry import SchemaRegistry

    original_init = SchemaRegistry.__init__
    built: list[object] = []
    built_lock = threading.Lock()

    def delayed_init(self: SchemaRegistry, *args: object, **kwargs: object) -> None:
        time.sleep(0.02)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        with built_lock:
            built.append(self)

    monkeypatch.setattr(SchemaRegistry, "__init__", delayed_init)

    registries: list[object] = []
    registries_lock = threading.Lock()

    def worker() -> None:
        registry = ingest_worker_module._runtime_schema_registry()
        with registries_lock:
            registries.append(registry)

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(registries) == 8
        assert len(built) == 1, f"concurrent first access built {len(built)} registries, expected exactly 1"
        assert len({id(registry) for registry in registries}) == 1
    finally:
        ingest_worker_module._SCHEMA_REGISTRY = original
