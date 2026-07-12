from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch import _MAX_APPEND_PLAN_PAYLOAD_BYTES, LiveBatchProcessor, _ArchiveFullWriteResult
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _AppendPlan,
    _AppendResult,
    _detect_provider_from_path_sample,
    _parse_path_as_session_artifact,
    encode_cursor_hash_authority,
    sha256_range_from_path,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.raw.models import UNSET
from polylogue.storage.sqlite.archive_tiers import archive as archive_tier_module
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source_write import read_archive_raw_session_envelope
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION


def _write_archive_blob(archive_root: Path, blob_hash: bytes | str, payload: bytes) -> None:
    blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else blob_hash.lower()
    blob_path = archive_root / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)


def _cursor_hash_authority(payload: bytes) -> str:
    return encode_cursor_hash_authority(
        sha256(payload).hexdigest(),
        sha256(payload[-64 * 1024 :]).hexdigest(),
        ctime_ns=0,
    )


def _append_plan(path: Path, payload: bytes, *, payload_hash: str) -> _AppendPlan:
    stat = path.stat()
    return _AppendPlan(
        path=path,
        source_name="codex",
        start_offset=0,
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=payload,
        payload_hash=payload_hash,
        cursor_fingerprint="base",
        bytes_read=len(payload),
    )


def _append_owner(archive_root: Path) -> object:
    cursor = CursorStore(archive_root / "append.sqlite")
    return SimpleNamespace(
        _cursor=cursor,
        _polylogue=SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=cursor._db_path)),
    )


def _raw_parse_state(archive_root: Path) -> tuple[int | None, str | None]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions").fetchone()
    assert row is not None
    return cast(tuple[int | None, str | None], row)


def test_full_ingest_heartbeats_small_file_groups_with_current_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_text('{"type":"session_meta","payload":{"id":"first"}}\n', encoding="utf-8")
    second.write_text('{"type":"session_meta","payload":{"id":"second"}}\n', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    events: list[tuple[str, Path | None, int | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del stage_payload
        del force
        events.append((phase, current_path, source_payload_read_bytes))

    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )

    result = processor._ingest_full_paths_sync([first, second], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [first, second]
    assert ("full_file_scan", first, 0) in events
    assert ("full_file_scan", second, first.stat().st_size) in events
    assert any(
        event == ("full_archive_write", second, first.stat().st_size + second.stat().st_size) for event in events
    )


def test_large_full_ingest_uses_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    source.write_text('{"type":"session_meta","payload":{"id":"large"}}\n', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )

    result = processor._ingest_full_paths_sync([source], source_name="codex")

    assert result.succeeded == [source]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "large"


def test_streaming_sized_full_ingest_uses_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    source.write_bytes(b'{"type":"session_meta","payload":{"id":"large"}}\n' + (b" " * (9 * 1024 * 1024)))
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("streaming-sized JSONL ingest must not materialize through parse_payload")
        ),
    )

    result = processor._ingest_full_paths_sync([source], source_name="codex")

    assert result.succeeded == [source]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1


def test_full_ingest_writes_archive_with_route_observability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"full-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    result = processor._ingest_full_paths_sync([source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 1
    assert result.changed_session_count == 1
    assert result.raw_fingerprints[source]
    assert {
        "full.provider_parse",
        "full.source_raw_write",
        "full.index_parsed_write",
        "full.index.session_upsert",
        "full.index.full_replace",
        "full.index.full_replace.clear_projection_rows",
        "full.index.full_replace.messages",
        "full.index.full_replace.blocks",
        "full.index.full_replace.fts_insert",
    }.issubset(result.stage_timings_s)
    with sqlite3.connect(source_db) as conn:
        raw_state = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions").fetchone()
        assert raw_state is not None
        assert raw_state[0] is not None
        assert raw_state[1] is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "full-v1"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None

    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        "archive_present_tiers": "source,index,ops",
        "archive_missing_tiers": "embeddings,user",
        "archive_tier_user_versions_json": json.dumps(
            {
                "embeddings": None,
                "index": INDEX_SCHEMA_VERSION,
                "ops": 1,
                "source": SOURCE_SCHEMA_VERSION,
                "user": None,
            },
            sort_keys=True,
        ),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 1,
        "payload_unavailable_file_count": 0,
        "payload_replayed_from_blob_file_count": 0,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "written_raw_count": 1,
        "ingested_session_count": 1,
        "ingested_message_count": 1,
        "payload_unavailable_file_count": 0,
        "payload_replayed_from_blob_file_count": 0,
    }


def test_streaming_full_ingest_writes_archive_from_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "stream-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"stream-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"large"}]}}\n'
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)

    result = processor._ingest_full_paths_sync([source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 1
    assert result.changed_session_count == 1
    with sqlite3.connect(source_db) as conn:
        raw_row = conn.execute("SELECT raw_id, blob_size FROM raw_sessions").fetchone()
        assert raw_row[1] == len(payload)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "stream-v1"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1

    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        "archive_present_tiers": "source,index,ops",
        "archive_missing_tiers": "embeddings,user",
        "archive_tier_user_versions_json": json.dumps(
            {
                "embeddings": None,
                "index": INDEX_SCHEMA_VERSION,
                "ops": 1,
                "source": SOURCE_SCHEMA_VERSION,
                "user": None,
            },
            sort_keys=True,
        ),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 0,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "written_raw_count": 1,
        "ingested_session_count": 1,
        "ingested_message_count": 1,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    assert raw_row[0] == result.raw_fingerprints[source]


def test_streaming_sized_browser_capture_json_uses_native_payload_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "browser-capture" / "chatgpt"
    root.mkdir(parents=True)
    source = root / "native-capture.json"
    native_payload = {
        "id": "native-large",
        "title": "Native large capture",
        "create_time": 1781442866.0,
        "update_time": 1781442966.0,
        "current_node": "assistant-node",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["user-node"]},
            "user-node": {
                "id": "user-node",
                "parent": "root",
                "children": ["assistant-node"],
                "message": {
                    "id": "native-u1",
                    "author": {"role": "user"},
                    "create_time": 1781442870.0,
                    "content": {"content_type": "text", "parts": ["Native user text"]},
                    "metadata": {},
                },
            },
            "assistant-node": {
                "id": "assistant-node",
                "parent": "user-node",
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "create_time": 1781442880.0,
                    "content": {"content_type": "text", "parts": ["Native answer text"]},
                    "metadata": {"model_slug": "gpt-native"},
                },
            },
        },
    }
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:native-large",
        "provenance": {
            "source_url": "https://chatgpt.com/c/native-large",
            "page_title": "ChatGPT - Native large capture",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-native-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "dom-fallback",
            "title": "DOM fallback title",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [{"provider_turn_id": "dom-u1", "role": "user", "text": "DOM fallback", "ordinal": 0}],
        },
        "raw_provider_payload": native_payload,
        "padding": "x" * 256,
    }
    source.write_text(json.dumps(capture_payload), encoding="utf-8")
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="browser-capture", root=root.parent),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)

    result = processor._ingest_full_paths_sync([source], source_name="browser-capture")

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 2
    assert result.raw_source_names[source] == "chatgpt"
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT origin, native_id FROM raw_sessions").fetchone() == (
            "chatgpt-export",
            "native-large",
        )
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id, title FROM sessions").fetchone() == (
            "native-large",
            "Native large capture",
        )
        assert (
            conn.execute(
                """
            SELECT group_concat(item, '|')
            FROM (
                SELECT messages.role || ':' || blocks.text AS item
                FROM messages
                JOIN blocks USING (message_id)
                ORDER BY messages.position, blocks.position
            )
            """
            ).fetchone()[0]
            == "user:Native user text|assistant:Native answer text"
        )


def test_generic_large_browser_capture_json_uses_prefix_detection_without_unknown_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "inbox"
    root.mkdir()
    source = root / "large-browser-capture.json"
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:generic-large",
        "provenance": {
            "source_url": "https://chatgpt.com/c/generic-large",
            "page_title": "ChatGPT - Generic capture",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "generic-large",
            "title": "Generic inbox browser capture",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Generic user text", "ordinal": 0},
                {"provider_turn_id": "a1", "role": "assistant", "text": "Generic answer text", "ordinal": 1},
            ],
        },
        "padding": "x" * 256,
    }
    source.write_text(json.dumps(capture_payload), encoding="utf-8")
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr("polylogue.sources.live.batch_support._STREAMING_FULL_INGEST_BYTES", 1)

    result = processor._ingest_full_paths_sync([source], source_name="inbox")

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 2
    assert result.raw_source_names[source] == "chatgpt"
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT origin, native_id FROM raw_sessions").fetchone() == (
            "chatgpt-export",
            "generic-large",
        )
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id, title, message_count FROM sessions").fetchone() == (
            "generic-large",
            "Generic inbox browser capture",
            2,
        )


def test_large_browser_capture_prefix_planning_does_not_materialize_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large-browser-capture.json"
    target.write_text(
        json.dumps(
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "prefix-only",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "x"}],
                },
                "provenance": {
                    "source_url": "https://chatgpt.com/c/prefix-only",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large browser-capture planning must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _detect_provider_from_path_sample(target, Provider.UNKNOWN) is Provider.CHATGPT
    assert _parse_path_as_session_artifact(target, provider=Provider.CHATGPT) is True


def test_full_ingest_bootstraps_archive_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "bootstrap-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"bootstrap-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"boot"}]}}\n'
    )
    source.write_bytes(payload)
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    result = processor._ingest_full_paths_sync([source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        assert (tmp_path / filename).exists()
    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_tiers": "source,index,embeddings,user,ops",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": True,
        "archive_present_tiers": "source,index,embeddings,user,ops",
        "archive_missing_tiers": "",
        "archive_tier_user_versions_json": json.dumps(
            {
                "embeddings": 1,
                "index": INDEX_SCHEMA_VERSION,
                "ops": 1,
                "source": SOURCE_SCHEMA_VERSION,
                "user": USER_SCHEMA_VERSION,
            },
            sort_keys=True,
        ),
    }


def test_fingerprint_file_streams_in_bounded_memory(tmp_path: Path) -> None:
    """``fingerprint_file`` must not load the whole file into memory.

    Regression: the previous implementation read the entire file via
    ``Path.read_bytes()``, producing an RSS peak proportional to file size.
    This test exercises the streaming path on a multi-megabyte synthetic
    file and asserts that the working set stays bounded by ``chunk_size``.
    """
    import hashlib
    import tracemalloc

    from polylogue.sources.live.batch_support import fingerprint_file

    payload = (b"x" * 4095 + b"\n") * 4096  # ~16 MiB, all lines newline-terminated
    target = tmp_path / "huge.jsonl"
    target.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    expected_last_nl = len(payload)  # ends in newline

    tracemalloc.start()
    try:
        fp, last_nl = fingerprint_file(target, chunk_size=64 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert fp == expected_hash
    assert last_nl == expected_last_nl
    # Peak Python-allocated memory must stay well under the file size. The
    # 1 MiB budget is generous (chunk_size is 64 KiB) but leaves room for
    # hasher state and chunk overhead without admitting a full-file read.
    assert peak < 1 * 1024 * 1024, f"fingerprint_file peak {peak} bytes is not bounded for a {len(payload)}-byte file"


def test_fingerprint_file_tracks_last_newline_across_chunk_boundary(tmp_path: Path) -> None:
    """The streaming fingerprint must locate the last newline even when it
    sits in an earlier chunk than the file tail."""
    from polylogue.sources.live.batch_support import fingerprint_file

    # 4 KiB of newline-terminated lines, followed by 4 KiB without any \n.
    head = (b"line\n") * 1000  # 5_000 bytes, ends with \n
    tail = b"y" * 5000  # no newline anywhere
    payload = head + tail
    target = tmp_path / "no-trailing-newline.jsonl"
    target.write_bytes(payload)

    _fp, last_nl = fingerprint_file(target, chunk_size=1024)
    assert last_nl == len(head), f"last_complete_newline should be at end-of-head ({len(head)}), got {last_nl}"


def test_fingerprint_file_empty_file(tmp_path: Path) -> None:
    import hashlib

    from polylogue.sources.live.batch_support import fingerprint_file

    target = tmp_path / "empty.jsonl"
    target.write_bytes(b"")

    fp, last_nl = fingerprint_file(target)
    assert fp == hashlib.sha256(b"").hexdigest()
    assert last_nl == 0


def test_large_non_jsonl_full_ingest_planning_does_not_read_whole_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.json"
    target.write_text('{"mapping": {}}\n', encoding="utf-8")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large full-ingest planning must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _detect_provider_from_path_sample(target, Provider.CHATGPT) is Provider.CHATGPT
    assert _parse_path_as_session_artifact(target, provider=Provider.CHATGPT) is True


def test_unclassified_large_non_jsonl_is_not_streamed_as_session_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "unknown.large"
    target.write_bytes(b"not-json")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("unclassified large files must not be materialized during planning")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _parse_path_as_session_artifact(target, provider=Provider.UNKNOWN) is False


def test_append_plan_chunks_large_tail_without_full_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    first_chunk = b'{"b":"' + (b"x" * (_MAX_APPEND_PLAN_PAYLOAD_BYTES - 128)) + b'"}\n'
    second_chunk = b'{"c":"' + (b"y" * 512) + b'"}\n'
    appended = first_chunk + second_chunk
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    assert plan.last_complete_newline == len(original) + len(first_chunk)
    assert plan.stat_size == len(original) + len(appended)
    assert plan.bytes_read == _MAX_APPEND_PLAN_PAYLOAD_BYTES
    assert plan.payload == first_chunk

    assert processor._record_append_cursor(plan) is True
    next_plan = processor._append_plan(path)
    assert isinstance(next_plan, _AppendPlan)
    assert next_plan.start_offset == len(original) + len(first_chunk)
    assert next_plan.last_complete_newline == len(original) + len(appended)
    assert next_plan.payload == second_chunk


def test_append_plan_defers_when_tail_has_no_complete_line(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    assert processor._append_plan(path) is _DEFER_APPEND


@pytest.mark.asyncio
async def test_inbox_browser_capture_json_replacement_uses_full_ingest(tmp_path: Path) -> None:
    from polylogue.api import Polylogue

    root = tmp_path / "inbox"
    root.mkdir()
    path = root / "capture.json"

    def capture(turns: list[dict[str, object]]) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:inbox-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/inbox-replacement",
                "page_title": "Inbox replacement",
                "captured_at": "2026-07-11T00:00:00+00:00",
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "inbox-replacement",
                "title": "Inbox replacement",
                "updated_at": "2026-07-11T00:00:01+00:00",
                "turns": turns,
            },
        }

    first_turn = {
        "provider_turn_id": "turn-1",
        "role": "user",
        "text": "first snapshot",
        "ordinal": 0,
    }
    replacement_turn = {
        "provider_turn_id": "turn-2",
        "role": "assistant",
        "text": "replacement snapshot",
        "ordinal": 1,
    }
    path.write_text(json.dumps(capture([first_turn])), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="inbox", root=root, suffixes=(".json", ".jsonl")),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await processor.ingest_files([path], emit_event=False)
        path.write_text(json.dumps(capture([first_turn, replacement_turn])), encoding="utf-8")
        second = await processor.ingest_files([path], emit_event=False)
        assert first.full_file_count == 1
        assert second.full_file_count == 1
        assert second.append_file_count == 0
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            source_indexes = conn.execute(
                "SELECT source_index FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms",
                (str(path),),
            ).fetchall()
        assert source_indexes == [(0,), (0,)]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_browser_capture_replacement_advances_membership_head_and_acquires_attachment(tmp_path: Path) -> None:
    """A mutable receiver snapshot must retain both raws but materialize the newer capture."""
    from polylogue.api import Polylogue

    root = tmp_path / "browser-capture"
    root.mkdir()
    path = root / "capture.json"
    asset_bytes = b"browser-capture-asset" * 37
    asset_hash = sha256(asset_bytes).digest()

    def capture(turns: list[dict[str, object]]) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:browser-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/browser-replacement",
                "captured_at": "2026-07-12T00:00:00+00:00",
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "browser-replacement",
                "title": "Browser replacement",
                "updated_at": "2026-07-12T00:00:01+00:00",
                "turns": turns,
            },
        }

    first_turn = {"provider_turn_id": "turn-1", "role": "user", "text": "make an asset", "ordinal": 0}
    acquired_turn = {
        "provider_turn_id": "turn-2",
        "role": "assistant",
        "text": "asset acquired",
        "ordinal": 1,
        "attachments": [
            {
                "provider_attachment_id": "asset-1",
                "message_provider_id": "turn-2",
                "name": "deliverable.bin",
                "mime_type": "application/octet-stream",
                "inline_base64": base64.b64encode(asset_bytes).decode("ascii"),
            }
        ],
    }
    divergent_turn = {
        "provider_turn_id": "turn-divergent",
        "role": "assistant",
        "text": "older divergent snapshot",
        "ordinal": 1,
    }
    path.write_text(json.dumps(capture([first_turn])), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await processor.ingest_files([path], emit_event=False)
        path.write_text(json.dumps(capture([first_turn, acquired_turn])), encoding="utf-8")
        replacement = await processor.ingest_files([path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids = [
                str(row[0])
                for row in source_conn.execute(
                    "SELECT raw_id FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms", (str(path),)
                )
            ]
            decisions = source_conn.execute(
                """
                SELECT raw_id, decision FROM raw_session_memberships
                WHERE logical_source_key = 'chatgpt:browser-replacement'
                """
            ).fetchall()
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            accepted_raw_id = index_conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-replacement'"
            ).fetchone()[0]
            attachment = index_conn.execute(
                "SELECT acquisition_status, byte_count, blob_hash FROM attachments WHERE display_name = 'deliverable.bin'"
            ).fetchone()

        assert first.full_file_count == replacement.full_file_count == 1
        assert len(raw_ids) == 2
        assert accepted_raw_id in raw_ids
        assert {decision for _raw_id, decision in decisions} == {"superseded_prefix", "applied"}
        assert dict(decisions)[accepted_raw_id] == "applied"
        assert attachment == ("acquired", len(asset_bytes), asset_hash)

        path.write_text(json.dumps(capture([first_turn])), encoding="utf-8")
        reverse = await processor.ingest_files([path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-replacement'"
                ).fetchone()[0]
                == accepted_raw_id
            )
        assert reverse.full_file_count == 1

        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_before_divergence = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
        path.write_text(json.dumps(capture([first_turn, divergent_turn])), encoding="utf-8")
        divergent = await processor.ingest_files([path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_after_divergence = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
            divergent_raw_id = (raw_ids_after_divergence - raw_ids_before_divergence).pop()
            divergent_decision = source_conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE raw_id = ? AND logical_source_key = 'chatgpt:browser-replacement'
                """,
                (divergent_raw_id,),
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-replacement'"
                ).fetchone()[0]
                == accepted_raw_id
            )
        assert divergent.full_file_count == 1
        assert divergent_decision == "ambiguous"
    finally:
        await archive.close()


def test_jsonl_stream_retains_append_plan(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"append-safe"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        source_name="inbox",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="inbox", root=root, suffixes=(".jsonl",)),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.payload.endswith(appended)


def test_incomplete_append_is_requeued_not_full_ingested(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        source_name="chatgpt",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    metrics = asyncio.run(processor.ingest_files([path], emit_event=False))

    assert metrics.full_file_count == 0
    assert metrics.append_file_count == 0
    assert metrics.failed_paths == [str(path)]


def test_codex_append_plan_uses_append_only_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-019e309f-7614-7381-a8ac-f9080f304ee6.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            source_index=-1,
            payload=original,
            acquired_at_ms=1_770_000_000_000,
        )
        blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
    _write_archive_blob(tmp_path, cast(bytes, blob_hash), original)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "019e309f-7614-7381-a8ac-f9080f304ee6",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_770_000_000_000,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base-cursor",
        tail_hash=_cursor_hash_authority(original),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    assert plan.payload.startswith(b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n')
    assert plan.payload.endswith(appended)


def test_codex_append_plan_reads_archive_file_set_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-019e309f-7614-7381-a8ac-f9080f304ee6.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            source_index=0,
            payload=original,
            acquired_at_ms=1_770_000_000_000,
        )
        blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
    _write_archive_blob(tmp_path, cast(bytes, blob_hash), original)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "019e309f-7614-7381-a8ac-f9080f304ee6",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_770_000_000_000,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base-cursor",
        tail_hash=_cursor_hash_authority(original),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.payload.startswith(b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n')
    assert plan.payload.endswith(appended)
    assert processor._latest_raw_fingerprint(path) == raw_id
    with cursor._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None


def test_latest_raw_fingerprint_ignores_archive_source_row_with_missing_blob(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "missing-blob.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"missing-blob"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    blob_hash = b"a" * 32
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-missing-blob", "codex-session", "missing-blob", str(path), 0, blob_hash, len(payload), 1),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    assert processor._latest_raw_fingerprint(path) is None

    _write_archive_blob(tmp_path, blob_hash, payload)

    assert processor._latest_raw_fingerprint(path) == "raw-missing-blob"


def test_append_ingest_preserves_successes_when_other_plan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok_payload = (
        b'{"type":"session_meta","payload":{"id":"append-ok","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"ok"}]}}\n'
    )
    bad_payload = b"{bad json}\n"

    class Owner:
        def __init__(self) -> None:
            self._cursor = CursorStore(tmp_path / "append.sqlite")
            self._polylogue = SimpleNamespace(
                archive_root=tmp_path,
                backend=SimpleNamespace(db_path=self._cursor._db_path),
            )

    plans = [
        _AppendPlan(
            path=tmp_path / "ok.jsonl",
            source_name="codex",
            start_offset=0,
            last_complete_newline=8,
            stat_size=8,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=ok_payload,
            payload_hash="ok",
            cursor_fingerprint="base",
            bytes_read=len(ok_payload),
        ),
        _AppendPlan(
            path=tmp_path / "bad.jsonl",
            source_name="unknown",
            start_offset=0,
            last_complete_newline=9,
            stat_size=9,
            st_dev=1,
            st_ino=2,
            mtime_ns=1,
            payload=bad_payload,
            payload_hash="bad",
            cursor_fingerprint="base",
            bytes_read=len(bad_payload),
        ),
    ]

    owner = Owner()
    result = ingest_append_plans(owner, plans)

    assert result.succeeded == []
    assert result.deferred == [plans[0]]
    assert result.failed == [plans[1]]
    assert result.worker_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            "SELECT parse_error, revision_authority FROM raw_sessions ORDER BY parse_error IS NOT NULL"
        ).fetchall()
        assert rows[0] == (None, "quarantined")
        assert rows[1][0]
        assert rows[1][1] == "quarantined"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize("protect_chain", [True, False], ids=["protected", "protection-disabled"])
def test_live_append_chain_survives_post_ingest_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protect_chain: bool,
) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"append-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    original_publish = ArchiveBlobPublisher.write_from_bytes
    published_payloads: list[bytes] = []

    def counted_publish(publisher: ArchiveBlobPublisher, raw: bytes) -> tuple[str, int]:
        published_payloads.append(raw)
        return original_publish(publisher, raw)

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_bytes", counted_publish)
    if not protect_chain:
        from polylogue.storage.raw_retention import RawRetentionAuthority

        def unsafe_retention_authority(conn: sqlite3.Connection, **_kwargs: object) -> RawRetentionAuthority:
            raw_ids = frozenset(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions"))
            return RawRetentionAuthority(protected_raw_ids=frozenset(), eligible_raw_ids=raw_ids)

        monkeypatch.setattr(
            "polylogue.storage.raw_retention.active_raw_retention_authority",
            unsafe_retention_authority,
        )

    append_chunks = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n',
        b'{"type":"response_item","payload":{"type":"message","id":"message-2","role":"user",'
        b'"content":[{"type":"input_text","text":"two"}]}}\n',
        b'{"type":"response_item","payload":{"type":"message","id":"message-3","role":"assistant",'
        b'"content":[{"type":"output_text","text":"three"}]}}\n',
    )
    results = [asyncio.run(processor.ingest_files([path]))]
    for chunk in append_chunks:
        with path.open("ab") as handle:
            handle.write(chunk)
        results.append(asyncio.run(processor.ingest_files([path])))

    assert results[0].full_file_count == 1
    assert results[0].succeeded_file_count == 1
    assert all(result.append_file_count == 1 for result in results[1:])
    append_identity = b'{"type":"session_meta","payload":{"id":"append-v1"}}\n'
    assert published_payloads == [payload, *(append_identity + chunk for chunk in append_chunks)]
    if not protect_chain:
        assert [result.succeeded_file_count for result in results] == [1, 1, 1, 0]
        assert [result.failed_file_count for result in results] == [0, 0, 0, 1]
        cursor_record = cursor.get_record(path)
        assert cursor_record is not None
        assert cursor_record.byte_offset == len(payload) + sum(len(chunk) for chunk in append_chunks[:2])
        assert cursor_record.failure_count == 1
        with sqlite3.connect(index_db) as conn:
            head_raw_id = str(conn.execute("SELECT accepted_raw_id FROM raw_revision_heads").fetchone()[0])
            assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
                "message-0",
                "message-1",
                "message-2",
            }
        with sqlite3.connect(source_db) as conn:
            predecessor_raw_id = str(
                conn.execute(
                    "SELECT predecessor_raw_id FROM raw_sessions WHERE raw_id = ?",
                    (head_raw_id,),
                ).fetchone()[0]
            )
            assert conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?",
                (predecessor_raw_id,),
            ).fetchone() == (0,)
        return

    assert all(result.succeeded_file_count == 1 for result in results)
    assert all(result.failed_file_count == 0 for result in results)
    expected_sessions = parse_payload(
        Provider.CODEX,
        [json.loads(line) for line in path.read_bytes().splitlines()],
        path.stem,
        source_path=str(path),
    )
    assert len(expected_sessions) == 1
    expected_session_hash = bytes.fromhex(session_content_hash(expected_sessions[0]))
    with sqlite3.connect(source_db) as conn:
        raw_rows = conn.execute(
            """SELECT raw_id, revision_kind, predecessor_raw_id, baseline_raw_id,
                      parsed_at_ms, parse_error
               FROM raw_sessions ORDER BY acquisition_generation"""
        ).fetchall()
        assert len(raw_rows) == 4
        assert [row[1] for row in raw_rows] == ["full", "append", "append", "append"]
        assert all(row[4] is not None and row[5] is None for row in raw_rows)
        raw_by_id = {str(row[0]): row for row in raw_rows}
    with sqlite3.connect(index_db) as conn:
        session_native_id, session_hash = conn.execute("SELECT native_id, content_hash FROM sessions").fetchone()
        assert session_native_id == "append-v1"
        assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
            "message-0",
            "message-1",
            "message-2",
            "message-3",
        }
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 4
        assert conn.execute(
            """SELECT b.search_text
               FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid
               ORDER BY b.message_id"""
        ).fetchall() == [("zero",), ("one",), ("two",), ("three",)]
        head_raw_id, accepted_hash = conn.execute(
            "SELECT accepted_raw_id, accepted_content_hash FROM raw_revision_heads"
        ).fetchone()
        head_raw_id = str(head_raw_id)
        assert session_hash == accepted_hash
        assert session_hash == expected_session_hash
        assert conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_revision_applications").fetchone()[0] == 4
        assert conn.execute(
            """SELECT decision, COUNT(DISTINCT raw_id)
               FROM raw_revision_applications GROUP BY decision ORDER BY decision"""
        ).fetchall() == [("applied_append", 3), ("selected_baseline", 1)]
        receipt_decisions = {
            str(raw_id): str(decision)
            for raw_id, decision in conn.execute("SELECT DISTINCT raw_id, decision FROM raw_revision_applications")
        }
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None
    chain: list[str] = []
    current_raw_id: str | None = head_raw_id
    while current_raw_id is not None:
        chain.append(current_raw_id)
        row = raw_by_id[current_raw_id]
        current_raw_id = str(row[2]) if row[2] is not None else None
    assert len(chain) == 4
    assert raw_by_id[chain[-1]][1] == "full"
    assert receipt_decisions == {
        chain[-1]: "selected_baseline",
        **dict.fromkeys(chain[:-1], "applied_append"),
    }
    cursor_record = cursor.get_record(path)
    assert cursor_record is not None
    assert cursor_record.byte_offset == len(payload) + sum(len(chunk) for chunk in append_chunks)


def test_full_ingest_cursor_stops_at_captured_blob_boundary_when_file_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "hot.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"hot-growth","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"captured","role":"user",'
        b'"content":[{"type":"input_text","text":"captured"}]}}\n'
    )
    appended_during_parse = (
        b'{"type":"response_item","payload":{"type":"message","id":"later","role":"assistant",'
        b'"content":[{"type":"output_text","text":"later"}]}}\n'
    )
    path.write_bytes(captured)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    grew = False

    def grow_after_acquisition(paths: list[Path]) -> tuple[set[Path], float, dict[str, float], list[object]]:
        nonlocal grew
        if not grew:
            with path.open("ab") as handle:
                handle.write(appended_during_parse)
            grew = True
        return set(paths), 0.0, {}, []

    monkeypatch.setattr(processor, "_converge_paths", grow_after_acquisition)

    first = asyncio.run(processor.ingest_files([path]))

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 1
    record = cursor.get_record(path)
    assert record is not None
    assert record.byte_size == len(captured)
    assert record.byte_offset == len(captured)
    assert record.last_complete_newline == len(captured)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(captured)
    assert plan.payload.endswith(appended_during_parse)

    second = asyncio.run(processor.ingest_files([path]))

    assert second.append_file_count == 1
    assert second.succeeded_file_count == 1
    final_record = cursor.get_record(path)
    assert final_record is not None
    assert final_record.byte_offset == len(captured) + len(appended_during_parse)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        full_blob_size, append_start, append_blob_hash = conn.execute(
            """SELECT
                   MAX(CASE WHEN revision_kind = 'full' THEN blob_size END),
                   MAX(CASE WHEN revision_kind = 'append' THEN append_start_offset END),
                   MAX(CASE WHEN revision_kind = 'append' THEN hex(blob_hash) END)
               FROM raw_sessions"""
        ).fetchone()
    assert full_blob_size == len(captured)
    assert append_start == len(captured)
    assert isinstance(append_blob_hash, str)
    from polylogue.storage.blob_store import BlobStore

    append_identity = b'{"type":"session_meta","payload":{"id":"hot-growth"}}\n'
    assert BlobStore(tmp_path / "blob").read_all(append_blob_hash.lower()) == (append_identity + appended_during_parse)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("captured",),
            ("later",),
        ]
        assert conn.execute(
            "SELECT b.search_text FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid ORDER BY b.message_id"
        ).fetchall() == [("captured",), ("later",)]
        session_hash = conn.execute("SELECT content_hash FROM sessions").fetchone()[0]
        accepted_hash = conn.execute("SELECT accepted_content_hash FROM raw_revision_heads").fetchone()[0]
        assert session_hash == accepted_hash


@pytest.mark.parametrize("replacement_mode", ["atomic", "in-place"])
def test_full_ingest_does_not_advance_cursor_across_same_size_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_mode: str,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "replaced.jsonl"
    replacement = root / "replacement.jsonl"
    payload_a = (
        b'{"type":"session_meta","payload":{"id":"atomic-replace-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-a","role":"user",'
        b'"content":[{"type":"input_text","text":"alpha"}]}}\n'
    )
    payload_b = (
        b'{"type":"session_meta","payload":{"id":"atomic-replace-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-b","role":"user",'
        b'"content":[{"type":"input_text","text":"bravo"}]}}\n'
    )
    assert len(payload_a) == len(payload_b)
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    original_stat = path.stat()
    original_identity = (original_stat.st_dev, original_stat.st_ino)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    replaced = False

    def replace_after_acquisition(paths: list[Path]) -> tuple[set[Path], float, dict[str, float], list[object]]:
        nonlocal replaced
        if not replaced:
            if replacement_mode == "atomic":
                replacement.replace(path)
            else:
                path.write_bytes(payload_b)
                current_stat = path.stat()
                os.utime(
                    path,
                    ns=(current_stat.st_atime_ns, max(current_stat.st_mtime_ns, original_stat.st_mtime_ns) + 1_000_000),
                )
            replaced = True
        return set(paths), 0.0, {}, []

    monkeypatch.setattr(processor, "_converge_paths", replace_after_acquisition)

    first = asyncio.run(processor.ingest_files([path]))

    assert first.succeeded_file_count == 1
    assert first.stale_cursor_write_count == 1
    if replacement_mode == "atomic":
        assert (path.stat().st_dev, path.stat().st_ino) != original_identity
    else:
        assert (path.stat().st_dev, path.stat().st_ino) == original_identity
    stale_cursor = cursor.get_record(path)
    assert stale_cursor is not None
    assert stale_cursor.byte_offset == 0
    assert stale_cursor.content_fingerprint is None
    assert LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == [("message-a",)]

    second = asyncio.run(processor.ingest_files([path]))

    assert second.full_file_count == 1
    assert second.succeeded_file_count == 1
    assert second.stale_cursor_write_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == len(payload_b)
    assert (final_cursor.st_dev, final_cursor.st_ino) == (path.stat().st_dev, path.stat().st_ino)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-a",),
            ("message-b",),
        ]
        assert conn.execute("SELECT search_text FROM blocks ORDER BY search_text").fetchall() == [
            ("alpha",),
            ("bravo",),
        ]


def test_archive_cursor_reconciliation_rejects_restored_mtime_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "archive-reconcile.jsonl"
    payload_a = b'{"type":"session_meta","payload":{"id":"archive-reconcile-a"}}\n'
    payload_b = b'{"type":"session_meta","payload":{"id":"archive-reconcile-b"}}\n'
    assert len(payload_a) == len(payload_b)
    path.write_bytes(payload_a)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert asyncio.run(processor.ingest_files([path])).succeeded_file_count == 1
    with sqlite3.connect(cursor._ops_db_path) as conn:
        conn.execute("DELETE FROM ingest_cursor WHERE source_path = ?", (str(path),))
        conn.commit()
    watcher = LiveWatcher(polylogue, (WatchSource(name="codex", root=root),), cursor=cursor)
    initial_stat = path.stat()
    original_hash = sha256_range_from_path

    def rewrite_after_hash(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        path.write_bytes(payload_b)
        rewritten_stat = path.stat()
        os.utime(path, ns=(rewritten_stat.st_atime_ns, initial_stat.st_mtime_ns))
        return result

    monkeypatch.setattr(live_watcher, "sha256_range_from_path", rewrite_after_hash)

    assert not watcher._reconcile_archived_cursor(path, stat=initial_stat)
    assert cursor.get_record(path) is None
    final_stat = path.stat()
    assert final_stat.st_mtime_ns == initial_stat.st_mtime_ns
    assert final_stat.st_ctime_ns != initial_stat.st_ctime_ns


def test_rejected_full_cursor_frontier_requires_reauthorization(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rejected-frontier.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"rejected-frontier"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-a","role":"user",'
        b'"content":[{"type":"input_text","text":"alpha"}]}}\n'
    )
    growth = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-b","role":"assistant",'
        b'"content":[{"type":"output_text","text":"bravo"}]}}\n'
    )
    path.write_bytes(captured)
    captured_stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    obsolete_offset = len(captured) + 1
    cursor.set(
        path,
        obsolete_offset,
        byte_offset=obsolete_offset,
        last_complete_newline=obsolete_offset,
        parser_fingerprint="test-parser",
        content_fingerprint="obsolete-frontier",
        tail_hash="obsolete-tail",
        source_name="codex",
        st_dev=captured_stat.st_dev,
        st_ino=captured_stat.st_ino,
        mtime_ns=captured_stat.st_mtime_ns,
        failure_count=2,
    )
    with path.open("ab") as handle:
        handle.write(growth)

    processor._record_full_cursor(
        path,
        raw_fingerprint=sha256(captured).hexdigest(),
        raw_byte_size=len(captured),
        source_name="codex",
        captured_content_hash=sha256(captured).hexdigest(),
        captured_file_observation=(
            captured_stat.st_dev,
            captured_stat.st_ino,
            captured_stat.st_size,
            captured_stat.st_mtime_ns,
            captured_stat.st_ctime_ns,
        ),
    )

    assert processor._last_cursor_write_stale is True
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert invalidated.failure_count == 2
    assert LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)
    assert processor._append_plan(path, cursor=invalidated) is None


def test_cursor_invalidation_lock_exhaustion_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "locked-invalidation.jsonl"
    path.write_bytes(b'{"type":"session_meta","payload":{"id":"locked-invalidation"}}\n')
    stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="accepted-frontier",
        tail_hash="accepted-tail",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=2,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(cursor, "_sync_cursor_record_to_ops", lambda _record: False)

    with pytest.raises(sqlite3.OperationalError, match="failed to persist cursor invalidation"):
        processor._invalidate_cursor_for_full_retry(path, source_name="codex", stat=stat)

    unchanged = cursor.get_record(path)
    assert unchanged is not None
    assert unchanged.byte_offset == stat.st_size
    assert unchanged.content_fingerprint == "accepted-frontier"
    assert unchanged.failure_count == 2


def test_append_plan_rejects_malformed_hash_authority(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "malformed-authority.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"malformed-authority"}}\n'
    path.write_bytes(original + b'{"type":"turn_context","payload":{}}\n')
    stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="accepted-frontier",
        tail_hash=f"sha256-prefix-v1:{sha256(original).hexdigest()}:invalid:0",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    record = cursor.get_record(path)
    assert record is not None
    assert processor._append_plan(path, cursor=record) is None


@pytest.mark.parametrize("rewrite_mode", ["atomic-replacement", "in-place-prefix"])
def test_append_cursor_forces_full_retry_after_source_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewrite_mode: str,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-replaced.jsonl"
    replacement = root / "append-replacement.jsonl"
    prefix_padding = b"p" * (70 * 1024)
    baseline_a = (
        b'{"type":"session_meta","payload":{"id":"append-replace-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0a","role":"user",'
        b'"content":[{"type":"input_text","text":"zeroa' + prefix_padding + b'"}]}}\n'
    )
    append_a = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-aa","role":"assistant",'
        b'"content":[{"type":"output_text","text":"alpha"}]}}\n'
    )
    replacement_b = (
        b'{"type":"session_meta","payload":{"id":"append-replace-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0b","role":"user",'
        b'"content":[{"type":"input_text","text":"zerob' + prefix_padding + b'"}]}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-bb","role":"assistant",'
        b'"content":[{"type":"output_text","text":"bravo"}]}}\n'
    )
    assert len(baseline_a + append_a) == len(replacement_b)
    assert len(baseline_a) > 64 * 1024
    path.write_bytes(baseline_a)
    replacement.write_bytes(replacement_b)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert asyncio.run(processor.ingest_files([path])).succeeded_file_count == 1
    with path.open("ab") as handle:
        handle.write(append_a)
    pre_rewrite_stat = path.stat()
    accepted_tail_before_rewrite = path.read_bytes()[-64 * 1024 :]
    replaced = False

    def replace_after_append(paths: list[Path]) -> tuple[set[Path], float, dict[str, float], list[object]]:
        nonlocal replaced
        if not replaced:
            if rewrite_mode == "atomic-replacement":
                replacement.replace(path)
            else:
                rewritten = path.read_bytes().replace(b"zeroa", b"zerob", 1)
                assert len(rewritten) == pre_rewrite_stat.st_size
                path.write_bytes(rewritten)
                current_stat = path.stat()
                os.utime(
                    path,
                    ns=(
                        current_stat.st_atime_ns,
                        pre_rewrite_stat.st_mtime_ns,
                    ),
                )
                restored_stat = path.stat()
                assert restored_stat.st_mtime_ns == pre_rewrite_stat.st_mtime_ns
                assert restored_stat.st_ctime_ns != pre_rewrite_stat.st_ctime_ns
            replaced = True
        return set(paths), 0.0, {}, []

    monkeypatch.setattr(processor, "_converge_paths", replace_after_append)

    appended = asyncio.run(processor.ingest_files([path]))

    assert appended.append_file_count == 1
    assert appended.succeeded_file_count == 1
    assert appended.stale_cursor_write_count == 1
    stale_cursor = cursor.get_record(path)
    assert stale_cursor is not None
    assert stale_cursor.byte_offset == 0
    assert stale_cursor.content_fingerprint is None
    assert LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-0a",),
            ("message-aa",),
        ]
        assert conn.execute("SELECT substr(search_text, 1, 5) FROM blocks ORDER BY search_text").fetchall() == [
            ("alpha",),
            ("zeroa",),
        ]

    if rewrite_mode == "in-place-prefix":
        assert b"zerob" in path.read_bytes()
        assert path.read_bytes()[-64 * 1024 :] == accepted_tail_before_rewrite
        return

    retried = asyncio.run(processor.ingest_files([path]))

    assert retried.full_file_count == 1
    assert retried.succeeded_file_count == 1
    assert retried.stale_cursor_write_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == len(replacement_b)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-0a",),
            ("message-0b",),
            ("message-aa",),
            ("message-bb",),
        ]


def test_rewrite_plus_growth_before_planning_fails_closed_to_full_route(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rewrite-before-plan.jsonl"
    padding = b"p" * (70 * 1024)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"rewrite-before-plan"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zeroa' + padding + b'"}]}}\n'
    )
    appended = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"alpha"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert asyncio.run(processor.ingest_files([path])).succeeded_file_count == 1
    rewritten = baseline.replace(b"zeroa", b"zerob", 1)
    assert rewritten[-64 * 1024 :] == baseline[-64 * 1024 :]
    path.write_bytes(rewritten + appended)

    second = asyncio.run(processor.ingest_files([path]))

    assert second.full_file_count == 1
    assert second.append_file_count == 0
    assert second.succeeded_file_count == 0
    assert second.failed_file_count == 1
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [("message-0",)]
        assert conn.execute("SELECT substr(search_text, 1, 5) FROM blocks ORDER BY search_text").fetchall() == [
            ("zeroa",),
        ]
    failed_cursor = cursor.get_record(path)
    assert failed_cursor is not None
    assert failed_cursor.byte_offset == len(baseline)
    assert failed_cursor.failure_count == 1
    assert failed_cursor.next_retry_at is not None


def test_incomplete_full_jsonl_capture_retries_without_losing_split_record(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "split-record.jsonl"
    prefix = (
        b'{"type":"session_meta","payload":{"id":"split-record"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    split_record = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}'
    )
    split_at = len(split_record) // 2
    path.write_bytes(prefix + split_record[:split_at])
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    first = asyncio.run(processor.ingest_files([path]))

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 0
    assert first.failed_file_count == 1
    failed_cursor = cursor.get_record(path)
    assert failed_cursor is not None
    assert failed_cursor.byte_offset == 0
    assert failed_cursor.content_fingerprint is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()[0]
        assert "complete record boundary" in str(parse_error)

    with path.open("ab") as handle:
        handle.write(split_record[split_at:])
    second = asyncio.run(processor.ingest_files([path]))

    assert second.full_file_count == 1
    assert second.append_file_count == 0
    assert second.succeeded_file_count == 1
    assert second.failed_file_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == path.stat().st_size
    assert final_cursor.failure_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]
        assert conn.execute(
            "SELECT b.search_text FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid ORDER BY b.message_id"
        ).fetchall() == [("zero",), ("one",)]


def test_captured_incomplete_jsonl_is_rejected_after_source_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "disappearing.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"disappearing"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"complete"}]}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-1"'
    )
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    original_ingest = processor._ingest_full_records_archive

    def remove_source_after_capture(*args: Any, **kwargs: Any) -> _ArchiveFullWriteResult:
        path.unlink()
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(processor, "_ingest_full_records_archive", remove_source_after_capture)

    result = asyncio.run(processor.ingest_files([path]))

    assert result.succeeded_file_count == 0
    assert result.failed_file_count == 1
    failed_cursor = cursor.get_record(path)
    assert failed_cursor is None or failed_cursor.byte_offset == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()[0]
        assert "complete record boundary" in str(parse_error)


def test_append_persistence_failure_preserves_frontier_for_next_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "retry.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"retry"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    append = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert asyncio.run(processor.ingest_files([path])).succeeded_file_count == 1
    accepted_cursor = cursor.get_record(path)
    assert accepted_cursor is not None

    def index_state() -> tuple[object, ...]:
        with sqlite3.connect(index_db) as conn:
            return (
                conn.execute(
                    "SELECT session_id, message_count, content_hash FROM sessions ORDER BY session_id"
                ).fetchall(),
                conn.execute("SELECT message_id, position, content_hash FROM messages ORDER BY message_id").fetchall(),
                conn.execute("SELECT block_id, message_id, search_text FROM blocks ORDER BY block_id").fetchall(),
                conn.execute("SELECT id, sz FROM messages_fts_docsize ORDER BY id").fetchall(),
                conn.execute(
                    """SELECT logical_source_key, accepted_raw_id, accepted_source_revision,
                              accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                              acquisition_generation, append_end_offset
                       FROM raw_revision_heads ORDER BY logical_source_key"""
                ).fetchall(),
                conn.execute(
                    """SELECT decision_id, raw_id, decision, accepted_raw_id,
                              accepted_source_revision, accepted_content_hash
                       FROM raw_revision_applications ORDER BY decision_id"""
                ).fetchall(),
            )

    accepted_index_state = index_state()
    with path.open("ab") as handle:
        handle.write(append)

    original_record = archive_tier_module.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected append persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_tier_module, "record_revision_application_sync", injected_failure)
    failed = asyncio.run(processor.ingest_files([path]))

    assert failed.succeeded_file_count == 0
    assert failed.failed_file_count == 1
    retry_cursor = cursor.get_record(path)
    assert retry_cursor is not None
    assert (
        retry_cursor.byte_size,
        retry_cursor.byte_offset,
        retry_cursor.last_complete_newline,
        retry_cursor.parser_fingerprint,
        retry_cursor.content_fingerprint,
        retry_cursor.tail_hash,
        retry_cursor.source_name,
        retry_cursor.st_dev,
        retry_cursor.st_ino,
        retry_cursor.mtime_ns,
    ) == (
        accepted_cursor.byte_size,
        accepted_cursor.byte_offset,
        accepted_cursor.last_complete_newline,
        accepted_cursor.parser_fingerprint,
        accepted_cursor.content_fingerprint,
        accepted_cursor.tail_hash,
        accepted_cursor.source_name,
        accepted_cursor.st_dev,
        accepted_cursor.st_ino,
        accepted_cursor.mtime_ns,
    )
    assert index_state() == accepted_index_state
    with sqlite3.connect(tmp_path / "source.db") as conn:
        retained_append = conn.execute(
            """SELECT revision_kind, predecessor_raw_id, append_start_offset,
                      append_end_offset, revision_authority, parsed_at_ms, parse_error
               FROM raw_sessions WHERE source_index = -1"""
        ).fetchone()
    assert retained_append is not None
    assert retained_append[0] == "append"
    assert retained_append[1] is not None
    assert retained_append[2:5] == (len(baseline), len(baseline) + len(append), "byte_proven")
    assert retained_append[5] is None
    assert "injected append persistence failure" in str(retained_append[6])

    cursor.reset_failures(path)
    monkeypatch.setattr("polylogue.sources.live.watcher._PARSER_FINGERPRINT", "test-parser")
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )
    assert watcher._needs_work(path)
    cursor.mark_failed(path)
    pending_retry = cursor.get_record(path)
    assert pending_retry is not None
    assert pending_retry.failure_count == 1

    retried = asyncio.run(processor.ingest_files([path]))

    assert retried.append_file_count == 1
    assert retried.succeeded_file_count == 1
    assert retried.failed_file_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == path.stat().st_size
    assert final_cursor.failure_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]


def test_failed_parser_upgrade_preserves_accepted_parser_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "parser-upgrade.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"parser-upgrade"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor_a = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="parser-a",
    )
    assert asyncio.run(processor_a.ingest_files([path])).succeeded_file_count == 1
    accepted = cursor.get_record(path)
    assert accepted is not None
    assert accepted.parser_fingerprint == "parser-a"

    with path.open("ab") as handle:
        handle.write(
            b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
            b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
        )
    processor_b = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="parser-b",
    )
    original_record = archive_tier_module.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected parser-upgrade persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_tier_module, "record_revision_application_sync", injected_failure)

    failed = asyncio.run(processor_b.ingest_files([path]))

    assert failed.full_file_count == 1
    assert failed.failed_file_count == 1
    retry = cursor.get_record(path)
    assert retry is not None
    assert retry.parser_fingerprint == "parser-a"
    assert retry.byte_offset == accepted.byte_offset
    assert retry.content_fingerprint == accepted.content_fingerprint

    cursor.reset_failures(path)
    retried = asyncio.run(processor_b.ingest_files([path]))

    assert retried.full_file_count == 1
    assert retried.append_file_count == 0
    assert retried.succeeded_file_count == 1
    final = cursor.get_record(path)
    assert final is not None
    assert final.parser_fingerprint == "parser-b"
    assert final.byte_offset == path.stat().st_size


def test_append_parse_failure_retains_typed_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-bad.jsonl"
    payload = b"{bad json}\n"
    path.write_bytes(payload)
    plan = _append_plan(path, payload, payload_hash="bad")
    owner = _append_owner(tmp_path)
    monkeypatch.setattr(
        "polylogue.sources.dispatch.parse_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected append parse failure")),
    )

    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.succeeded == []
    assert result.failed == [plan]
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected append parse failure" in parse_error
    assert len(parse_error) <= 2000
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions").fetchone()[0])
        envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert envelope.parse_error == parse_error
    assert envelope.detection_warnings == (parse_error[:500],)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_append_archive_lock_propagates_for_watcher_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-locked.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"append-locked"}}\n'
    path.write_bytes(payload)
    plan = _append_plan(path, payload, payload_hash="locked")
    owner = _append_owner(tmp_path)

    monkeypatch.setattr(
        ArchiveStore,
        "write_raw_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        ingest_append_plans(cast(Any, owner), [plan])


def test_full_parse_failure_retains_typed_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-bad.jsonl"
    source.write_bytes(b"{bad json}\n")
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected full parse failure")),
    )

    result = processor._ingest_full_paths_sync([source], source_name="codex")

    assert source in result.failed
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected full parse failure" in parse_error
    assert len(parse_error) <= 2000
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_full_archive_lock_propagates_for_watcher_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-locked.jsonl"
    source.write_bytes(b'{"type":"session_meta","payload":{"id":"full-locked"}}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        ArchiveStore,
        "write_raw_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        processor._ingest_full_paths_sync([source], source_name="codex")


def test_append_index_failure_never_marks_raw_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-index-fail.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"index-fail"}}\n'
    path.write_bytes(payload)
    plan = _append_plan(path, payload, payload_hash="index-fail")
    owner = _append_owner(tmp_path)

    def fail_index(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.IntegrityError("injected index commit failure")

    monkeypatch.setattr(ArchiveStore, "_write_parsed_precedence_result", fail_index)
    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected index commit failure" in parse_error
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_append_multi_session_failure_does_not_finalize_after_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-multi.jsonl"
    payload = b"{}\n"
    path.write_bytes(payload)
    plan = _append_plan(path, payload, payload_hash="multi")
    owner = _append_owner(tmp_path)
    sessions = [
        ParsedSession(source_name=Provider.CODEX, provider_session_id="multi-1", messages=[]),
        ParsedSession(source_name=Provider.CODEX, provider_session_id="multi-2", messages=[]),
    ]
    monkeypatch.setattr("polylogue.sources.dispatch.parse_payload", lambda *_args, **_kwargs: sessions)
    original_write = ArchiveStore._write_parsed_precedence_result
    write_count = 0

    def fail_second_index(
        archive: ArchiveStore,
        session: ParsedSession,
        **kwargs: object,
    ) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise sqlite3.IntegrityError("injected second-session index failure")
        return original_write(archive, session, **cast(Any, kwargs))

    monkeypatch.setattr(ArchiveStore, "_write_parsed_precedence_result", fail_second_index)

    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "second-session index failure" in parse_error
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_full_multi_session_failure_retries_without_success_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-multi.jsonl"
    source.write_bytes(b"{}\n")
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    sessions = [
        ParsedSession(source_name=Provider.CODEX, provider_session_id="full-multi-1", messages=[]),
        ParsedSession(source_name=Provider.CODEX, provider_session_id="full-multi-2", messages=[]),
    ]
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr("polylogue.sources.live.batch.parse_stream_payload", lambda *_args, **_kwargs: sessions)
    original_write = ArchiveStore._write_parsed_precedence_result
    write_count = 0

    def fail_second_index(
        archive: ArchiveStore,
        session: ParsedSession,
        **kwargs: object,
    ) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise sqlite3.IntegrityError("injected full second-session index failure")
        return original_write(archive, session, **cast(Any, kwargs))

    monkeypatch.setattr(ArchiveStore, "_write_parsed_precedence_result", fail_second_index)
    archive_results: list[_ArchiveFullWriteResult] = []
    original_full_write = processor._ingest_full_records_archive

    def capture_full_write(*args: Any, **kwargs: Any) -> _ArchiveFullWriteResult:
        outcome = original_full_write(*args, **kwargs)
        archive_results.append(outcome)
        return outcome

    monkeypatch.setattr(processor, "_ingest_full_records_archive", capture_full_write)

    first = processor._ingest_full_paths_sync([source], source_name="codex")

    assert first.succeeded == []
    assert source in first.failed
    assert archive_results[0].raw_ids == {}
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "full second-session index failure" in parse_error
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    retry = processor._ingest_full_paths_sync([source], source_name="codex")

    assert retry.succeeded == [source]
    assert retry.failed == []
    assert archive_results[1].raw_ids
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is not None
    assert parse_error is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_live_multi_session_divergence_reopens_raw_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_bytes(b'{"first":true}\n')
    second.write_bytes(b'{"second":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    parsed_batches = iter(
        [
            [session("shared", "base", "left"), session("safe-1", "one")],
            [session("shared", "base", "right"), session("safe-2", "two")],
            [session("shared", "base", "left"), session("safe-1", "one")],
        ]
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )

    assert processor._ingest_full_paths_sync([first], source_name="codex").failed == []
    second_result = processor._ingest_full_paths_sync([second], source_name="codex")
    assert second_result.failed == [second]
    assert second_result.succeeded == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE decision = 'ambiguous'").fetchone() == (
            2,
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NULL").fetchone() == (2,)
    with sqlite3.connect(index_db) as conn:
        # The first accepted branch remains queryable; the later divergence is
        # nonterminal debt and has no deletion authority.
        assert set(conn.execute("SELECT native_id FROM sessions")) == {
            ("safe-1",),
            ("safe-2",),
            ("shared",),
        }


def test_single_session_full_terminally_supersedes_older_membership_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    older = root / "older.jsonl"
    bundle.write_bytes(b'{"bundle":true}\n')
    older.write_bytes(b'{"older":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    bundle_sessions = [session("shared", "base", "new"), session("safe", "one")]
    parsed_batches = iter([bundle_sessions, [session("shared", "base")]])
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda _archive, _raw_id: bundle_sessions,
    )

    assert processor._ingest_full_paths_sync([bundle], source_name="codex").failed == []
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        rejected_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=older.read_bytes(),
            source_path=str(older),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            rejected_raw_id,
            RawRevisionEnvelope(
                logical_source_key="codex:shared",
                kind=RawRevisionKind.FULL,
                source_revision=sha256(older.read_bytes()).hexdigest(),
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    older_result = processor._ingest_full_paths_sync([older], source_name="codex")

    assert older_result.succeeded == [older]
    assert older_result.failed == []
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT message_count FROM sessions WHERE native_id = 'shared'").fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.parsed_at_ms IS NOT NULL, r.parse_error,
                   r.logical_source_key, r.revision_kind
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex:shared'
            """,
            (str(older),),
        ).fetchone() == ("superseded_prefix", 1, None, None, "unknown")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _unclassified, revision_keys = archive.raw_revision_rebuild_selection([rejected_raw_id])
        _membership_raws, membership_keys = archive.expand_raw_membership_selection([rejected_raw_id])
    assert revision_keys == ()
    assert "codex:shared" in membership_keys


@pytest.mark.parametrize(
    ("bundle_texts", "succeeds", "census_head"),
    [
        (("base",), True, False),
        (("base", "different"), False, False),
        (("base", "new", "later"), False, False),
        (("base", "new", "later"), False, True),
    ],
)
def test_bundle_replay_respects_unconvertible_single_session_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_texts: tuple[str, ...],
    succeeds: bool,
    census_head: bool,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    current = root / "current.jsonl"
    older_bundle = root / "older-bundle.jsonl"
    current.write_bytes(b'{"current":true}\n')
    older_bundle.write_bytes(b'{"bundle":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    current_session = session("shared", "base", "new")
    bundle_sessions = [session("shared", *bundle_texts), session("safe", "one")]
    parsed_batches = iter([[current_session], bundle_sessions])
    current_raw_id: list[str] = []
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda archive, raw_id: (
            [current_session] if Path(archive.raw_revision_material(raw_id)[2]) == current else bundle_sessions
        ),
    )

    assert processor._ingest_full_paths_sync([current], source_name="codex").failed == []
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex:shared'"
        ).fetchone()
        assert row is not None
        current_raw_id.append(str(row[0]))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        source_revision = (
            archive._ensure_source_conn()
            .execute(
                "SELECT source_revision FROM raw_sessions WHERE raw_id = ?",
                (current_raw_id[0],),
            )
            .fetchone()
        )
        assert source_revision is not None
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"append":true}\n',
            source_path=str(current),
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_raw_id,
            RawRevisionEnvelope(
                "codex:shared",
                RawRevisionKind.APPEND,
                "append-blocker",
                0,
                predecessor_source_revision=str(source_revision[0]),
                append_start_offset=1,
                append_end_offset=2,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        assert archive.convertible_full_revision_raw_ids("codex:shared") == ()
        if census_head:
            archive.replace_raw_membership_census(
                current_raw_id[0],
                [current_session],
                parser_fingerprint="test-parser",
                censused_at_ms=2,
            )
    with sqlite3.connect(index_db) as conn:
        head_before = conn.execute(
            "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier "
            "FROM raw_revision_heads WHERE logical_source_key = 'codex:shared'"
        ).fetchone()
        assert head_before is not None

    result = processor._ingest_full_paths_sync([older_bundle], source_name="codex")

    assert result.failed == ([] if succeeds else [older_bundle])
    assert result.succeeded == ([older_bundle] if succeeds else [])
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT message_count FROM sessions WHERE native_id = 'shared'").fetchone() == (2,)
        head_after = conn.execute(
            "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier "
            "FROM raw_revision_heads WHERE logical_source_key = 'codex:shared'"
        ).fetchone()
        assert head_after == ((current_raw_id[0], "semantic", 2) if succeeds else head_before)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_session_memberships WHERE raw_id = ?",
            (current_raw_id[0],),
        ).fetchone() == ((1,) if census_head else (0,))
        decisions = conn.execute(
            "SELECT decision FROM raw_session_memberships WHERE logical_source_key = 'codex:shared' AND raw_id != ?",
            (current_raw_id[0],),
        ).fetchall()
        if succeeds:
            assert decisions == [("superseded_prefix",)]


def test_single_session_full_cannot_overwrite_divergent_membership_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    divergent = root / "divergent.jsonl"
    bundle.write_bytes(b'{"bundle":true}\n')
    divergent.write_bytes(b'{"divergent":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    bundle_sessions = [session("shared", "base", "left"), session("safe", "one")]
    parsed_batches = iter([bundle_sessions, [session("shared", "base", "right", "extra")]])
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda _archive, _raw_id: bundle_sessions,
    )

    assert processor._ingest_full_paths_sync([bundle], source_name="codex").failed == []
    divergent_result = processor._ingest_full_paths_sync([divergent], source_name="codex")

    assert divergent_result.succeeded == []
    assert divergent_result.failed == [divergent]
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT m.position, b.search_text
            FROM messages AS m
            JOIN blocks AS b USING (message_id)
            WHERE m.session_id = 'codex-session:shared'
            ORDER BY m.position, b.position
            """
        ).fetchall() == [(0, "base"), (1, "left")]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.parsed_at_ms, r.parse_error
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex:shared'
            """,
            (str(divergent),),
        ).fetchone() == ("ambiguous", None, None)


def test_single_session_full_advances_authorized_metadata_only_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    metadata_update = root / "metadata-update.jsonl"
    bundle.write_bytes(b'{"bundle":true}\n')
    metadata_update.write_bytes(b'{"metadata_update":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, title: str, updated_at: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            title=title,
            updated_at=updated_at,
            messages=[ParsedMessage(provider_message_id=f"{native_id}-0", role=Role.USER, text="same content")],
        )

    older = session("shared", "old title", "2026-01-01T00:00:00Z")
    newer = session("shared", "new title", "2026-01-02T00:00:00Z")
    bundle_sessions = [older, session("safe", "safe", "2026-01-01T00:00:00Z")]
    parsed_batches = iter([bundle_sessions, [newer]])
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda _archive, _raw_id: bundle_sessions,
    )

    assert processor._ingest_full_paths_sync([bundle], source_name="codex").failed == []
    update_result = processor._ingest_full_paths_sync([metadata_update], source_name="codex")

    assert update_result.succeeded == [metadata_update]
    assert update_result.failed == []
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT s.title, s.updated_at_ms, h.accepted_frontier_kind,
                   h.accepted_content_hash = s.content_hash
            FROM sessions AS s
            JOIN raw_revision_heads AS h USING (session_id)
            WHERE s.native_id = 'shared'
            """
        ).fetchone() == ("new title", 1767312000000, "semantic", 1)


def test_bundle_promotes_prior_single_full_into_membership_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    single = root / "single.jsonl"
    bundle = root / "bundle.jsonl"
    single.write_bytes(b'{"single":true}\n')
    bundle.write_bytes(b'{"bundle":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    single_session = session("shared", "base")
    bundle_sessions = [session("shared", "base", "new"), session("safe", "one")]
    parsed_batches = iter([[single_session], bundle_sessions])
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        lambda *_args, **_kwargs: next(parsed_batches),
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda _archive, _raw_id: [single_session],
    )

    assert processor._ingest_full_paths_sync([single], source_name="codex").failed == []
    bundle_result = processor._ingest_full_paths_sync([bundle], source_name="codex")

    assert bundle_result.succeeded == [bundle]
    assert bundle_result.failed == []
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT s.message_count, h.accepted_frontier_kind
            FROM sessions AS s
            JOIN raw_revision_heads AS h USING (session_id)
            WHERE s.native_id = 'shared'
            """
        ).fetchone() == (2, "semantic")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.logical_source_key, r.revision_kind
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex:shared'
            """,
            (str(single),),
        ).fetchone() == ("superseded_prefix", None, "unknown")


def test_append_crash_after_index_commit_repairs_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    path = tmp_path / "append-crash.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"crash-retry"}}\n'
    path.write_bytes(payload)
    plan = _append_plan(path, payload, payload_hash="crash")
    owner = _append_owner(tmp_path)
    original_finalize = ArchiveStore.finalize_raw_parse_state
    crashed = False

    def crash_after_index(
        archive: ArchiveStore,
        raw_id: str,
        *,
        state: object,
    ) -> None:
        nonlocal crashed
        parsed_at = getattr(state, "parsed_at", UNSET)
        if parsed_at is not UNSET and not crashed:
            assert (
                archive._conn.execute("SELECT 1 FROM sessions WHERE native_id = 'crash-retry'").fetchone() is not None
            )
            crashed = True
            raise SimulatedProcessCrash
        original_finalize(archive, raw_id, state=cast(Any, state))

    monkeypatch.setattr(ArchiveStore, "finalize_raw_parse_state", crash_after_index)
    with pytest.raises(SimulatedProcessCrash):
        ingest_append_plans(cast(Any, owner), [plan])

    assert _raw_parse_state(tmp_path) == (None, None)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    monkeypatch.setattr(ArchiveStore, "finalize_raw_parse_state", original_finalize)
    retry = ingest_append_plans(cast(Any, owner), [plan])

    assert retry.succeeded == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is not None
    assert parse_error is None
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_append_ingest_bootstraps_archive_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-bootstrap.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"append-bootstrap","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    path.write_bytes(payload)
    cursor = CursorStore(tmp_path / "append.sqlite")

    class Owner:
        _cursor = cursor
        _polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))

    stat = path.stat()
    plan = _AppendPlan(
        path=path,
        source_name="codex",
        start_offset=0,
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=payload,
        payload_hash="payload-hash",
        cursor_fingerprint="base",
        bytes_read=len(payload),
    )

    result = ingest_append_plans(Owner(), [plan])

    assert result.succeeded == []
    assert result.deferred == [plan]
    assert result.failed == []
    for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        assert (tmp_path / filename).exists()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_live_raw_compaction_ignores_cursor_db_without_source_db(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    cursor_db = tmp_path / "live.sqlite"
    cursor = CursorStore(cursor_db)
    with cursor._connect() as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                blob_size INTEGER NOT NULL,
                acquired_at TEXT NOT NULL
            );
            INSERT INTO raw_sessions
                (raw_id, source_path, source_index, blob_size, acquired_at)
            VALUES
                ('raw-old', '/tmp/old.jsonl', 0, 10, '2026-01-01T00:00:00+00:00');
            """
        )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    processor._compact_superseded_raw_snapshots([path])

    with cursor._connect() as conn:
        rows = conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
    assert rows == [("raw-old",)]


@pytest.mark.asyncio
async def test_live_append_plans_flush_in_bounded_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"{index}.jsonl" for index in range(5)]
    for path in paths:
        path.write_text('{"type":"session_meta","payload":{"id":"bounded"}}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    groups: list[list[Path]] = []

    def fake_append_plan(path: Path, *, cursor: object | None = None) -> _AppendPlan:
        del cursor
        return _AppendPlan(
            path=path,
            source_name="codex",
            start_offset=0,
            last_complete_newline=10,
            stat_size=10,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=b"payload\n",
            payload_hash="tail",
            cursor_fingerprint="base",
            bytes_read=10,
        )

    def fake_ingest_append_plans(plans: list[_AppendPlan]) -> _AppendResult:
        groups.append([plan.path for plan in plans])
        return _AppendResult(succeeded=list(plans), failed=[], worker_count=1)

    monkeypatch.setattr(processor, "_append_plan", fake_append_plan)
    monkeypatch.setattr(processor, "_ingest_append_plans", fake_ingest_append_plans)
    monkeypatch.setattr(processor, "_converge_paths", lambda paths: (paths, 0.0, {}, []))
    monkeypatch.setattr(processor, "_record_append_cursor", lambda plan: True)
    monkeypatch.setattr(processor, "_record_convergence_outcome", lambda path, debts: None)
    monkeypatch.setattr("polylogue.sources.live.batch._append_plan_group_ready", lambda plans: len(plans) >= 2)

    metrics = await processor.ingest_files(paths, emit_event=False)

    assert groups == [paths[:2], paths[2:4], paths[4:]]
    assert metrics.append_file_count == 5
    assert metrics.full_file_count == 0
    with sqlite3.connect(cursor._ops_db_path) as conn:
        stage_payloads = [
            (str(row[0]), json.loads(row[1]))
            for row in conn.execute(
                """
                SELECT stage, payload_json
                FROM daemon_stage_events
                WHERE stage IN ('append_parse', 'convergence', 'cursor_update', 'completed')
                ORDER BY observed_at_ms, rowid
                """
            ).fetchall()
        ]
    route_payloads = [(stage, payload) for stage, payload in stage_payloads if payload.get("storage_route")]
    assert route_payloads
    assert {payload["storage_route"] for _, payload in route_payloads} == {"archive_append"}
    assert ("cursor_update", "archive_append") in [
        (stage, str(payload.get("storage_route"))) for stage, payload in route_payloads
    ]
    assert ("completed", "archive_append") in [
        (stage, str(payload.get("storage_route"))) for stage, payload in route_payloads
    ]
