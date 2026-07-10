from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch import _MAX_APPEND_PLAN_PAYLOAD_BYTES, LiveBatchProcessor, _ArchiveFullWriteResult
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _AppendPlan,
    _AppendResult,
    _detect_provider_from_path_sample,
    _parse_path_as_session_artifact,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.raw.models import UNSET
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
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    assert processor._append_plan(path) is _DEFER_APPEND


def test_browser_capture_json_replacement_bypasses_append_plan(tmp_path: Path) -> None:
    root = tmp_path / "browser-capture" / "chatgpt"
    root.mkdir(parents=True)
    path = root / "capture.json"
    path.write_text('{"polylogue_capture_kind":"browser_llm_session"}', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    old_size = path.stat().st_size
    cursor.set(
        path,
        old_size,
        byte_offset=old_size,
        last_complete_newline=old_size,
        parser_fingerprint="test-parser",
        content_fingerprint="old-fingerprint",
        source_name="chatgpt",
    )
    path.write_text('{"polylogue_capture_kind":"browser_llm_session","turns":["replacement"]}', encoding="utf-8")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="browser-capture", root=root.parent, suffixes=(".json",)),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    assert processor._append_plan(path) is None


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

    assert result.succeeded == [plans[0]]
    assert result.failed == [plans[1]]
    assert result.worker_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "append-ok"


def test_append_ingest_writes_archive_file_set_through_archive_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"append-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
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

    first = asyncio.run(processor.ingest_files([path]))
    appended = (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant",'
        b'"content":[{"type":"output_text","text":"hello"}]}}\n'
    )
    with path.open("ab") as handle:
        handle.write(appended)
    second = asyncio.run(processor.ingest_files([path]))

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 1
    assert second.append_file_count == 1
    assert second.succeeded_file_count == 1
    assert published_payloads == [payload, b'{"type":"session_meta","payload":{"id":"append-v1"}}\n' + appended]
    with sqlite3.connect(source_db) as conn:
        raw_state = conn.execute(
            "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE source_index = -1"
        ).fetchone()
        assert raw_state is not None
        assert raw_state[0] is not None
        assert raw_state[1] is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "append-v1"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None


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

    assert result.succeeded == [plan]
    assert result.failed == []
    for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        assert (tmp_path / filename).exists()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "append-bootstrap"


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
