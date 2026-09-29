"""Regressions for Codex review findings on source acquisition routes (worker 9)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.config import Source
from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources import assembly_chatgpt, dispatch, drive
from polylogue.sources.drive.types import DriveFile
from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source
from polylogue.sources.live import WatchSource
from polylogue.sources.live import cold_build, hook_paste_enrichment
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import jsonl_complete_prefix
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.parse_prefetch import live_parse_path_worker
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, write_lease
from tests.infra.archive_templates import bootstrap_archive_root


_TIMESTAMP = "2026-06-02T00:00:00Z"


def _codex_file(path: Path) -> bytes:
    records = [
        {"type": "session_meta", "payload": {"id": "w9-session", "timestamp": _TIMESTAMP}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "w9-message",
                "role": "user",
                "content": [{"type": "input_text", "text": "worker nine regression"}],
            },
        },
    ]
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _processor(archive_root: Path, source_root: Path, provider: str) -> LiveBatchProcessor:
    index_db = archive_root / "index.db"
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name=provider, root=source_root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="w9-regression-parser",
    )


def test_w9_asset_acquisition_rejects_symlink_leaves(tmp_path: Path) -> None:
    """W9-90-01: the unfixed acquisition publishes the outside symlink's bytes."""
    export = tmp_path / "export"
    audio = export / "conversation" / "audio"
    audio.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not part of the selected export")
    (audio / "file_abc.wav").symlink_to(outside)
    good = audio / "file_def.wav"
    good.write_bytes(b"selected regular file")

    acquired = assembly_chatgpt._acquire_asset_blobs_from_directory(export, BlobStore(tmp_path / "blob"))

    assert acquired
    assert {blob[0] for blob in acquired.values()} == {sha256(good.read_bytes()).hexdigest()}
    assert all("abc" not in key for key in acquired)


def test_w9_path_worker_recognizes_project_keys_beyond_prefix(tmp_path: Path) -> None:
    """W9-90-15: the real path worker's bounded probe used to lose project keys."""
    payload = {
        "uuid": "w9-project",
        "description": "x" * 32768,
        "docs": [],
        "prompt_template": "Use exact evidence when answering.",
        "is_starter_project": False,
    }
    source = tmp_path / "project.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = dispatch.detect_provider(payload)
    assert expected is Provider.CLAUDE_AI
    artifact = live_parse_path_worker(
        Provider.UNKNOWN.value, str(source), source.stem,
        is_stream=False, shard_directory=str(tmp_path / "prepared"),
    )
    try:
        assert artifact.error is None
        assert artifact.resolved_provider is expected
    finally:
        artifact.discard()


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_w9_drive_lowering_preserves_gemini_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrapper_depth: int,
) -> None:
    """W9-90-16: the production Gemini parser receives None before the fix."""
    payload: Any = {
        "sessionId": "w9-gemini", "projectHash": "w9-project",
        "startTime": _TIMESTAMP, "lastUpdated": _TIMESTAMP,
        "messages": [{"id": "m", "type": "user", "content": "hello"}],
    }
    for _ in range(wrapper_depth):
        payload = [payload]
    source_path = str(tmp_path / "chats" / "session.json")
    original = dispatch.local_agent.parse_gemini_cli
    seen: list[str | Path | None] = []

    def record_path(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("source_path"))
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch.local_agent, "parse_gemini_cli", record_path)
    sessions = dispatch.parse_payload(Provider.GEMINI, payload, "fallback", source_path=source_path)
    assert sessions
    assert seen == [source_path]


class _DriveClient:
    def __init__(self, name: str, payload: bytes) -> None:
        self.file = DriveFile("w9-file", name, "application/json", _TIMESTAMP, len(payload))
        self.payload = payload
        self.downloads = 0

    def resolve_folder_id(self, _folder: str) -> str:
        return "w9-folder"

    def iter_json_files(self, _folder_id: str) -> Any:
        yield self.file

    def download_bytes(self, _file_id: str) -> bytes:
        self.downloads += 1
        return self.payload


def _cached_drive_source(root: Path, name: str, payload: bytes) -> tuple[Source, Path, _DriveClient]:
    root.mkdir(parents=True, exist_ok=True)
    source = Source(name="gemini", folder="AI Studio", path=root)
    cache = drive.drive_cache_file_path(root, name)
    cache.write_bytes(payload)
    cache.with_name(cache.name + ".revision").write_text(_TIMESTAMP, encoding="utf-8")
    return source, cache, _DriveClient(name, payload)


@pytest.mark.parametrize("known_revision", [False, True])
def test_w9_drive_cache_accepts_null_jsonl(
    tmp_path: Path, known_revision: bool,
) -> None:
    """W9-90-23: either production cache route redownloads valid null records."""
    payload = b'null\n{"value":1}\n'
    source, cache, client = _cached_drive_source(tmp_path / "cache", "session.jsonl", payload)
    records = list(drive.iter_drive_raw_data(
        source=source, client=cast(Any, client), blob_store=BlobStore(tmp_path / "blob"),
        known_mtimes={str(cache): _TIMESTAMP} if known_revision else None,
    ))
    assert client.downloads == 0
    assert len(records) == (0 if known_revision else 1)
    if records:
        assert records[0].blob_hash == sha256(payload).hexdigest()


@pytest.mark.parametrize("failure_at", ["write", "close"])
def test_w9_drive_failed_cache_publication_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str,
) -> None:
    """W9-90-24: write/close failures leaked the temporary in the public route."""
    source, cache, client = _cached_drive_source(tmp_path / "cache", "session.json", b"{")
    client.payload = b'{"replacement":true}'
    before = set(cache.parent.iterdir())
    original = drive.tempfile.NamedTemporaryFile

    class FailingTemporary:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.handle = original(*args, **kwargs)
            self.name = self.handle.name

        def __enter__(self) -> FailingTemporary:
            return self

        def write(self, raw: bytes) -> None:
            self.handle.write(raw[:1])
            if failure_at == "write":
                raise OSError("injected cache write failure")
            self.handle.write(raw[1:])

        def __exit__(self, *_args: Any) -> None:
            self.handle.close()
            if failure_at == "close":
                raise OSError("injected cache close failure")

    monkeypatch.setattr(drive.tempfile, "NamedTemporaryFile", FailingTemporary)
    with pytest.raises(OSError, match="injected cache"):
        list(drive.iter_drive_raw_data(
            source=source, client=cast(Any, client), blob_store=BlobStore(tmp_path / "blob"),
        ))
    assert client.downloads == 1
    assert cache.read_bytes() == b"{"
    assert set(cache.parent.iterdir()) == before


def test_w9_drive_unchanged_large_integer_cache_does_not_download(tmp_path: Path) -> None:
    """W9-90-25: use_float validation rejects an integer the JSON cache accepts."""
    payload = json.dumps({"integer": 2**128}).encode()
    source, cache, client = _cached_drive_source(tmp_path / "cache", "session.json", payload)
    records = list(drive.iter_drive_raw_data(
        source=source, client=cast(Any, client), blob_store=BlobStore(tmp_path / "blob"),
        known_mtimes={str(cache): _TIMESTAMP},
    ))
    assert records == []
    assert client.downloads == 0


def test_w9_trajectory_preflight_reports_degraded_steps(tmp_path: Path) -> None:
    """W9-91-03: unsupported step caveats previously still reported supported."""
    source = tmp_path / "trajectory.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript("""
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES ('w9-trajectory', 'w9-cascade');
            INSERT INTO steps VALUES (0, 'message', 'v1', '{"role":"user","text":"hello"}');
            INSERT INTO steps VALUES (1, 'future-step', 'v999', '{}');
        """)
    result = preflight_import_source(source)
    assert result.supported_count == 1
    assert result.caveats
    assert result.status is ImportPreflightStatus.DEGRADED


def test_w9_completed_ingest_repeats_materialized_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W9-91-15: losing cursor_update must not leave a completed attempt at zero."""
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    source = tmp_path / "codex" / "session.jsonl"
    _codex_file(source)
    processor = _processor(archive_root, source.parent, "codex")
    original = processor._cursor.update_ingest_attempt
    dropped: list[str] = []

    def drop_cursor_progress(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("phase") == "cursor_update":
            dropped.append("cursor_update")
            return False
        return original(*args, **kwargs)

    monkeypatch.setattr(processor._cursor, "update_ingest_attempt", drop_cursor_progress)
    metrics = asyncio.run(processor.ingest_files([source], emit_event=False, whole_archive_convergence=False))
    assert metrics.succeeded_file_count == 1
    assert dropped
    with sqlite3.connect(archive_root / "ops.db") as connection:
        assert connection.execute(
            "SELECT materialized_count FROM ingest_attempts WHERE status = 'completed'"
        ).fetchall() == [(1,)]


def test_w9_source_only_protobuf_does_not_invoke_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W9-91-17: raw-only acquisition incorrectly called the language server."""
    from polylogue.sources import source_parsing

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    source = tmp_path / "antigravity" / "conversations" / "w9.pb"
    source.parent.mkdir(parents=True)
    payload = b"\x08\x01"
    source.write_bytes(payload)
    processor = _processor(archive_root, source.parent, "antigravity")

    def unexpected_converter(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("source-only acquisition invoked the Antigravity converter")

    monkeypatch.setattr(source_parsing, "iter_antigravity_language_server_sessions", unexpected_converter)
    set_degraded(DegradedReason(code="schema_version_mismatch", message="derived tier unavailable", derived_only=True))
    try:
        result = processor._ingest_full_paths_sync([source], source_name="antigravity")
    finally:
        clear_degraded()
    assert result.succeeded == [source]
    assert result.failed == []
    with sqlite3.connect(archive_root / "source.db") as connection:
        assert connection.execute("SELECT lower(hex(blob_hash)), parsed_at_ms FROM raw_sessions").fetchall() == [
            (sha256(payload).hexdigest(), None),
        ]


def test_w9_source_only_zip_preserves_declared_sidecar_provider(tmp_path: Path) -> None:
    """W9-92-01: source-only storage stamps UNKNOWN on a Claude-owned sidecar."""
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    source_root = tmp_path / "inbox"
    source_root.mkdir()
    bundle = source_root / "sidecar.zip"
    payload = json.dumps([{
        "uuid": "not-a-standalone-session", "name": "embedded tool output",
        "created_at": _TIMESTAMP, "updated_at": _TIMESTAMP,
        "chat_messages": [{"uuid": "m", "sender": "human", "text": "embedded", "created_at": _TIMESTAMP}],
    }]).encode()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("tool-results/dump.json", payload)
    processor = _processor(archive_root, source_root, "unknown")
    set_degraded(DegradedReason(code="schema_version_mismatch", message="derived tier unavailable", derived_only=True))
    try:
        result = processor._ingest_full_paths_sync([bundle], source_name="unknown")
    finally:
        clear_degraded()
    assert result.succeeded == [bundle]
    with sqlite3.connect(archive_root / "source.db") as connection:
        assert connection.execute("SELECT origin, lower(hex(blob_hash)) FROM raw_sessions").fetchall() == [
            (origin_from_provider(Provider.CLAUDE_CODE).value, sha256(payload).hexdigest()),
        ]


def test_w9_jsonl_boundary_keeps_newline_fast_path() -> None:
    """W9-92-11: the old regex makes this production boundary call scan lines."""
    class NoLineWalk(bytes):
        def __getitem__(self, key: Any) -> Any:
            value = super().__getitem__(key)
            return NoLineWalk(value) if isinstance(value, bytes) else value

        def find(self, *_args: Any, **_kwargs: Any) -> int:
            pytest.fail("ordinary newline-terminated JSONL entered the per-line count")

    boundary = jsonl_complete_prefix(NoLineWalk(b'{}\n' * 100))
    assert boundary.record_count == 100
    assert boundary.prefix_size == 300
    assert boundary.incomplete_tail is False
    for payload, expected in [(b'{}\n\n{}\n', 2), (b'\n{}\n', 1), (b'{}\n \t\r\n{}\n', 2)]:
        assert jsonl_complete_prefix(payload).record_count == expected


def test_w9_ops_holder_accepts_symlinked_database(tmp_path: Path) -> None:
    """W9-92-12: the old holder checks the symlink spelling for SQLite's SHM."""
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    physical = tmp_path / "physical-ops.db"
    (root / "ops.db").replace(physical)
    (root / "ops.db").symlink_to(physical)
    with write_lease("w9-holder", archive_root=root):
        holder = cold_build._hold_ops_checkpoints(root)
    try:
        assert holder is not None
        assert physical.with_name(physical.name + "-shm").exists()
    finally:
        if holder is not None:
            holder.close()


@pytest.mark.parametrize("count,total,sealed,expected_eta", [(1, 1, True, 0.0), (1, 2, True, None), (1, 1, False, None)])
def test_w9_completed_cold_build_eta_does_not_expire(
    monkeypatch: pytest.MonkeyPatch, count: int, total: int, sealed: bool, expected_eta: float | None,
) -> None:
    """W9-92-13: a sealed complete cached receipt count loses ETA after stalling."""
    generation = object.__new__(cold_build.ColdBuildGeneration)
    generation._accepted_progress_lock = threading.Lock()
    generation._accepted_progress_total = total
    generation._accepted_progress_count = count
    generation._accepted_progress_valid = True
    generation._accepted_progress_started_at = 0.0
    generation._accepted_progress_last_advanced_at = 1.0
    generation._accepted_progress_denominator_sealed = sealed
    monkeypatch.setattr(cold_build.time, "monotonic", lambda: 1000.0)
    assert generation.accepted_progress[3] == expected_eta


def test_w9_hook_paste_uses_archive_lease_for_generation_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W9-92-18: the generation directory is not the root named by the lease."""
    root = tmp_path / "archive"
    index = root / "generations" / "w9" / "index.db"
    index.parent.mkdir(parents=True)
    epoch_ms = int(datetime(2026, 6, 2, tzinfo=UTC).timestamp() * 1000)
    with sqlite3.connect(index) as connection:
        connection.executescript("""
            CREATE TABLE sessions (session_id TEXT, native_id TEXT, paste_count INTEGER);
            CREATE TABLE messages (
                message_id TEXT, session_id TEXT, role TEXT, has_paste INTEGER,
                paste_boundary TEXT, occurred_at_ms INTEGER, position INTEGER
            );
            INSERT INTO sessions VALUES ('w9-session', 'w9-native', 0);
        """)
        connection.execute(
            "INSERT INTO messages VALUES ('w9-message', 'w9-session', 'user', 0, NULL, ?, 0)", (epoch_ms,),
        )
    event = {"session_id": "w9-native", "timestamp": _TIMESTAMP, "event_type": "UserPromptSubmit"}
    monkeypatch.setattr(hook_paste_enrichment, "_archive_index_path", lambda _path: index)
    monkeypatch.setattr(hook_paste_enrichment, "_archive_source_path", lambda _path: root / "source.db")
    monkeypatch.setattr(hook_paste_enrichment, "_iter_hook_paste_events", lambda *_args: [event])
    with arm_write_lease_enforcement(), write_lease("w9-paste", archive_root=root):
        assert hook_paste_enrichment.enrich_paste_from_hooks(root / "ops.db") == 1
    with sqlite3.connect(index) as connection:
        assert connection.execute("SELECT has_paste FROM messages").fetchall() == [(1,)]
        assert connection.execute("SELECT paste_count FROM sessions").fetchall() == [(1,)]
