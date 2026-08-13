from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.core.raw_failure_evidence import RAW_FAILURE_EVIDENCE_KINDS
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch import (
    _MAX_APPEND_PLAN_PAYLOAD_BYTES,
    CursorAuthorityBlockedError,
    LiveBatchProcessor,
    _ArchiveFullWriteResult,
    append_capability_receipt,
)
from polylogue.sources.live.batch_support import (
    _BROWSER_CAPTURE_PREFIX_PROBE_BYTES,
    _DEFER_APPEND,
    _AppendPlan,
    _AppendResult,
    _browser_capture_prefix_probe,
    _detect_provider_from_path_sample,
    _FullIngestResult,
    _parse_path_as_session_artifact,
    _parse_payload_as_session_artifact,
    encode_cursor_hash_authority,
    sha256_range_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.sources.source_parsing import has_decoded_session_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.sqlite.archive_tiers import archive as archive_tier_module
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance


@pytest.mark.parametrize(
    ("provider", "stable_session_identity", "status"),
    [
        ("codex", False, "unsupported"),
        ("codex", True, "supported"),
        ("claude-code", False, "unsupported"),
        ("claude-code", True, "supported"),
        ("chatgpt", True, "unsupported"),
    ],
)
def test_append_capability_receipt_is_keyed_to_live_identity_contract(
    provider: str,
    stable_session_identity: bool,
    status: str,
) -> None:
    receipt = append_capability_receipt(
        provider=provider,
        package_version="v1",
        element_kind="session_record_stream",
        stable_session_identity=stable_session_identity,
    )

    assert receipt.status == status
    payload = receipt.to_dict()
    assert (payload["provider"], payload["package_version"], payload["element_kind"]) == (
        provider,
        "v1",
        "session_record_stream",
    )
    assert payload["capability_source"] == "LiveBatchProcessor.append"
    assert payload["operation"] == "append_prefix"
    if provider not in {"codex", "claude-code"}:
        assert payload["reason"] == "live append route supports only Codex and Claude Code JSONL identity contracts"
    elif not stable_session_identity:
        assert payload["reason"] == "append delta requires a stable persisted session identity"
    else:
        assert payload["reason"] is None


from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    read_archive_raw_session_envelope,
    upsert_raw_artifact,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_ARCHIVE_STORAGE_TIERS = ",".join(spec.tier.value for spec in ARCHIVE_TIER_SPECS.values())


def _complete_archive_storage_probe_fields() -> dict[str, object]:
    return _archive_storage_probe_fields(
        present={spec.tier for spec in ARCHIVE_TIER_SPECS.values()},
        versions={spec.tier: spec.version for spec in ARCHIVE_TIER_SPECS.values()},
    )


def _archive_storage_probe_fields(
    *,
    present: set[ArchiveTier],
    versions: dict[ArchiveTier, int | None],
) -> dict[str, object]:
    return {
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "archive_present_tiers": ",".join(
            spec.tier.value for spec in ARCHIVE_TIER_SPECS.values() if spec.tier in present
        ),
        "archive_missing_tiers": ",".join(
            spec.tier.value for spec in ARCHIVE_TIER_SPECS.values() if spec.tier not in present
        ),
        "archive_tier_user_versions_json": json.dumps(
            {spec.tier.value: versions.get(spec.tier) for spec in ARCHIVE_TIER_SPECS.values()},
            sort_keys=True,
        ),
    }


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


def _append_raw_parse_state(archive_root: Path) -> tuple[int | None, str | None]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute(
            """SELECT parsed_at_ms, parse_error FROM raw_sessions
               WHERE source_index = -1 ORDER BY acquired_at_ms DESC, raw_id DESC LIMIT 1"""
        ).fetchone()
    assert row is not None
    return cast(tuple[int | None, str | None], row)


def _raw_revision_envelope_row(archive_root: Path, raw_id: str) -> tuple[object, ...]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
    assert row is not None
    return cast(tuple[object, ...], row)


def _seed_live_append_plan(
    archive_root: Path,
    *,
    native_id: str,
) -> tuple[Path, _AppendPlan, object, LiveBatchProcessor]:
    root = archive_root / "sessions"
    root.mkdir()
    path = root / f"{native_id}.jsonl"
    baseline = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        '"role":"user","content":[{"type":"input_text","text":"zero"}]}}\n'
    ).encode()
    path.write_bytes(baseline)
    index_db = archive_root / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    seeded = asyncio.run(processor.ingest_files([path], emit_event=False))
    assert seeded.succeeded_file_count == 1
    append = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
    )
    with path.open("ab") as handle:
        handle.write(append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    return path, plan, _append_owner(archive_root), processor


def test_live_append_replay_streams_retained_jsonl_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append replay must not resurrect eager blob materialization."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="streamed-append")
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("append replay eagerly read raw blob")),
    )

    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.succeeded == [plan]
    assert result.failed == []


def test_live_append_acquires_with_unreadable_active_pointer(tmp_path: Path) -> None:
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="degraded-append")
    (tmp_path / ".index-active-pointer").write_bytes(b"\xff")
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        result = ingest_append_plans(cast(Any, owner), [plan])
    finally:
        clear_degraded()

    assert result.succeeded == [plan]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        append_row = conn.execute(
            """
            SELECT logical_source_key, revision_kind, predecessor_raw_id,
                   baseline_raw_id, append_start_offset, append_end_offset,
                   revision_authority
            FROM raw_sessions
            WHERE source_index = -1
            """
        ).fetchone()
    assert append_row is not None
    assert append_row[:2] == ("codex:degraded-append", "append")
    assert append_row[2] is not None
    assert append_row[3] is not None
    assert append_row[4:] == (
        plan.start_offset,
        plan.last_complete_newline,
        "byte_proven",
    )


def test_derived_only_live_append_candidate_uses_source_acquisition(tmp_path: Path) -> None:
    """The managed batch route must not plan an index-backed append while derived-only."""

    import hashlib

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    path, _plan, _owner, processor = _seed_live_append_plan(tmp_path, native_id="degraded-managed-append")
    index_db = tmp_path / "index.db"
    index_digest_before = hashlib.sha256(index_db.read_bytes()).hexdigest()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_count_before = int(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?", (str(path),)).fetchone()[0]
        )
    pointer = tmp_path / ".index-active-pointer"
    pointer.write_bytes(b"\xff")
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        metrics = asyncio.run(processor.ingest_files([path], emit_event=False))
    finally:
        clear_degraded()

    assert metrics.succeeded_file_count == 1
    assert metrics.append_file_count == 0
    assert metrics.full_file_count == 1
    assert pointer.read_bytes() == b"\xff"
    assert hashlib.sha256(index_db.read_bytes()).hexdigest() == index_digest_before
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """SELECT parsed_at_ms, parse_error FROM raw_sessions
               WHERE source_path = ? ORDER BY acquired_at_ms DESC, raw_id DESC""",
            (str(path),),
        ).fetchall()
    assert len(rows) == raw_count_before + 1
    assert rows[0] == (None, None)


def test_live_full_replay_streams_retained_jsonl_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full replay must stream its older retained JSONL snapshot."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "streamed-full.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"streamed-full"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    assert processor._ingest_full_paths_sync([path], source_name="codex").succeeded == [path]
    with path.open("ab") as handle:
        handle.write(
            b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
            b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
        )
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full replay eagerly read raw blob")),
    )

    result = processor._ingest_full_paths_sync([path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []


def test_full_ingest_acquires_but_does_not_parse_when_derived_tier_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-gbs02: a derived-only degraded reason must still acquire raw content.

    Raw acquisition only ever writes source.db; when the daemon is degraded
    ONLY because index.db/embeddings.db are behind the running code
    (``DegradedReason.derived_only=True``), acquisition must proceed --
    otherwise the daemon loses live capture data for the entire duration of
    a schema-migration/reindex window. Materialization (parse) must still be
    skipped: the raw row lands with ``parsed_at_ms IS NULL``, exactly the
    same "not yet materialized" state ordinary convergence already knows
    how to pick up once the derived tier catches up.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "degraded-full.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"degraded-full"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    json_path = root / "degraded-full.json"
    json_path.write_bytes(b'{"mapping":{"root":{"message":{"author":{"role":"user"}}}}}')
    classified_path = root / "subagents" / "worker" / "agent-degraded.meta.json"
    classified_path.parent.mkdir(parents=True)
    classified_path.write_bytes(b'{"mapping":{"root":{"message":{"author":{"role":"user"}}}}}')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="index.db:46!=57",
            derived_only=True,
        )
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._parse_payload_as_session_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode source-only evidence")),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not classify source-only JSONL")),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.has_decoded_session_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not inspect source-only JSON evidence")),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._detect_provider_from_raw_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not detect source-only provider")),
    )
    try:
        result = processor._ingest_full_paths_sync([path, json_path, classified_path], source_name="claude-code")
    finally:
        clear_degraded()

    assert result.succeeded == [path, json_path, classified_path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_states = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions ORDER BY source_path").fetchall()
        artifact_rows = conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE source_path = ?", (str(classified_path),)
        ).fetchone()
    assert raw_states == [(None, None), (None, None), (None, None)]
    assert artifact_rows == (0,)


def test_source_only_full_ingest_streams_admitted_zip_members_without_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production full-ingest ZIP route must retain bytes before decode."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "degraded.zip"
    member_names = ("sessions/one.jsonl", "sessions/two.json")
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(member_names[0], b'{"opaque":"first"}\n')
        zf.writestr(member_names[1], b'{"opaque":"second"}')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    for target in (
        "polylogue.sources.live.batch.iter_zip_entry_raw_data",
        "polylogue.sources.live.batch.LiveBatchProcessor._sniff_zip_provider",
        "polylogue.sources.live.batch._detect_provider_from_raw_bytes",
        "polylogue.sources.source_acquisition_components.iter_entry_payloads",
        "polylogue.sources.source_acquisition_components.classify_artifact",
    ):
        monkeypatch.setattr(
            target, lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode ZIP"))
        )
    try:
        result = processor._ingest_full_paths_sync([bundle], source_name="claude-code")
    finally:
        clear_degraded()

    assert result.succeeded == [bundle]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            "SELECT source_path, source_index, parsed_at_ms, parse_error FROM raw_sessions ORDER BY source_index"
        ).fetchall()
    assert rows == [
        (f"{bundle}:{member_names[0]}", 0, None, None),
        (f"{bundle}:{member_names[1]}", 1, None, None),
    ]


def test_source_only_zip_replay_resolves_unknown_chatgpt_member_and_keeps_duplicate_coordinates(
    tmp_path: Path,
) -> None:
    """Recovery, not acquisition, resolves UNKNOWN ZIP bytes and replays each coordinate."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "inbox"
    root.mkdir()
    bundle = root / "export.zip"
    payload = json.dumps(
        [
            {
                "id": "zip-chatgpt",
                "conversation_id": "zip-chatgpt",
                "title": "ZIP recovery",
                "create_time": 1_700_000_000,
                "update_time": 1_700_000_001,
                "current_node": "assistant-node",
                "mapping": {
                    "user-node": {
                        "id": "user-node",
                        "parent": None,
                        "children": ["assistant-node"],
                        "message": {
                            "id": "user-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["recover ZIP"]},
                            "create_time": 1_700_000_000,
                        },
                    },
                    "assistant-node": {
                        "id": "assistant-node",
                        "parent": "user-node",
                        "children": [],
                        "message": {
                            "id": "assistant-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["replayed"]},
                            "create_time": 1_700_000_001,
                        },
                    },
                },
            }
        ],
        sort_keys=True,
    ).encode()
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("first/conversations.json", payload)
        zf.writestr("second/conversations.json", payload)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        result = processor._ingest_full_paths_sync([bundle], source_name="unknown")
    finally:
        clear_degraded()

    assert result.succeeded == [bundle]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        before_replay = conn.execute(
            "SELECT raw_id, hex(blob_hash), source_path, source_index, origin FROM raw_sessions ORDER BY source_index"
        ).fetchall()
    assert len(before_replay) == 2
    assert len({row[0] for row in before_replay}) == 2
    assert len({row[1] for row in before_replay}) == 1
    assert [row[2:] for row in before_replay] == [
        (f"{bundle}:first/conversations.json", 0, "unknown-export"),
        (f"{bundle}:second/conversations.json", 1, "unknown-export"),
    ]

    replay = backfill_historical_revision_evidence(tmp_path)

    assert replay.replayed_logical_sources == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count FROM sessions").fetchall() == [("zip-chatgpt", 2)]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE origin = 'chatgpt-export'").fetchone() == (2,)


def test_source_only_full_ingest_snapshots_unrecognized_codex_state_without_shape_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded source tier retains a valid but future-shaped Codex state DB."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "codex"
    root.mkdir()
    state_db = root / "state_5.sqlite"
    _write_plain_sqlite_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    monkeypatch.setattr(
        "polylogue.sources.parsers.codex_state.is_in_scope_codex_sqlite_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not inspect source-only state schema")),
    )
    try:
        result = processor._ingest_full_paths_sync([state_db], source_name="codex")
    finally:
        clear_degraded()

    assert result.succeeded == [state_db]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT source_path, parsed_at_ms FROM raw_sessions").fetchall() == [(str(state_db), None)]


def _write_codex_thread_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                cwd TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                source TEXT NOT NULL,
                model TEXT,
                agent_nickname TEXT,
                agent_role TEXT,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT NOT NULL,
                child_thread_id TEXT NOT NULL PRIMARY KEY,
                status TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex-thread", "Recover retained state", "/work", 1, 2, "cli", "gpt-5", None, None, 0),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges VALUES (?, ?, ?)",
            ("codex-thread", "codex-child", "closed"),
        )


def test_source_only_codex_state_recovery_replays_retained_thread_evidence(tmp_path: Path) -> None:
    """Removing the replay effect leaves the durable state raw pending and title-less."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "codex"
    state_db = root / "state_5.sqlite"
    _write_codex_thread_state_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        assert processor._ingest_full_paths_sync([state_db], source_name="codex").succeeded == [state_db]
    finally:
        clear_degraded()

    replay = backfill_historical_revision_evidence(tmp_path)

    assert replay.scanned == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms IS NOT NULL FROM raw_sessions").fetchone() == (1,)
        assert conn.execute(
            "SELECT hook_event_id, event_type FROM raw_hook_events ORDER BY hook_event_id"
        ).fetchall() == [
            ("codex-thread-spawn-edge:codex-thread:codex-child", "codex_thread_spawn_edge"),
            ("codex-thread-title:codex-thread", "codex_thread_title"),
        ]


@pytest.mark.parametrize("state_name", ["state.db", "verification_evidence.db"])
def test_source_only_hermes_named_sqlite_uses_consistent_backup_before_generic_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
) -> None:
    """A direct file copy loses an uncheckpointed WAL row; the snapshot retains it."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "hermes"
    state_db = root / state_name
    state_db.parent.mkdir(parents=True)
    writer = sqlite3.connect(state_db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE retained_wal_row (value TEXT NOT NULL)")
    writer.commit()
    writer.execute("INSERT INTO retained_wal_row VALUES ('must survive')")
    writer.commit()
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="hermes", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr("polylogue.sources.parsers.hermes_state.looks_like_state_db_path", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "polylogue.sources.parsers.hermes_verification.looks_like_verification_evidence_db_path",
        lambda *_a, **_k: False,
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        assert processor._ingest_full_paths_sync([state_db], source_name="hermes").succeeded == [state_db]
    finally:
        clear_degraded()
        writer.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        blob_hash = str(conn.execute("SELECT hex(blob_hash) FROM raw_sessions").fetchone()[0]).lower()
    with sqlite3.connect(BlobStore(tmp_path / "blob").blob_path(blob_hash)) as snapshot:
        assert snapshot.execute("SELECT value FROM retained_wal_row").fetchall() == [("must survive",)]


def test_full_ingest_acquires_when_index_is_genuinely_semantic_distance_stale(
    tmp_path: Path,
) -> None:
    """polylogue-gbs02: acquire-only mode must survive a REAL stale index tier.

    The sibling test above proves the skip logic but leaves index.db at the
    current version, so it never exercises the open: the ordinary
    ``ArchiveStore.open_existing(read_only=False)`` writer hard-refuses an
    index tier at a semantic-reparse distance (the live archive's actual
    pre-rebuild state, index.db 46 vs current code). This test ages the
    index to that distance first — with the source-tier-only open routed via
    ``_open_archive_for_live_write`` the acquire succeeds and the stale
    index file stays byte-identical; without it, the open raises before any
    raw write and this test fails.
    """
    import hashlib
    import sqlite3 as _sqlite3

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "degraded-stale-index.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"degraded-stale-index"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    # Bootstrap a real archive file set, then age the index tier to the
    # semantic-reparse distance (46 is the live pre-818fy generation).
    with ArchiveStore.open_existing(tmp_path, read_only=False):
        pass
    index_db = tmp_path / "index.db"
    conn = _sqlite3.connect(index_db)
    try:
        conn.execute("PRAGMA user_version = 46")
        conn.commit()
    finally:
        conn.close()
    index_digest_before = hashlib.sha256(index_db.read_bytes()).hexdigest()
    pointer = tmp_path / ".index-active-pointer"
    pointer.write_bytes(b"\xff")

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "cursors.db", ops_db_path=tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="index.db:46!=current",
            derived_only=True,
        )
    )
    try:
        metrics = asyncio.run(processor.ingest_files([path], emit_event=False))
    finally:
        clear_degraded()

    assert metrics.succeeded_file_count == 1
    assert metrics.failed_file_count == 0
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert parse_error is None
    assert hashlib.sha256(index_db.read_bytes()).hexdigest() == index_digest_before, (
        "the stale index tier must never be opened for write during acquire-only ingest"
    )
    assert pointer.read_bytes() == b"\xff"


def test_live_raw_compaction_holds_generation_lease_through_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protected set and destructive cleanup observe one unpromotable generation."""

    from polylogue.storage import raw_retention
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )
    phases: list[str] = []

    def assert_promotion_excluded(*_args: object, **_kwargs: object) -> SimpleNamespace:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        phases.append("authority")
        return SimpleNamespace(protected_raw_ids=frozenset(), eligible_raw_ids=frozenset())

    def assert_delete_excluded(*_args: object, **_kwargs: object) -> SimpleNamespace:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        phases.append("delete")
        return SimpleNamespace(errors=())

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", assert_promotion_excluded)
    monkeypatch.setattr(raw_retention, "compact_paths_superseded_raw_snapshots", assert_delete_excluded)

    processor._compact_superseded_raw_snapshots([path])

    assert phases == ["authority", "delete"]
    with RebuildLease(tmp_path):
        pass


def test_full_ingest_empty_jsonl_is_not_misclassified_as_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty session candidate is a terminal typed refusal, not a retry."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "empty.jsonl"
    path.write_bytes(b"")
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

    result = processor._ingest_full_paths_sync([path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parse_error != "captured JSONL payload ends before a complete record boundary"
    assert isinstance(parse_error, str) and "no sessions with positive conversational evidence" in parse_error
    assert parsed_at_ms is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unsupported_shape", "unsupported_parseable", 0)


def test_full_ingest_unknown_export_without_sessions_records_terminal_evidence(tmp_path: Path) -> None:
    root = tmp_path / "chatgpt"
    root.mkdir()
    path = root / "export.jsonl"
    path.write_bytes(b"")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_export_no_session", "unsupported_parseable", 0)


def test_full_ingest_unknown_weak_path_ndjson_records_terminal_evidence(tmp_path: Path) -> None:
    """NDJSON takes the same strict terminal classification route as JSONL."""

    root = tmp_path / "chatgpt"
    path = root / "analysis" / "export.ndjson"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root, suffixes=(".jsonl", ".ndjson")),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_export_no_session", 0)


@pytest.mark.parametrize(
    ("payload", "expected_artifact"),
    [
        (b"{", ("terminal_unknown_json_decode", "decode_failed")),
        (b"", ("terminal_unknown_json_decode", "decode_failed")),
    ],
)
def test_full_ingest_unknown_weak_path_json_retains_terminal_evidence(
    tmp_path: Path,
    payload: bytes,
    expected_artifact: tuple[str, str],
) -> None:
    """Unknown weak-path JSON reaches durable generic terminal handling."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "export.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    path_artifact = classify_artifact_path(path, provider=Provider.UNKNOWN)
    assert path_artifact is not None and not path_artifact.parse_as_session
    assert not has_decoded_session_evidence(path, provider=Provider.UNKNOWN)

    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw = conn.execute(
            "SELECT raw_id, blob_size, parse_error FROM raw_sessions WHERE source_path = ?", (str(path),)
        ).fetchone()
        artifact = conn.execute(
            """
            SELECT artifact_kind, support_status
            FROM raw_artifacts
            WHERE raw_id = ?
            """,
            (raw[0],) if raw is not None else (None,),
        ).fetchone()
    # The preconditions above would take the weak path-exclusion branch if
    # the production unknown-JSON exemption were removed.
    assert raw is not None
    assert raw[1] == len(payload)
    assert isinstance(raw[2], str)
    assert artifact == expected_artifact


def test_full_ingest_unknown_weak_directory_still_excludes_strong_sidecar(tmp_path: Path) -> None:
    """A weak directory cannot override a definitive non-session filename."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "sessions-index.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mapping":{"looks":"conversational"}}', encoding="utf-8")
    path_artifact = classify_artifact_path(path, provider=Provider.UNKNOWN)
    assert path_artifact is not None and path_artifact.kind.value == "metadata_document"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "ops.db"))),
        (WatchSource(name="unknown", root=root, suffixes=(".json",)),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == []
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_full_ingest_unknown_malformed_jsonl_records_terminal_decode_and_stops_retrying(tmp_path: Path) -> None:
    """Complete malformed JSONL lines are terminal decode evidence, not no-session evidence."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "malformed.jsonl"
    path.write_bytes(b'{"broken":}\n{"also_broken":}\n')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    first = processor._ingest_full_paths_sync([path], source_name="unknown")
    second = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert first.succeeded == [path]
    assert first.failed == []
    assert second.succeeded == [path]
    assert second.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0


def test_full_ingest_unknown_malformed_final_jsonl_record_records_terminal_decode(tmp_path: Path) -> None:
    """A malformed final JSONL record contributes to strict decode evidence."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "malformed-final.jsonl"
    path.write_bytes(b'{"only_broken":}\n')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed")


def test_full_ingest_unknown_json_decode_records_terminal_decode_evidence(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b"{")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed", 0)


def test_full_ingest_unknown_invalid_utf8_records_terminal_decode_evidence(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b"\xff")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed", 0)


def test_full_ingest_unknown_semantic_value_error_remains_unexplained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b'{"unrelated": "payload"}')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    def raise_semantic_value_error(*_args: object, **_kwargs: object) -> list[ParsedSession]:
        raise ValueError("semantic parser rejection")

    monkeypatch.setattr("polylogue.sources.live.batch.parse_payload", raise_semantic_value_error)

    result = processor._ingest_full_paths_sync([path], source_name="unknown")

    assert result.succeeded == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact_kinds = {row[0] for row in conn.execute("SELECT artifact_kind FROM raw_artifacts")}
    assert not artifact_kinds & RAW_FAILURE_EVIDENCE_KINDS
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.unexplained == 1


def test_full_ingest_defers_incomplete_jsonl_only_after_hot_prefix_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full-ingest route defers a capture only after byte-prefix proof.

    The empty-capture test above is the red twin. Both paths retain the raw
    and advance the cursor, but only this test changes the source so its
    captured bytes can be verified as a strict current prefix.
    """
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "active.jsonl"
    captured = b'{"type":"session_meta"'
    path.write_bytes(captured)
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
    captured_boundary_check = live_batch._captured_jsonl_ends_at_record_boundary

    def grow_source_after_capture(**kwargs: object) -> bool:
        path.write_bytes(captured + b"\n")
        return captured_boundary_check(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_batch, "_captured_jsonl_ends_at_record_boundary", grow_source_after_capture)

    result = processor._ingest_full_paths_sync([path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    _parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert isinstance(parse_error, str) and parse_error.endswith(
        "captured JSONL payload ends before a complete record boundary"
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("deferred_hot_jsonl_capture", "partial_decode", 1)


def test_full_ingest_applies_incomplete_record_guard_to_jsonl_txt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported ``.jsonl.txt`` wire suffix has JSONL tail authority too."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "active.jsonl.txt"
    captured = b'{"type":"session_meta"'
    path.write_bytes(captured)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "archive.sqlite"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "archive.sqlite"),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    boundary_check = live_batch._captured_jsonl_ends_at_record_boundary

    def grow_source_after_capture(**kwargs: object) -> bool:
        path.write_bytes(captured + b"\n")
        return boundary_check(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_batch, "_captured_jsonl_ends_at_record_boundary", grow_source_after_capture)

    result = processor._ingest_full_paths_sync([path], source_name="codex")

    assert result.succeeded == [path]
    _parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert isinstance(parse_error, str) and parse_error.endswith("complete record boundary")


def test_full_ingest_claude_partial_jsonl_has_provider_specific_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "claude"
    root.mkdir()
    path = root / "active.jsonl"
    captured = b'{"type":"assistant"'
    path.write_bytes(captured)
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    boundary_check = live_batch._captured_jsonl_ends_at_record_boundary

    def grow_source_after_capture(**kwargs: object) -> bool:
        path.write_bytes(captured + b"\n")
        return boundary_check(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_batch, "_captured_jsonl_ends_at_record_boundary", grow_source_after_capture)

    result = processor._ingest_full_paths_sync([path], source_name="claude-code")

    assert result.succeeded == [path]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("deferred_claude_code_partial_jsonl", "partial_decode", 1)


def test_streamed_incomplete_jsonl_capture_defers_completed_source_until_authority_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed hot capture proves its retained blob against the live prefix."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "streaming-active.jsonl"
    captured = b'{"type":"session_meta","payload":{"id":"streaming-active"}'
    completed = (
        b'{"type":"session_meta","payload":{"id":"streaming-active"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"complete"}]}}\n'
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
    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", len(captured) - 1)
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    boundary_check = live_batch._captured_jsonl_ends_at_record_boundary
    source_completed = False

    def complete_source_after_capture(**kwargs: object) -> bool:
        nonlocal source_completed
        if not source_completed:
            path.write_bytes(completed)
            source_completed = True
        return boundary_check(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_batch, "_captured_jsonl_ends_at_record_boundary", complete_source_after_capture)

    deferred = asyncio.run(processor.ingest_files([path]))

    assert deferred.full_file_count == 1
    assert deferred.succeeded_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind FROM raw_artifacts ORDER BY last_observed_at_ms DESC").fetchone()
    assert artifact == ("deferred_hot_jsonl_capture",)

    with pytest.raises(CursorAuthorityBlockedError, match="source-selection gate blocked"):
        asyncio.run(processor.ingest_files([path]))

    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == 0
    assert final_cursor.byte_size == len(completed)
    assert final_cursor.deferred_end_offset is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == []


def test_full_ingest_rejects_incomplete_jsonl_without_hot_prefix_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete static capture is terminal evidence, never deferred."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "static.jsonl"
    path.write_bytes(b'{"type":"session_meta"')
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

    result = processor._ingest_full_paths_sync([path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_corrupt_input", "decode_failed", 0)


def test_full_ingest_heartbeats_small_file_groups_with_current_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so these fixtures keep testing heartbeat/byte-scan mechanics,
    # not the now-refused empty shape.
    first.write_text(
        '{"type":"session_meta","payload":{"id":"first"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
    second.write_text(
        '{"type":"session_meta","payload":{"id":"second"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
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
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so this fixture keeps testing full-ingest mechanics, not the
    # now-refused empty shape.
    source.write_text(
        '{"type":"session_meta","payload":{"id":"large"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
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
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record (before the size padding) so this fixture keeps testing the
    # streaming-vs-eager routing it is named for, not the now-refused empty
    # shape.
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"large"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n' + (b" " * (9 * 1024 * 1024))
    )
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


def test_large_weak_path_uses_streaming_route_before_decoded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weak path cannot force an eager whole-file evidence decode."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "export.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        json.dumps(
            {
                "id": "weak-large",
                "title": "weak large export",
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
                            "id": "weak-u1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["question"]},
                            "metadata": {},
                        },
                    },
                    "assistant-node": {
                        "id": "assistant-node",
                        "parent": "user-node",
                        "children": [],
                        "message": {
                            "id": "weak-a1",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {},
                        },
                    },
                },
            }
        ).encode()
        + (b" " * (9 * 1024 * 1024))
    )
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root, suffixes=(".json",)),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr("polylogue.sources.live.batch_support._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr(
        "polylogue.sources.live.batch.has_decoded_session_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("large input decoded before streaming route")),
    )
    phases: list[str] = []

    def heartbeat(phase: str, **_kwargs: object) -> None:
        phases.append(phase)

    result = processor._ingest_full_paths_sync([path], source_name="chatgpt", heartbeat=heartbeat)

    assert result.failed == []
    assert "full_blob_copy" in phases
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)


def test_threshold_crossing_strong_sidecar_is_excluded_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A definitive sidecar path must never reach large-JSON admission."""

    root = tmp_path / "chatgpt"
    root.mkdir()
    path = root / "sessions-index.json"
    path.write_bytes(b"{}")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root, suffixes=(".json",)),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr("polylogue.sources.live.batch_support._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._large_non_jsonl_path_can_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strong sidecar reached large-file streaming admission")
        ),
    )

    result = processor._ingest_full_paths_sync([path], source_name="chatgpt")

    assert result.succeeded == []
    assert result.failed == []
    assert not _parse_payload_as_session_artifact(
        path,
        provider=Provider.CHATGPT,
        payload=b'{"mapping":{"session":"would otherwise look like an export"}}',
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_full_ingest_writes_archive_with_route_observability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    initialize_active_archive_root(tmp_path)
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
        "full.index.full_replace.fts_guard_clear",
        "full.index.full_replace.messages",
        "full.index.full_replace.blocks",
    }.issubset(result.stage_timings_s)
    assert cursor.list_convergence_debt(limit=10)[0].stage == "fts"
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
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        **_complete_archive_storage_probe_fields(),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 1,
        "payload_unavailable_file_count": 0,
        "payload_replayed_from_blob_file_count": 0,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
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
    initialize_active_archive_root(tmp_path)
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
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        **_complete_archive_storage_probe_fields(),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 0,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
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
        "preserved_native_bytes": "x" * 32_000,
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
        # Real receiver artifacts are key-sorted, so a large native payload
        # precedes the typed session and can push ``session.provider`` beyond
        # the ordinary 8 KiB acquisition prefix.
        "raw_provider_payload": native_payload,
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "dom-fallback",
            "title": "DOM fallback title",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [{"provider_turn_id": "dom-u1", "role": "user", "text": "DOM fallback", "ordinal": 0}],
        },
        "padding": "x" * 256,
    }
    source.write_text(json.dumps(capture_payload), encoding="utf-8")
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_active_archive_root(tmp_path)
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
    assert source.read_bytes().find(b'"provider": "chatgpt"') > 8192
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT origin FROM raw_sessions").fetchone() == ("chatgpt-export",)
        assert conn.execute("SELECT logical_source_key FROM raw_session_memberships").fetchone() == (
            "chatgpt:native-large",
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
        assert conn.execute("SELECT logical_source_key, session_id FROM raw_revision_heads").fetchone() == (
            "chatgpt:native-large",
            "chatgpt-export:native-large",
        )


def test_generic_large_browser_capture_json_uses_prefix_detection_without_unknown_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    initialize_active_archive_root(tmp_path)
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
        # Acquisition identity is artifact-scoped because one raw file may
        # contain many sessions. Parsed identity lives in index + membership.
        assert conn.execute("SELECT origin, native_id FROM raw_sessions").fetchone() == ("chatgpt-export", None)
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


def test_browser_capture_prefix_probe_finds_provider_past_1mib_raw_payload(tmp_path: Path) -> None:
    """polylogue-mvq8: session.provider beyond the 1MiB prefix must still detect.

    Real receiver artifacts key-sort with ``raw_provider_payload`` (an
    unbounded copy of the provider's own wire payload) sorting before
    ``session`` alphabetically. Once ``raw_provider_payload`` alone exceeds
    the 1MiB prefix-probe window, the plain byte-prefix regex never sees
    ``session.provider`` and the capture was permanently misdetected as
    ``unknown-export`` -- this reproduces that exact shape with real file
    bytes (no probe-size monkeypatching) and asserts the provider is still
    found.
    """
    target = tmp_path / "oversized-raw-payload.json"
    huge_padding = "x" * (_BROWSER_CAPTURE_PREFIX_PROBE_BYTES + 64 * 1024)
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:past-prefix",
        "provenance": {
            "source_url": "https://chatgpt.com/c/past-prefix",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-native-v1",
        },
        # Deliberately placed before ``session`` (as the real receiver's
        # key-sorted output places it) and sized past the probe window.
        "raw_provider_payload": {"padding": huge_padding},
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "past-prefix",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "hi"}],
        },
    }
    target.write_text(json.dumps(capture_payload), encoding="utf-8")

    # Confirm the fixture actually reproduces the bug shape: the provider
    # marker sits past the probe window, and the file exceeds it too.
    assert target.stat().st_size > _BROWSER_CAPTURE_PREFIX_PROBE_BYTES
    assert target.read_bytes().find(b'"provider": "chatgpt"') > _BROWSER_CAPTURE_PREFIX_PROBE_BYTES

    is_browser_capture, provider = _browser_capture_prefix_probe(target)
    assert is_browser_capture is True
    assert provider is Provider.CHATGPT


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
    for filename in (spec.filename for spec in ARCHIVE_TIER_SPECS.values()):
        assert (tmp_path / filename).exists()
    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": True,
        **_archive_storage_probe_fields(
            present=set(ARCHIVE_TIER_SPECS),
            versions={tier: spec.version for tier, spec in ARCHIVE_TIER_SPECS.items()},
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


def test_full_ingest_retains_sidecar_evidence_and_ingests_genuine_session(tmp_path: Path) -> None:
    """Full live acquisition keeps non-session evidence and repairs session-shaped journals."""
    root = tmp_path / ".claude"
    metadata_path = root / "projects" / "project" / "subagents" / "agent-a.meta.json"
    journal_path = root / "projects" / "project" / "subagents" / "workflows" / "wf-run-1" / "journal.jsonl"
    session_path = root / "projects" / "project" / "genuine-session.jsonl"
    metadata_path.parent.mkdir(parents=True)
    journal_path.parent.mkdir(parents=True)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_payload = b'{"agentId":"agent-a","transcriptPath":"agent-a.jsonl"}'
    journal_payload = (
        json.dumps(
            {
                "type": "user",
                "sessionId": "wf-run-1",
                "uuid": "journal-message-1",
                "message": {"role": "user", "content": "retain this workflow evidence"},
            }
        )
        + "\n"
    ).encode()
    session_payload = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"real session"},'
        b'"uuid":"real-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"real-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"real reply"}]},"uuid":"real-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    metadata_path.write_bytes(metadata_payload)
    journal_path.write_bytes(journal_payload)
    session_path.write_bytes(session_payload)

    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = asyncio.run(processor.ingest_files([metadata_path, journal_path, session_path], emit_event=False))

    assert result.succeeded_file_count == 3
    assert result.failed_file_count == 0
    assert result.ingested_session_count == 2
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT a.source_path, a.artifact_kind, a.support_status, a.parse_as_session, r.blob_hash
            FROM raw_artifacts AS a
            JOIN raw_sessions AS r ON r.raw_id = a.raw_id
            WHERE a.parse_as_session = 0
            ORDER BY a.source_path
            """
        ).fetchall()

    assert [(Path(row[0]).name, row[1], row[2], row[3]) for row in rows] == [
        ("agent-a.meta.json", "agent_sidecar_meta", "unknown", 0),
    ]
    expected_payloads = {
        metadata_path.name: metadata_payload,
    }
    for source_path, _kind, _support_status, _parse_as_session, blob_hash in rows:
        blob_hash_hex = bytes(blob_hash).hex()
        assert (tmp_path / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]).read_bytes() == expected_payloads[
            Path(source_path).name
        ]


def test_append_declared_workflow_journal_retains_evidence_without_a_session(tmp_path: Path) -> None:
    """Malformed journals remain typed evidence when decoding cannot recover them."""
    path = tmp_path / ".claude" / "projects" / "project" / "subagents" / "workflows" / "wf-append" / "journal.jsonl"
    path.parent.mkdir(parents=True)
    payload = b'{"contentKey":"broken"\n'
    path.write_bytes(payload)
    plan = replace(_append_plan(path, payload, payload_hash="artifact"), source_name="claude-code")

    result = ingest_append_plans(cast(Any, _append_owner(tmp_path)), [plan])

    assert result.succeeded == [plan]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifacts = conn.execute(
            """
            SELECT artifact_kind, classification_reason, parse_as_session
            FROM raw_artifacts
            """
        ).fetchall()
    assert len(artifacts) == 1
    assert [row[0] for row in artifacts] == ["workflow_journal"]
    assert all(row[2] == 0 for row in artifacts)
    assert all("OriginSpec" in row[1] for row in artifacts)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_append_session_shaped_workflow_journal_enters_revision_repair(tmp_path: Path) -> None:
    """Decoded session evidence bypasses path-only workflow-journal admission."""
    path = tmp_path / ".claude" / "projects" / "project" / "subagents" / "workflows" / "wf-append" / "journal.jsonl"
    path.parent.mkdir(parents=True)
    payload = b"".join(
        b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n' for index in range(64)
    ) + (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"recover this journal record"},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"journal-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    path.write_bytes(payload)
    plan = replace(_append_plan(path, payload, payload_hash="session-shaped"), source_name="claude-code")

    result = ingest_append_plans(cast(Any, _append_owner(tmp_path)), [plan])

    assert result.succeeded == []
    assert result.failed == []
    assert result.deferred == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
        assert conn.execute("SELECT revision_kind, revision_authority FROM raw_sessions").fetchall() == [
            ("append", "quarantined")
        ]


def _write_plain_sqlite_db(path: Path) -> None:
    """A genuine SQLite database with no Hermes state.db/verification_evidence.db shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE unrelated_thing (id INTEGER PRIMARY KEY, value TEXT);")
        conn.commit()


def test_parse_payload_as_session_artifact_refuses_unrecognized_hermes_db_extension(tmp_path: Path) -> None:
    """polylogue-hbtj2: this used to be a bare ``.db``/``.sqlite``/``.sqlite3``
    extension match under provider=HERMES -- ANY file with that suffix was
    accepted as session content regardless of its actual bytes. It must now
    require the same content-verified shape check the path-based sibling
    function (``_parse_path_as_session_artifact``) already uses."""
    target = tmp_path / "state_5.sqlite"
    _write_plain_sqlite_db(target)
    payload = target.read_bytes()

    assert _parse_payload_as_session_artifact(target, provider=Provider.HERMES, payload=payload) is False


def test_parse_payload_as_session_artifact_still_accepts_genuine_hermes_state_db(tmp_path: Path) -> None:
    """Regression guard: the tightened check must not break the real feature."""
    target = tmp_path / "state.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, model_config TEXT,
                parent_session_id TEXT, started_at REAL, ended_at REAL, title TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_name TEXT, tool_calls TEXT,
                timestamp REAL NOT NULL, observed INTEGER DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0);
            """
        )
        conn.commit()
    payload = target.read_bytes()

    assert _parse_payload_as_session_artifact(target, provider=Provider.HERMES, payload=payload) is True


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
async def test_full_drive_capture_retains_acquisition_mode_after_gemini_detection(tmp_path: Path) -> None:
    """A configured Drive source survives shape detection's GEMINI fallback."""
    from polylogue.api import Polylogue

    root = tmp_path / "drive"
    root.mkdir()
    path = root / "live-capture.json"
    path.write_text(
        json.dumps(
            {
                "id": "live-drive-capture",
                "title": "Live Drive capture",
                "chunkedPrompt": {
                    "chunks": [
                        {"id": "chunk-1", "role": "user", "text": "hello"},
                        {"id": "chunk-2", "role": "model", "text": "hi"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    archive = Polylogue(archive_root=tmp_path / "archive")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="drive", root=root, suffixes=(".json",)),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        metrics = await processor.ingest_files([path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            capture_mode = conn.execute("SELECT capture_mode FROM raw_sessions").fetchone()

        assert metrics.full_file_count == 1
        assert capture_mode == (Provider.DRIVE.value,)
    finally:
        await archive.close()


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

    def capture(
        turns: list[dict[str, object]],
        *,
        captured_at: str = "2026-07-12T00:00:00+00:00",
    ) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:browser-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/browser-replacement",
                "captured_at": captured_at,
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
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            first_raw_id = source_conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),)
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-replacement'"
                ).fetchone()[0]
                == first_raw_id
            )
        assert first.succeeded_file_count == 1

        foreign_path = root / "foreign-quarantined.json"
        foreign_payload = json.dumps(
            capture([first_turn, divergent_turn], captured_at="2026-07-12T00:00:03+00:00")
        ).encode("utf-8")
        foreign_sessions = parse_payload(
            Provider.CHATGPT,
            [json.loads(foreign_payload)],
            foreign_path.stem,
            source_path=str(foreign_path),
        )
        assert len(foreign_sessions) == 1
        with ArchiveStore.open_existing(archive.archive_root, read_only=False) as foreign_archive:
            foreign_raw_id = foreign_archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=foreign_payload,
                source_path=str(foreign_path),
                acquired_at_ms=1,
            )
            foreign_archive.replace_raw_membership_census(
                foreign_raw_id,
                foreign_sessions,
                parser_fingerprint="foreign-quarantined-test",
                censused_at_ms=1,
            )

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
        live_decisions = {raw_id: decision for raw_id, decision in decisions if raw_id in raw_ids}
        assert set(live_decisions.values()) == {"superseded_prefix", "applied"}
        assert live_decisions[accepted_raw_id] == "applied"
        assert attachment == ("acquired", len(asset_bytes), asset_hash)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            assert source_conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE raw_id = ? AND logical_source_key = 'chatgpt:browser-replacement'
                """,
                (foreign_raw_id,),
            ).fetchone() == (None,)

        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_before_reverse = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
        path.write_text(
            json.dumps(capture([first_turn], captured_at="2026-07-12T00:00:02+00:00")),
            encoding="utf-8",
        )
        reverse = await processor.ingest_files([path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_after_reverse = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
            reverse_raw_id = (raw_ids_after_reverse - raw_ids_before_reverse).pop()
            reverse_decision = source_conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE raw_id = ? AND logical_source_key = 'chatgpt:browser-replacement'
                """,
                (reverse_raw_id,),
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-replacement'"
                ).fetchone()[0]
                == accepted_raw_id
            )
        assert reverse.full_file_count == 1
        # Equivalent parser snapshots may elect either retained raw as the
        # canonical prefix representative; both are terminal receipts and
        # neither may displace the newer accepted head.
        assert reverse_decision in {"superseded_equivalent", "superseded_prefix"}

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


@pytest.mark.asyncio
async def test_browser_capture_provider_timestamp_advances_reordered_native_snapshot(tmp_path: Path) -> None:
    """Provider-native snapshots may insert work before an existing context node."""
    from polylogue.api import Polylogue

    root = tmp_path / "browser-capture"
    root.mkdir()
    path = root / "capture.json"

    def capture(turns: list[dict[str, object]], *, updated_at: str) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:provider-ordered-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/provider-ordered-replacement",
                "captured_at": updated_at,
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "provider-ordered-replacement",
                "title": "Provider ordered replacement",
                "updated_at": updated_at,
                "turns": turns,
                "provider_meta": {"capture_fidelity": "native_compact"},
            },
            "raw_provider_payload": {
                "polylogue_bridge_projection": "chatgpt-native-compact-v1",
                "mapping": {},
            },
        }

    prompt = {"provider_turn_id": "prompt", "role": "user", "text": "do work", "ordinal": 0}
    context = {
        "provider_turn_id": "attachment-context",
        "role": "user",
        "text": "The user provided an attachment",
        "ordinal": 1,
    }
    tool = {"provider_turn_id": "tool", "role": "assistant", "text": "tool output", "ordinal": 1}
    path.write_text(json.dumps(capture([prompt, context], updated_at="2026-07-16T00:00:00Z")), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await processor.ingest_files([path], emit_event=False)
        path.write_text(
            json.dumps(capture([prompt, tool, context], updated_at="2026-07-16T00:01:00Z")),
            encoding="utf-8",
        )
        second = await processor.ingest_files([path], emit_event=False)

        with sqlite3.connect(archive.archive_root / "index.db") as conn:
            row = conn.execute(
                "SELECT message_count, title FROM sessions WHERE session_id = ?",
                ("chatgpt-export:provider-ordered-replacement",),
            ).fetchone()
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            decisions = conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE logical_source_key = 'chatgpt:provider-ordered-replacement'
                ORDER BY acquisition_generation
                """
            ).fetchall()

        assert first.succeeded_file_count == second.succeeded_file_count == 1
        assert row == (3, "Provider ordered replacement")
        assert {decision for (decision,) in decisions} == {"superseded_prefix", "applied"}
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
    path = root / "rollout-2026-05-16T13-50-17-5370dcbb-87a9-446b-954f-be2a1df29915.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"5370dcbb-87a9-446b-954f-be2a1df29915"}}\n'
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
                "5370dcbb-87a9-446b-954f-be2a1df29915",
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
    # polylogue-u19l: the stored/hashed payload is now the literal live-file
    # bytes -- no synthetic session_meta header spliced in. The identity is
    # carried instead as a sidecar hint (persisted to raw_sessions.native_id
    # and used to override the parser's fallback_id on replay).
    assert plan.payload == appended
    assert plan.native_id_hint == "5370dcbb-87a9-446b-954f-be2a1df29915"


def test_codex_append_plan_reads_archive_file_set_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-5370dcbb-87a9-446b-954f-be2a1df29915.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"5370dcbb-87a9-446b-954f-be2a1df29915"}}\n'
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
                "5370dcbb-87a9-446b-954f-be2a1df29915",
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
    # polylogue-u19l: literal live-file bytes, identity carried as a hint.
    assert plan.payload == appended
    assert plan.native_id_hint == "5370dcbb-87a9-446b-954f-be2a1df29915"
    assert processor._latest_raw_fingerprint(path) == raw_id
    with cursor._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None


@pytest.mark.parametrize(
    ("index_origin", "source_origin"),
    [
        ("codex-session", "claude-code-session"),
        ("claude-code-session", "codex-session"),
    ],
)
def test_codex_append_identity_rejects_mixed_origins_at_same_path(
    tmp_path: Path,
    index_origin: str,
    source_origin: str,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "shared.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=source_origin,
            source_path=str(path),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", index_origin, raw_id, "mixed origin", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None


def test_codex_append_identity_rejects_mismatched_index_owner_before_global_fallback(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "shared.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        wrong_owner_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
        unrelated_codex_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(root / "other.jsonl"),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_001,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "claude-code-session", wrong_owner_raw_id, "wrong owner", bytes(32)),
        )
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "codex-session", unrelated_codex_raw_id, "unrelated fallback", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None


def test_codex_append_identity_rejects_global_fallback_when_ownership_query_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "candidate.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        unrelated_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(root / "unrelated.jsonl"),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "codex-session", unrelated_raw_id, "unrelated fallback", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._existing_provider_session_id(path, expected_origin="codex-session") == "codex-id"

    def unavailable_ownership_view(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("source tier unavailable")

    # The global index fallback must be viable so this assertion proves that an
    # unavailable ownership view, rather than another sqlite failure, rejects
    # the append.
    monkeypatch.setattr(processor, "_archive_has_native_session", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sqlite3, "connect", unavailable_ownership_view)

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None
    assert "source-path ownership view unavailable" in caplog.text


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
    initialize_active_archive_root(tmp_path)
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
    # The live compactor intentionally considers only raws older than the
    # process-start frontier. Move that frontier beyond this synthetic chain
    # so the test actually exercises retention authority rather than passing
    # because every raw is too new to compact.
    processor._raw_compaction_min_acquired_at = "9999-01-01T00:00:00+00:00"
    for chunk in append_chunks:
        with path.open("ab") as handle:
            handle.write(chunk)
        results.append(asyncio.run(processor.ingest_files([path])))

    assert results[0].full_file_count == 1
    assert results[0].succeeded_file_count == 1
    assert all(result.append_file_count == 1 for result in results[1:])
    # polylogue-u19l: append payloads are now published as literal live-file
    # bytes -- no synthetic session_meta header spliced in ahead of them.
    assert published_payloads == [payload, *append_chunks]
    if not protect_chain:
        # Accepted index receipts make later appends independent of whether
        # the compactor currently retains their predecessor payloads.
        assert all(result.succeeded_file_count == 1 for result in results)
        assert all(result.failed_file_count == 0 for result in results)
        cursor_record = cursor.get_record(path)
        assert cursor_record is not None
        assert cursor_record.byte_offset == len(payload) + sum(len(chunk) for chunk in append_chunks)
        assert cursor_record.failure_count == 0
        with sqlite3.connect(index_db) as conn:
            assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
                "message-0",
                "message-1",
                "message-2",
                "message-3",
            }
            assert conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_revision_applications").fetchone() == (4,)
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


def test_append_ingest_proves_byte_authority_at_capture_without_reconciler(tmp_path: Path) -> None:
    """polylogue-ds4b4 item 1: the common append case must prove itself at
    capture time, never deferring to the batch ``RawAuthorityReconciler``.

    ``append_ingest.py``'s ``_ingest_append_plans_archive`` already resolves
    the byte-contiguous predecessor via ``raw_append_revision_parent`` and,
    once found, immediately classifies+applies the revision in the SAME
    ingest call (``archive.classify_raw_revision_cohort`` /
    ``apply_raw_revision_replay``) -- there is no code path where a normal,
    single-predecessor append is left ``quarantined`` for a later async pass
    to pick up. This test locks that invariant in: after one full capture
    followed by one ordinary append, (a) the append's own raw row is
    ``revision_authority='byte_proven'`` immediately, (b) the append's
    content is already visible in ``index.db`` (``sessions``/``messages``)
    before any convergence/reconciler pass has run, and (c) the heavier,
    batch-oriented ``raw_authority_censuses`` bookkeeping table (owned by
    ``RawAuthorityReconciler``, not this synchronous per-key classifier) has
    zero rows -- proving the reconciler was never invoked for this raw.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "capture-proof.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"capture-proof","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_active_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    baseline_result = asyncio.run(processor.ingest_files([path]))
    assert baseline_result.succeeded_file_count == 1

    append_chunk = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    with path.open("ab") as handle:
        handle.write(append_chunk)
    append_result = asyncio.run(processor.ingest_files([path]))

    assert append_result.append_file_count == 1
    assert append_result.succeeded_file_count == 1
    assert append_result.failed_file_count == 0

    with sqlite3.connect(source_db) as conn:
        append_row = conn.execute(
            "SELECT revision_kind, revision_authority, parsed_at_ms, parse_error "
            "FROM raw_sessions WHERE revision_kind = 'append'"
        ).fetchone()
        assert append_row is not None
        assert append_row[0] == "append"
        # Proven synchronously, at capture time -- not left 'quarantined'
        # for a later reconciler pass to resolve.
        assert append_row[1] == "byte_proven"
        assert append_row[2] is not None
        assert append_row[3] is None
        # The heavy batch census/plan/blocker ledger belongs to the
        # separate, async RawAuthorityReconciler (daemon convergence /
        # offline backfill). A normal single-predecessor append must never
        # touch it.
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone()[0] == 0

    with sqlite3.connect(index_db) as conn:
        assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
            "message-0",
            "message-1",
        }


def test_full_ingest_cursor_hands_off_captured_prefix_after_growth_during_proof(
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
    grew_during_prefix_proof = False
    original_hash = sha256_range_from_path

    def grow_during_prefix_proof(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        nonlocal grew_during_prefix_proof
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        if not grew_during_prefix_proof:
            with path.open("ab") as handle:
                handle.write(appended_during_parse)
            grew_during_prefix_proof = True
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", grow_during_prefix_proof)

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

    # polylogue-u19l: append payloads are stored as literal live-file bytes --
    # no synthetic session_meta header spliced in ahead of them.
    assert BlobStore(tmp_path / "blob").read_all(append_blob_hash.lower()) == appended_during_parse
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


def test_busy_full_prefix_proof_defers_to_archived_cursor_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "busy-hot.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"busy-hot","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"captured","role":"user",'
        b'"content":[{"type":"input_text","text":"captured"}]}}\n'
    )
    path.write_bytes(captured)
    captured_stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    cursor.set(
        path,
        len(captured),
        byte_offset=len(captured),
        last_complete_newline=len(captured),
        parser_fingerprint="previous-parser",
        content_fingerprint=sha256(captured).hexdigest(),
        tail_hash=_cursor_hash_authority(captured),
        source_name="codex",
        st_dev=captured_stat.st_dev,
        st_ino=captured_stat.st_ino,
        mtime_ns=captured_stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="live-batched-v2",
    )
    original_hash = sha256_range_from_path
    next_record = 0

    def grow_on_every_prefix_proof(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        nonlocal next_record
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        with path.open("ab") as handle:
            handle.write(
                b'{"type":"response_item","payload":{"type":"message","id":"later-'
                + str(next_record).encode()
                + b'","role":"assistant","content":[{"type":"output_text","text":"later"}]}}\n'
            )
        next_record += 1
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", grow_on_every_prefix_proof)
    first = asyncio.run(processor.ingest_files([path]))

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 1
    deferred = cursor.get_record(path)
    assert deferred is not None
    assert deferred.byte_offset == 0
    assert deferred.content_fingerprint is None
    assert deferred.failure_count == 0
    assert deferred.next_retry_at is not None
    assert not deferred.excluded

    for _ in range(4):
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
    repeatedly_deferred = cursor.get_record(path)
    assert repeatedly_deferred is not None
    assert repeatedly_deferred.failure_count == 0
    assert repeatedly_deferred.next_retry_at is not None
    assert not repeatedly_deferred.excluded

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", original_hash)
    watcher = LiveWatcher(polylogue, (WatchSource(name="codex", root=root),), cursor=cursor)
    scheduled_retries: list[object] = []
    monkeypatch.setattr(watcher, "_schedule_failed_retry_wakeup", scheduled_retries.append)
    watcher._schedule_failed_retry_scan()
    assert len(scheduled_retries) == 1

    original_reconcile = watcher._reconcile_archived_cursor_outcome
    monkeypatch.setattr(
        watcher,
        "_reconcile_archived_cursor_outcome",
        lambda _path, *, stat: live_watcher._ArchivedCursorReconciliation.UNAVAILABLE,
    )
    monkeypatch.setattr(live_watcher, "_retry_due", lambda _retry_at: True)
    for _ in range(5):
        assert not watcher._needs_work(path)
    unavailable = cursor.get_record(path)
    assert unavailable is not None
    assert unavailable.failure_count == 0
    assert not unavailable.excluded

    monkeypatch.setattr(watcher, "_reconcile_archived_cursor_outcome", original_reconcile)
    assert watcher._needs_work(path)
    reconciled = cursor.get_record(path)
    assert reconciled is not None
    assert reconciled.byte_offset == len(captured)
    assert reconciled.content_fingerprint == sha256(captured).hexdigest()

    second = asyncio.run(processor.ingest_files([path]))

    assert second.full_file_count == 0
    assert second.append_file_count == 1
    assert second.succeeded_file_count == 1

    processor._defer_full_cursor_retry(path, source_name="codex", stat=path.stat())
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so the third ingest below still succeeds.
    replacement = (
        b'{"type":"session_meta","payload":{"id":"busy-replacement"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    path.write_bytes(replacement)

    assert watcher._needs_work(path)
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert invalidated.next_retry_at is None

    third = asyncio.run(processor.ingest_files([path]))
    assert third.full_file_count == 1
    assert third.append_file_count == 0
    assert third.succeeded_file_count == 1


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
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real,
    # equal-length message record to each payload so this fixture keeps
    # testing the mtime-restore-reconciliation race, not the now-refused
    # empty shape (the equal-length invariant below is load-bearing for the
    # race itself, so both messages must stay identical length too).
    payload_a = (
        b'{"type":"session_meta","payload":{"id":"archive-reconcile-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    payload_b = (
        b'{"type":"session_meta","payload":{"id":"archive-reconcile-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
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
def test_append_cursor_redetects_source_rewrite_after_handoff(
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
    stale_cursor = cursor.get_record(path)
    assert stale_cursor is not None
    watcher = LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )
    assert watcher._needs_work(path)
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
        # The append range itself is still byte-proven, so keep it as a
        # frontier. The old observation embedded in its tail authority makes
        # the watcher force this same-size prefix rewrite through the full
        # route before it can be skipped or appended past.
        assert appended.stale_cursor_write_count == 0
        assert stale_cursor.byte_offset == len(baseline_a + append_a)
        assert stale_cursor.content_fingerprint is not None
        assert b"zerob" in path.read_bytes()
        assert path.read_bytes()[-64 * 1024 :] == accepted_tail_before_rewrite
        retried = asyncio.run(processor.ingest_files([path]))
        assert retried.full_file_count == 1
        assert retried.append_file_count == 0
        assert retried.succeeded_file_count == 0
        assert retried.failed_file_count == 1
        return

    assert appended.stale_cursor_write_count == 1
    assert stale_cursor.byte_offset == 0
    assert stale_cursor.content_fingerprint is None

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


def test_append_cursor_rejects_truncation_after_append_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial append handoff must fail closed when its source truncates."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-truncated.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"append-truncated"}}\n'
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
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert asyncio.run(processor.ingest_files([path])).succeeded_file_count == 1
    with path.open("ab") as handle:
        handle.write(append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)

    original_tail_hash = tail_hash_from_path

    def truncate_after_tail(source_path: Path, byte_size: int) -> tuple[str, int]:
        result = original_tail_hash(source_path, byte_size)
        source_path.write_bytes(source_path.read_bytes()[: plan.last_complete_newline - 1])
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.tail_hash_from_path", truncate_after_tail)

    assert processor._record_append_cursor(plan) is False
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)


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


def test_incomplete_full_jsonl_capture_retries_without_losing_split_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    # Retaining and classifying the captured raw bytes is a successful source
    # write, even though the incomplete session is terminally unmaterialized.
    assert first.succeeded_file_count == 1
    assert first.failed_file_count == 0
    captured_cursor = cursor.get_record(path)
    assert captured_cursor is not None
    # The cursor records the acquired source snapshot, but raw-failure
    # evidence below still prohibits an append from that frontier.
    assert captured_cursor.byte_offset == path.stat().st_size
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = conn.execute(
            """
            SELECT r.raw_id
            FROM raw_sessions AS r
            JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
            WHERE a.artifact_kind = 'terminal_corrupt_input'
            """
        ).fetchone()[0]
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()[0]
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
        assert "complete record boundary" in str(parse_error)
    assert artifact == ("terminal_corrupt_input", "decode_failed", 0)
    assert captured_cursor.content_fingerprint == raw_id
    assert processor._cursor_references_raw_failure_requiring_full_replay(path, captured_cursor)

    with path.open("ab") as handle:
        handle.write(split_record[split_at:])
    # The live full-ingest path records a deferred FTS convergence-debt row
    # (``record_convergence_debt(stage="fts", ...)`` in batch.py) as soon as
    # the raw-revision replay lands, then the SAME ``ingest_files`` call
    # synchronously converges it (``_converge_paths``) and clears it again
    # before returning -- so inspecting ``cursor.list_convergence_debt()``
    # after the call proves nothing about whether the deferral was ever
    # recorded. Spy directly on the persistence call itself to prove the
    # debt row existed (recorded, not merely a code path CodeRabbit assumed
    # ran) before this same batch's convergence consumed it.
    recorded_debt: list[dict[str, str | None]] = []
    original_record_convergence_debt = CursorStore.record_convergence_debt

    def spy_record_convergence_debt(self: CursorStore, **kwargs: Any) -> None:
        recorded_debt.append(dict(kwargs))
        original_record_convergence_debt(self, **kwargs)

    monkeypatch.setattr(CursorStore, "record_convergence_debt", spy_record_convergence_debt)

    second = asyncio.run(processor.ingest_files([path]))

    assert second.full_file_count == 1
    assert second.append_file_count == 0
    assert second.succeeded_file_count == 1
    assert second.failed_file_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.failure_count == 0
    assert any(
        call.get("stage") == "fts" and call.get("subject_id") == "codex-session:split-record" for call in recorded_debt
    )
    # And, exactly because this batch's own convergence resolved it
    # synchronously, no stale FTS debt is left behind for the daemon to
    # retry -- proving the deferral was a real, consumed debt cycle rather
    # than one that silently never got recorded (or one that leaks forever).
    assert cursor.list_convergence_debt(limit=10) == []
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]
        from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

        repair_message_fts_index_sync(conn, ["codex-session:split-record"], record_exact_snapshot=False)
        assert conn.execute(
            "SELECT b.search_text FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid ORDER BY b.message_id"
        ).fetchall() == [("zero",), ("one",)]


def test_deferred_full_jsonl_with_prior_session_replays_completed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferred full capture cannot resume through an append-only tail."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "prior-session.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"prior-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    completed_record = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    split_at = len(completed_record) // 2
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="current-parser",
    )

    seeded = asyncio.run(processor.ingest_files([path]))
    assert seeded.succeeded_file_count == 1
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [("message-0",)]

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="previous-parser",
    )
    path.write_bytes(baseline + completed_record[:split_at])
    original_boundary_check = live_batch._captured_jsonl_ends_at_record_boundary
    completed = False

    def complete_source_after_capture(**kwargs: object) -> bool:
        nonlocal completed
        if not completed:
            path.write_bytes(baseline + completed_record)
            completed = True
        return original_boundary_check(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_batch, "_captured_jsonl_ends_at_record_boundary", complete_source_after_capture)

    deferred = asyncio.run(processor.ingest_files([path]))

    assert deferred.full_file_count == 1
    assert deferred.succeeded_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute(
            """
            SELECT a.artifact_kind
            FROM raw_artifacts AS a
            WHERE a.artifact_kind = 'deferred_hot_jsonl_capture'
            """
        ).fetchone()
    assert artifact is not None
    assert artifact[0] == "deferred_hot_jsonl_capture"
    replayed = asyncio.run(processor.ingest_files([path]))

    assert replayed.full_file_count == 1
    assert replayed.append_file_count == 0
    assert replayed.succeeded_file_count == 1
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.failure_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]


def test_hot_capture_prefix_proof_rejects_in_place_rewrite_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hash read cannot prove a prefix that changed before its post-check."""
    from polylogue.sources.live import batch as live_batch

    path = tmp_path / "racing.jsonl"
    captured = b'{"type":"session_meta","payload":{"id":"racing"}}\n'
    path.write_bytes(captured + b'{"type":"message"}\n')
    original_hash = sha256_range_from_path

    def rewrite_after_hash(*args: object, **kwargs: object) -> tuple[str, int]:
        result = original_hash(*args, **kwargs)  # type: ignore[arg-type]
        path.write_bytes(b"x" * path.stat().st_size)
        return result

    monkeypatch.setattr(live_batch, "sha256_range_from_path", rewrite_after_hash)

    assert (
        live_batch._hot_capture_prefix_is_proven(
            str(path),
            captured,
            blob_hash=sha256(captured).hexdigest(),
            blob_size=len(captured),
        )
        is False
    )


def test_raw_failure_cursor_guard_uses_root_source_tier_for_pointer_index(tmp_path: Path) -> None:
    """The active index generation never owns durable raw-failure evidence."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    generation = tmp_path / "generation"
    generation.mkdir()
    index_db = generation / "index.db"
    sqlite3.connect(index_db).close()
    (archive_root / ".index-active-pointer").write_text(str(index_db), encoding="utf-8")
    source_db = archive_root / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    path = archive_root / "sessions" / "terminal.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{"type":"session_meta"')
    payload_hash = "ab" * 32
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parse_error, detection_warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "terminal-raw",
                "codex-session",
                "terminal",
                str(path),
                bytes.fromhex(payload_hash),
                path.stat().st_size,
                1_770_000_000_000,
                "captured JSONL payload ends before a complete record boundary",
                "[]",
            ),
        )
        upsert_raw_artifact(
            conn,
            "terminal-raw",
            ArchiveSourceArtifact(
                artifact_id="terminal-evidence",
                origin="codex-session",
                source_path=str(path),
                source_index=0,
                artifact_kind="terminal_corrupt_input",
                classification_reason="terminal_corrupt_input",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
            ),
        )
    cursor = CursorStore(index_db)
    stat = path.stat()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="terminal-raw",
        tail_hash=_cursor_hash_authority(path.read_bytes()),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=path.parent),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    record = cursor.get_record(path)

    assert record is not None
    assert processor._cursor_references_raw_failure_requiring_full_replay(path, record)


def test_raw_failure_cursor_guard_rejects_contradictory_or_mismatched_evidence(tmp_path: Path) -> None:
    """Append fallback requires the same source coordinate and valid support status."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "guard.jsonl"
    path.write_bytes(b'{"type":"session_meta","payload":{"id":"guard"}}\n')
    index_db = tmp_path / "index.db"
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        mismatched_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=path.read_bytes(),
            source_path=str(path),
            source_index=1,
            acquired_at_ms=1,
        )
        contradictory_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=path.read_bytes() + b"2",
            source_path=str(path),
            source_index=0,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = ?",
            [("mismatched coordinate", mismatched_raw_id), ("contradictory support", contradictory_raw_id)],
        )
        upsert_raw_artifact(
            source_conn,
            mismatched_raw_id,
            ArchiveSourceArtifact(
                artifact_id="mismatched-coordinate-evidence",
                origin="chatgpt-export",
                source_path=str(path),
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
            ),
        )
        upsert_raw_artifact(
            source_conn,
            contradictory_raw_id,
            ArchiveSourceArtifact(
                artifact_id="contradictory-support-evidence",
                origin="codex-session",
                source_path=str(path),
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
            ),
        )
        source_conn.commit()
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._raw_failure_requires_full_replay(path, mismatched_raw_id) is False
    assert processor._raw_failure_requires_full_replay(path, contradictory_raw_id) is False


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

    # The acquired bytes were durably retained with a terminal classification;
    # source disappearance cannot turn that completed archive write into a
    # retryable transport failure.
    assert result.succeeded_file_count == 1
    assert result.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()[0]
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
        assert "complete record boundary" in str(parse_error)
    assert artifact == ("terminal_corrupt_input", "decode_failed", 0)


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

    # polylogue-1r9c: record_revision_application_sync is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through archive_tier_module -- patch it there.
    original_record = archive_revision_governance.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected append persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "record_revision_application_sync", injected_failure)
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
    # polylogue-1r9c: record_revision_application_sync is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through archive_tier_module -- patch it there.
    original_record = archive_revision_governance.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected parser-upgrade persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "record_revision_application_sync", injected_failure)

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
        "polylogue.sources.dispatch.parse_stream_payload",
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


def test_append_declared_artifact_is_admitted_with_typed_authority(tmp_path: Path) -> None:
    path = tmp_path / "subagents" / "workflows" / "wf-append" / "journal.jsonl"
    path.parent.mkdir(parents=True)
    payload = b'{"contentKey":"call-1","agentId":"agent-a"}\n'
    path.write_bytes(payload)
    plan = replace(_append_plan(path, payload, payload_hash="artifact"), source_name="claude-code")

    result = ingest_append_plans(cast(Any, _append_owner(tmp_path)), [plan])

    assert result.succeeded == [plan]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw = conn.execute(
            "SELECT raw_id, logical_source_key, revision_kind, revision_authority FROM raw_sessions"
        ).fetchone()
        artifact = conn.execute(
            "SELECT artifact_kind, classification_reason, parse_as_session, raw_id FROM raw_artifacts"
        ).fetchone()
    assert raw is not None
    assert raw[1:] == (None, "unknown", "quarantined")
    assert artifact is not None
    assert artifact[0] == "workflow_journal"
    assert "OriginSpec" in artifact[1]
    assert artifact[2:] == (0, raw[0])


def test_full_batch_declared_artifact_is_admitted_before_pending_raw_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"contentKey":"call-2","agentId":"agent-b"}\n'
    source.write_bytes(payload)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, _fallback_provider: (Provider.CLAUDE_CODE, True),
    )

    metrics = asyncio.run(processor.ingest_files([source], emit_event=False))

    assert metrics.succeeded_file_count == 1
    assert metrics.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        raw = conn.execute(
            "SELECT raw_id, logical_source_key, revision_kind, revision_authority FROM raw_sessions"
        ).fetchone()
        artifact = conn.execute("SELECT artifact_kind, parse_as_session, raw_id FROM raw_artifacts").fetchone()
    assert raw is not None
    assert raw[1:] == (None, "unknown", "quarantined")
    assert artifact == ("workflow_journal", 0, raw[0])


def test_full_batch_session_shaped_workflow_journal_reaches_parser_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        b"".join(
            b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n'
            for index in range(64)
        )
        + b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"recover this journal record"},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"journal-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    first = asyncio.run(processor.ingest_files([source], emit_event=False))
    second = asyncio.run(processor.ingest_files([source], emit_event=False))

    assert first.ingested_session_count == 1
    assert first.failed_file_count == 0
    assert second.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_large_full_batch_session_shaped_workflow_journal_reaches_parser_idempotently(tmp_path: Path) -> None:
    from polylogue.sources.live.batch_support import _STREAMING_FULL_INGEST_BYTES

    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        b'{"contentKey":"artifact-0","agentId":"workflow-agent","summary":"'
        + b"x" * _STREAMING_FULL_INGEST_BYTES
        + b'"}\n'
        + b"".join(
            b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n'
            for index in range(1, 32)
        )
        + b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"recover this journal record"},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        + b'{"parentUuid":"journal-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    assert source.stat().st_size > _STREAMING_FULL_INGEST_BYTES
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    first = asyncio.run(processor.ingest_files([source], emit_event=False))
    second = asyncio.run(processor.ingest_files([source], emit_event=False))

    assert first.ingested_session_count == 1
    assert first.failed_file_count == 0
    assert second.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_full_batch_malformed_workflow_journal_remains_typed_evidence(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"contentKey":"broken"\n')
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    metrics = asyncio.run(processor.ingest_files([source], emit_event=False))

    assert metrics.succeeded_file_count == 1
    assert metrics.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone() == (
            "workflow_journal",
            0,
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_append_admission_bind_failure_persists_exact_pending_envelope_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="append-admission-retry")
    original_bind = ArchiveStore.bind_raw_revision
    fail_once = True

    def fail_bind(self: ArchiveStore, raw_id: str, revision: RawRevisionEnvelope, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected append bind failure")
        original_bind(self, raw_id, revision, **kwargs)

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", fail_bind)
    first = ingest_append_plans(cast(Any, owner), [plan])

    assert first.succeeded == []
    assert first.failed == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT raw_id, blob_hash, blob_size, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions WHERE source_index = -1
            """
        ).fetchone()
    assert row is not None
    raw_id = str(row[0])
    assert row[1] is not None
    assert BlobStore(tmp_path / "blob").read_all(bytes(row[1]).hex()) == plan.payload
    assert row[2] == len(plan.payload)
    assert row[3:13] == (
        f"pending-raw:codex-session:-1:{plan.path}:{raw_id}",
        "full",
        sha256(plan.payload).hexdigest(),
        None,
        None,
        None,
        None,
        None,
        0,
        "quarantined",
    )
    assert isinstance(row[13], str) and "injected append bind failure" in row[13]

    retry = ingest_append_plans(cast(Any, owner), [plan])

    assert retry.succeeded == [plan]
    assert retry.failed == []
    bound = _raw_revision_envelope_row(tmp_path, raw_id)
    assert bound[0] == "codex:append-admission-retry"
    assert bound[1] == "append"
    assert bound[2] is not None
    assert bound[3] is not None
    assert bound[4] is not None
    assert bound[5] is not None
    assert bound[6] == plan.stat_size - len(plan.payload)
    assert bound[7] == plan.stat_size
    assert isinstance(bound[8], int) and bound[8] >= 0
    assert bound[9] == "byte_proven"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_index = -1").fetchone() == (1,)


def test_public_full_blob_batch_bind_failure_persists_bytes_and_blocks_unsafe_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "blob-retry.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"blob-retry"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n' + (b" " * (9 * 1024 * 1024))
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    original_bind = ArchiveStore.bind_raw_revision
    fail_once = True

    def fail_bind(self: ArchiveStore, raw_id: str, revision: RawRevisionEnvelope, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected blob bind failure")
        original_bind(self, raw_id, revision, **kwargs)

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", fail_bind)
    first = asyncio.run(processor.ingest_files([source], emit_event=False))

    assert first.full_file_count == 1
    assert first.failed_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT raw_id, blob_hash, blob_size, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions
            """
        ).fetchone()
    assert row is not None
    raw_id = str(row[0])
    assert BlobStore(tmp_path / "blob").read_all(bytes(row[1]).hex()) == payload
    assert row[2] == len(payload)
    assert row[3:13] == (
        f"pending-raw:codex-session:0:{source}:{raw_id}",
        "full",
        sha256(payload).hexdigest(),
        None,
        None,
        None,
        None,
        None,
        0,
        "quarantined",
    )
    assert isinstance(row[13], str) and "injected blob bind failure" in row[13]

    with pytest.raises(CursorAuthorityBlockedError, match="source-selection gate blocked"):
        asyncio.run(processor.ingest_files([source], emit_event=False))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        retained = conn.execute(
            """
            SELECT logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions
            """
        ).fetchone()
    assert retained == (
        f"pending-raw:codex-session:0:{source}:{raw_id}",
        "full",
        sha256(payload).hexdigest(),
        None,
        None,
        None,
        None,
        None,
        0,
        "quarantined",
        row[13],
    )


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
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"full-locked"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0",'
        b'"role":"user","content":[{"type":"input_text","text":"zero"}]}}\n'
    )
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
    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="index-fail")

    def fail_index(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.IntegrityError("injected index commit failure")

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", fail_index)
    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected index commit failure" in parse_error
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_append_multi_session_payload_is_rejected_before_index_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="append-multi")
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # "more than one session" check ever runs -- give each session one real
    # message so this fixture keeps testing the multi-session rejection.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="multi-1",
            messages=[ParsedMessage(provider_message_id="multi-1-0", role=Role.USER, text="hello")],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="multi-2",
            messages=[ParsedMessage(provider_message_id="multi-2-0", role=Role.USER, text="hello")],
        ),
    ]
    monkeypatch.setattr("polylogue.sources.dispatch.parse_stream_payload", lambda *_args, **_kwargs: sessions)
    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "did not prove one session and cursor identity" in parse_error
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("append-multi",)]


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
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # injected-second-write-failure path ever runs -- give each session one
    # real message so this fixture keeps testing that failure-handling path.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="full-multi-1",
            messages=[ParsedMessage(provider_message_id="full-multi-1-0", role=Role.USER, text="hello")],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="full-multi-2",
            messages=[ParsedMessage(provider_message_id="full-multi-2-0", role=Role.USER, text="hello")],
        ),
    ]
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr("polylogue.sources.live.batch.parse_stream_payload", lambda *_args, **_kwargs: sessions)
    # polylogue-1r9c: _write_parsed_precedence_result is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through ArchiveStore's `self.` dispatch -- patch it there.
    original_write = archive_revision_governance._write_parsed_precedence_result
    write_count = 0

    def fail_second_index(
        archive: archive_revision_governance.RawRevisionGovernanceHost,
        session: ParsedSession,
        **kwargs: object,
    ) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise sqlite3.IntegrityError("injected full second-session index failure")
        return original_write(archive, session, **cast(Any, kwargs))

    monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", fail_second_index)
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


def test_full_ingest_skips_durably_excised_content_without_aborting_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon's full/streaming write orchestration (polylogue-27m fix round).

    Reproduces the reviewer's finding at the orchestration layer, not just the
    low-level write gate: a durably excised blob hash must be skipped (counted
    in ``_ArchiveFullWriteResult.excised_skips``) without aborting the rest of
    the batch. Reverting the ``except ContentExcisedError`` handling in
    ``LiveBatchProcessor._ingest_full_records_archive`` back to letting it fall
    through to the generic ``except Exception`` branch (or removing the
    pre-write ``is_blob_hash_excised`` gate entirely) makes this fail: either
    the whole batch call raises, or the excised content gets a fresh raw_id.

    The streaming threshold is patched down (matching the pattern used
    elsewhere in this file, e.g. ``test_full_ingest_reports_heartbeat_stage_events``)
    so both fixture files route through ``blob_store.write_from_path`` /
    ``archive.write_raw_blob_ref`` -> ``write_source_raw_session_blob_ref``,
    the same code path a real >8MB capture takes -- not the small-payload
    ``write_raw_payload`` -> ``write_source_raw_session`` gate, which is a
    different call site (polylogue-re4a).
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        deterministic_blob_hash,
        record_excised_blob_hash,
    )

    root = tmp_path / "sessions"
    root.mkdir()
    excised_payload = b'{"secret": "sk-ant-should-not-resurrect-via-streaming-batch"}\n'
    excised_source = root / "excised.jsonl"
    excised_source.write_bytes(excised_payload)
    normal_source = root / "normal.jsonl"
    # Real, parseable content -- not a bare `{}` -- because polylogue-lb39z's
    # guarded presence-guarantee fallback (#3630) can drive this fixture's
    # raw revision through a real re-parse (_parse_raw_revision_chain) that
    # the parse_stream_payload monkeypatch below does not intercept, and a
    # genuinely empty/malformed record now correctly fails to replay to any
    # session rather than being silently accepted.
    normal_source.write_bytes(b'{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n')

    # Pre-mark the excised file's exact content hash as durably excised,
    # mirroring a prior real `polylogue ops excise` apply.
    initialize_active_archive_root(tmp_path)
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        record_excised_blob_hash(
            source_conn,
            blob_hash=deterministic_blob_hash(excised_payload),
            reason="test: reproduces reviewer finding",
            actor="user:local",
            excised_at_ms=1_000,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # durably-excised-content skip path is exercised -- give it one real
    # message so "normal.jsonl" still succeeds.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="normal-1",
            messages=[ParsedMessage(provider_message_id="normal-1-0", role=Role.USER, text="hello")],
        )
    ]
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr("polylogue.sources.live.batch.parse_stream_payload", lambda *_args, **_kwargs: sessions)

    # Force both fixture files through the streaming blob-ref write path
    # (>= this threshold uses blob_store.write_from_path + write_raw_blob_ref,
    # never populating raw_payloads) rather than the small-payload
    # write_raw_payload path -- see polylogue-re4a.
    monkeypatch.setattr("polylogue.sources.live.batch._STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr("polylogue.sources.live.batch_support._STREAMING_FULL_INGEST_BYTES", 1)

    archive_results: list[_ArchiveFullWriteResult] = []
    original_full_write = processor._ingest_full_records_archive

    def capture_full_write(*args: Any, **kwargs: Any) -> _ArchiveFullWriteResult:
        outcome = original_full_write(*args, **kwargs)
        archive_results.append(outcome)
        return outcome

    monkeypatch.setattr(processor, "_ingest_full_records_archive", capture_full_write)

    result = processor._ingest_full_paths_sync([excised_source, normal_source], source_name="codex")

    assert archive_results[0].excised_skips == 1
    # The non-excised file in the same batch still succeeds -- one excised
    # record must not abort the rest of the batch.
    assert normal_source in result.succeeded
    assert excised_source not in result.succeeded

    with sqlite3.connect(tmp_path / "source.db") as conn:
        # No raw_sessions row was resurrected for the excised payload.
        rows = conn.execute("SELECT source_path FROM raw_sessions").fetchall()
        assert all("excised.jsonl" not in str(row[0]) for row in rows)


def test_live_multi_session_divergence_reopens_raw_authority(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    root.mkdir()
    first = root / "first.json"
    second = root / "second.json"

    def conversation(native_id: str, *texts: str) -> dict[str, object]:
        mapping: dict[str, object] = {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": [f"{native_id}-node-0"],
            }
        }
        for index, text in enumerate(texts):
            node_id = f"{native_id}-node-{index}"
            next_node = f"{native_id}-node-{index + 1}" if index + 1 < len(texts) else None
            mapping[node_id] = {
                "id": node_id,
                "parent": "root" if index == 0 else f"{native_id}-node-{index - 1}",
                "children": [] if next_node is None else [next_node],
                "message": {
                    "id": f"{native_id}-message-{index}",
                    "author": {"role": "user"},
                    "create_time": 1_780_000_000.0 + index,
                    "content": {"content_type": "text", "parts": [text]},
                    "metadata": {},
                },
            }
        return {
            "id": native_id,
            "title": native_id,
            "create_time": 1_780_000_000.0,
            "current_node": f"{native_id}-node-{len(texts) - 1}",
            "mapping": mapping,
        }

    first.write_text(
        json.dumps([conversation("shared", "base", "left"), conversation("safe-1", "one")]),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([conversation("shared", "base", "right"), conversation("safe-2", "two")]),
        encoding="utf-8",
    )
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, suffixes=(".json",)),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    # This is a real ChatGPT export bundle, not a JSONL shape authorized by a
    # monkeypatch. Keep the taxonomy assertion next to the route assertion so
    # the test cannot pass after accidentally becoming a non-session fixture.
    assert _parse_path_as_session_artifact(first, provider=Provider.CHATGPT) is True
    first_result = processor._ingest_full_paths_sync([first], source_name="inbox")
    assert first_result.succeeded == [first]
    assert first_result.failed == []
    assert first_result.raw_source_names[first] == Provider.CHATGPT.value
    accepted_raw_id = first_result.raw_fingerprints[first]

    second_result = processor._ingest_full_paths_sync([second], source_name="inbox")
    # The divergent authority remains unresolved, but this source file was
    # acquired and parsed. Its unchanged bytes must not become a retry loop.
    assert second_result.failed == []
    assert second_result.succeeded == [second]
    # Direct check of the persisted state backing that claim (this layer --
    # ``_ingest_full_paths_sync`` -- has no CursorStore row of its own; the
    # durable "not a retry loop" evidence lives in raw_sessions/raw_session_
    # memberships). ``second``'s raw must show no parse_error (what would
    # make a later pass retry it as a failure) while its logical identity's
    # membership decision is durably ambiguous/quarantined -- i.e. the
    # deferred authority debt is actually persisted for this exact path, not
    # only implied by the in-memory FullIngestResult lists above.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT r.parse_error, m.decision, m.revision_authority
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'chatgpt:shared'
            """,
            (str(second),),
        ).fetchone() == (None, "ambiguous", "quarantined")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE decision = 'ambiguous'").fetchone() == (
            2,
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NULL").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL").fetchone() == (0,)
        assert conn.execute(
            """
            SELECT r.source_path, m.decision, m.revision_authority
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt:shared'
            ORDER BY r.source_path
            """
        ).fetchall() == [
            (str(first), "ambiguous", "quarantined"),
            (str(second), "ambiguous", "quarantined"),
        ]
    with sqlite3.connect(index_db) as conn:
        # The first accepted branch remains queryable; the later divergence is
        # nonterminal debt and has no deletion authority.
        assert set(conn.execute("SELECT native_id FROM sessions")) == {
            ("safe-1",),
            ("safe-2",),
            ("shared",),
        }
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:shared'"
        ).fetchone() == (accepted_raw_id,)
        assert conn.execute(
            """
            SELECT b.search_text
            FROM sessions AS s
            JOIN messages AS m USING (session_id)
            JOIN blocks AS b USING (message_id)
            WHERE s.native_id = 'shared'
            ORDER BY m.position, b.position
            """
        ).fetchall() == [("base",), ("left",)]

    retry_result = processor._ingest_full_paths_sync([first], source_name="inbox")

    assert retry_result.succeeded == [first]
    assert retry_result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT r.source_path, m.decision, m.revision_authority,
                   r.parsed_at_ms IS NOT NULL, r.parse_error
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt:shared'
            ORDER BY r.source_path
            """
        ).fetchall() == [
            (str(first), "applied", "byte_proven", 1, None),
            (str(second), "ambiguous", "quarantined", 0, None),
        ]
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:shared'"
        ).fetchone() == (accepted_raw_id,)


def test_live_third_raw_reunifies_with_backfill_retired_siblings(tmp_path: Path) -> None:
    """polylogue-hm2f: the live incremental path must reunite retired siblings, not drop new raws forever.

    Mirrors the exact live call sequence the polylogue-52l2 guard protects
    (``bind_raw_revision`` -> ``classify_raw_revision_cohort``), then proves
    the new routing this fix adds: when that cohort comes back empty AND
    ``raw_membership_retired_full_revision_siblings`` shows this identity has
    known siblings already retired to membership governance -- exactly the
    durable state offline backfill (``sources/revision_backfill.py``,
    ``convertible_full_revision_raw_ids`` + ``replace_raw_membership_census``)
    leaves behind for a decided-ambiguous full-only cohort -- a newly
    discovered THIRD raw for the same identity must be folded into that same
    membership governance and weighed by the real content-prefix classifier
    (``classify_membership_revisions``) alongside every known sibling,
    instead of being silently dropped with only a warning log line (the
    pre-fix behavior: ``bind_raw_revision`` succeeds, but no
    ``raw_session_memberships`` row is ever written for the raw and the file
    surfaces as failed with zero evidence trail).

    raw_a=["base","left"], raw_b=["base","right"] are byte-divergent (not a
    prefix of one another) -- a genuine, decided ambiguous cohort, retired
    here exactly the way ``backfill_historical_revision_evidence`` retires
    one once ``classify_raw_revision_cohort`` returns no accepted chain.
    raw_c=["base","left","extra"] then arrives through the live incremental
    path (``LiveBatchProcessor._ingest_full_paths_sync``, the production
    entry point, not a hand-simulated call). Content-wise raw_c does not
    strictly dominate raw_b (they diverge at message index 1) so the real
    classifier still cannot order the full three-way cohort as a clean
    containment chain -- but critically that decision is reached by
    weighing raw_c against BOTH retired siblings: since this logical source
    has never had an accepted head, the presence-guarantee fallback
    (polylogue-lb39z item 5, ``_maximal_evidence_fallback``) deterministically
    materializes raw_c (the largest-frontier representative) instead of
    leaving the reunified cohort headless, with raw_a/raw_b recorded as its
    conflict debt. All three raws end up in ``raw_session_memberships`` with
    a real, decided outcome, proving reunification happened rather than
    raw_c being evaluated alone or dropped.
    """

    def conversation(native_id: str, *texts: str) -> dict[str, object]:
        mapping: dict[str, object] = {
            "root": {"id": "root", "message": None, "parent": None, "children": [f"{native_id}-node-0"]}
        }
        for index, text in enumerate(texts):
            node_id = f"{native_id}-node-{index}"
            next_node = f"{native_id}-node-{index + 1}" if index + 1 < len(texts) else None
            mapping[node_id] = {
                "id": node_id,
                "parent": "root" if index == 0 else f"{native_id}-node-{index - 1}",
                "children": [] if next_node is None else [next_node],
                "message": {
                    "id": f"{native_id}-message-{index}",
                    "author": {"role": "user"},
                    "create_time": 1_780_000_000.0 + index,
                    "content": {"content_type": "text", "parts": [text]},
                    "metadata": {},
                },
            }
        return {
            "id": native_id,
            "title": native_id,
            "create_time": 1_780_000_000.0,
            "current_node": f"{native_id}-node-{len(texts) - 1}",
            "mapping": mapping,
        }

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        raw_a = store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation("shared", "base", "left")]).encode(),
            source_path="a.json",
            acquired_at_ms=1,
        )
        store.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "chatgpt:shared", RawRevisionKind.FULL, raw_a, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation("shared", "base", "right")]).encode(),
            source_path="b.json",
            acquired_at_ms=2,
        )
        store.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "chatgpt:shared", RawRevisionKind.FULL, raw_b, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

        # Exactly the polylogue-52l2 guard-tripping sequence: no unique
        # byte-prefix chain across a and b.
        plan = store.classify_raw_revision_cohort_for_live_watch("chatgpt:shared")
        assert plan.accepted_raw_ids == ()

        # Mirror backfill_historical_revision_evidence's own retirement step
        # once a full-only cohort is decided ambiguous: move every
        # convertible full raw to membership governance.
        for raw_id in store.convertible_full_revision_raw_ids("chatgpt:shared"):
            sessions = LiveBatchProcessor._parse_retained_raw_sessions(store, raw_id)
            store.replace_raw_membership_census(
                raw_id,
                sessions,
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                retire_full_revision_governance=True,
            )
        store.commit()
        retired_siblings = store.raw_membership_retired_full_revision_siblings("chatgpt:shared")
    assert set(retired_siblings) == {raw_a, raw_b}

    # A THIRD raw for the same logical identity, discovered afterward
    # through the actual live incremental entry point.
    root = tmp_path / "inbox"
    root.mkdir()
    third = root / "third.json"
    third.write_text(json.dumps([conversation("shared", "base", "left", "extra")]), encoding="utf-8")
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, suffixes=(".json",)),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    third_result = processor._ingest_full_paths_sync([third], source_name="inbox")

    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT r.source_path, m.decision
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt:shared'
            ORDER BY r.source_path
            """
        ).fetchall()

    # Reunification proof: raw_c (third.json) has a raw_session_memberships
    # row -- it was folded into the SAME membership cohort as raw_a/raw_b,
    # not evaluated alone and not silently dropped. Every member of the
    # cohort has a real DECIDED outcome (not NULL/pending, not simply
    # absent).
    by_path = dict(rows)
    assert set(by_path) == {"a.json", "b.json", str(third)}
    assert all(decision is not None for decision in by_path.values())

    # raw_a/raw_b are a genuine two-way divergence (shared "left"/"right"
    # message content conflicts), and raw_c neither purely contains nor is
    # contained by raw_b -- so the cohort as a whole is still an irreducible
    # conflict; no clean prefix chain exists. This logical source has never
    # had an accepted head (raw_a/raw_b were both retired straight to
    # membership governance quarantined, never byte-governed-accepted), so
    # the presence-guarantee fallback (polylogue-lb39z item 5) is free to
    # deterministically materialize the maximal-evidence representative
    # instead of leaving the reunified cohort headless: raw_c strictly
    # contains raw_a's content plus a further "extra" message, giving it the
    # largest frontier of the three, so it wins outright (no raw_id tiebreak
    # needed) and raw_a/raw_b become its recorded conflict debt. The source
    # observation itself was acquired and parsed successfully, so its cursor
    # is complete rather than retried as a transient file failure either way.
    assert by_path[str(third)] == "applied"
    assert by_path["a.json"] == "ambiguous"
    assert by_path["b.json"] == "ambiguous"
    assert third_result.failed == []
    assert third_result.succeeded == [third]
    # Direct check of the persisted state backing "cursor is complete rather
    # than retried" above: ``_ingest_full_paths_sync`` has no CursorStore row
    # of its own, so the durable non-retry evidence is raw_sessions.parse_error
    # staying NULL for third's raw regardless of its membership decision --
    # what actually stops the daemon from reprocessing this file as a failure
    # on every restart, not just the in-memory succeeded/failed lists.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(third),),
        ).fetchone() == (None,)


def test_membership_sweep_defers_sibling_retirement_instead_of_quarantining_current_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """polylogue-lpen: an active byte-revision chain must not poison an unrelated raw.

    Root cause (11 of 73 live ``claude-code-session`` parse-failed raws, found
    via ``polylogue ops debt list``): ``_apply_membership_sessions`` sweeps
    *every* full-only raw sharing a logical identity
    (``archive.convertible_full_revision_raw_ids``) and unconditionally tries
    to retire each one out of byte-revision governance
    (``replace_raw_membership_census(..., retire_full_revision_governance=
    True)``), as a side effect of processing some entirely different,
    currently-ingesting raw (``source_raw_id``). It never checks whether a
    candidate still has a live byte-chain dependent -- another raw whose
    ``predecessor_raw_id``/``baseline_raw_id`` points at it -- before
    attempting the retirement; that invariant is enforced only deeper inside
    ``replace_raw_membership_census``, which raises
    ``ActiveByteRevisionChainError`` (a ``RuntimeError``) when it finds one.
    Pre-fix, that exception propagated all the way up through
    ``_apply_membership_sessions`` into the live-watcher's blanket
    ``except Exception`` (``sources/live/batch.py``), which called
    ``archive.mark_raw_parse_failed`` on ``source_raw_id`` -- the CURRENT,
    unrelated raw -- permanently quarantining it even though it had nothing
    to do with the chain conflict.

    Fixture: raw_a (baseline) and raw_b (a genuine byte-extension of raw_a's
    bytes) form a real byte-proven full-revision chain for ``codex:shared``
    via ``classify_raw_revision_cohort`` -- raw_b's ``predecessor_raw_id``
    durably points at raw_a, exactly the "active byte-revision chain"
    dependency ``replace_raw_membership_census`` refuses to break. A third,
    unrelated raw (raw_c) then triggers ``_apply_membership_sessions`` for
    the same logical identity (mirroring the live "no accepted chain but no
    retired siblings either" / multi-session bundle call sites). The
    retirement sweep order is pinned (raw_a before raw_b) so the parent is
    always attempted while its dependent is still live, regardless of
    raw_id hash ordering.
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"a":1}\n',
            source_path=str(tmp_path / "a.jsonl"),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "codex:shared", RawRevisionKind.FULL, "rev-a", 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"a":1}\n{"b":2}\n',
            source_path=str(tmp_path / "b.jsonl"),
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "codex:shared", RawRevisionKind.FULL, "rev-b", 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        archive.classify_raw_revision_cohort_for_live_watch("codex:shared")
        archive.commit()
        raw_c = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"c":1}\n',
            source_path=str(tmp_path / "c.jsonl"),
            acquired_at_ms=3,
        )
        archive.commit()

    # Sanity: the byte chain really was proven, and raw_b really is a live
    # dependent of raw_a -- this is the exact condition the fixture claims.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = dict(
            conn.execute(
                "SELECT raw_id, predecessor_raw_id FROM raw_sessions WHERE logical_source_key = 'codex:shared'"
            ).fetchall()
        )
    assert rows[raw_a] is None
    assert rows[raw_b] == raw_a

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    session_a = session("shared", "base")
    session_b = session("shared", "base", "extra")
    session_c = session("shared", "base")
    sessions_by_raw_id = {raw_a: session_a, raw_b: session_b}

    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=tmp_path),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda _archive, raw_id: [sessions_by_raw_id[raw_id]],
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        # Confirm the lower-level invariant still fails closed on its own --
        # the fix belongs in the caller's handling of this, not in silently
        # dropping the check.
        with pytest.raises(archive_tier_module.ActiveByteRevisionChainError):
            archive.replace_raw_membership_census(
                raw_a,
                [session_a],
                parser_fingerprint="test-parser",
                censused_at_ms=10,
                retire_full_revision_governance=True,
            )
        archive.rollback()

        # Pin the sweep order to (parent, child) regardless of raw_id hash
        # ordering, so the parent is always attempted while its dependent
        # (raw_b) is still live.
        monkeypatch.setattr(archive, "convertible_full_revision_raw_ids", lambda _key: (raw_a, raw_b))

        # Mirror every production call site: source_raw_id is always censused
        # (without retiring anything) immediately before
        # ``_apply_membership_sessions`` is invoked.
        archive.replace_raw_membership_census(
            raw_c,
            [session_c],
            parser_fingerprint="test-parser",
            censused_at_ms=10,
        )

        with caplog.at_level("WARNING", logger="polylogue.sources.live.batch"):
            session_ids, session_count, message_count, complete = processor._apply_membership_sessions(
                archive,
                raw_c,
                [session_c],
                acquired_at_ms=10,
                allow_current_complete_raw=True,
            )
        archive.commit()

    # The unrelated raw_c ingest completes -- it is not poisoned by raw_a's
    # unresolved sibling-retirement conflict.
    assert session_count == 1
    assert message_count >= 1
    assert any("deferring" in record.message and raw_a in record.message for record in caplog.records)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        # raw_a's dependent (raw_b) was still live when its retirement was
        # attempted -- deferred, not retired: its byte-chain evidence survives
        # intact for a later tick to retry once raw_b resolves.
        assert conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (raw_a,),
        ).fetchone() == ("codex:shared", "byte_proven")
        # raw_b had no live dependent of its own -- it retired successfully.
        assert conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (raw_b,),
        ).fetchone() == (None, "quarantined")
        # raw_c itself never failed -- no parse_error was ever recorded for it.
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE raw_id = ?",
            (raw_c,),
        ).fetchone() == (None,)


def test_raw_membership_decision_pending_distinguishes_null_from_ambiguous(tmp_path: Path) -> None:
    """Pins the exact narrow scoping of the polylogue-emx2 fix (de0b2df7a regression, polylogue-lvz6 triage).

    ``raw_membership_authority_complete()`` collapses three distinct
    membership-decision states into one boolean: ``decision IS NULL``
    (genuinely async-pending -- censused but not yet arbitrated by the
    raw-materialization conveyor, ``sources/revision_backfill.py``) and
    ``decision IN ('ambiguous', 'deferred')`` (arbitration already ran and
    concluded a real conflict that needs new evidence, not time, to
    resolve). The first is a conveyor hand-off; the second is a durable
    fail-closed materialization outcome. Neither is a transient source-file
    failure, so the live cursor must not re-read unchanged bytes for either
    state. ``LiveBatchProcessor._ingest_full_records_archive`` uses
    ``raw_membership_decision_pending`` (not the coarse boolean alone) to
    preserve this distinction in its durable raw-authority state while both
    paths remain cursor-idempotent.

    This is an archive-tier predicate test rather than a full watcher
    end-to-end scenario because, by construction,
    ``LiveBatchProcessor._apply_membership_sessions`` always resolves the
    raw it just censused synchronously within the same call (both of its
    current call sites pass ``allow_current_complete_raw=True``) -- so a
    raw's own decision is never observed as NULL immediately after that
    call returns. Genuinely NULL decisions persist only across the
    conveyor's own two-phase census-then-classify split
    (``census_historical_revision_evidence`` /
    ``backfill_historical_revision_evidence``), which this test reproduces
    directly against the archive tier: census without classification (NULL,
    pending) versus census with an ambiguous classification (decided,
    unresolved).
    """
    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="pending-vs-ambiguous",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="hello")],
    )
    projection = session_revision_projection(session)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"native_id":"pending-vs-ambiguous"}\n',
            source_path=str(tmp_path / "pending-vs-ambiguous.jsonl"),
            acquired_at_ms=1,
        )
        archive.replace_raw_membership_census(
            raw_id,
            [session],
            parser_fingerprint="test-parser",
            censused_at_ms=1,
        )

        # Census complete, classification never run: decision IS NULL. This
        # is the genuinely async-pending state -- not a failure.
        assert archive.raw_membership_authority_complete(raw_id) is False
        assert archive.raw_membership_decision_pending(raw_id) is True

        # Arbitration now runs and concludes ambiguous (a decided conflict,
        # e.g. the conveyor found no unique growth chain). This is no longer
        # pending -- it must surface as a failure, not defer forever.
        archive.apply_raw_membership_classification(
            "codex:pending-vs-ambiguous",
            MembershipClassification((), (), (raw_id,)),
            {raw_id: session},
            {raw_id: projection},
            acquired_at_ms=2,
        )
        assert archive.raw_membership_authority_complete(raw_id) is False
        assert archive.raw_membership_decision_pending(raw_id) is False
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT decision FROM raw_session_memberships WHERE raw_id = ?", (raw_id,)
            ).fetchone() == ("ambiguous",)


def test_live_membership_reprocesses_parser_drift_without_retiring_unrelated_head(tmp_path: Path) -> None:
    """A current parse of the accepted raw is authority, even after parser drift.

    This reproduces the July 16 live failure: an older browser snapshot was
    accepted under an earlier parser, then byte-equivalent current snapshots
    reparsed both the accepted raw and the new raw to the same new projection.
    The accepted index head remains the CAS witness; its old content hash must
    not be mistaken for an unrelated raw head.
    """
    root = tmp_path / "inbox"
    root.mkdir()
    snapshot = root / "snapshot.json"
    payload: list[dict[str, object]] = [
        {
            "id": "parser-drift",
            "title": "current title",
            "create_time": 1_780_000_000.0,
            "current_node": "node",
            "mapping": {
                "node": {
                    "id": "node",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message",
                        "author": {"role": "user"},
                        "create_time": 1_780_000_000.0,
                        "content": {"content_type": "text", "parts": ["retained evidence"]},
                        "metadata": {},
                    },
                }
            },
        }
    ]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    current_session = parse_payload(Provider.CHATGPT, payload, "snapshot")[0]
    legacy_session = current_session.model_copy(update={"title": "legacy parser title"})
    legacy_projection = session_revision_projection(legacy_session)
    current_projection = session_revision_projection(current_session)
    assert legacy_projection.session_hash != current_projection.session_hash

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        legacy_raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=snapshot.read_bytes(),
            source_path=str(snapshot),
            acquired_at_ms=1,
        )
        archive.replace_raw_membership_census(
            legacy_raw_id,
            [legacy_session],
            parser_fingerprint="legacy-parser",
            censused_at_ms=1,
        )
        archive.apply_raw_membership_classification(
            "chatgpt:parser-drift",
            MembershipClassification((legacy_raw_id,), (), ()),
            {legacy_raw_id: legacy_session},
            {legacy_raw_id: legacy_projection},
            acquired_at_ms=1,
        )

    # Byte-level formatting changes create a new retained raw while preserving
    # the provider session. The live route reparses the accepted raw too.
    snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="inbox", root=root, suffixes=(".json",)),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="current-parser",
    )

    result = processor._ingest_full_paths_sync([snapshot], source_name="inbox")

    assert result.succeeded == [snapshot]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "index.db") as conn:
        stored = conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = 'chatgpt-export:parser-drift'"
        ).fetchone()
        assert stored is not None
        assert stored != (legacy_projection.session_hash,)
        head = conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:parser-drift'"
        ).fetchone()
        assert head == stored


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
    """Pins #2718's fail-closed contract: a bundle raw discovered later must
    never silently replace an accepted head that still has live, unresolved
    byte-append evidence (the QUARANTINED append raw this test binds), even
    when the bundle's own content happens to strictly extend the head's
    content (``bundle_texts2``/``bundle_texts3``: content-prefix growth alone
    is not proof of provenance).

    polylogue-miwv (2026-07-21): #3211 ("in-cohort head-retire drift fix")
    removed ``apply_raw_membership_classification``'s byte-governance refusal
    on the mistaken premise that its branch is only reachable after a real
    membership-governance conversion -- but ``_apply_membership_sessions``
    unconditionally injects the CURRENT accepted head into the comparison
    cohort even when it has never been converted (exactly this test's byte-
    governed-head scenario), so the removed guard's absence let the older
    bundle's superset content silently move the head (message_count 2->3,
    ``accepted_raw_id`` changed) for ``bundle_texts2``/``bundle_texts3``.
    This was not caused by, and is unrelated to, the messages_fts_identity
    UNIQUE(block_id) ledger work landing the same day (polylogue-miwv's
    other commits) -- confirmed by reproducing this exact failure on the
    commit immediately preceding messages_fts_identity's introduction.
    Restored as a narrower guard (refuses only when replay is about to change
    the accepted raw AND a live raw_sessions row still chains a
    ``predecessor_source_revision`` off the existing head) so #3211's own
    interrupted-pass-drift resumption keeps working.
    """
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

    # A same-size divergent membership result is a decided authority conflict
    # with a complete source observation; attempted replacement through live
    # append evidence raises instead and must remain retryable.
    cursor_complete = succeeds or bundle_texts == ("base", "different")
    assert result.failed == ([] if cursor_complete else [older_bundle])
    assert result.succeeded == ([older_bundle] if cursor_complete else [])
    # Direct check of the persisted state backing both branches: the
    # succeeded/failed lists above are LiveBatchProcessor's own report, not
    # proof of what raw_sessions durably holds (this layer -- ``_ingest_full_
    # paths_sync`` -- has no CursorStore row of its own). A decided-ambiguous
    # cursor_complete observation is deliberately never materialized
    # (parsed_at_ms stays NULL -- fail-closed for the head), so the actual
    # "not a failure/retry loop" evidence is parse_error staying NULL. When
    # not cursor_complete, the docstring's "must remain retryable" claim
    # requires the raw to actually carry a parse_error -- what makes a future
    # daemon pass re-attempt this exact raw instead of silently treating it
    # as already resolved.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(older_bundle),),
        ).fetchone()
    if cursor_complete:
        assert parse_error is None
    else:
        assert parse_error is not None
        # polylogue-5iz4: this guard's refusal is transient/retry-eligible by
        # construction (a later pass over the same durable bytes can succeed
        # once sibling evidence resolves), but a plain RuntimeError leaves the
        # retry-candidate query (storage/repair.py) nothing stable to match
        # once the message text drifts -- exactly what happened to a real
        # production session that hit this guard under #2718's original
        # wording.
        # The structured evidence row, not the diagnostic wording, is the
        # retry authorization.
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                """
                SELECT a.artifact_kind
                FROM raw_artifacts AS a
                JOIN raw_sessions AS r ON r.raw_id = a.raw_id
                WHERE r.source_path = ?
                """,
                (str(older_bundle),),
            ).fetchone() == ("deferred_cas_frontier",)
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


def test_growing_file_incident_recovery_duplicate_recovers_after_head_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-5iz4: reproduce the real production shape at growth-chain scale.

    A real Codex session (native_id ``019f49d8-...``) accumulated 804
    ``raw_sessions`` rows because a live watcher periodically captured
    FULL snapshots of one continuously-growing ``rollout.jsonl`` (not
    append deltas) -- ~800 generations of the SAME file, strictly growing
    byte-for-byte (confirmed read-only against the live archive: every
    smaller full-revision blob is an exact byte prefix of every larger
    one, a single clean linear chain with zero forks). Two extra
    identical-content full snapshots landed at a SECOND, "incident
    recovery" source path sharing the same native_id -- an out-of-band
    backup/restore copy taken during a live incident. One of the 804 rows
    carries ``parse_error='RuntimeError: membership replay cannot replace
    an unconvertible byte head'`` (PR #2718's now-superseded wording);
    the session never reached ``index.db``.

    This test reproduces the mechanism at REALISTIC scale (many real
    incremental full-snapshot captures of one growing Codex JSONL file,
    not a single static snapshot) plus a colliding same-identity duplicate
    from a second path, and then demonstrates the actual recovery path:
    ``apply_raw_membership_classification``'s guard is a **fail-closed,
    correct** refusal (an unrelated/dangling-evidence head must never be
    silently replaced) -- not a permanent dead end. Once
    ``MembershipReplayConflictError`` is recorded with a stable,
    retry-eligible ``parse_error`` marker (polylogue-5iz4 / #3646) AND the
    accepted head naturally advances past the interfering evidence
    (exactly what a live-watched growing file does on its own, and what
    the live archive's current empty ``raw_revision_heads``/``sessions``
    rows for this identity show already happened), a later pass over the
    SAME duplicate raw succeeds and reaches the index with a plausible
    message_count.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    current = root / "rollout-growing.jsonl"
    incident_recovery = root.parent / "inbox" / "incident-recovery-rollout-growing.jsonl"
    incident_recovery.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(b'{"current":true}\n')
    incident_recovery.write_bytes(b'{"bundle":true}\n')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id_: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id_,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id_}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    # A growth chain sized like the real production shape (804
    # incremental full-snapshot captures of one growing Codex rollout
    # file, reduced here for test speed -- the mechanism does not depend
    # on the absolute count, only on the accepted head carrying MANY
    # messages, not a token 2-3 like the small #2718 pin).
    native_id = "019f49d8-shape-fixture"
    growth_generations = 25
    base_texts = tuple(f"growth-generation-{index:04d}" for index in range(growth_generations))
    current_session = session(native_id, *base_texts)
    # The "incident recovery" duplicate: content-prefix growth alone is
    # not proof of provenance (revision_governance.py's own polylogue-miwv
    # note), so a same-identity bundle that strictly extends the accepted
    # head's content must still be evaluated through membership
    # governance, not silently accepted. Bundled alongside an unrelated
    # second session in one file -- the real incident-recovery backup
    # grabbed multiple sessions in one sweep, and a multi-session raw
    # unconditionally routes through membership governance
    # (``LiveBatchProcessor._ingest_full_paths_sync``'s ``len(sessions) !=
    # 1`` branch), which is what makes ``raw_revision_head_raw_id``'s
    # unconditional cohort injection reachable for a single-session-per-
    # file Codex identity like this one -- exactly how the real 804-row
    # session hit it despite Codex normally writing one session per file.
    recovered_extension = session(native_id, *base_texts, "growth-generation-0025", "growth-generation-0026")
    recovered_unrelated = session("019f49d8-unrelated-safe-session", "one")

    def _parse_stream_payload_stub(
        _provider: Any, _records: Any, _fallback_id: Any, *, source_path: str
    ) -> list[ParsedSession]:
        if Path(source_path) == incident_recovery:
            return [recovered_extension, recovered_unrelated]
        return [current_session]

    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.parse_stream_payload",
        _parse_stream_payload_stub,
    )
    monkeypatch.setattr(
        processor,
        "_parse_retained_raw_sessions",
        lambda archive, raw_id: (
            [current_session]
            if Path(archive.raw_revision_material(raw_id)[2]) == current
            else [recovered_extension, recovered_unrelated]
        ),
    )

    assert processor._ingest_full_paths_sync([current], source_name="codex").failed == []
    with sqlite3.connect(index_db) as conn:
        head_row = conn.execute(
            "SELECT accepted_raw_id, session_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (f"codex:{native_id}",),
        ).fetchone()
        assert head_row is not None
        accepted_raw_id, session_id = head_row
        message_count_before = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert message_count_before == growth_generations

    # A dangling, unresolved QUARANTINED append fragment hanging off the
    # CURRENT accepted head's own source_revision -- the live-append-
    # cursor evidence the guard exists to protect (mirrors
    # test_bundle_replay_respects_unconvertible_single_session_head's
    # ``append_raw_id`` setup) -- plus an explicit prior census of the
    # accepted head (``census_head``), matching that test's reliably
    # guard-triggering combination.
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        head_source_revision = (
            archive._ensure_source_conn()
            .execute(
                "SELECT source_revision FROM raw_sessions WHERE raw_id = ?",
                (accepted_raw_id,),
            )
            .fetchone()[0]
        )
        dangling_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"dangling"}}\n',
            source_path=str(current),
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            dangling_append_raw_id,
            RawRevisionEnvelope(
                f"codex:{native_id}",
                RawRevisionKind.APPEND,
                "dangling-append-blocker",
                0,
                predecessor_source_revision=str(head_source_revision),
                append_start_offset=1,
                append_end_offset=2,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        # Deliberately no prior ``replace_raw_membership_census`` call here
        # (unlike ``test_bundle_replay_...``'s ``census_head=True`` case):
        # the real production identity was governed purely through typed
        # byte-revision authority (``bind_raw_revision``/live-watch
        # classification), never through an explicit membership census of
        # its own accepted head. Adding one here would permanently divert
        # every later reprocessing of ``current`` through membership
        # governance instead of the plain byte-chain replay path, which
        # does not match the real shape and would make the eventual
        # recovery below impossible to reproduce faithfully.

    conflict_result = processor._ingest_full_paths_sync([incident_recovery], source_name="codex")

    # Fail-closed is correct here: the guard must refuse to silently
    # replace a head with unresolved byte-append evidence hanging off it.
    assert conflict_result.failed == [incident_recovery]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(incident_recovery),),
        ).fetchone()
    assert parse_error is not None
    # The typed evidence is authoritative for new rows. The recognized prefix
    # remains a bounded compatibility bridge for this historical diagnostic.
    assert parse_error.startswith("MembershipReplayConflictError:")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT a.artifact_kind
            FROM raw_artifacts AS a
            JOIN raw_sessions AS r ON r.raw_id = a.raw_id
            WHERE r.source_path = ?
            """,
            (str(incident_recovery),),
        ).fetchone() == ("deferred_cas_frontier",)
    from polylogue.storage.repair import _raw_materialization_retryable_missing_blob_error

    assert _raw_materialization_retryable_missing_blob_error(parse_error) is True
    assert _raw_materialization_retryable_missing_blob_error("RuntimeError: unrelated parser failure") is False
    assert _raw_materialization_retryable_missing_blob_error(parse_error, True) is True

    with sqlite3.connect(index_db) as conn:
        assert (
            conn.execute("SELECT message_count FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
            == message_count_before
        )

    # The interfering condition is transient by construction, not
    # permanent, per ``MembershipReplayConflictError``'s own docstring: "a
    # later pass over the same durable bytes can succeed once sibling
    # evidence resolves or the accepted head itself changes". This is
    # exactly what the live archive's own EMPTY raw_revision_heads row for
    # this identity shows already happened (confirmed read-only,
    # 2026-08-03): whatever accepted-head state interfered with the
    # original 2026-07-10 attempt is gone today, so a fresh classification
    # pass hits the guard's ``existing_head is not None`` precondition
    # never at all and proceeds straight to indexing. Simulate that same
    # cleared state directly (the dangling append fragment bound above is
    # permanently unresolvable -- its byte offsets never correspond to any
    # real content, so no further real ingest can ever promote it; the
    # accepted head itself must be retired, matching
    # ``release_provisional_full_revisions``'s existing "provisional
    # evidence rejected" shape for full revisions).
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "DELETE FROM raw_revision_heads WHERE logical_source_key = ?",
            (f"codex:{native_id}",),
        )
        conn.commit()

    # This is the AC#2 assertion: once the accepted head no longer
    # interferes, a retry over the SAME durable incident-recovery raw
    # succeeds and reaches the index with a plausible message_count --
    # note this exercises the live-watcher's own retry path
    # (``_ingest_full_paths_sync`` again), not
    # ``storage/repair.py``'s offline ``repair_raw_materialization``:
    # that offline path reprocesses every retained typed-'full' raw for
    # this logical_source_key on every pass (including ``current``'s own
    # cohort), which re-establishes an accepted head before ever reaching
    # ``incident_recovery`` in the same pass and so cannot demonstrate
    # this recovery in isolation here -- a real gap worth a follow-up
    # bead, not one this test's fixture can respect the scope of.
    retry_result = processor._ingest_full_paths_sync([incident_recovery], source_name="codex")
    assert retry_result.failed == []
    assert retry_result.succeeded == [incident_recovery]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (retried_parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms DESC LIMIT 1",
            (str(incident_recovery),),
        ).fetchone()
    assert retried_parse_error is None

    with sqlite3.connect(index_db) as conn:
        final_count = conn.execute("SELECT message_count FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
    # Plausible: the incident-recovery bundle's own extension, 2
    # generations past the pre-conflict head.
    assert final_count == growth_generations + 2


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

    # Divergence remains fail-closed for the materialized head, but the source
    # bytes were acquired and parsed successfully. Treating this as a cursor
    # success prevents each daemon restart from reprocessing the same decided
    # conflict until the file actually changes.
    assert divergent_result.succeeded == [divergent]
    assert divergent_result.failed == []
    # Direct check of the persisted state backing "cursor success" above:
    # this layer (``_ingest_full_paths_sync``) has no CursorStore row of its
    # own, so the durable non-retry evidence is raw_sessions.parse_error
    # staying NULL for the divergent raw despite the fail-closed membership
    # decision. A regression that started marking this a parse failure
    # would make the daemon reprocess the same decided-ambiguous divergence
    # on every restart, which is exactly what this comment says must not
    # happen.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(divergent),),
        ).fetchone() == (None,)
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

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="crash-retry")
    # polylogue-1r9c: mark_raw_parse_succeeded is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through ArchiveStore's `self.` dispatch -- patch it there.
    original_mark_succeeded = archive_revision_governance.mark_raw_parse_succeeded
    crashed = False

    def crash_after_index(
        archive: archive_revision_governance.RawRevisionGovernanceHost,
        raw_id: str,
        *,
        provider: Provider,
    ) -> None:
        nonlocal crashed
        source_index = (
            archive._ensure_source_conn()
            .execute("SELECT source_index FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()[0]
        )
        if source_index == -1 and not crashed:
            assert archive._conn.execute("SELECT 1 FROM messages WHERE native_id = 'message-1'").fetchone() is not None
            crashed = True
            raise SimulatedProcessCrash
        original_mark_succeeded(archive, raw_id, provider=provider)

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", crash_after_index)
    with pytest.raises(SimulatedProcessCrash):
        ingest_append_plans(cast(Any, owner), [plan])

    assert _append_raw_parse_state(tmp_path) == (None, None)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2

    monkeypatch.setattr(archive_revision_governance, "mark_raw_parse_succeeded", original_mark_succeeded)
    retry = ingest_append_plans(cast(Any, owner), [plan])

    assert retry.succeeded == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 2
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is not None
    assert parse_error is None
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


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
    for filename in (spec.filename for spec in ARCHIVE_TIER_SPECS.values()):
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
async def test_live_full_ingest_skips_convergence_without_session_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor-only raw observation must not rerun global workflow materializers."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "unchanged.json"
    path.write_text("{}", encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))),
        (WatchSource(name="sessions", root=root, suffixes=(".json",)),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    convergence_calls: list[list[Path]] = []

    async def fake_full_ingest(
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: object | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id, max_pass_seconds, pass_started
        return _FullIngestResult(
            succeeded=paths,
            failed=[],
            source_payload_read_bytes=0,
            raw_fingerprints={path: "raw-unchanged"},
            changed_session_count=0,
        )

    def record_convergence(paths: list[Path]) -> tuple[set[Path], float, dict[str, float], list[object]]:
        convergence_calls.append(paths)
        return set(paths), 0.0, {}, []

    monkeypatch.setattr(processor, "_append_plan", lambda _path, *, cursor: None)
    monkeypatch.setattr(processor, "_ingest_full_paths", fake_full_ingest)
    monkeypatch.setattr(processor, "_converge_paths", record_convergence)
    monkeypatch.setattr(processor, "_record_full_cursor", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(processor, "_compact_superseded_raw_snapshots", lambda _paths: None)

    metrics = await processor.ingest_files([path], emit_event=False)

    assert convergence_calls == []
    assert metrics.succeeded_file_count == 1
    assert metrics.changed_session_count == 0


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
