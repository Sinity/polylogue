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
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, TitleSource
from polylogue.pipeline.services import ingest_worker as ingest_worker_module
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.sources.live import WatchSource
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord
from tests.infra.archive_templates import run_off_event_loop


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
        canonical_source_path=source_path,
        payload_provider=Provider.CODEX,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
        sidecar_snapshot=sidecar_snapshot,
    )


def _ingest_title(record: RawSessionRecord, tmp_path: Path, store: BlobStore) -> tuple[str | None, str | None]:
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(store.root))
    assert result.error is None, result.error
    assert result.sessions, "expected one materializable session"
    parsed = result.sessions[0].parsed_session
    source = parsed.title_source
    return parsed.title, str(source) if source is not None else None


@pytest.mark.parametrize("late_session", [False, True])
def test_canonical_worker_classifies_complete_checkpoint_stream_with_declared_path(
    blob_store: BlobStore,
    tmp_path: Path,
    late_session: bool,
) -> None:
    rows: list[dict[str, object]] = [{"type": "file-history-snapshot"}] * 65
    if late_session:
        rows.append(
            {
                "type": "user",
                "uuid": "message",
                "sessionId": "session",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "preserve the actual late turn"},
            }
        )
    content = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    raw_id, size = blob_store.write_from_bytes(content)
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_CODE.value,
        payload_provider=Provider.CLAUDE_CODE,
        source_path=str(tmp_path / ".claude" / "projects" / "project" / "session.jsonl"),
        canonical_source_path=str(tmp_path / ".claude" / "projects" / "project" / "session.jsonl"),
        source_index=None,
        blob_size=size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))
    assert result.error is None
    if late_session:
        assert result.sessions
        assert result.sessions[0].parsed_session.messages[0].text == "preserve the actual late turn"
    else:
        assert not result.sessions
        assert result.evidence_ref == "artifact_not_session:file_history_snapshot"


def test_canonical_ingest_applies_thread_name(blob_store: BlobStore, tmp_path: Path) -> None:
    """The daemon worker resolves the provider thread name, like direct ingest."""
    session_id = "aaaa1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "please fix the ingest bug")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "session_index.jsonl").write_text(
        json.dumps({"id": session_id, "thread_name": "Ingest bug hunt"}) + "\n",
        encoding="utf-8",
    )

    record = _record(
        blob_store,
        content,
        source_path=str(rollout),
        sidecar_snapshot={"thread_names": {session_id: "Ingest bug hunt"}},
    )
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

    record = _record(
        blob_store,
        content,
        source_path=str(rollout),
        sidecar_snapshot={"history_titles": {session_id: "Wire the Hermes bridge"}},
    )
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


def test_missing_snapshot_does_not_attempt_on_demand_enrichment(
    blob_store: BlobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct worker call treats absent acquisition evidence as ordinary absence."""
    from polylogue.sources import assembly_codex

    session_id = "eeee1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "please fix the ingest bug")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "session_index.jsonl").write_text(
        json.dumps({"id": session_id, "thread_name": "Ingest bug hunt"}) + "\n",
        encoding="utf-8",
    )

    discovery_calls: list[object] = []

    def recording_discover(self: object, paths: object) -> object:
        discovery_calls.append(paths)
        raise OSError("on-demand discovery must never run inside a worker")

    monkeypatch.setattr(assembly_codex.CodexAssemblySpec, "discover_sidecars", recording_discover)

    record = _record(blob_store, content, source_path=str(rollout))
    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None
    assert result.sessions, "expected the record to materialize without optional evidence"
    assert discovery_calls == [], "worker consulted the ambient source tree for sidecars"
    # The runtime root carries a session_index naming the thread "Ingest bug
    # hunt"; only a worker that read it off disk could surface that title.
    assert result.sessions[0].parsed_session.title == "please fix the ingest bug"


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


# ---------------------------------------------------------------------------
# polylogue-ximhz: provider metadata and attachment joins resolve from
# retained evidence, with the original files gone.
#
# Every case below acquires its evidence through the ordinary live route
# (``LiveBatchProcessor.ingest_files``), unlinks every original path, and then
# runs the canonical raw-record ingest. Nothing is hand-seeded into the source
# tier, so the join under test is the production one.
#
# Anti-vacuity: drop the retained map -- delete the ``sessions-index.json`` /
# ``history.jsonl`` / ``conversation_asset_file_names.json`` raw row, as
# ``test_dropping_the_retained_asset_map_loses_the_resolved_name`` does -- and
# the matching fidelity assertion fails. Reverting the
# ``resolve_retained_assembly_evidence`` call in ``_enrich_parsed_sessions``
# fails all of them at once.
# ---------------------------------------------------------------------------

_CLAUDE_SESSION_ID = "aaaaaaaa-1111-2222-3333-444444444444"
_CLAUDE_TS = "2026-07-20T10:00:00.000Z"
_CLAUDE_TS_MS = 1784541600000
_CHATGPT_ASSET_ID = "file-ABCdef123"
_CHATGPT_ASSET_BYTES = b"\x89PNG\r\n\x1a\nsynthetic asset payload"


def _claude_transcript() -> bytes:
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": _CLAUDE_SESSION_ID,
            "timestamp": _CLAUDE_TS,
            "message": {"role": "user", "content": "first prompt"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": _CLAUDE_SESSION_ID,
            "timestamp": "2026-07-20T10:00:01.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        },
    ]
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")


def _chatgpt_export_document() -> bytes:
    return json.dumps(
        [
            {
                "id": "conv-1",
                "title": "Asset conversation",
                "create_time": 1750000000.0,
                "update_time": 1750000100.0,
                "mapping": {
                    "m1": {
                        "id": "m1",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m1",
                            "author": {"role": "user"},
                            "create_time": 1750000000.0,
                            "content": {
                                "content_type": "multimodal_text",
                                "parts": [
                                    {
                                        "content_type": "image_asset_pointer",
                                        "asset_pointer": f"file-service://{_CHATGPT_ASSET_ID}",
                                        "size_bytes": len(_CHATGPT_ASSET_BYTES),
                                    },
                                    "look at this",
                                ],
                            },
                            "metadata": {"attachments": [{"id": _CHATGPT_ASSET_ID, "size": len(_CHATGPT_ASSET_BYTES)}]},
                        },
                    }
                },
            }
        ]
    ).encode("utf-8")


async def _acquire_evidence(archive_root: Path, source: WatchSource, paths: list[Path]) -> None:
    """Acquire declared non-session evidence through the live source route and its owners."""
    import polylogue.sources.live.watcher as live_watcher
    from tests.infra.live_batch import prepared_live_batch_processor

    archive_root.mkdir(parents=True, exist_ok=True)
    async with prepared_live_batch_processor(
        archive_root, (source,), parser_fingerprint=live_watcher._PARSER_FINGERPRINT
    ) as processor:
        await processor.ingest_files(paths, emit_event=False)


def _retained_artifact_kinds(archive_root: Path) -> dict[str, str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        return {
            str(source_path): str(kind)
            for kind, source_path in conn.execute("SELECT artifact_kind, source_path FROM raw_artifacts")
        }
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_codex_retained_root_sidecar_titles_survive_without_a_live_tree(
    blob_store: BlobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained Codex root sidecars title one rollout without minting another.

    The daemon's default ``codex-state`` source admits the two exact root
    coordinates despite having no JSONL suffix intake, and nothing else of
    JSONL shape under the install root. Once those bytes and the rollout
    disappear, the canonical ingest worker must resolve the title from the
    archive itself.

    Anti-vacuity (polylogue-ez5b9, 11.F069): with the default source's
    path-rule escape hatch closed, it refuses both sidecars and they are
    never retained.
    """
    from polylogue.sources.live.source_selection import deepest_source_for_path
    from polylogue.sources.live.watcher import default_sources

    monkeypatch.setenv("HOME", str(tmp_path))
    archive_root = tmp_path / "archive"
    session_id = "retained-codex-sidecar-thread"
    content = _codex_stream(session_id, "opening prompt must not win")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    codex_root = rollout.parents[2]
    index_path = codex_root / "session_index.jsonl"
    history_path = codex_root / "history.jsonl"
    index_path.write_text(json.dumps({"id": session_id, "thread_name": "Retained curated title"}) + "\n")
    history_path.write_text(json.dumps({"session_id": session_id, "ts": 1, "text": "Retained history title"}) + "\n")

    codex_sources = tuple(
        source for source in default_sources() if source.name in {"codex", "codex-state", "codex-memories"}
    )
    source = next(source for source in codex_sources if source.name == "codex-state")
    assert source.root == codex_root
    assert source.accepts(index_path)
    assert source.accepts(history_path)
    for unrelated in (
        codex_root / "sessions" / "nested" / "session_index.jsonl",
        codex_root / "other.jsonl",
        codex_root / "log" / "history.jsonl",
        codex_root / "skills" / "tool" / "memories" / "note.md",
        rollout,
    ):
        assert not source.accepts(unrelated), unrelated
    owners = {path: deepest_source_for_path(path, codex_sources) for path in (index_path, history_path, rollout)}
    assert {path: owner.name for path, owner in owners.items() if owner is not None} == {
        index_path: "codex-state",
        history_path: "codex-state",
        rollout: "codex",
    }
    await _acquire_evidence(archive_root, source, [index_path, history_path])
    assert _retained_artifact_kinds(archive_root) == {
        str(history_path): "prompt_history_log",
        str(index_path): "session_index",
    }

    for path in (index_path, history_path, rollout):
        path.unlink()
    record = _record(blob_store, content, source_path=str(rollout))
    result = ingest_record(record, str(archive_root), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None, result.error
    assert len(result.sessions) == 1
    assert result.sessions[0].parsed_session.title == "Retained curated title"
    assert result.sessions[0].parsed_session.title_source is TitleSource.ORIGIN


@pytest.mark.asyncio
async def test_claude_index_and_history_resolve_with_the_original_tree_gone(
    blob_store: BlobStore, tmp_path: Path
) -> None:
    """The curated title and paste evidence come back from retained bytes."""
    from polylogue.sources.live import WatchSource
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    archive_root = tmp_path / "archive"
    claude_home = tmp_path / "live" / ".claude"
    project = claude_home / "projects" / "-realm-project-x"
    project.mkdir(parents=True)
    transcript = project / f"{_CLAUDE_SESSION_ID}.jsonl"
    transcript.write_bytes(_claude_transcript())
    index_path = project / "sessions-index.json"
    index_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "sessionId": _CLAUDE_SESSION_ID,
                        "fullPath": str(transcript),
                        "summary": "Curated index title",
                        "messageCount": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    history_path = claude_home / "history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "display": "first prompt",
                "timestamp": _CLAUDE_TS_MS,
                "sessionId": _CLAUDE_SESSION_ID,
                "pastedContents": {"1": {"type": "text", "content": "pasted body"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    await _acquire_evidence(
        archive_root,
        WatchSource(
            name="claude-code",
            root=project.parent,
            suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
        ),
        [index_path],
    )
    await _acquire_evidence(
        archive_root,
        WatchSource(name="claude-code-history", root=claude_home, suffixes=()),
        [history_path],
    )
    kinds = _retained_artifact_kinds(archive_root)
    assert kinds.get(str(index_path)) == "session_index"
    assert kinds.get(str(history_path)) == "prompt_history_log"

    content = _claude_transcript()
    raw_id, blob_size = blob_store.write_from_bytes(content)
    for path in (index_path, history_path, transcript):
        path.unlink()

    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="claude-code",
        source_path=str(transcript),
        canonical_source_path=str(transcript),
        payload_provider=Provider.CLAUDE_CODE,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-07-20T10:00:00+00:00",
        file_mtime=None,
    )
    result = ingest_record(record, str(archive_root), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None, result.error
    parsed = result.sessions[0].parsed_session
    assert parsed.title == "Curated index title"
    assert str(parsed.title_source) == TitleSource.ORIGIN.value
    user_message = next(message for message in parsed.messages if message.role == "user")
    assert [span.source_marker for span in user_message.paste_spans] == ["1"]


@pytest.mark.asyncio
async def test_claude_retained_index_is_scoped_to_its_own_install(blob_store: BlobStore, tmp_path: Path) -> None:
    """A second install's index never titles the first install's session."""
    from polylogue.sources.live import WatchSource
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    archive_root = tmp_path / "archive"
    other = tmp_path / "live" / "install-b" / ".claude" / "projects" / "-realm-project-x"
    other.mkdir(parents=True)
    other_index = other / "sessions-index.json"
    other_index.write_text(
        json.dumps({"entries": [{"sessionId": _CLAUDE_SESSION_ID, "fullPath": "", "summary": "Other install title"}]}),
        encoding="utf-8",
    )
    await _acquire_evidence(
        archive_root,
        WatchSource(
            name="claude-code",
            root=other.parent,
            suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
        ),
        [other_index],
    )
    assert _retained_artifact_kinds(archive_root).get(str(other_index)) == "session_index"

    mine = tmp_path / "live" / "install-a" / ".claude" / "projects" / "-realm-project-x"
    transcript = mine / f"{_CLAUDE_SESSION_ID}.jsonl"
    raw_id, blob_size = blob_store.write_from_bytes(_claude_transcript())
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="claude-code",
        source_path=str(transcript),
        canonical_source_path=str(transcript),
        payload_provider=Provider.CLAUDE_CODE,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-07-20T10:00:00+00:00",
        file_mtime=None,
    )
    result = ingest_record(record, str(archive_root), "advisory", blob_root_str=str(blob_store.root))

    assert result.error is None, result.error
    parsed = result.sessions[0].parsed_session
    assert parsed.title == "first prompt"
    assert str(parsed.title_source) == TitleSource.HEURISTIC.value


def _write_chatgpt_export(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    asset = root / f"{_CHATGPT_ASSET_ID}.dat"
    asset.write_bytes(_CHATGPT_ASSET_BYTES)
    library = root / "library_files.json"
    library.write_text(json.dumps([{"id": _CHATGPT_ASSET_ID, "name": "diagram.png"}]), encoding="utf-8")
    names = root / "conversation_asset_file_names.json"
    names.write_text(json.dumps({_CHATGPT_ASSET_ID: "diagram.png"}), encoding="utf-8")
    conversations = root / "conversations.json"
    conversations.write_bytes(_chatgpt_export_document())
    return asset, library, names, conversations


async def _acquire_chatgpt_export(archive_root: Path, root: Path) -> tuple[Path, Path, Path, Path]:
    from polylogue.sources.live import WatchSource

    asset, library, names, conversations = _write_chatgpt_export(root)
    await _acquire_evidence(
        archive_root,
        WatchSource(name="chatgpt", root=root, suffixes=(".json",)),
        [asset, library, names],
    )
    return asset, library, names, conversations


def _chatgpt_resolution_event(parsed: object) -> dict[str, object]:
    events = [
        event
        for event in parsed.session_events  # type: ignore[attr-defined]
        if event.event_type == "chatgpt_asset_resolution"
    ]
    assert events, "expected one asset-resolution event"
    return dict(events[0].payload)


def _chatgpt_replay(archive_root: Path, blob_store: BlobStore, conversations: Path) -> object:
    raw_id, blob_size = blob_store.write_from_bytes(_chatgpt_export_document())
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="chatgpt",
        source_path=str(conversations),
        canonical_source_path=str(conversations),
        payload_provider=Provider.CHATGPT,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-07-20T10:00:00+00:00",
        file_mtime=None,
    )
    result = ingest_record(record, str(archive_root), "advisory", blob_root_str=str(blob_store.root))
    assert result.error is None, result.error
    assert result.sessions, "expected one materializable session"
    return result.sessions[0].parsed_session


@pytest.mark.asyncio
async def test_chatgpt_asset_identity_and_bytes_resolve_with_the_export_gone(
    blob_store: BlobStore, tmp_path: Path
) -> None:
    """Attachment name and payload come back from retained export evidence."""
    archive_root = tmp_path / "archive"
    root = tmp_path / "live" / "chatgpt-export"
    asset, library, names, conversations = await _acquire_chatgpt_export(archive_root, root)

    kinds = _retained_artifact_kinds(archive_root)
    assert kinds.get(str(asset)) == "export_asset"
    assert kinds.get(str(library)) == "export_asset_index"
    assert kinds.get(str(names)) == "export_asset_index"

    for path in (asset, library, names, conversations):
        path.unlink()

    parsed = _chatgpt_replay(archive_root, blob_store, conversations)
    attachment = parsed.attachments[0]  # type: ignore[attr-defined]
    assert attachment.provider_file_id == _CHATGPT_ASSET_ID
    assert attachment.name == "diagram.png"
    assert attachment.precomputed_blob is not None
    blob_hash, size = attachment.precomputed_blob
    assert size == len(_CHATGPT_ASSET_BYTES)
    # The payload is readable from the archive's own blob store, which is
    # where acquisition retained it -- not from the original export path.
    assert BlobStore(archive_root / "blob").read_all(blob_hash) == _CHATGPT_ASSET_BYTES
    assert _chatgpt_resolution_event(parsed)["blob_acquired"] is True


@pytest.mark.asyncio
async def test_dropping_the_retained_asset_map_loses_the_resolved_name(blob_store: BlobStore, tmp_path: Path) -> None:
    """The retained map, not the original file, is what the name depends on."""
    import sqlite3

    archive_root = tmp_path / "archive"
    root = tmp_path / "live" / "chatgpt-export"
    asset, library, names, conversations = await _acquire_chatgpt_export(archive_root, root)
    for path in (asset, library, names, conversations):
        path.unlink()

    conn = sqlite3.connect(archive_root / "source.db")
    try:
        conn.execute("DELETE FROM raw_artifacts WHERE artifact_kind = 'export_asset_index'")
        conn.execute("DELETE FROM raw_sessions WHERE source_path LIKE '%asset_file_names.json'")
        conn.execute("DELETE FROM raw_sessions WHERE source_path LIKE '%library_files.json'")
        conn.commit()
    finally:
        conn.close()

    parsed = _chatgpt_replay(archive_root, blob_store, conversations)
    attachment = parsed.attachments[0]  # type: ignore[attr-defined]
    assert attachment.name != "diagram.png"


@pytest.mark.asyncio
async def test_the_same_asset_id_in_two_exports_never_cross_binds(blob_store: BlobStore, tmp_path: Path) -> None:
    """Scope is identity: one export's map never names another export's asset."""
    archive_root = tmp_path / "archive"
    other_root = tmp_path / "live" / "other-export"
    other_asset, other_library, other_names, _other_conversations = await _acquire_chatgpt_export(
        archive_root, other_root
    )
    assert _retained_artifact_kinds(archive_root).get(str(other_asset)) == "export_asset"

    mine = tmp_path / "live" / "my-export"
    mine.mkdir(parents=True)
    conversations = mine / "conversations.json"

    parsed = _chatgpt_replay(archive_root, blob_store, conversations)
    attachment = parsed.attachments[0]  # type: ignore[attr-defined]
    assert attachment.name != "diagram.png"
    assert attachment.precomputed_blob is None


@pytest.mark.asyncio
async def test_a_late_asset_map_resolves_on_the_next_convergence(blob_store: BlobStore, tmp_path: Path) -> None:
    """Late metadata is an attributable outcome, then an ordinary resolution.

    Convergence over the same retained conversation is idempotent and
    re-reads the source tier, so an export map acquired after the
    conversation resolves on the next pass instead of being permanently
    missed. Before the map lands the resolution is explicitly unresolved,
    which is what makes the second assertion non-vacuous.
    """
    archive_root = tmp_path / "archive"
    root = tmp_path / "live" / "chatgpt-export"
    root.mkdir(parents=True)
    conversations = root / "conversations.json"
    conversations.write_bytes(_chatgpt_export_document())

    before = _chatgpt_replay(archive_root, blob_store, conversations)
    assert before.attachments[0].name != "diagram.png"  # type: ignore[attr-defined]
    assert before.attachments[0].precomputed_blob is None  # type: ignore[attr-defined]

    asset, library, names, _conversations = await _acquire_chatgpt_export(archive_root, root)
    for path in (asset, library, names, conversations):
        path.unlink()

    after = _chatgpt_replay(archive_root, blob_store, conversations)
    assert after.attachments[0].name == "diagram.png"  # type: ignore[attr-defined]
    assert after.attachments[0].precomputed_blob is not None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_retained_replay_archives_every_duplicate_asset_rendition(blob_store: BlobStore, tmp_path: Path) -> None:
    """Replay from retained evidence keys renditions as live discovery does.

    Anti-vacuity: keep only the first retained member per asset id and replay
    yields one attachment carrying the ``p0`` bytes while ``p1`` is never
    referenced.
    """
    from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec

    archive_root = tmp_path / "archive"
    root = tmp_path / "live" / "chatgpt-export"
    single_asset, library, names, conversations = _write_chatgpt_export(root)
    single_asset.unlink()
    renditions = {
        root / f"{_CHATGPT_ASSET_ID}-p0.png": b"\x89PNG page zero",
        root / f"{_CHATGPT_ASSET_ID}-p1.png": b"\x89PNG page one",
    }
    for path, payload in renditions.items():
        path.write_bytes(payload)
    from polylogue.sources.live import WatchSource

    await _acquire_evidence(
        archive_root,
        WatchSource(name="chatgpt", root=root, suffixes=(".json",)),
        [*renditions, library, names],
    )
    live_keys = set(
        ChatGPTAssemblySpec()
        .discover_sidecars([conversations], blob_store=BlobStore(tmp_path / "live-blobs"))
        .get("chatgpt_asset_blobs", {})
    )
    for path in (*renditions, library, names, conversations):
        path.unlink()

    replayed = _chatgpt_replay(archive_root, blob_store, conversations)
    attachments = list(replayed.attachments)  # type: ignore[attr-defined]
    assert len(live_keys) == 2
    assert len(attachments) == 2
    blobs = [attachment.precomputed_blob for attachment in attachments]
    assert all(blob is not None for blob in blobs)
    stored = BlobStore(archive_root / "blob")
    assert sorted(stored.read_all(blob[0]) for blob in blobs if blob is not None) == sorted(renditions.values())
    # The retained route names each rendition by the same member coordinate.
    assert {attachment.provider_attachment_id.rsplit("#", 1)[-1] for attachment in attachments} == {
        key.rsplit("#", 1)[-1] for key in live_keys
    }


def _write_chatgpt_zip(zip_path: Path, library_names: Sequence[str]) -> None:
    """A ChatGPT export ZIP whose ``library_files.json`` may repeat."""
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversations.json", _chatgpt_export_document())
        for index, name in enumerate(library_names):
            payload = json.dumps([{"file_id": _CHATGPT_ASSET_ID, "file_name": name}])
            if index == 0:
                archive.writestr("library_files.json", payload)
                continue
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("library_files.json", payload)


def _resolved_library_names(archive_root: Path, zip_path: Path) -> tuple[str | None, str | None]:
    """The asset name live assembly and retained replay each resolve."""
    import sqlite3

    from polylogue.archive.revision_authority import raw_receipt_order_sql
    from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
    from polylogue.sources.retained_assembly import retained_chatgpt_sidecars
    from polylogue.storage.sqlite.archive_tiers.source_write import read_raw_captured_zip_coordinate
    from tests.infra.retained_jsonl import prepared_source_fixture

    live_index = ChatGPTAssemblySpec().discover_sidecars([zip_path])["chatgpt_asset_index"]
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT r.raw_id FROM raw_sessions r WHERE r.source_path=? ORDER BY "
            + raw_receipt_order_sql("r")
            + " DESC LIMIT 1",
            (f"{zip_path}:conversations.json",),
        ).fetchone()
        assert row is not None
        coordinate = read_raw_captured_zip_coordinate(conn, str(row[0]))
        assert coordinate is not None
        with prepared_source_fixture(archive_root) as source_read:
            retained = retained_chatgpt_sidecars(
                source_read,
                BlobStore(archive_root / "blob"),
                session_source_path=f"{zip_path}:conversations.json",
                captured_zip_coordinate=coordinate,
            )
    finally:
        conn.close()
    retained_index = retained["chatgpt_asset_index"]
    try:
        live = live_index.resolve_dat(_CHATGPT_ASSET_ID)
        replayed = retained_index.resolve_dat(_CHATGPT_ASSET_ID)
        return (live.name if live else None, replayed.name if replayed else None)
    finally:
        retained_index.close()
        live_index.close()


@pytest.mark.asyncio
async def test_retained_zip_sidecar_binds_the_member_live_assembly_binds(blob_store: BlobStore, tmp_path: Path) -> None:
    """Replay binds the first duplicate member of one acquisition, as live does.

    A later acquisition of the export still supersedes the earlier one.

    Anti-vacuity: rank the duplicates by receipt order alone and replay binds
    the later-received ``second.png`` member while live assembly binds the
    first in central-directory order.
    """
    from polylogue.sources.live import WatchSource

    archive_root = tmp_path / "archive"
    root = tmp_path / "live" / "exports"
    root.mkdir(parents=True)
    zip_path = root / "chatgpt-export.zip"
    source = WatchSource(name="chatgpt", root=root, suffixes=(".json", ".zip"))

    _write_chatgpt_zip(zip_path, ["first.png", "second.png"])
    await _acquire_evidence(archive_root, source, [zip_path])
    assert run_off_event_loop(lambda: _resolved_library_names(archive_root, zip_path)) == ("first.png", "first.png")

    _write_chatgpt_zip(zip_path, ["reexported.png"])
    await _acquire_evidence(archive_root, source, [zip_path])
    assert run_off_event_loop(lambda: _resolved_library_names(archive_root, zip_path)) == (
        "reexported.png",
        "reexported.png",
    )


@pytest.mark.asyncio
async def test_source_walk_stamps_one_acquisition_time_per_zip_pass(tmp_path: Path) -> None:
    """Every member of one ZIP pass carries the pass's acquisition time.

    The timestamp describes acquisition only. Exact completed Source item
    membership establishes duplicate-member grouping independently of clocks.

    Anti-vacuity: stamp each record with its own clock in
    ``iter_raw_record_stream`` and the two ``library_files.json`` members get
    different acquisition times.
    """
    from polylogue.config import Source
    from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream

    root = tmp_path / "inbox"
    root.mkdir()
    first_zip = root / "chatgpt-export.zip"
    second_zip = root / "chatgpt-export-later.zip"
    _write_chatgpt_zip(first_zip, ["first.png", "second.png"])
    _write_chatgpt_zip(second_zip, ["later.png"])

    records = [
        record
        async for record in iter_raw_record_stream(
            Source(name="chatgpt", path=root), blob_store=BlobStore(tmp_path / "archive" / "blob")
        )
    ]

    by_container: dict[str, set[str | None]] = {}
    for record in records:
        container = record.source_path.split(".zip:", 1)[0]
        by_container.setdefault(container, set()).add(record.acquired_at)
    first_members = [record for record in records if record.source_path.startswith(f"{first_zip}:")]
    assert len([record for record in first_members if record.source_path.endswith("library_files.json")]) == 2
    assert all(len(times) == 1 for times in by_container.values()), by_container
    assert len(by_container) == 2
