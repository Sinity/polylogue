"""Provider assembly values in retained parsing come from acquired evidence."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Provider, Role, TitleSource
from polylogue.sources.live import WatchSource
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.retained_replay import publish_retained_payload


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


async def _retained_session(
    archive_root: Path,
    provider: Provider,
    content: bytes,
    source_path: str,
) -> Session:
    _raw_id, session_ids = await publish_retained_payload(
        archive_root,
        provider=provider,
        payload=content,
        source_path=source_path,
        acquired_at_ms=1,
    )
    assert session_ids, "retained payload must materialize a session"
    repository = SessionRepository(
        backend=SQLiteBackend(db_path=archive_root / "index.db"),
        archive_root=archive_root,
    )
    try:
        session = await repository.get(session_ids[0])
        assert session is not None
        return session
    finally:
        await repository.close()


def _retained_session_sync(
    archive_root: Path,
    provider: Provider,
    content: bytes,
    source_path: str,
) -> Session:
    return run_off_event_loop(lambda: asyncio.run(_retained_session(archive_root, provider, content, source_path)))


def _title_fields(session: Session) -> tuple[str | None, str | None]:
    title = session.title
    source = session.title_source
    return title, str(source) if source is not None else None


def test_retained_replay_classifies_complete_checkpoint_stream_with_declared_path(tmp_path: Path) -> None:
    """A late real turn wins over a long prefix of non-session checkpoint rows."""
    rows: list[dict[str, object]] = [{"type": "file-history-snapshot"}] * 65
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
    session = _retained_session_sync(
        tmp_path / "archive",
        Provider.CLAUDE_CODE,
        content,
        str(tmp_path / ".claude" / "projects" / "project" / "session.jsonl"),
    )
    assert list(session.messages)[0].text == "preserve the actual late turn"


def test_retained_replay_applies_thread_name_from_acquired_evidence(tmp_path: Path) -> None:
    """The retained root index titles a Codex session after source files disappear."""
    from polylogue.sources.live import WatchSource
    from tests.infra.live_batch import prepared_live_batch_processor

    async def scenario() -> tuple[str | None, str | None]:
        archive_root = tmp_path / "archive"
        session_id = "aaaa1111-2222-3333-4444-555566667777"
        content = _codex_stream(session_id, "please fix the ingest bug")
        rollout = _codex_runtime_root(tmp_path, session_id, content)
        root = rollout.parents[2]
        index_path = root / "session_index.jsonl"
        index_path.write_text(json.dumps({"id": session_id, "thread_name": "Ingest bug hunt"}) + "\n")
        import polylogue.sources.live.watcher as live_watcher

        async with prepared_live_batch_processor(
            archive_root,
            (WatchSource(name="codex-state", root=root),),
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        ) as processor:
            await processor.ingest_files([index_path], emit_event=False)
        session = await _retained_session(archive_root, Provider.CODEX, content, str(rollout))
        return _title_fields(session)

    assert run_off_event_loop(lambda: asyncio.run(scenario())) == ("Ingest bug hunt", TitleSource.ORIGIN.value)


def test_retained_replay_uses_history_title_without_thread_name(tmp_path: Path) -> None:
    """Without a thread name, retained authored history supplies the title."""
    from polylogue.sources.live import WatchSource
    from tests.infra.live_batch import prepared_live_batch_processor

    async def scenario() -> tuple[str | None, str | None]:
        archive_root = tmp_path / "archive"
        session_id = "bbbb1111-2222-3333-4444-555566667777"
        content = _codex_stream(session_id, "opening prompt typed by the operator")
        rollout = _codex_runtime_root(tmp_path, session_id, content)
        history = rollout.parents[2] / "history.jsonl"
        history.write_text(json.dumps({"session_id": session_id, "ts": 1, "text": "Wire the Hermes bridge"}) + "\n")
        import polylogue.sources.live.watcher as live_watcher

        async with prepared_live_batch_processor(
            archive_root,
            (WatchSource(name="codex-state", root=rollout.parents[2]),),
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        ) as processor:
            await processor.ingest_files([history], emit_event=False)
        session = await _retained_session(archive_root, Provider.CODEX, content, str(rollout))
        return _title_fields(session)

    assert run_off_event_loop(lambda: asyncio.run(scenario())) == (
        "Wire the Hermes bridge",
        TitleSource.ORIGIN.value,
    )


def test_retained_replay_uses_message_fallback_when_no_sidecars_exist(tmp_path: Path) -> None:
    session_id = "cccc1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "refactor the daemon status loop")
    title, title_source = _title_fields(
        _retained_session_sync(
            tmp_path / "archive",
            Provider.CODEX,
            content,
            str(tmp_path / "gone" / "sessions" / f"rollout-{session_id}.jsonl"),
        )
    )
    assert title == "refactor the daemon status loop"
    assert title_source == TitleSource.HEURISTIC.value


def test_retained_replay_does_not_discover_sidecars_from_ambient_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only acquired sidecar evidence can affect replay, even if a live tree exists."""
    from polylogue.sources import assembly_codex

    session_id = "eeee1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "please fix the ingest bug")
    rollout = _codex_runtime_root(tmp_path, session_id, content)
    (rollout.parents[2] / "session_index.jsonl").write_text(
        json.dumps({"id": session_id, "thread_name": "Ambient title must be ignored"}) + "\n",
        encoding="utf-8",
    )

    def reject_discovery(self: object, paths: object) -> object:
        raise AssertionError("retained replay attempted ambient sidecar discovery")

    monkeypatch.setattr(assembly_codex.CodexAssemblySpec, "discover_sidecars", reject_discovery)
    title, title_source = _title_fields(
        _retained_session_sync(tmp_path / "archive", Provider.CODEX, content, str(rollout))
    )
    assert title == "please fix the ingest bug"
    assert title_source == TitleSource.HEURISTIC.value


# ---------------------------------------------------------------------------
# polylogue-ximhz: provider metadata and attachment joins resolve from
# retained evidence, with the original files gone.
#
# Every case below acquires its evidence through the ordinary live route,
# unlinks every original path, then publishes the provider bytes through the
# retained replay owner. Nothing is hand-seeded into the source tier.
#
# Anti-vacuity: remove the ChatGPT map's artifact declarations while preserving
# its raw rows and membership census; the matching fidelity assertion fails.
# Reverting the retained assembly lookup makes the fidelity assertions fail.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained Codex root sidecars title one rollout without minting another.

    The daemon's default ``codex-state`` source admits the two exact root
    coordinates despite having no JSONL suffix intake, and nothing else of
    JSONL shape under the install root. Once those bytes and the rollout
    disappear, the retained replay owner must resolve the title from the
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
    parsed = await _retained_session(archive_root, Provider.CODEX, content, str(rollout))
    assert parsed.title == "Retained curated title"
    assert parsed.title_source is TitleSource.ORIGIN


@pytest.mark.asyncio
async def test_claude_index_and_history_resolve_with_the_original_tree_gone(tmp_path: Path) -> None:
    """The curated title and paste evidence come back from retained bytes."""
    from polylogue.sources.live import WatchSource

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
        ),
        [index_path],
    )
    await _acquire_evidence(
        archive_root,
        WatchSource(name="claude-code-history", root=claude_home),
        [history_path],
    )
    kinds = _retained_artifact_kinds(archive_root)
    assert kinds.get(str(index_path)) == "session_index"
    assert kinds.get(str(history_path)) == "prompt_history_log"

    content = _claude_transcript()
    for path in (index_path, history_path, transcript):
        path.unlink()

    parsed = await _retained_session(archive_root, Provider.CLAUDE_CODE, content, str(transcript))
    assert parsed.title == "Curated index title"
    user_message = next(message for message in parsed.messages if message.role == "user")
    assert user_message.has_paste
    with sqlite3.connect(archive_root / "index.db") as conn:
        paste_markers = conn.execute(
            "SELECT source_marker FROM paste_spans WHERE session_id = ? ORDER BY position",
            (str(parsed.id),),
        ).fetchall()
    assert [row[0] for row in paste_markers] == ["1"]
    assert str(parsed.title_source) == TitleSource.ORIGIN.value


def test_retained_claude_paste_output_sink_closes_when_consumer_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A yielded clone stays readable until downstream cancellation closes it."""
    from polylogue.sources import revision_backfill
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.sources.parsers.claude.history import HistoryEntry, HistoryPaste
    from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore

    source_store = SqliteMessageStore(tmp_path / "source.sqlite")
    source_sink = source_store.new_sink()
    source_sink.append(
        ParsedMessage(
            provider_message_id="user-1",
            role=Role.USER,
            text="first prompt",
            timestamp="2026-01-01T00:00:00Z",
        )
    )
    sealed = SqliteMessageSink(source_store.path, source_sink.session_ordinal, count=1)
    source_store.conn.commit()
    source = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-1",
        messages=[],
    ).model_copy(update={"messages": sealed})
    entry = HistoryEntry(
        display="first prompt",
        timestamp_ms=1767225600000,
        project=None,
        session_id="session-1",
        pastes=(HistoryPaste(paste_id="1", paste_type="text", content="pasted body", has_content=True),),
    )
    monkeypatch.setattr(
        revision_backfill,
        "_retained_enrichment_sidecar_data",
        lambda **_: {"history_paste_index": {"session-1": [entry]}},
    )
    iterator = revision_backfill.iter_enriched_sessions_from_retained_read(
        object(),  # type: ignore[arg-type]
        Provider.CLAUDE_CODE,
        "session.jsonl",
        [source],
        captured_zip_coordinate=None,
    )
    output_path: Path | None = None
    try:
        enriched = next(iterator)
        assert isinstance(enriched.messages, SqliteMessageSink)
        output_path = enriched.messages.path
        assert output_path != source_store.path
        assert output_path.exists()
        assert next(iter(enriched.messages)).paste_spans[0].source_marker == "1"
        assert next(iter(sealed)).paste_spans == []
    finally:
        iterator.close()
        source_store.close()
    assert output_path is not None
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_claude_retained_index_is_scoped_to_its_own_install(tmp_path: Path) -> None:
    """A second install's index never titles the first install's session."""
    from polylogue.sources.live import WatchSource

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
        ),
        [other_index],
    )
    assert _retained_artifact_kinds(archive_root).get(str(other_index)) == "session_index"

    mine = tmp_path / "live" / "install-a" / ".claude" / "projects" / "-realm-project-x"
    transcript = mine / f"{_CLAUDE_SESSION_ID}.jsonl"
    parsed = await _retained_session(archive_root, Provider.CLAUDE_CODE, _claude_transcript(), str(transcript))
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
        WatchSource(name="chatgpt", root=root, layout=export_drop_layout((".json",))),
        [asset, library, names],
    )
    return asset, library, names, conversations


def _chatgpt_resolution_event(parsed: Session) -> dict[str, object]:
    events = [event for event in parsed.session_events if event.event_type == "chatgpt_asset_resolution"]
    assert events, "expected one asset-resolution event"
    return dict(events[0].payload)


def _chatgpt_message_attachments(parsed: Session) -> list[Attachment]:
    """Return message-owned attachments through the ordinary read contract."""
    return [attachment for message in parsed.messages for attachment in message.attachments]


def _chatgpt_replay(archive_root: Path, conversations: Path) -> Session:
    return _retained_session_sync(archive_root, Provider.CHATGPT, _chatgpt_export_document(), str(conversations))


def _stored_chatgpt_asset_evidence(
    archive_root: Path, session: Session
) -> list[tuple[str, str | None, str, bytes | None]]:
    """Read durable attachment identity and acquisition facts for one session."""
    with sqlite3.connect(archive_root / "index.db") as conn:
        rows = conn.execute(
            """
            SELECT ids.native_id, attachments.display_name, attachments.acquisition_status, attachments.blob_hash
            FROM attachment_refs AS refs
            JOIN attachments USING (attachment_id)
            JOIN attachment_native_ids AS ids USING (ref_id)
            WHERE refs.session_id = ? AND ids.id_kind = 'attachment'
            ORDER BY ids.native_id
            """,
            (str(session.id),),
        ).fetchall()
    return [
        (
            str(row[0]),
            str(row[1]) if row[1] is not None else None,
            str(row[2]),
            bytes(row[3]) if row[3] is not None else None,
        )
        for row in rows
    ]


@pytest.mark.asyncio
async def test_chatgpt_asset_identity_and_bytes_resolve_with_the_export_gone(tmp_path: Path) -> None:
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

    parsed = _chatgpt_replay(archive_root, conversations)
    [attachment] = _chatgpt_message_attachments(parsed)
    assert attachment.name == "diagram.png"
    [(native_id, display_name, acquisition_status, blob_hash)] = _stored_chatgpt_asset_evidence(archive_root, parsed)
    assert native_id == _CHATGPT_ASSET_ID
    assert display_name == "diagram.png"
    assert acquisition_status == "acquired"
    assert blob_hash is not None
    # The payload is readable from the archive's own blob store, which is
    # where acquisition retained it -- not from the original export path.
    assert BlobStore(archive_root / "blob").read_all(blob_hash.hex()) == _CHATGPT_ASSET_BYTES
    assert _chatgpt_resolution_event(parsed)["blob_acquired"] is True


@pytest.mark.asyncio
async def test_dropping_the_retained_asset_map_loses_the_resolved_name(tmp_path: Path) -> None:
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
        # Remove the retained map declarations while preserving the raw rows
        # and their complete membership census. Deleting raw_sessions directly
        # would manufacture an invalid archive state with orphan memberships.
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE artifact_kind = 'export_asset_index'").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                """
            SELECT COUNT(*) FROM raw_session_memberships AS membership
            LEFT JOIN raw_sessions AS raw USING (raw_id)
            WHERE raw.raw_id IS NULL
            """
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()

    parsed = _chatgpt_replay(archive_root, conversations)
    [attachment] = _chatgpt_message_attachments(parsed)
    assert attachment.name != "diagram.png"


@pytest.mark.asyncio
async def test_the_same_asset_id_in_two_exports_never_cross_binds(tmp_path: Path) -> None:
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

    parsed = _chatgpt_replay(archive_root, conversations)
    [attachment] = _chatgpt_message_attachments(parsed)
    assert attachment.name != "diagram.png"
    [(_native_id, display_name, acquisition_status, blob_hash)] = _stored_chatgpt_asset_evidence(archive_root, parsed)
    assert display_name is None
    assert acquisition_status != "acquired"
    assert blob_hash is None


@pytest.mark.asyncio
async def test_a_late_asset_map_resolves_on_the_next_convergence(tmp_path: Path) -> None:
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

    before = _chatgpt_replay(archive_root, conversations)
    [before_attachment] = _chatgpt_message_attachments(before)
    assert before_attachment.name != "diagram.png"
    [before_asset] = _stored_chatgpt_asset_evidence(archive_root, before)
    assert before_asset[3] is None

    asset, library, names, _conversations = await _acquire_chatgpt_export(archive_root, root)
    for path in (asset, library, names, conversations):
        path.unlink()

    after = _chatgpt_replay(archive_root, conversations)
    [after_attachment] = _chatgpt_message_attachments(after)
    assert after_attachment.name == "diagram.png"
    [after_asset] = _stored_chatgpt_asset_evidence(archive_root, after)
    assert after_asset[3] is not None


@pytest.mark.asyncio
async def test_retained_replay_archives_every_duplicate_asset_rendition(tmp_path: Path) -> None:
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
        WatchSource(name="chatgpt", root=root, layout=export_drop_layout((".json",))),
        [*renditions, library, names],
    )
    live_keys = set(
        ChatGPTAssemblySpec()
        .discover_sidecars([conversations], blob_store=BlobStore(tmp_path / "live-blobs"))
        .get("chatgpt_asset_blobs", {})
    )
    for path in (*renditions, library, names, conversations):
        path.unlink()

    replayed = _chatgpt_replay(archive_root, conversations)
    attachments = _chatgpt_message_attachments(replayed)
    assert len(live_keys) == 2
    assert len(attachments) == 2
    asset_rows = _stored_chatgpt_asset_evidence(archive_root, replayed)
    assert len(asset_rows) == 2
    assert all(row[3] is not None for row in asset_rows)
    stored = BlobStore(archive_root / "blob")
    assert sorted(stored.read_all(row[3].hex()) for row in asset_rows if row[3] is not None) == sorted(
        renditions.values()
    )
    # The retained route names each rendition by the same member coordinate.
    assert {row[0].rsplit("#", 1)[-1] for row in asset_rows} == {key.rsplit("#", 1)[-1] for key in live_keys}


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
async def test_retained_zip_sidecar_binds_the_member_live_assembly_binds(tmp_path: Path) -> None:
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
    source = WatchSource(name="chatgpt", root=root, layout=export_drop_layout((".json", ".zip")))

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
