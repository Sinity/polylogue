from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.excision import (
    resolve_session_excision_target_from_root,
)
from tests.infra.live_ingest import write_index_session
from tests.infra.retained_jsonl import prepared_source_fixture


def _converge_work_events(root: Path, raw_ids: tuple[str, ...], *, retained_replay: bool = False) -> None:
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def converge() -> None:
        async with prepared_live_convergence_owner(root) as owner:
            receipts = (
                (await owner.replay_retained_raw_ids(raw_ids)).require_complete()
                if retained_replay
                else (await owner.ingest_retained_raw_ids(raw_ids)).require_complete()
            )
            assert receipts

    asyncio.run(converge())


def test_agent_work_event_uses_append_ingest_and_is_idempotent(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="work-event-session",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="start")],
                git_branch="feature/work-events",
                git_repository_url="https://example.invalid/repo.git",
                git_commit_hash="a" * 40,
                pending_drafts=[{"text": "draft", "role": "user"}],
            ),
        )
        admitted = archive.admit_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg", "evidence_refs": ["message:m1"]},
            event_id="evt-1",
            summary="searched the source tree",
        )
        archive.admit_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg", "evidence_refs": ["message:m1"]},
            event_id="evt-1",
            summary="searched the source tree",
        )
        with pytest.raises(ValueError, match="work event type"):
            archive.admit_work_event(
                session_id=session_id,
                event_type="unknown",
                payload={},
                event_id="evt-invalid",
                summary="invalid",
            )
        with pytest.raises(ValueError, match="work event id"):
            archive.admit_work_event(
                session_id=session_id,
                event_type="tool_run",
                payload={},
                event_id=" ",
                summary="invalid",
            )
        other_session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="other-work-event-session",
                messages=[ParsedMessage(provider_message_id="m2", role=Role.USER, text="continue")],
            ),
        )
        other_admitted = archive.admit_work_event(
            session_id=other_session_id,
            event_type="decision",
            payload={"decision": "continue"},
            event_id="evt-1",
            summary="continued work",
        )
    _converge_work_events(tmp_path, (str(admitted["raw_id"]), str(other_admitted["raw_id"])))
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        rows = archive._conn.execute(
            "SELECT event_type, json_extract(payload_json, '$.summary'), payload_json "
            "FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [("tool_run", "searched the source tree")]
        # Anti-vacuity: treating the acquired event as a skeletal replacement
        # clears the existing session metadata during prepared publication.
        preserved = archive._conn.execute(
            "SELECT git_branch, git_repository_url, commit_hash, pending_drafts_json "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert preserved[0:3] == (
            "feature/work-events",
            "https://example.invalid/repo.git",
            "a" * 40,
        )
        assert preserved[3] == '[{"role":"user","text":"draft"}]'
        # Anti-vacuity: changing the retained-raw replay discriminator or
        # envelope parser to provider-only parsing makes this assertion fail.
        raw_id = str(admitted["raw_id"])
        with prepared_source_fixture(tmp_path) as source_read:
            replayed = parse_retained_raw_sessions(source_read, raw_id)
        assert len(replayed) == 1
        assert replayed[0].session_events[0].event_type == "tool_run"
        assert replayed[0].session_events[0].payload["event_id"] == "evt-1"
        assert (
            archive._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            == 1
        )

        assert (
            archive._conn.execute(
                "SELECT COUNT(*) FROM session_events WHERE session_id = ?", (other_session_id,)
            ).fetchone()[0]
            == 1
        )


_HEADER_COLUMNS = (
    "title",
    "session_kind",
    "display_name",
    "active_leaf_message_id",
    "pending_drafts_json",
    "git_branch",
    "git_repository_url",
    "commit_hash",
    "instructions_text",
    "reported_cost_usd",
    "provider_project_ref",
    "created_at_ms",
    "updated_at_ms",
)


def _session_state(
    archive: ArchiveStore, session_id: str
) -> tuple[tuple[object, ...], tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    conn = archive._conn
    header = conn.execute(f"SELECT {', '.join(_HEADER_COLUMNS)} FROM sessions WHERE session_id = ?", (session_id,))
    messages = conn.execute(
        "SELECT message_id, is_active_leaf FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
    )
    working_dirs = conn.execute(
        "SELECT path FROM session_working_dirs WHERE session_id = ? ORDER BY position", (session_id,)
    )
    return (tuple(header.fetchone()), tuple(map(tuple, messages)), tuple(map(tuple, working_dirs)))


@pytest.mark.parametrize("replay_route", ["acquired-resume", "retained-replay", "cold-writer"])
def test_retained_work_event_replay_keeps_the_reconstructed_session(tmp_path: Path, replay_route: str) -> None:
    """Live append and replay of a retained work event change only the events.

    Anti-vacuity: write the replayed skeleton as an ordinary session and the
    upsert clears the git metadata, drafts, project identity and active leaf
    (and a same-raw full replay replaces the transcript); the header snapshot
    and message rows then differ from the reconstructed session's.
    """
    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="replayed-work-event-session",
                title="Reconstructed title",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-02T00:00:00+00:00",
                messages=[
                    ParsedMessage(provider_message_id="m1", role=Role.USER, text="start"),
                    ParsedMessage(provider_message_id="m2", role=Role.ASSISTANT, text="reply"),
                ],
                active_leaf_message_provider_id="m2",
                git_branch="feature/replay",
                git_repository_url="https://example.invalid/replay.git",
                git_commit_hash="b" * 40,
                pending_drafts=[{"text": "draft", "role": "user"}],
                provider_project_ref="project:replay",
                display_name="Replay display",
                instructions_text="be careful",
                reported_cost_usd=1.25,
                working_directories=["/work/replay"],
            ),
        )
        reconstructed = _session_state(archive, session_id)
        assert reconstructed[0][3] is not None

        admitted = archive.admit_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg"},
            event_id="evt-replay",
            summary="searched",
        )
    raw_id = str(admitted["raw_id"])
    _converge_work_events(tmp_path, (raw_id,))
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert _session_state(archive, session_id) == reconstructed
        with prepared_source_fixture(tmp_path) as source_read:
            (replayed,) = parse_retained_raw_sessions(source_read, raw_id)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive._conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))
        archive._conn.execute("UPDATE sessions SET content_hash = zeroblob(32) WHERE session_id = ?", (session_id,))
        archive._conn.commit()
        if replay_route == "cold-writer":
            # The cold-build writer shortcut: the event raw reaches a session
            # this generation already holds.
            from tests.infra.index_writer import write_fixture_index_session

            with archive._owned_read_connection(archive.source_db_path) as source_conn:
                write_fixture_index_session(
                    archive._conn,
                    replayed,
                    raw_id=raw_id,
                    source_conn=source_conn,
                    merge_append=True,
                    fresh_build=True,
                )
    if replay_route != "cold-writer":
        _converge_work_events(tmp_path, (str(raw_id),), retained_replay=replay_route == "retained-replay")
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert _session_state(archive, session_id) == reconstructed
        events = archive._conn.execute(
            "SELECT event_type, json_extract(payload_json, '$.event_id') FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        assert [tuple(row) for row in events] == [("tool_run", "evt-replay")]


def test_session_excision_reaches_its_retained_work_events(tmp_path: Path) -> None:
    """Excising a session also removes the work events retained for it.

    Anti-vacuity: each work event is its own logical source, so drop the
    work-event seed from excision resolution and neither the transcript's
    revision closure nor ``sessions.raw_id`` reaches the event raw.
    """

    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_ids = [
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="start")],
                ),
            )
            for native_id in ("excised-work-event-session", "kept-work-event-session")
        ]
        event_raw_by_session = {}
        for session_id in session_ids:
            admitted = archive.admit_work_event(
                session_id=session_id,
                event_type="decision",
                payload={"decision": "continue"},
                event_id="evt-excise",
                summary="private summary",
            )
            event_raw_by_session[session_id] = str(admitted["raw_id"])

    _converge_work_events(tmp_path, tuple(event_raw_by_session.values()))
    target = resolve_session_excision_target_from_root(tmp_path, session_ids[0])

    raw_ids = {raw.raw_id for raw in target.raw_targets}
    assert event_raw_by_session[session_ids[0]] in raw_ids
    assert event_raw_by_session[session_ids[1]] not in raw_ids


def test_public_work_event_uses_resident_owner_and_preserves_duplicate_result(tmp_path: Path) -> None:
    import asyncio

    from polylogue.api import Polylogue
    from tests.infra.daemon_operations import daemon_serving_archive

    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="public-work-event",
                messages=[ParsedMessage(provider_message_id="original", role=Role.USER, text="original message")],
            ),
        )

    async def record() -> None:
        async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as api:
            first = await api.record_work_event(
                session_id, event_id="public-event", event_type="decision", summary="retain the original message"
            )
            repeated = await api.record_work_event(
                session_id, event_id="public-event", event_type="decision", summary="retain the original message"
            )
            assert first["event_id"] == repeated["event_id"] == "public-event"
            assert first["session_id"] == repeated["session_id"] == session_id
            assert first["content_changed"] is True
            assert repeated["content_changed"] is False

    with daemon_serving_archive(tmp_path, session_derivation=True) as stack:
        asyncio.run(record())
        from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

        refused = stack.client.operation(
            "mutation.facade.record_work_event",
            {
                "session_id": session_id,
                "event_id": "schema-refused-event",
                "event_type": "decision",
                "summary": "must not acquire with stale schema authority",
                "payload": {},
            },
            archive_root=str(tmp_path),
            index_schema_version=INDEX_SCHEMA_VERSION + 1,
        )
        assert refused is not None and refused["outcome"] == "failed"
        assert refused["error"]["detail"] == "schema_version_mismatch"
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert (
            archive.source_connection.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE logical_source_key LIKE 'agent-work-event:%'"
            ).fetchone()[0]
            == 1
        )
        index_connection = archive.index_connection
        assert index_connection is not None
        assert (
            index_connection.execute(
                "SELECT COUNT(*) FROM session_events WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            == 1
        )
        stored = archive.read_session(session_id)
        assert [
            (message.native_id, message.role, tuple(block.text for block in message.blocks))
            for message in stored.messages
        ] == [("original", "user", ("original message",))]


def test_empty_message_work_event_does_not_read_transcript_occurrences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import write

    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="event-occurrences",
                messages=[ParsedMessage(provider_message_id="original", role=Role.USER, text="original transcript")],
            ),
        )
        admitted = archive.admit_work_event(
            session_id=session_id,
            event_type="decision",
            event_id="occurrence-event",
            summary="retain transcript",
            payload={},
        )

    def reject_transcript_census(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise AssertionError("empty incoming messages cannot consume stored occurrence offsets")

    monkeypatch.setattr(write, "_stored_content_occurrences", reject_transcript_census)
    _converge_work_events(tmp_path, (str(admitted["raw_id"]),))
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        stored = archive.read_session(session_id)
        assert [
            (message.native_id, message.role, tuple(block.text for block in message.blocks))
            for message in stored.messages
        ] == [("original", "user", ("original transcript",))]
        index_connection = archive.index_connection
        assert index_connection is not None
        assert (
            index_connection.execute(
                "SELECT COUNT(*) FROM session_events WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            == 1
        )
