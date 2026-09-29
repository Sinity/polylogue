from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session


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
        archive.append_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg", "evidence_refs": ["message:m1"]},
            event_id="evt-1",
            summary="searched the source tree",
        )
        archive.append_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg", "evidence_refs": ["message:m1"]},
            event_id="evt-1",
            summary="searched the source tree",
        )
        with pytest.raises(ValueError, match="work event type"):
            archive.append_work_event(
                session_id=session_id,
                event_type="unknown",
                payload={},
                event_id="evt-invalid",
                summary="invalid",
            )
        with pytest.raises(ValueError, match="work event id"):
            archive.append_work_event(
                session_id=session_id,
                event_type="tool_run",
                payload={},
                event_id=" ",
                summary="invalid",
            )
        rows = archive._conn.execute(
            "SELECT event_type, json_extract(payload_json, '$.summary'), payload_json "
            "FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [("tool_run", "searched the source tree")]
        # Anti-vacuity: omitting any copied session field from append_work_event
        # lets the skeletal upsert clear durable session metadata.
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
        raw_id = (
            archive._ensure_source_conn()
            .execute("SELECT raw_id FROM raw_sessions WHERE source_path LIKE 'agent-work-event:%' LIMIT 1")
            .fetchone()[0]
        )
        replayed = parse_retained_raw_sessions(archive, raw_id)
        assert len(replayed) == 1
        assert replayed[0].session_events[0].event_type == "tool_run"
        assert replayed[0].session_events[0].payload["event_id"] == "evt-1"
        assert (
            archive._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            == 1
        )

        other_session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="other-work-event-session",
                messages=[ParsedMessage(provider_message_id="m2", role=Role.USER, text="continue")],
            ),
        )
        archive.append_work_event(
            session_id=other_session_id,
            event_type="decision",
            payload={"decision": "continue"},
            event_id="evt-1",
            summary="continued work",
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


@pytest.mark.parametrize("source_index", [-1, 0, None], ids=["append-replay", "full-replay", "cold-build-write"])
def test_retained_work_event_replay_keeps_the_reconstructed_session(tmp_path: Path, source_index: int | None) -> None:
    """Live append and replay of a retained work event change only the events.

    Anti-vacuity: write the replayed skeleton as an ordinary session and the
    upsert clears the git metadata, drafts, project identity and active leaf
    (and a same-raw full replay replaces the transcript); the header snapshot
    and message rows then differ from the reconstructed session's.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import _index_parsed_for_retained_raw

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

        archive.append_work_event(
            session_id=session_id,
            event_type="tool_run",
            payload={"tool_name": "rg"},
            event_id="evt-replay",
            summary="searched",
        )
        assert _session_state(archive, session_id) == reconstructed

        (raw_id,) = (
            archive._ensure_source_conn()
            .execute("SELECT raw_id FROM raw_sessions WHERE source_path LIKE 'agent-work-event:%'")
            .fetchone()
        )
        (replayed,) = parse_retained_raw_sessions(archive, raw_id)
        archive._conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))
        archive._conn.execute("UPDATE sessions SET content_hash = zeroblob(32) WHERE session_id = ?", (session_id,))
        archive._conn.commit()
        if source_index is None:
            # The cold-build writer shortcut: the event raw reaches a session
            # this generation already holds.
            from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

            write_parsed_session_to_archive(archive._conn, replayed, raw_id=raw_id, fresh_build=True)
        else:
            _index_parsed_for_retained_raw(
                archive,
                replayed,
                raw_id=raw_id,
                source_index=source_index,
                stage_timings_s=None,
                stage_timing_prefix="replay",
                manage_transaction=True,
                preacquired_attachment_blobs={},
                finalize_raw_parse=False,
                revision_authoritative=True,
            )

        assert _session_state(archive, session_id) == reconstructed
        events = archive._conn.execute(
            "SELECT event_type, json_extract(payload_json, '$.event_id') FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        assert [tuple(row) for row in events] == [("tool_run", "evt-replay")]
