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
        raw_id = archive._conn.execute(
            "SELECT raw_id FROM raw_sessions WHERE source_path LIKE 'agent-work-event:%' LIMIT 1"
        ).fetchone()[0]
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
