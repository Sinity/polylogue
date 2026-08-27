from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def test_agent_work_event_uses_append_ingest_and_is_idempotent(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path, initialize=True, read_only=False) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="work-event-session",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="start")],
            )
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
            "SELECT event_type, summary, payload_json FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [("tool_run", "searched the source tree")]
        assert (
            archive._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            == 1
        )

        other_session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="other-work-event-session",
                messages=[ParsedMessage(provider_message_id="m2", role=Role.USER, text="continue")],
            )
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
