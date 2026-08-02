"""Regression coverage for polylogue-cuxz.8: ``stop_reason`` must survive the
bulk query-execution path, not just the single-session
``repository.get()``/``message_from_record`` path.

``messages.stop_reason`` was written on ingest (schema v46) and read back
into the full-session envelope (``read_archive_session_envelope`` and its
sibling ``_fetch_message_window``/``_row_to_archive_message`` helpers), and
``storage/hydrators.py::message_from_record`` already threaded it onto the
domain ``Message`` for the single-session ``repository.get()`` path
(polylogue-2qx.4). But ``ArchiveMessageRow``/``ArchiveBlockRow`` -- the row
types shared by *every* archive-tier read helper, including the bulk
``find``/``sessions where ...`` query path in
``archive/query/archive_execution.py`` -- never carried the field at all, so
``SessionFilter(...).list()`` (the production route behind the CLI/MCP query
grammar) silently dropped it. This proves the value survives the full
write -> storage-row -> domain-``Message`` chain for that path too.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.storage_records import db_setup


def _write_session_with_stop_reason(db_path: Path, *, native_id: str, stop_reason: str | None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id=native_id,
            title="Refused turn",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="Do something unsafe.",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Do something unsafe.")],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.ASSISTANT,
                    text="I can't help with that.",
                    position=1,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="I can't help with that.")],
                    stop_reason=stop_reason,
                ),
            ],
        )
        write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


def test_archive_session_envelope_reads_stop_reason(tmp_path: Path) -> None:
    """The single-session envelope read (``read_archive_session_envelope``) exposes ``stop_reason``."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass

    _write_session_with_stop_reason(db_path, native_id="claude-stop-1", stop_reason="refusal")

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("claude-stop-1")
        envelope = archive.read_session(session_id)
        assistant_messages = [m for m in envelope.messages if m.role == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].stop_reason == "refusal"


@pytest.mark.asyncio
async def test_session_filter_query_path_exposes_stop_reason(workspace_env: dict[str, Path]) -> None:
    """``SessionFilter.list()`` -- the production route behind the CLI/MCP
    query grammar (``find``/``sessions where ...``) -- must also expose
    ``Message.stop_reason``. Before threading ``stop_reason`` through
    ``ArchiveMessageRow`` (this bead), ``archive_execution.py``'s
    ``_message_to_domain`` silently dropped it even though the single-session
    ``repository.get()`` path already carried it.
    """
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_session_with_stop_reason(db_path, native_id="claude-stop-2", stop_reason="max_tokens")

    plan = SessionQueryPlan(origins=("claude-code-session",), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    assert len(sessions) == 1
    assistant_messages = [m for m in sessions[0].messages if m.role == Role.ASSISTANT]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].stop_reason == "max_tokens"
