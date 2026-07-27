"""Regression coverage for polylogue-ih67: ``title_source`` must be queryable.

``sessions.title_source`` was already written on ingest (Codex/Gemini/Claude
assembly stamp ``TitleSource.ORIGIN``/``HEURISTIC`` on the parsed session) and
even read back into the full-session envelope (``read_archive_session_
envelope``), but it was silently dropped everywhere else on the read path:

- ``ArchiveStore.read_summary``/``list_summaries`` never selected the column,
  so ``ArchiveSessionSummary`` had no ``title_source`` field at all.
- ``_session_to_session``/``_summary_to_domain`` (``archive/query/archive_
  execution.py`` and the equivalent helpers in ``api/archive.py``) never
  copied ``title_source`` from the storage row onto the ``Session``/
  ``SessionSummary`` domain models, even for the full-envelope path where the
  storage layer already had the value.
- ``SessionListRowPayload``/``SessionSummaryPayload`` (CLI/MCP/API output)
  never exposed it.

So title provenance was persisted but not queryable anywhere above the raw
SQL row — AC#5 of polylogue-ih67. This test proves the value survives the
full write -> storage-summary -> domain-model -> surface-payload chain.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.core.enums import BlockType, Provider, Role, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.storage_records import db_setup


def _write_codex_session(db_path: Path, *, native_id: str, title: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            title=title,
            title_source=TitleSource.ORIGIN,
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="hi",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                ),
            ],
        )
        write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


def test_archive_store_summary_reads_expose_title_source(tmp_path: Path) -> None:
    """``ArchiveStore.read_summary``/``list_summaries`` select ``title_source``."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass  # bootstrap the schema, then reopen for a plain-connection write below.

    _write_codex_session(db_path, native_id="codex-ts-1", title="Fix the flaky test")

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("codex-ts-1")
        summary = archive.read_summary(session_id)
        assert summary.title_source == "origin"

        listed = archive.list_summaries(origin="codex-session", limit=10, offset=0)
        assert [s.title_source for s in listed if s.session_id == session_id] == ["origin"]


@pytest.mark.asyncio
async def test_session_filter_summary_exposes_title_source(workspace_env: dict[str, Path]) -> None:
    """``SessionFilter.list_summaries()`` yields a domain ``SessionSummary`` with ``title_source`` set."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-ts-2", title="Deploy the daemon fix")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()
    assert len(summaries) == 1
    assert summaries[0].title_source == TitleSource.ORIGIN


@pytest.mark.asyncio
async def test_session_filter_full_session_exposes_title_source(workspace_env: dict[str, Path]) -> None:
    """A full ``Session`` read (not just the summary) also carries ``title_source``."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-ts-3", title="Investigate the ingest stall")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    assert len(sessions) == 1
    assert sessions[0].title_source == TitleSource.ORIGIN


def test_session_list_row_payload_carries_title_source() -> None:
    """The CLI/MCP row payload surfaces ``title_source`` as its plain string value."""
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId
    from polylogue.surfaces.payloads import SessionListRowPayload, SessionSummaryPayload

    session = Session(
        id=SessionId("codex-session:codex-ts-4"),
        origin=Origin.CODEX_SESSION,
        title="Ship the release",
        title_source=TitleSource.HEURISTIC,
        messages=MessageCollection(messages=[]),
    )
    row = SessionListRowPayload.from_session(session)
    assert row.title_source == "heuristic"
    summary_payload = SessionSummaryPayload.from_session(session)
    assert summary_payload.title_source == "heuristic"
