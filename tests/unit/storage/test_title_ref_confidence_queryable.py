"""Regression coverage for polylogue-ih67: a title provenance ref + confidence field.

``TitleSource`` (``sessions.title_source``) only records which coarse
strategy won (origin/heuristic/...). AC#5 of polylogue-ih67 additionally
wants a *specific* reference to the exact evidence row that produced a
title, plus a 0..1 confidence signal for that resolution. This test proves
``title_ref``/``title_confidence`` survive the full
write -> storage-summary/envelope -> domain-model -> surface-payload chain,
the same way ``test_title_source_queryable.py`` proved it for
``title_source``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, Role, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.storage_records import db_setup

_TITLE_REF = "codex-history:codex-tr-native"
_TITLE_CONFIDENCE = 0.9


def _write_codex_session(db_path: Path, *, native_id: str, title: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            title=title,
            title_source=TitleSource.ORIGIN,
            title_ref=_TITLE_REF,
            title_confidence=_TITLE_CONFIDENCE,
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


def test_archive_store_summary_reads_expose_title_ref_confidence(tmp_path: Path) -> None:
    """``ArchiveStore.read_summary``/``list_summaries`` select ``title_ref``/``title_confidence``."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass  # bootstrap the schema, then reopen for a plain-connection write below.

    _write_codex_session(db_path, native_id="codex-tr-1", title="Fix the flaky test")

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("codex-tr-1")
        summary = archive.read_summary(session_id)
        assert summary.title_ref == _TITLE_REF
        assert summary.title_confidence == pytest.approx(_TITLE_CONFIDENCE)

        listed = archive.list_summaries(origin="codex-session", limit=10, offset=0)
        matching = [s for s in listed if s.session_id == session_id]
        assert [s.title_ref for s in matching] == [_TITLE_REF]
        assert matching[0].title_confidence == pytest.approx(_TITLE_CONFIDENCE)


@pytest.mark.asyncio
async def test_session_filter_summary_exposes_title_ref_confidence(workspace_env: dict[str, Path]) -> None:
    """``SessionFilter.list_summaries()`` yields a domain ``SessionSummary`` with ref/confidence set."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-tr-2", title="Deploy the daemon fix")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()
    assert len(summaries) == 1
    assert summaries[0].title_ref == _TITLE_REF
    assert summaries[0].title_confidence == pytest.approx(_TITLE_CONFIDENCE)


@pytest.mark.asyncio
async def test_session_filter_full_session_exposes_title_ref_confidence(workspace_env: dict[str, Path]) -> None:
    """A full ``Session`` read (not just the summary) also carries ref/confidence."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-tr-3", title="Investigate the ingest stall")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    assert len(sessions) == 1
    assert sessions[0].title_ref == _TITLE_REF
    assert sessions[0].title_confidence == pytest.approx(_TITLE_CONFIDENCE)


def test_session_list_row_payload_carries_title_ref_confidence() -> None:
    """The CLI/MCP row payload surfaces ``title_ref``/``title_confidence``."""
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId
    from polylogue.surfaces.payloads import session_list_envelope_from_domain, session_summary_envelope_from_domain

    session = Session(
        id=SessionId("codex-session:codex-tr-4"),
        origin=Origin.CODEX_SESSION,
        title="Ship the release",
        title_source=TitleSource.HEURISTIC,
        title_ref="message:codex-session:codex-tr-4:m1",
        title_confidence=0.5,
        messages=MessageCollection(messages=[]),
    )
    row = session_list_envelope_from_domain(session)
    assert row.title_ref == "message:codex-session:codex-tr-4:m1"
    assert row.title_confidence == pytest.approx(0.5)
    summary_payload = session_summary_envelope_from_domain(session)
    assert summary_payload.title_ref == "message:codex-session:codex-tr-4:m1"
    assert summary_payload.title_confidence == pytest.approx(0.5)


def test_assembly_codex_sets_ref_and_confidence_per_resolution_lane() -> None:
    """Each Codex title-resolution lane stamps a distinct ref + confidence (not just title_source)."""
    from polylogue.sources.assembly_codex import CodexAssemblySpec

    spec = CodexAssemblySpec()
    cid = "native-1"

    thread_name_result = spec.enrich_session(
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=cid,
            title=cid,
            messages=[],
        ),
        {"thread_names": {cid: "Thread Name Title"}},
    )
    assert thread_name_result.title_source == TitleSource.ORIGIN
    assert thread_name_result.title_ref == f"codex-thread-name:{cid}"
    assert thread_name_result.title_confidence == pytest.approx(1.0)

    history_result = spec.enrich_session(
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=cid,
            title=cid,
            messages=[],
        ),
        {"history_titles": {cid: "History title"}},
    )
    assert history_result.title_ref == f"codex-history:{cid}"
    assert history_result.title_confidence == pytest.approx(0.9)

    state_db_result = spec.enrich_session(
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=cid,
            title=cid,
            messages=[],
        ),
        {"state_titles": {cid: "State DB title"}},
    )
    assert state_db_result.title_ref == f"codex-state-db:{cid}"
    assert state_db_result.title_confidence == pytest.approx(0.75)

    message_result = spec.enrich_session(
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=cid,
            title=cid,
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.USER,
                    text="please fix the thing",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                ),
            ],
        ),
        {},
    )
    assert message_result.title_source == TitleSource.HEURISTIC
    assert message_result.title_ref == "message:msg-1"
    assert message_result.title_confidence == pytest.approx(0.5)
