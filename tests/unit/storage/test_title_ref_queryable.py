"""Regression coverage for polylogue-ih67 AC#5: the title provenance ref.

``TitleSource`` (``sessions.title_source``) only records which coarse
strategy won (origin/heuristic/...); ``title_ref`` names the *specific*
evidence row that produced the title. This test proves ``title_ref``
survives the full write -> storage-summary/envelope -> domain-model ->
surface-payload chain, the same way ``test_title_source_queryable.py``
proved it for ``title_source``.
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


def test_archive_store_summary_reads_expose_title_ref(tmp_path: Path) -> None:
    """``ArchiveStore.read_summary``/``list_summaries`` select ``title_ref``."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass  # bootstrap the schema, then reopen for a plain-connection write below.

    _write_codex_session(db_path, native_id="codex-tr-1", title="Fix the flaky test")

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("codex-tr-1")
        summary = archive.read_summary(session_id)
        assert summary.title_ref == _TITLE_REF

        listed = archive.list_summaries(origin="codex-session", limit=10, offset=0)
        matching = [s for s in listed if s.session_id == session_id]
        assert [s.title_ref for s in matching] == [_TITLE_REF]


@pytest.mark.asyncio
async def test_session_filter_summary_exposes_title_ref(workspace_env: dict[str, Path]) -> None:
    """``SessionFilter.list_summaries()`` yields a domain ``SessionSummary`` with ref/confidence set."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-tr-2", title="Deploy the daemon fix")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()
    assert len(summaries) == 1
    assert summaries[0].title_ref == _TITLE_REF


@pytest.mark.asyncio
async def test_session_filter_full_session_exposes_title_ref(workspace_env: dict[str, Path]) -> None:
    """A full ``Session`` read (not just the summary) also carries ref/confidence."""
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    _write_codex_session(db_path, native_id="codex-tr-3", title="Investigate the ingest stall")

    plan = SessionQueryPlan(origins=("codex-session",), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    assert len(sessions) == 1
    assert sessions[0].title_ref == _TITLE_REF


def test_session_list_row_payload_carries_title_ref() -> None:
    """The CLI/MCP row payload surfaces ``title_ref``."""
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
        messages=MessageCollection(messages=[]),
    )
    row = session_list_envelope_from_domain(session)
    assert row.title_ref == "message:codex-session:codex-tr-4:m1"
    summary_payload = session_summary_envelope_from_domain(session)
    assert summary_payload.title_ref == "message:codex-session:codex-tr-4:m1"


def test_assembly_codex_sets_a_distinct_ref_per_resolution_lane() -> None:
    """Each Codex title-resolution lane stamps a distinct ref, not just title_source."""
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


def test_canonical_index_ddl_declares_no_retired_session_columns() -> None:
    """polylogue-k1eiz: the retired derived columns are gone from fresh DDL.

    Anti-vacuity: reintroducing either column to ``SESSIONS_SPEC`` (or to any
    other table in ``INDEX_DDL``) makes this red. ``title_confidence`` was a
    fabricated heuristic score restating evidence ``title_source``/
    ``title_ref`` already carry; ``run_settings_json`` had no reader.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import SESSIONS_SPEC
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL

    declared = {column.name for column in SESSIONS_SPEC.all_columns}
    assert "title_confidence" not in declared
    assert "run_settings_json" not in declared
    assert "title_confidence" not in INDEX_DDL
    assert "run_settings_json" not in INDEX_DDL
    # The retained provenance pair must still be declared -- a test that
    # passed by emptying the spec would prove nothing.
    assert {"title_source", "title_ref"} <= declared


def test_fresh_index_database_has_no_retired_session_columns(tmp_path: Path) -> None:
    """The same retirement, proven against a materialized fresh archive."""
    db_path = tmp_path / "fresh-index.db"
    _write_codex_session(db_path, native_id="codex-tr-ddl", title="Ship the release")
    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    finally:
        conn.close()
    assert "title_confidence" not in columns
    assert "run_settings_json" not in columns
    assert {"title_source", "title_ref"} <= columns
