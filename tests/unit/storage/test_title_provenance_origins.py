"""Production-route title provenance coverage for active origin parsers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.core.enums import Origin, TitleSource
from polylogue.core.sources import origin_from_provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.browser_capture import parse as parse_browser_capture
from polylogue.sources.parsers.hermes_state import parse_state_db
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import session_summary_envelope_from_domain
from tests.infra.live_ingest import write_session_sync
from tests.infra.storage_records import db_setup
from tests.unit.sources.parsers.test_hermes_state import _write_state_db
from tests.unit.sources.parsers.test_origin_regression_pack import (
    ORIGIN_FIXTURES,
    OriginFixture,
    _browser_capture_snapshot_payload,
)

_TITLE_CASES = (
    ("chatgpt-export", "ChatGPT regression fixture"),
    ("antigravity-session", "Antigravity regression fixture"),
    ("grok-export", "Grok regression fixture"),
    ("browser-capture", "Browser snapshot regression fixture"),
    ("hermes-state", "Outcome fixture"),
)


def _origin_fixture(label: str) -> OriginFixture:
    return next(fixture for fixture in ORIGIN_FIXTURES if fixture.label == label)


def _parse_title_case(label: str, tmp_path: Path) -> ParsedSession:
    if label == "hermes-state":
        state_path = tmp_path / "hermes" / "state.db"
        _write_state_db(state_path, tool_contents=['{"output":"ok","exit_code":0}'])
        return parse_state_db(state_path)[0]
    if label == "browser-capture":
        return parse_browser_capture(_browser_capture_snapshot_payload(), "fallback")
    fixture = _origin_fixture(label)
    return cast(ParsedSession, fixture.parse_fn(fixture.payload, fixture.session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("label,expected_title", _TITLE_CASES, ids=[case[0] for case in _TITLE_CASES])
async def test_provider_title_survives_production_ingest_and_public_session_conversion(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    label: str,
    expected_title: str,
) -> None:
    """Parser evidence must survive storage, full-session reads, and labels.

    The production symbols exercised are the selected parser, ``write_session_sync``
    and ``write_parsed_session_to_archive`` for materialization,
    ``ArchiveStore.read_summary``/``read_session``, ``SessionFilter.list`` plus
    ``_session_to_session``, and ``session_summary_envelope_from_domain``.
    Removing the parser's ``TitleSource.ORIGIN`` assignment makes the storage
    summary title ``None`` and changes the full-session display label to a
    structural fallback, so the assertions fail without changing this oracle.
    """
    parsed = _parse_title_case(label, tmp_path)
    archive_root = workspace_env["archive_root"]
    db_path = db_setup(workspace_env)
    with ArchiveStore(archive_root, initialize=True, read_only=False):
        pass

    session_id = write_session_sync(db_path, parsed)
    origin = origin_from_provider(parsed.source_name)
    assert isinstance(origin, Origin)

    with ArchiveStore(archive_root, initialize=False, read_only=True) as archive:
        stored_summary = archive.read_summary(session_id)
        stored_full = archive.read_session(session_id)

    assert stored_summary.title == expected_title
    assert stored_summary.title_source == TitleSource.ORIGIN.value
    assert stored_summary.display_label == expected_title
    assert stored_full.title == expected_title
    assert stored_full.title_source == TitleSource.ORIGIN.value

    plan = SessionQueryPlan(origins=(origin.value,), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    session = next(item for item in sessions if str(item.id) == session_id)
    assert session.title == expected_title
    assert session.title_source is TitleSource.ORIGIN
    assert session.display_title == expected_title

    public_summary = session_summary_envelope_from_domain(session)
    assert public_summary.title == expected_title
    assert public_summary.title_source == TitleSource.ORIGIN.value


@pytest.mark.asyncio
async def test_provider_title_degrades_when_parser_provenance_is_removed(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Removing typed parser provenance must make the public title disappear."""
    parsed = _parse_title_case("grok-export", tmp_path)
    expected_title = parsed.title
    assert expected_title == "Grok regression fixture"
    assert parsed.title_source is TitleSource.ORIGIN
    mutated = parsed.model_copy(update={"title_source": None})

    archive_root = workspace_env["archive_root"]
    db_path = db_setup(workspace_env)
    with ArchiveStore(archive_root, initialize=True, read_only=False):
        pass

    session_id = write_session_sync(db_path, mutated)
    with ArchiveStore(archive_root, initialize=False, read_only=True) as archive:
        stored_summary = archive.read_summary(session_id)

    assert stored_summary.title is None
    assert stored_summary.title_source is None
    assert stored_summary.display_label != expected_title

    plan = SessionQueryPlan(origins=(Origin.GROK_EXPORT.value,), limit=10)
    sessions = await SessionFilter(archive_root=archive_root, query_plan=plan).list()
    session = next(item for item in sessions if str(item.id) == session_id)
    assert session.title is None
    assert session.title_source is None
    assert session.display_title != expected_title

    public_summary = session_summary_envelope_from_domain(session)
    assert public_summary.title != expected_title
    assert public_summary.title_source is None
