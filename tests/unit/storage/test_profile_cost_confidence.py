"""Profile cost confidence reads the evidence the canonical writer stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.mappers_insight_profiles import _row_to_session_profile_record
from tests.infra.session_profiles import write_session_profile


@pytest.mark.parametrize(("estimated", "provenance"), [(True, "unknown"), (False, "provider_reported")])
def test_stored_evidence_decides_cost_confidence(tmp_path: Path, estimated: bool, provenance: str) -> None:
    """Ignoring the stored evidence changes reported charges into estimates."""
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('cost', 'codex-session', ?)", (bytes(32),)
        )
        write_session_profile(
            conn, "codex-session:cost", evidence={"cost_is_estimated": estimated, "cost_provenance": provenance}
        )
        row = conn.execute("SELECT * FROM session_profiles WHERE session_id = 'codex-session:cost'").fetchone()
        record = _row_to_session_profile_record(row)
    assert record.cost_is_estimated is estimated
    assert record.cost_provenance == provenance


@pytest.mark.parametrize(("estimated", "provenance"), [(True, "catalog_priced"), (False, "provider_reported")])
@pytest.mark.asyncio
async def test_public_profile_record_keeps_canonical_estimated_evidence(
    workspace_env: dict[str, Path],
    estimated: bool,
    provenance: str,
) -> None:
    import json
    from typing import cast

    from polylogue.analysis.archive import SessionCostInsightQuery
    from polylogue.api import Polylogue
    from polylogue.mcp.server import build_server
    from polylogue.services import RuntimeServices
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async
    from tests.infra.storage_records import SessionBuilder, db_setup, materialize_session_insights

    path = db_setup(workspace_env)
    builder = (
        SessionBuilder(path, "profile-cost-evidence")
        .provider("claude-code")
        .add_message("m-1", role="assistant", text="done")
    )
    builder.save()
    session_id = builder.native_session_id()
    run_off_event_loop(lambda: materialize_session_insights(path))
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        write_session_profile(
            conn, session_id, evidence={"cost_is_estimated": estimated, "cost_provenance": provenance}
        )
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model_name, input_tokens, catalog_cost_usd, provider_cost_usd) "
            "VALUES (?, 'test-priced-model', 100, 1.0, ?)",
            (session_id, None if estimated else 0.0),
        )
    async with Polylogue(archive_root=path.parent, db_path=path) as poly:
        record = await poly.get_session_profile_record(session_id)
        insight = await poly.get_session_profile_insight(session_id)
        assert record is not None and insight is not None and insight.evidence is not None
        assert record.cost_is_estimated is estimated
        assert insight.evidence.cost_is_estimated is estimated
        [cost] = await poly.list_session_cost_insights(SessionCostInsightQuery(limit=1))
        assert cost.estimate.total_usd == (1.0 if estimated else 0.0)
        assert cost.estimate.status == ("priced" if estimated else "exact")
        assert cost.estimate.confidence == (0.7 if estimated else 1.0)
        with installed_runtime_services(path.parent):
            server = cast(MCPServerUnderTest, build_server(services=RuntimeServices(config=poly.config)))
            result = json.loads(
                cast(
                    str,
                    await invoke_surface_async(
                        server._tool_manager._tools["query"].fn,
                        projection="costs",
                        limit=1,
                    ),
                )
            )
        estimate = result["session_costs"][0]["estimate"]
        expected = cost.estimate.model_dump(mode="json")
        # Required fields must be physically present; model defaults cannot
        # conceal an omitted basis or a missing zero-valued provider amount.
        fields = ("status", "confidence", "total_usd", "basis", "missing_reasons", "provenance")
        assert {field: estimate[field] for field in fields} == {field: expected[field] for field in fields}
        assert estimate.get("unavailable_reason") is None


@pytest.mark.parametrize("reported", [0.0, 12.5])
@pytest.mark.asyncio
async def test_profile_read_routes_preserve_exact_provider_money(
    workspace_env: dict[str, Path], reported: float
) -> None:
    import aiosqlite

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.query_models import SessionProfileListQuery
    from polylogue.storage.sqlite.queries.session_insight_profile_reads import (
        get_session_profile,
        get_session_profiles_batch,
        list_session_profiles,
    )
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.index_writer import write_fixture_index_session
    from tests.infra.storage_records import db_setup, materialize_session_insights

    path = db_setup(workspace_env)
    session_id = "codex-session:provider-profile-cost"

    def write() -> None:
        with connect_measured(path) as conn:
            conn.row_factory = sqlite3.Row
            write_fixture_index_session(
                conn,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="provider-profile-cost",
                    reported_cost_usd=reported,
                    messages=[
                        ParsedMessage(
                            provider_message_id="m",
                            role=Role.ASSISTANT,
                            model_name="gpt-4o",
                            input_tokens=100,
                            output_tokens=0,
                            cache_read_tokens=0,
                            cache_write_tokens=0,
                            text="done",
                        )
                    ],
                ),
            )
        materialize_session_insights(path)

    run_off_event_loop(write)
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        single = await get_session_profile(conn, session_id)
        batch = await get_session_profiles_batch(conn, [session_id])
        listed = await list_session_profiles(conn, SessionProfileListQuery())
        for record in (single, batch[session_id], listed[0]):
            assert record is not None
            assert record.total_cost_usd == reported
            assert record.cost_provenance == "provider_reported"
            assert not record.cost_is_estimated
            assert record.total_input_tokens == 100
