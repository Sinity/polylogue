"""Insight provenance contracts.

Every materialized insight row must record:

- ``materialized_at`` (computed_at): wall-clock instant the row was produced;
- ``input_high_water_mark``: the latest source change timestamp folded in;
- ``input_row_count``: number of source rows that produced the insight.

These contracts let downstream readers reason about staleness without
re-deriving the comparison logic. See ``polylogue/analysis/provenance.py``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from polylogue.analysis.provenance import HasProvenance, is_stale
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.storage_records import SessionBuilder, db_setup, materialize_session_insights


def _open_archive(db_path: Path) -> sqlite3.Connection:
    """Raw read connection to the index.db (bypasses v22 guard)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture()
def provenance_db(workspace_env: Mapping[str, Path]) -> Path:
    """Build a small archive and materialize session insights against it."""
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "prov-1").provider("chatgpt").title("alpha").add_message(
        role="user", text="hello"
    ).add_message(role="assistant", text="hi").save()
    SessionBuilder(db_path, "prov-2").provider("claude-code").title("beta").add_message(
        role="user", text="please refactor"
    ).add_message(role="assistant", text="ok").add_message(role="user", text="thanks").save()

    materialize_session_insights(db_path)
    return db_path


class TestProfileProvenance:
    def test_profile_has_materialized_at_and_input_provenance(self, provenance_db: Path) -> None:
        # Native provenance lives on each derived session row; the profile's
        # input_row_count must equal the session's message_count.
        with closing(_open_archive(provenance_db)) as conn:
            rows = conn.execute(
                """
                SELECT sp.session_id, sp.materialized_at, sp.input_high_water_mark,
                       sp.input_row_count, s.message_count
                FROM session_profiles sp
                JOIN sessions s ON s.session_id = sp.session_id
                ORDER BY sp.session_id
                """
            ).fetchall()
        assert rows, "expected materialized session_profile provenance rows"
        for row in rows:
            assert row["materialized_at"], f"missing materialized_at for {row['session_id']}"
            assert int(row["input_row_count"]) == int(row["message_count"]), (
                f"input_row_count must match message_count for {row['session_id']}; "
                f"got input_row_count={row['input_row_count']} message_count={row['message_count']}"
            )

    def test_profile_provenance_round_trips_via_archive(self, provenance_db: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(provenance_db.parent) as archive:
            profiles = archive.list_session_profile_insights(tier="merged")
        assert profiles, "expected at least one materialized profile"
        for profile in profiles:
            assert profile.provenance.materialized_at
            # The profile provenance must record the source it folded in.
            assert profile.provenance.materializer_version is not None
            assert profile.provenance.materializer_version >= 0


class TestStalenessDetection:
    """``is_stale`` correctly classifies fresh and stale insight records."""

    def _record(self, hwm: str | None, rows: int) -> HasProvenance:
        # lightweight stand-in implementing the HasProvenance protocol
        from dataclasses import dataclass

        @dataclass
        class _R:
            materialized_at: str
            materializer_version: int
            input_high_water_mark: str | None
            input_high_water_mark_source: str | None
            input_row_count: int

        return _R(
            materialized_at="2026-01-01T00:00:00+00:00",
            materializer_version=5,
            input_high_water_mark=hwm,
            input_high_water_mark_source="provider_ts" if hwm else None,
            input_row_count=rows,
        )

    def test_fresh_when_source_matches_input_hwm(self) -> None:
        hwm = "2026-03-01T00:00:00+00:00"
        record = self._record(hwm, rows=10)
        verdict = is_stale(record, source_high_water_mark=hwm, source_row_count=10)
        assert verdict.stale is False
        assert verdict.reason == "fresh"

    def test_stale_when_source_advances_past_input_hwm(self) -> None:
        record = self._record("2026-03-01T00:00:00+00:00", rows=10)
        later = (datetime.fromisoformat("2026-03-01T00:00:00+00:00") + timedelta(days=1)).isoformat()
        verdict = is_stale(record, source_high_water_mark=later)
        assert verdict.stale is True
        assert "source has changed" in verdict.reason

    def test_stale_when_source_row_count_exceeds_input(self) -> None:
        record = self._record("2026-03-01T00:00:00+00:00", rows=5)
        verdict = is_stale(record, source_high_water_mark=None, source_row_count=10)
        assert verdict.stale is True
        assert "row count" in verdict.reason

    def test_stale_when_insight_missing_hwm_but_source_known(self) -> None:
        record = self._record(None, rows=5)
        verdict = is_stale(record, source_high_water_mark="2026-03-01T00:00:00+00:00")
        assert verdict.stale is True
        assert "input_high_water_mark" in verdict.reason

    def test_fresh_when_both_provenance_unknown(self) -> None:
        record = self._record(None, rows=0)
        verdict = is_stale(record, source_high_water_mark=None, source_row_count=None)
        assert verdict.stale is False


class TestAggregateProvenance:
    def test_tag_rollup_session_count_matches_tagged_sessions(self, provenance_db: Path) -> None:
        # The archive tag rollup is a computed read model rather than a
        # materialized table with a separate input_row_count column. The
        # agreement law is therefore that the rollup's session_count
        # equals the number of sessions actually carrying that tag.
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(provenance_db.parent) as archive:
            rollups = archive.list_session_tag_rollup_insights()

        with closing(_open_archive(provenance_db)) as conn:
            for rollup in rollups:
                actual = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM session_tags WHERE tag = ?",
                    (rollup.tag,),
                ).fetchone()[0]
                assert int(rollup.session_count) == int(actual), (
                    f"tag {rollup.tag!r}: rollup session_count={rollup.session_count} != tagged session count={actual}"
                )


@pytest.mark.asyncio
async def test_public_insight_versions_require_recorded_profile_evidence(
    workspace_env: Mapping[str, Path],
) -> None:
    """Stamping today's version for a missing profile breaks this public read."""
    import json
    from typing import cast

    from polylogue.api import Polylogue
    from polylogue.mcp.server import build_server
    from polylogue.services import RuntimeServices
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async

    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "version-evidence")
        .provider("claude-code")
        .add_message("m-1", role="user", text="question")
        .add_message("m-2", role="assistant", text="answer")
    )
    builder.save()
    session_id = builder.native_session_id()
    run_off_event_loop(lambda: materialize_session_insights(db_path))
    with closing(_open_archive(db_path)) as conn, conn:
        conn.execute("UPDATE session_profiles SET materializer_version = 3 WHERE session_id = ?", (session_id,))
    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as poly:
        profile = await poly.get_session_profile_insight(session_id)
        thread = await poly.get_thread_insight(session_id)
        latency = await poly.get_session_latency_profile_insight(session_id)
        assert profile is not None and thread is not None and latency is not None
        assert profile.provenance.materializer_version == 3
        assert profile.inference_provenance is not None
        assert profile.inference_provenance.inference_version == 3
        assert profile.enrichment_provenance is not None
        assert profile.enrichment_provenance.enrichment_version == 3
        assert thread.provenance.materializer_version == latency.provenance.materializer_version == 3
    with closing(_open_archive(db_path)) as conn, conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as poly:
        assert await poly.get_session_profile_insight(session_id) is None
        thread = await poly.get_thread_insight(session_id)
        latency = await poly.get_session_latency_profile_insight(session_id)
        assert thread is not None and latency is not None
        for insight in (thread, latency):
            assert insight.provenance.materializer_version is None
            assert insight.provenance.materialized_at is None
            assert insight.model_dump(mode="json")["provenance"]["materializer_version"] is None
        with installed_runtime_services(db_path.parent):
            server = cast(MCPServerUnderTest, build_server(services=RuntimeServices(config=poly.config)))
            result = json.loads(
                cast(
                    str,
                    await invoke_surface_async(
                        server._tool_manager._tools["query"].fn,
                        projection="threads",
                        limit=1,
                    ),
                )
            )
        assert len(result["threads"]) == 1
        assert result["threads"][0]["provenance"].get("materializer_version") is None
    with ArchiveStore.open_existing(db_path.parent) as archive:
        assert archive.get_session_profile_record(session_id) is None


def test_unknown_provenance_schema_preserves_nullable_version_types() -> None:
    from polylogue.analysis.archive_models import ArchiveEnrichmentProvenance, ArchiveInferenceProvenance

    for model, field, family in (
        (ArchiveInferenceProvenance, "inference_version", "inference_family"),
        (ArchiveEnrichmentProvenance, "enrichment_version", "enrichment_family"),
    ):
        payload = {"materializer_version": None, field: None, family: "archive"}
        value = model.model_validate(payload)
        assert value.model_dump()[field] is None
        schema = model.model_json_schema()
        assert {entry["type"] for entry in schema["properties"][field]["anyOf"]} == {"integer", "null"}
