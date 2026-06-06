"""Temporal source taxonomy contracts (#1276).

Every materialized insight row must carry an ``input_high_water_mark_source``
naming the *clock* the HWM was sampled from. The taxonomy is a closed set
of six values; each materialization path picks the value that matches the
clock it actually used.

These tests pin:

- the literal value set is exactly the six tokens the issue specifies;
- the three classifier helpers return the expected tokens for each input
  shape;
- each materialization path (profiles, work events, phases, threads,
  day summaries, tag rollups) sets the source tag on every row it
  produces.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.insights.temporal_source import (
    TEMPORAL_SOURCE_VALUES,
    classify_aggregate_hwm_source,
    classify_profile_hwm_source,
    classify_thread_hwm_source,
    is_valid_temporal_source,
)
from tests.infra.storage_records import SessionBuilder, db_setup


class TestTemporalSourceLiteral:
    def test_literal_values_match_taxonomy(self) -> None:
        assert (
            frozenset(
                {
                    "provider_ts",
                    "hook_event_ts",
                    "sort_key",
                    "file_mtime",
                    "materialization_ts",
                    "fallback_date",
                }
            )
            == TEMPORAL_SOURCE_VALUES
        )

    @pytest.mark.parametrize(
        "value",
        sorted(
            [
                "provider_ts",
                "hook_event_ts",
                "sort_key",
                "file_mtime",
                "materialization_ts",
                "fallback_date",
            ]
        ),
    )
    def test_is_valid_temporal_source_accepts_known_tokens(self, value: str) -> None:
        assert is_valid_temporal_source(value) is True

    def test_is_valid_temporal_source_rejects_unknown(self) -> None:
        assert is_valid_temporal_source("wall_clock") is False
        assert is_valid_temporal_source("") is False


class TestClassifierHelpers:
    def test_profile_with_provider_timestamp_is_provider_ts(self) -> None:
        assert classify_profile_hwm_source(datetime(2026, 5, 1, tzinfo=UTC)) == "provider_ts"

    def test_profile_without_timestamp_is_fallback_date(self) -> None:
        assert classify_profile_hwm_source(None) == "fallback_date"

    def test_thread_with_end_time_is_provider_ts(self) -> None:
        assert classify_thread_hwm_source(datetime(2026, 5, 1, tzinfo=UTC)) == "provider_ts"

    def test_thread_without_end_time_is_fallback_date(self) -> None:
        assert classify_thread_hwm_source(None) == "fallback_date"

    def test_aggregate_with_inputs_is_provider_ts(self) -> None:
        assert classify_aggregate_hwm_source(["2026-05-01T00:00:00+00:00"]) == "provider_ts"

    def test_aggregate_without_inputs_is_fallback_date(self) -> None:
        assert classify_aggregate_hwm_source([]) == "fallback_date"


@pytest.fixture()
def temporal_source_db(workspace_env: Mapping[str, Path]) -> Path:
    """Build a small archive and materialize session insights against it."""
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "tsrc-1").provider("chatgpt").title("alpha").add_message(
        role="user", text="hello"
    ).add_message(role="assistant", text="hi").save()
    SessionBuilder(db_path, "tsrc-2").provider("claude-code").title("beta").add_message(
        role="user", text="please refactor"
    ).add_message(role="assistant", text="ok").add_message(role="user", text="done").save()

    from polylogue.api import Polylogue

    async def _rebuild() -> None:
        archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
        try:
            await archive.rebuild_insights()
        finally:
            await archive.close()

    import asyncio

    asyncio.run(_rebuild())
    return db_path


@pytest.mark.xfail(
    reason=(
        "Archive gap (#1781): the archive insight_materialization table does "
        "not yet persist input_high_water_mark_source; the per-record "
        "classifier value is computed but dropped by the archive rebuild. "
        "Carrying it is a schema-touching change tracked in #1781 (overlaps "
        "the #1743 archive redesign)."
    ),
    strict=False,
)
class TestMaterializationPathsCarryTaxonomy:
    """Every materialization path must set ``input_high_water_mark_source``."""

    def _assert_all_rows_tagged(
        self,
        db_path: Path,
        table: str,
    ) -> None:
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if has_table is None:
                pytest.skip(f"{table} not present")
            has_column = any(
                str(row[1]) == "input_high_water_mark_source"
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            )
            assert has_column, f"{table} missing input_high_water_mark_source column"
            rows = conn.execute(f"SELECT input_high_water_mark_source FROM {table}").fetchall()
        if not rows:
            pytest.skip(f"{table} has no materialized rows")
        for row in rows:
            source = row["input_high_water_mark_source"]
            assert source, f"{table} row missing input_high_water_mark_source"
            assert is_valid_temporal_source(str(source)), (
                f"{table} row has invalid temporal source {source!r}; must be one of {sorted(TEMPORAL_SOURCE_VALUES)}"
            )

    def test_session_profiles_tagged(self, temporal_source_db: Path) -> None:
        self._assert_all_rows_tagged(temporal_source_db, "session_profiles")

    def test_session_work_events_tagged(self, temporal_source_db: Path) -> None:
        self._assert_all_rows_tagged(temporal_source_db, "session_work_events")

    def test_session_phases_tagged(self, temporal_source_db: Path) -> None:
        self._assert_all_rows_tagged(temporal_source_db, "session_phases")

    def test_work_threads_tagged(self, temporal_source_db: Path) -> None:
        self._assert_all_rows_tagged(temporal_source_db, "work_threads")

    def test_session_tag_rollups_tagged(self, temporal_source_db: Path) -> None:
        self._assert_all_rows_tagged(temporal_source_db, "session_tag_rollups")


@pytest.mark.xfail(
    reason=(
        "Archive gap (#1781): the archive store does not persist "
        "input_high_water_mark_source, so the archive profile record cannot "
        "round-trip it. Tracked in #1781."
    ),
    strict=False,
)
class TestProvenanceRecordRoundTrips:
    """The record-level field round-trips through the storage mappers."""

    def test_profile_record_round_trips_source_tag(self, temporal_source_db: Path) -> None:
        import sqlite3

        from polylogue.storage.sqlite.queries.mappers_insight_profiles import (
            _row_to_session_profile_record,
        )

        conn = sqlite3.connect(temporal_source_db)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM session_profiles LIMIT 1").fetchone()
        finally:
            conn.close()
        assert row is not None
        record = _row_to_session_profile_record(row)
        assert record.input_high_water_mark_source is not None
        assert is_valid_temporal_source(record.input_high_water_mark_source)
