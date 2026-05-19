"""Insight provenance contracts.

Every materialized insight row must record:

- ``materialized_at`` (computed_at): wall-clock instant the row was produced;
- ``input_high_water_mark``: the latest source change timestamp folded in;
- ``input_row_count``: number of source rows that produced the insight.

These contracts let downstream readers reason about staleness without
re-deriving the comparison logic. See ``polylogue/insights/provenance.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from polylogue.insights.provenance import HasProvenance, is_stale
from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def provenance_db(workspace_env: Mapping[str, Path]) -> Path:
    """Build a small archive and materialize session insights against it."""
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "prov-1").provider("chatgpt").title("alpha").add_message(
        role="user", text="hello"
    ).add_message(role="assistant", text="hi").save()
    ConversationBuilder(db_path, "prov-2").provider("claude-code").title("beta").add_message(
        role="user", text="please refactor"
    ).add_message(role="assistant", text="ok").add_message(role="user", text="thanks").save()

    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)
        conn.commit()
    return db_path


class TestProfileProvenance:
    def test_profile_has_materialized_at_and_input_provenance(self, provenance_db: Path) -> None:
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(provenance_db) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")
            rows = conn.execute(
                """
                SELECT conversation_id, materialized_at, input_high_water_mark,
                       input_row_count, message_count
                FROM session_profiles
                ORDER BY conversation_id
                """
            ).fetchall()
        assert rows, "expected materialized session_profiles rows"
        for row in rows:
            assert row["materialized_at"], f"missing materialized_at for {row['conversation_id']}"
            # input_row_count must equal message_count for per-conversation profile
            assert int(row["input_row_count"]) == int(row["message_count"]), (
                f"input_row_count must match message_count for {row['conversation_id']}; "
                f"got input_row_count={row['input_row_count']} message_count={row['message_count']}"
            )

    def test_profile_provenance_round_trips_via_record(self, provenance_db: Path) -> None:
        from polylogue.storage.sqlite.connection import open_connection
        from polylogue.storage.sqlite.queries.mappers_insight_profiles import (
            _row_to_session_profile_record,
        )

        with open_connection(provenance_db) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")
            row = conn.execute("SELECT * FROM session_profiles LIMIT 1").fetchone()
        assert row is not None
        record = _row_to_session_profile_record(row)
        assert record.materialized_at
        assert record.input_row_count == record.message_count


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
    def test_day_summary_input_row_count_matches_session_count(self, provenance_db: Path) -> None:
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(provenance_db) as conn:
            has_day = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='day_session_summaries'"
            ).fetchone()
            if has_day is None:
                pytest.skip("day_session_summaries table not present")
            rows = conn.execute("SELECT input_row_count, conversation_count FROM day_session_summaries").fetchall()
        if not rows:
            pytest.skip("no day summaries materialized")
        for row in rows:
            assert int(row["input_row_count"]) == int(row["conversation_count"])

    def test_tag_rollup_input_row_count_matches_conversation_count(self, provenance_db: Path) -> None:
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(provenance_db) as conn:
            has_tags = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_tag_rollups'"
            ).fetchone()
            if has_tags is None:
                pytest.skip("session_tag_rollups table not present")
            rows = conn.execute("SELECT input_row_count, conversation_count FROM session_tag_rollups").fetchall()
        for row in rows:
            assert int(row["input_row_count"]) == int(row["conversation_count"])
