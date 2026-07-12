"""Consumer-facing temporal-confidence contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import get_args

import pytest

from polylogue.insights.temporal_source import (
    TIME_CONFIDENCE_VALUES,
    TemporalSource,
    time_confidence_for_record,
    time_confidence_for_source,
    time_confidence_for_sources,
    weakest_source,
)
from polylogue.storage.insights.session.records import SessionProfileRecord
from tests.infra.storage_records import SessionBuilder, db_setup

_TEMPORAL_SOURCES: tuple[TemporalSource, ...] = get_args(TemporalSource)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("provider_ts", "recorded"),
        ("hook_event_ts", "recorded"),
        ("sort_key", "estimated"),
        ("file_mtime", "estimated"),
        ("materialization_ts", "unknown"),
        ("fallback_date", "unknown"),
        (None, "unknown"),
        ("legacy-unrecognized-source", "unknown"),
    ],
)
def test_time_confidence_projects_source_without_fabricating_event_time(source: str | None, expected: str) -> None:
    """The production source projection keeps materialization/fallback time unknown."""

    assert time_confidence_for_source(source) == expected


@pytest.mark.parametrize("right", _TEMPORAL_SOURCES)
@pytest.mark.parametrize("left", _TEMPORAL_SOURCES)
def test_time_confidence_uses_the_weakest_temporal_source(left: TemporalSource, right: TemporalSource) -> None:
    """Aggregate confidence delegates to the production provenance lattice."""

    weakest = weakest_source(left, right)
    assert time_confidence_for_sources((left, right)) == time_confidence_for_source(weakest)


def test_timeless_profile_record_round_trips_as_unknown_time(
    workspace_env: Mapping[str, Path],
) -> None:
    """The library profile-record route never upgrades timeless provenance."""

    db_path = db_setup(workspace_env)
    session = SessionBuilder(db_path, "timeless-profile").provider("chatgpt").title("timeless")
    session.conv = session.conv.model_copy(update={"created_at": None, "updated_at": None, "sort_key": None})
    session.add_message(role="user", text="no provider timestamp", timestamp=None).add_message(
        role="assistant", text="still no provider timestamp", timestamp=None
    ).save()

    from polylogue.api import Polylogue

    async def read_record() -> SessionProfileRecord | None:
        archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
        try:
            await archive.rebuild_insights()
            return await archive.get_session_profile_record(session.native_session_id())
        finally:
            await archive.close()

    record = asyncio.run(read_record())
    assert record is not None
    # Current rebuilt rows may carry the explicit fallback tag or the older
    # absent tag. Both mean event time is unknown; the consumer projection
    # must stay correct while the storage propagation path converges.
    assert record.input_high_water_mark_source in {None, "fallback_date"}
    assert time_confidence_for_record(record) == "unknown"
    assert time_confidence_for_record(record) in TIME_CONFIDENCE_VALUES
