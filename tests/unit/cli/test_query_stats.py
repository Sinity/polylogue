from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.archive.stats import ArchiveStats
from polylogue.cli.query_stats import output_stats_sql


@pytest.fixture
def fixed_local_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", previous)
        time.tzset()


@pytest.mark.asyncio
async def test_structured_stats_keep_canonical_dates_while_text_is_local(
    capsys: pytest.CaptureFixture[str], fixed_local_timezone: None
) -> None:
    min_timestamp = datetime(2026, 7, 1, 1, tzinfo=timezone.utc).timestamp()
    max_timestamp = datetime(2026, 7, 2, 1, tzinfo=timezone.utc).timestamp()
    filter_chain = MagicMock()
    filter_chain.describe.return_value = []
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=2, total_messages=4))
    repo.aggregate_message_stats = AsyncMock(
        return_value={
            "total": 4,
            "user": 2,
            "assistant": 2,
            "words_approx": 10,
            "attachment_refs": 0,
            "distinct_attachments": 0,
            "origins": {"claude-ai-export": 2},
            "min_sort_key": min_timestamp,
            "max_sort_key": max_timestamp,
        }
    )
    env = MagicMock()

    await output_stats_sql(env, filter_chain, repo, output_format="json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["date_range"] == "2026-07-01 to 2026-07-02"

    await output_stats_sql(env, filter_chain, repo, output_format="text")

    env.ui.console.print.assert_any_call("Date range: 2026-06-30 to 2026-07-01")
