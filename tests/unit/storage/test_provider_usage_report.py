from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.usage import provider_usage_report_from_connection


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_provider_usage_report_keeps_events_cumulative_and_rollups_separate(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="provider-usage-report",
        title="provider usage report",
        models_used=["gpt-5-codex"],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
            ),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "last_token_usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "cached_input_tokens": 3,
                        "cache_write_tokens": 4,
                        "reasoning_output_tokens": 5,
                    },
                    "total_token_usage": {
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "cached_input_tokens": 30,
                        "cache_write_tokens": 40,
                        "reasoning_output_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            ),
            ParsedSessionEvent(event_type="token_count", payload={"type": "token_count"}),
        ],
    )

    write_parsed_session_to_archive(conn, session)
    report = provider_usage_report_from_connection(conn, archive_root=tmp_path)

    assert "not a precise cost report" in " ".join(report.caveats)
    row = report.origins[0]
    assert row.origin == "codex-session"
    assert row.session_count == 1
    assert row.provider_event_count == 2
    assert row.token_count_event_count == 2
    assert row.zero_token_event_count == 1
    assert row.missing_model_event_count == 1
    assert row.provider_request_usage.to_dict() == {
        "input_tokens": 2,
        "output_tokens": 1,
        "cached_input_tokens": 3,
        "cache_write_tokens": 4,
        "reasoning_output_tokens": 5,
        "total_tokens": 0,
    }
    assert row.provider_cumulative_usage.to_dict() == {
        "input_tokens": 20,
        "output_tokens": 10,
        "cached_input_tokens": 30,
        "cache_write_tokens": 40,
        "reasoning_output_tokens": 50,
        "total_tokens": 150,
    }
    assert row.model_rollup_usage.to_dict() == {
        "input_tokens": 20,
        "output_tokens": 60,
        "cached_input_tokens": 30,
        "cache_write_tokens": 40,
        "reasoning_output_tokens": 0,
        "total_tokens": 150,
    }
    assert "zero-token provider events" in " ".join(row.caveats)
    assert row.sample_missing_model_sessions == ("codex-session:provider-usage-report",)
    assert row.sample_zero_token_sessions == ("codex-session:provider-usage-report",)


def test_provider_usage_report_handles_empty_origin_filter(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")

    report = provider_usage_report_from_connection(conn, archive_root=tmp_path, origin="codex-session")

    assert report.origins == ()
    assert "no sessions found for origin 'codex-session'" in report.caveats
