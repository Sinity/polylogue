from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.usage import provider_usage_coverage_matrix, provider_usage_report_from_connection


def _connect(path: Path, tier: ArchiveTier = ArchiveTier.INDEX) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, tier)
    return conn


def _insert_raw_session(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    native_id: str,
    origin: str = "codex-session",
    parse_error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms, parse_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (raw_id, origin, native_id, f"{native_id}.jsonl", 0, bytes(32), 2, 1, parse_error),
    )


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
    assert row.provider == "codex"
    assert row.declared_coverage == "exact"
    assert row.coverage_state == "exact_provider_telemetry"
    assert "cached_input_tokens" in row.cache_semantics
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


def test_provider_usage_report_exposes_source_debt_and_stale_rollups(tmp_path: Path) -> None:
    index_conn = _connect(tmp_path / "index.db")
    source_conn = _connect(tmp_path / "source.db", ArchiveTier.SOURCE)
    _insert_raw_session(source_conn, raw_id="raw-materialized", native_id="provider-usage-report")
    _insert_raw_session(source_conn, raw_id="raw-missing", native_id="missing")
    _insert_raw_session(source_conn, raw_id="raw-error", native_id="bad", parse_error="bad json")
    source_conn.commit()
    source_conn.close()

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
        ],
    )

    write_parsed_session_to_archive(index_conn, session, raw_id="raw-materialized")
    index_conn.execute(
        """
        UPDATE session_model_usage
        SET input_tokens = 0, output_tokens = 0, cache_read_tokens = 0, cache_write_tokens = 0
        WHERE session_id = ? AND model_name = ?
        """,
        ("codex-session:provider-usage-report", "gpt-5-codex"),
    )

    report = provider_usage_report_from_connection(index_conn, archive_root=tmp_path)

    row = report.origins[0]
    assert row.coverage_state == "acquired_not_materialized"
    assert row.raw_session_count == 3
    assert row.raw_parse_error_count == 1
    assert row.acquired_not_materialized_count == 1
    assert row.sample_acquired_not_materialized_raw_ids == ("raw-missing",)
    assert row.stale_rollup_session_count == 1
    assert row.sample_stale_rollup_sessions == ("codex-session:provider-usage-report",)
    assert "acquired but not materialized" in " ".join(row.caveats)
    assert "stale relative to provider usage events" in " ".join(row.caveats)


def test_provider_usage_coverage_matrix_marks_estimate_only_exports(tmp_path: Path) -> None:
    matrix = {item.origin: item for item in provider_usage_coverage_matrix()}

    assert matrix["codex-session"].status == "exact"
    assert "cached_input_tokens" in matrix["codex-session"].cache_semantics
    assert matrix["claude-code-session"].event_types == ("message_usage",)
    assert matrix["chatgpt-export"].status == "estimate_only"

    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-usage-report",
        title="chatgpt usage report",
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="hello world",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello world")],
            ),
        ],
    )

    write_parsed_session_to_archive(conn, session)
    report = provider_usage_report_from_connection(conn, archive_root=tmp_path)

    row = report.origins[0]
    assert row.origin == "chatgpt-export"
    assert row.provider == "chatgpt"
    assert row.declared_coverage == "estimate_only"
    assert row.coverage_state == "estimate_only"
    assert row.provider_event_count == 0
    assert "exact provider telemetry unavailable" in " ".join(row.caveats)
