from __future__ import annotations

import hashlib
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


def _write_blob(archive_root: Path, payload: str) -> bytes:
    content = payload.encode()
    digest = hashlib.sha256(content).digest()
    blob_path = archive_root / "blob" / digest.hex()[:2] / digest.hex()[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(content)
    return digest


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
    conn.execute(
        """
        INSERT INTO session_provider_usage_events (
            session_id, position, provider_event_type, payload_json
        ) VALUES (?, ?, 'token_count', '{}')
        """,
        ("codex-session:provider-usage-report", 99),
    )
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
    # Derived rollup stores disjoint billing lanes, not the raw cumulative
    # counters. Output = 10 (reasoning is already inside output, not re-added).
    # This fixture has cached (30) > input (20) — impossible in real Codex data
    # but a guard case: fresh input = max(20 - 30, 0) clamps to 0 rather than
    # going negative.
    assert row.model_rollup_usage.to_dict() == {
        "input_tokens": 0,
        "output_tokens": 10,
        "cached_input_tokens": 30,
        "cache_write_tokens": 40,
        "reasoning_output_tokens": 0,
        "total_tokens": 80,
    }
    assert row.model_rollup_grain == "physical_session"
    assert row.logical_model_rollup_grain == "logical_session_model_high_water"
    assert row.logical_model_rollup_usage == row.model_rollup_usage
    assert "zero-token provider events" in " ".join(row.caveats)
    assert row.sample_missing_model_sessions == ("codex-session:provider-usage-report",)
    assert row.sample_zero_token_sessions == ("codex-session:provider-usage-report",)


def test_provider_usage_report_labels_physical_and_logical_model_rollups(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    conn.execute(
        """
        INSERT INTO sessions (
            origin, native_id, title, session_kind,
            created_at_ms, updated_at_ms, message_count, word_count, content_hash
        ) VALUES
            ('claude-code-session', 'root', 'root', 'standard',
             1, 1, 1, 1, zeroblob(32)),
            ('claude-code-session', 'child-a', 'child a', 'standard',
             2, 2, 1, 1, zeroblob(32)),
            ('claude-code-session', 'child-b', 'child b', 'standard',
             3, 3, 1, 1, zeroblob(32)),
            ('codex-session', 'codex-root', 'codex root', 'standard',
             4, 4, 1, 1, zeroblob(32)),
            ('codex-session', 'codex-child', 'codex child', 'standard',
             5, 5, 1, 1, zeroblob(32))
        """
    )
    conn.executemany(
        """
        INSERT INTO session_profiles (session_id, logical_session_id, materialized_at, source_name)
        VALUES (?, ?, 'now', 'claude-code-session')
        """,
        [
            ("claude-code-session:root", "claude-code-session:root"),
            ("claude-code-session:child-a", "claude-code-session:root"),
            ("claude-code-session:child-b", "claude-code-session:root"),
            ("codex-session:codex-root", "codex-session:codex-root"),
            ("codex-session:codex-child", "codex-session:codex-root"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO session_model_usage (
            session_id, model_name, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, message_count
        ) VALUES (?, 'claude-sonnet', ?, ?, ?, ?, 1)
        """,
        [
            ("claude-code-session:root", 100, 10, 1000, 50),
            ("claude-code-session:child-a", 140, 12, 1300, 60),
            ("claude-code-session:child-b", 90, 50, 800, 20),
            ("codex-session:codex-root", 1000, 100, 10_000, 0),
            ("codex-session:codex-child", 1200, 150, 12_000, 0),
        ],
    )

    report = provider_usage_report_from_connection(conn, archive_root=tmp_path)

    row = next(item for item in report.origins if item.origin == "claude-code-session")
    assert row.model_rollup_grain == "physical_session"
    assert row.model_rollup_usage.to_dict() == {
        "input_tokens": 330,
        "output_tokens": 72,
        "cached_input_tokens": 3100,
        "cache_write_tokens": 130,
        "reasoning_output_tokens": 0,
        "total_tokens": 3632,
    }
    assert row.logical_model_rollup_grain == "logical_session_model_high_water"
    assert row.logical_model_rollup_usage.to_dict() == {
        "input_tokens": 140,
        "output_tokens": 50,
        "cached_input_tokens": 1300,
        "cache_write_tokens": 60,
        "reasoning_output_tokens": 0,
        "total_tokens": 1550,
    }
    payload = row.to_dict()
    assert payload["model_rollup_grain"] == "physical_session"
    assert payload["logical_model_rollup_grain"] == "logical_session_model_high_water"
    assert report.model_rollup_grain == "physical_session"
    assert report.model_rollup_usage.to_dict() == {
        "input_tokens": 2530,
        "output_tokens": 322,
        "cached_input_tokens": 25100,
        "cache_write_tokens": 130,
        "reasoning_output_tokens": 0,
        "total_tokens": 28082,
    }
    assert report.logical_model_rollup_grain == "logical_session_model_high_water"
    assert report.logical_model_rollup_usage.to_dict() == {
        "input_tokens": 1340,
        "output_tokens": 200,
        "cached_input_tokens": 13300,
        "cache_write_tokens": 60,
        "reasoning_output_tokens": 0,
        "total_tokens": 14900,
    }
    report_payload = report.to_dict()
    assert report_payload["model_rollup_grain"] == "physical_session"
    assert report_payload["logical_model_rollup_grain"] == "logical_session_model_high_water"


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


def test_provider_usage_report_treats_codex_cumulative_as_session_global(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="provider-usage-model-switch",
        title="provider usage model switch",
        models_used=["gpt-5-codex", "o4-mini"],
        messages=[],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "gpt-5-codex",
                    "total_token_usage": {"input_tokens": 100, "output_tokens": 20},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "model": "o4-mini",
                    "total_token_usage": {"input_tokens": 50, "output_tokens": 10},
                },
            ),
            ParsedSessionEvent(
                event_type="token_count",
                payload={
                    "type": "token_count",
                    "total_token_usage": {"input_tokens": 999, "output_tokens": 999},
                },
            ),
        ],
    )

    write_parsed_session_to_archive(conn, session)
    report = provider_usage_report_from_connection(conn, archive_root=tmp_path)

    row = report.origins[0]
    assert row.stale_rollup_session_count == 0
    assert row.sample_stale_rollup_sessions == ()
    assert row.model_rollup_usage.to_dict() == {
        "input_tokens": 50,
        "output_tokens": 10,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 60,
    }


def test_provider_usage_report_ignores_codex_metadata_only_raw_rows(tmp_path: Path) -> None:
    index_conn = _connect(tmp_path / "index.db")
    source_conn = _connect(tmp_path / "source.db", ArchiveTier.SOURCE)
    blob_hash = _write_blob(
        tmp_path,
        '{"timestamp":"2026-01-01T00:00:00Z","type":"session_meta","payload":{"id":"meta-only"}}\n',
    )
    source_conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash,
            blob_size, acquired_at_ms, parsed_at_ms, validation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "raw-codex-meta-only",
            "codex-session",
            "meta-only",
            "rollout-meta-only.jsonl",
            0,
            blob_hash,
            91,
            1,
            2,
            "passed",
        ),
    )
    source_conn.commit()
    source_conn.close()

    report = provider_usage_report_from_connection(index_conn, archive_root=tmp_path, origin="codex-session")

    row = report.origins[0]
    assert row.raw_session_count == 1
    assert row.acquired_not_materialized_count == 0
    assert row.sample_acquired_not_materialized_raw_ids == ()
    assert row.coverage_state == "no_sessions"


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
