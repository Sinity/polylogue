"""Tests for OTLP span-to-work-event correlation (#1686)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from polylogue.insights.otlp_correlation import (
    SessionLLMTiming,
    SessionToolTiming,
    correlate_spans_to_work_events,
    get_session_llm_timing,
    get_session_tool_timing,
)


def _init_db_with_otlp_table(db_path: str) -> None:
    """Create the otlp_spans table directly in a test database."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS otlp_spans (
            span_id         TEXT PRIMARY KEY,
            trace_id        TEXT NOT NULL,
            parent_span_id  TEXT,
            agent_id        TEXT,
            parent_agent_id TEXT,
            session_id      TEXT,
            operation_name  TEXT NOT NULL,
            start_time_unix_ns INTEGER NOT NULL,
            end_time_unix_ns   INTEGER NOT NULL,
            duration_ms     INTEGER NOT NULL,
            status_code     INTEGER NOT NULL DEFAULT 0,
            status_message  TEXT,
            attributes_json TEXT NOT NULL DEFAULT '{}',
            ingested_at     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_otlp_spans_session
            ON otlp_spans(session_id)
            WHERE session_id IS NOT NULL;
    """)
    conn.commit()
    conn.close()


def _init_work_events_table(db_path: str) -> None:
    """Create a minimal session_work_events table for testing."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session_work_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            materializer_version INTEGER NOT NULL DEFAULT 5,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            input_high_water_mark TEXT,
            input_high_water_mark_source TEXT,
            input_row_count INTEGER NOT NULL DEFAULT 0,
            source_name TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            heuristic_label TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            start_index INTEGER NOT NULL DEFAULT 0,
            end_index INTEGER NOT NULL DEFAULT 0,
            start_time TEXT,
            end_time TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            canonical_session_date TEXT,
            summary TEXT NOT NULL,
            file_paths_json TEXT,
            tools_used_json TEXT,
            evidence_payload_json TEXT NOT NULL DEFAULT '{}',
            inference_payload_json TEXT NOT NULL DEFAULT '{}',
            search_text TEXT NOT NULL,
            inference_version INTEGER NOT NULL DEFAULT 1,
            inference_family TEXT NOT NULL DEFAULT 'heuristic_session_semantics'
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT,
            session_id TEXT,
            role TEXT,
            text TEXT,
            sort_key REAL
        );
    """)
    conn.commit()
    conn.close()


def _insert_otlp_span(
    db_path: str,
    *,
    span_id: str = "span-1",
    trace_id: str = "trace-1",
    session_id: str = "session-1",
    operation_name: str = "tool.Read",
    start_time_unix_ns: int = 1_700_000_000_000_000_000,
    end_time_unix_ns: int = 1_700_000_000_001_000_000,
    duration_ms: int = 1000,
    status_code: int = 0,
    status_message: str | None = None,
    attributes_json: str = "{}",
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO otlp_spans
            (span_id, trace_id, session_id, operation_name,
             start_time_unix_ns, end_time_unix_ns, duration_ms,
             status_code, status_message, attributes_json, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2024-01-15T10:00:00')
        """,
        (
            span_id,
            trace_id,
            session_id,
            operation_name,
            start_time_unix_ns,
            end_time_unix_ns,
            duration_ms,
            status_code,
            status_message,
            attributes_json,
        ),
    )
    conn.commit()
    conn.close()


def _insert_work_event(
    db_path: str,
    *,
    event_id: str = "we-1",
    session_id: str = "session-1",
    heuristic_label: str = "tool_use",
    start_time: str = "2024-01-15T10:00:00",
    end_time: str = "2024-01-15T10:00:01",
    duration_ms: int = 1000,
    tools_used_json: str = '["Read"]',
    event_index: int = 0,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO session_work_events
            (event_id, session_id, materializer_version, materialized_at,
             source_name, event_index, heuristic_label, confidence,
             start_time, end_time, duration_ms, summary,
             tools_used_json, evidence_payload_json, inference_payload_json,
             search_text)
        VALUES (?, ?, 5, '2024-01-15T10:00:00', 'claude-code',
                ?, ?, 0.8, ?, ?, ?, '', ?, '{}', '{}', '')
        """,
        (
            event_id,
            session_id,
            event_index,
            heuristic_label,
            start_time,
            end_time,
            duration_ms,
            tools_used_json,
        ),
    )
    conn.commit()
    conn.close()


def _insert_message(
    db_path: str,
    *,
    msg_id: str = "msg-1",
    session_id: str = "session-1",
    role: str = "assistant",
    text: str = "",
    sort_key: float = 1705312830.0,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO messages (message_id, session_id, role, text, sort_key)
        VALUES (?, ?, ?, ?, ?)
        """,
        (msg_id, session_id, role, text, sort_key),
    )
    conn.commit()
    conn.close()


# ── Correlation tests ────────────────────────────────────────────────────


class TestCorrelateSpansToWorkEvents:
    def test_correlation_with_matching_overlap(self, tmp_path: Path) -> None:
        """Spans that overlap with work events in time are matched."""
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)
        _init_work_events_table(db_path)

        # Span from 10:00:00 to 10:00:01 (1s duration)
        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="tool.Read",
            start_time_unix_ns=1_700_000_000_000_000_000,  # ~2023-11-14T22:13:20
            end_time_unix_ns=1_700_000_000_001_000_000,  # +1s in ns
            duration_ms=1000,
        )
        # Work event at same time window
        _insert_work_event(
            db_path,
            event_id="we-1",
            session_id="session-1",
            start_time="2023-11-14T22:13:20",
            end_time="2023-11-14T22:13:21",
            tools_used_json='["Read"]',
        )

        results = correlate_spans_to_work_events(db_path, "session-1")
        assert len(results) == 1
        assert results[0]["otlp_enriched"] is True
        otlp_spans = cast("list[dict[str, object]]", results[0]["otlp_spans"])
        assert len(otlp_spans) == 1
        assert otlp_spans[0]["span_id"] == "span-1"
        assert otlp_spans[0]["evidence_source"] == "otlp_span"

    def test_empty_otlp_table_returns_unenriched(self, tmp_path: Path) -> None:
        """When otlp_spans has no rows, work events are returned unenriched."""
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)
        _init_work_events_table(db_path)

        _insert_work_event(
            db_path,
            event_id="we-1",
            session_id="session-1",
        )

        results = correlate_spans_to_work_events(db_path, "session-1")
        assert len(results) == 1
        assert results[0]["otlp_enriched"] is False
        assert results[0]["otlp_spans"] == []

    def test_partial_overlap_mixed_evidence(self, tmp_path: Path) -> None:
        """Some work events match spans, others don't."""
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)
        _init_work_events_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="tool.Read",
            start_time_unix_ns=1_700_000_000_000_000_000,
            end_time_unix_ns=1_700_000_000_001_000_000,
            duration_ms=1000,
        )

        # First work event overlaps the span
        _insert_work_event(
            db_path,
            event_id="we-1",
            session_id="session-1",
            start_time="2023-11-14T22:13:20",
            end_time="2023-11-14T22:13:21",
            event_index=0,
        )
        # Second work event does not overlap
        _insert_work_event(
            db_path,
            event_id="we-2",
            session_id="session-1",
            start_time="2023-11-15T00:00:00",
            end_time="2023-11-15T00:00:01",
            event_index=1,
        )

        results = correlate_spans_to_work_events(db_path, "session-1")
        assert len(results) == 2
        assert results[0]["otlp_enriched"] is True
        assert results[1]["otlp_enriched"] is False


# ── Tool timing tests ─────────────────────────────────────────────────────


class TestGetSessionToolTiming:
    def test_happy_path_with_otlp_data(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="tool.Read",
            duration_ms=500,
            status_code=0,
        )
        _insert_otlp_span(
            db_path,
            span_id="span-2",
            session_id="session-1",
            operation_name="tool.Write",
            duration_ms=300,
            status_code=2,
            status_message="file not found",
        )

        timing = get_session_tool_timing(db_path, "session-1")
        assert isinstance(timing, SessionToolTiming)
        assert timing.session_id == "session-1"
        assert timing.evidence_available is True
        assert timing.total_tools_with_otlp == 2
        assert timing.total_tools_total == 2
        assert len(timing.tool_timings) == 2
        assert timing.tool_timings[0].tool_name == "Read"
        assert timing.tool_timings[0].duration_ms == 500
        assert timing.tool_timings[0].status == "ok"
        assert timing.tool_timings[0].evidence_source == "otlp_span"
        assert timing.tool_timings[0].span_id == "span-1"
        assert timing.tool_timings[1].tool_name == "Write"
        assert timing.tool_timings[1].status == "error"
        assert timing.tool_timings[1].evidence_source == "otlp_span"

    def test_fallback_when_no_otlp_data(self, tmp_path: Path) -> None:
        """When otlp_spans table exists but is empty, fall back to work events."""
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)
        _init_work_events_table(db_path)

        _insert_work_event(
            db_path,
            event_id="we-1",
            session_id="session-1",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:00:01",
            duration_ms=1000,
            tools_used_json='["Read"]',
        )

        timing = get_session_tool_timing(db_path, "session-1")
        assert timing.evidence_available is False
        assert timing.total_tools_with_otlp == 0
        assert len(timing.tool_timings) == 1
        assert timing.tool_timings[0].tool_name == "Read"
        assert timing.tool_timings[0].evidence_source == "message_gap_estimate"

    def test_tool_spans_only_non_tool_spans_filtered(self, tmp_path: Path) -> None:
        """LLM spans should not appear in tool timing results."""
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-tool",
            session_id="session-1",
            operation_name="tool.Read",
            duration_ms=100,
        )
        _insert_otlp_span(
            db_path,
            span_id="span-llm",
            session_id="session-1",
            operation_name="llm.generate",
            duration_ms=5000,
        )

        timing = get_session_tool_timing(db_path, "session-1")
        assert len(timing.tool_timings) == 1
        assert timing.tool_timings[0].tool_name == "Read"

    def test_as_dict_output_shape(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="tool.Read",
            duration_ms=500,
        )

        timing = get_session_tool_timing(db_path, "session-1")
        payload = timing.as_dict()
        assert payload["session_id"] == "session-1"
        assert payload["evidence_available"] is True
        tool_timings = cast("list[dict[str, object]]", payload["tool_timings"])
        assert len(tool_timings) == 1
        assert tool_timings[0]["tool_name"] == "Read"
        assert tool_timings[0]["evidence_source"] == "otlp_span"


# ── LLM timing tests ──────────────────────────────────────────────────────


class TestGetSessionLLMTiming:
    def test_llm_timing_from_spans(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="llm.generate",
            duration_ms=5000,
            attributes_json='{"model_name": "claude-sonnet-4-20250514"}',
        )

        timing = get_session_llm_timing(db_path, "session-1")
        assert isinstance(timing, SessionLLMTiming)
        assert timing.evidence_available is True
        assert len(timing.llm_timings) == 1
        assert timing.llm_timings[0].duration_ms == 5000
        assert timing.llm_timings[0].model_name == "claude-sonnet-4-20250514"
        assert timing.llm_timings[0].evidence_source == "otlp_span"

    def test_llm_fallback_to_message_gaps(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)
        _init_work_events_table(db_path)

        _insert_message(
            db_path,
            msg_id="msg-1",
            session_id="session-1",
            role="user",
            sort_key=1705312800.0,
        )
        _insert_message(
            db_path,
            msg_id="msg-2",
            session_id="session-1",
            role="assistant",
            sort_key=1705312805.0,
        )

        timing = get_session_llm_timing(db_path, "session-1")
        assert timing.evidence_available is False
        assert len(timing.llm_timings) >= 1
        assert timing.llm_timings[0].evidence_source == "message_gap_estimate"

    def test_llm_as_dict_output_shape(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        _init_db_with_otlp_table(db_path)

        _insert_otlp_span(
            db_path,
            span_id="span-1",
            session_id="session-1",
            operation_name="llm.generate",
            duration_ms=5000,
        )

        timing = get_session_llm_timing(db_path, "session-1")
        payload = timing.as_dict()
        assert payload["session_id"] == "session-1"
        assert payload["evidence_available"] is True
        llm_timings = cast("list[dict[str, object]]", payload["llm_timings"])
        assert len(llm_timings) == 1
        assert llm_timings[0]["evidence_source"] == "otlp_span"
