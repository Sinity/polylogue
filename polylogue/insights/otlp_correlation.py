"""OTLP span-to-work-event correlation (#1686).

Enriches session work events with exact wallclock timing from OTLP spans,
and provides per-tool-call and LLM timing breakdowns. When OTLP data is
absent, falls back to message-gap estimates with appropriate provenance
markers.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

# ── Result models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolTimingEntry:
    """Timing for a single tool call, with evidence provenance."""

    tool_name: str
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    status: str = "unknown"
    evidence_source: str = "message_gap_estimate"
    """'otlp_span' for exact OTLP timing, 'message_gap_estimate' for inferred."""

    span_id: str | None = None
    """OTLP span_id when evidence_source is 'otlp_span'."""


@dataclass(frozen=True)
class LLMTimingEntry:
    """LLM thinking/inference timing for a message turn."""

    turn_index: int
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    model_name: str | None = None
    evidence_source: str = "message_gap_estimate"
    span_id: str | None = None


@dataclass(frozen=True)
class SessionToolTiming:
    """Per-tool timing breakdown for a session from OTLP evidence."""

    session_id: str
    tool_timings: list[ToolTimingEntry] = field(default_factory=list)
    evidence_available: bool = False
    total_tools_with_otlp: int = 0
    total_tools_total: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "tool_timings": [
                {
                    "tool_name": t.tool_name,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "duration_ms": t.duration_ms,
                    "status": t.status,
                    "evidence_source": t.evidence_source,
                    "span_id": t.span_id,
                }
                for t in self.tool_timings
            ],
            "evidence_available": self.evidence_available,
            "total_tools_with_otlp": self.total_tools_with_otlp,
            "total_tools_total": self.total_tools_total,
        }


@dataclass(frozen=True)
class SessionLLMTiming:
    """LLM timing breakdown for a session."""

    session_id: str
    llm_timings: list[LLMTimingEntry] = field(default_factory=list)
    evidence_available: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "llm_timings": [
                {
                    "turn_index": t.turn_index,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "duration_ms": t.duration_ms,
                    "model_name": t.model_name,
                    "evidence_source": t.evidence_source,
                    "span_id": t.span_id,
                }
                for t in self.llm_timings
            ],
            "evidence_available": self.evidence_available,
        }


# ── Query helpers ────────────────────────────────────────────────────────


@contextlib.contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with Row factory and always close it.

    A bare ``with sqlite3.connect(...)`` only commits on exit, never closes —
    these are read-only query helpers, so the per-call connection would leak.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _has_otlp_data(db_path: str) -> bool:
    """Check whether any OTLP spans exist in the archive."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT 1 FROM otlp_spans LIMIT 1").fetchone()
        return row is not None


def _query_spans_for_session(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Return all OTLP spans for a session, ordered by start_time."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT span_id, trace_id, parent_span_id, agent_id,
                   operation_name, start_time_unix_ns, end_time_unix_ns,
                   duration_ms, status_code, status_message,
                   attributes_json
            FROM otlp_spans
            WHERE session_id = ?
            ORDER BY start_time_unix_ns ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def _query_work_events(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Return session work events for a session."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM session_work_events WHERE conversation_id = ? ORDER BY event_index ASC",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def _query_messages(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Return messages for a session.

    Returns rows with ``message_id``, ``role``, and ``sort_key`` (epoch seconds).
    The ``sort_key`` mirrors ``Conversation.sort_key`` semantics — present for
    every message, stable within a source family.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT message_id, role, sort_key
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sort_key ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def _unix_ns_to_iso(unix_ns: int) -> str:
    """Convert Unix nanoseconds to ISO 8601 string (UTC)."""
    seconds = unix_ns / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.isoformat()


# ── Work-event correlation ───────────────────────────────────────────────


def correlate_spans_to_work_events(db_path: str, session_id: str) -> list[dict[str, object]]:
    """Join otlp_spans with session_work_events on time overlap.

    Returns enriched work events with OTLP timing where available.
    Each result includes both the work event fields and any matching
    span data, plus an ``otlp_enriched`` boolean flag.
    """
    spans = _query_spans_for_session(db_path, session_id)

    # Index spans by time range for fast overlap lookup
    indexed_spans: list[dict[str, object]] = [
        {
            **cast(dict[str, object], s),
            "start_time_iso": _unix_ns_to_iso(cast(int, s["start_time_unix_ns"])),
            "end_time_iso": _unix_ns_to_iso(cast(int, s["end_time_unix_ns"])),
        }
        for s in spans
    ]

    work_event_rows = _query_work_events(db_path, session_id)

    results: list[dict[str, object]] = []
    for we in work_event_rows:
        we_start = cast("str | None", we.get("start_time"))
        we_end = cast("str | None", we.get("end_time"))
        tools_used: list[str] = []
        tools_json = we.get("tools_used_json")
        if tools_json and isinstance(tools_json, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                tools_used = json.loads(tools_json)

        # Find overlapping spans — a span overlaps if its time range
        # intersects the work event's time range.
        matched_spans: list[dict[str, object]] = []
        for span in indexed_spans:
            span_start = cast(str, span["start_time_iso"])
            span_end = cast(str, span["end_time_iso"])
            has_overlap = True
            if we_start and span_end < we_start:
                has_overlap = False
            if we_end and span_start > we_end:
                has_overlap = False
            if has_overlap:
                matched_spans.append(
                    {
                        "span_id": span["span_id"],
                        "trace_id": span["trace_id"],
                        "operation_name": span["operation_name"],
                        "start_time": span["start_time_iso"],
                        "end_time": span["end_time_iso"],
                        "duration_ms": span["duration_ms"],
                        "status_code": span["status_code"],
                        "status_message": span.get("status_message"),
                        "evidence_source": "otlp_span",
                    }
                )

        enriched: dict[str, object] = {
            "event_id": we["event_id"],
            "conversation_id": we["conversation_id"],
            "heuristic_label": we["heuristic_label"],
            "start_time": we_start,
            "end_time": we_end,
            "duration_ms": we["duration_ms"],
            "summary": we["summary"],
            "tools_used": tools_used,
            "otlp_enriched": len(matched_spans) > 0,
            "otlp_spans": matched_spans,
        }
        results.append(enriched)

    return results


# ── Tool timing ──────────────────────────────────────────────────────────


def get_session_tool_timing(db_path: str, session_id: str) -> SessionToolTiming:
    """Return per-tool-call timing breakdown from OTLP evidence.

    Matches OTLP spans with operation names containing 'tool' or matching
    known tool-call patterns against session work events. Falls back to
    message-gap estimates when OTLP data is unavailable.
    """
    if not _has_otlp_data(db_path):
        return _tool_timing_from_work_events(db_path, session_id)

    spans = _query_spans_for_session(db_path, session_id)
    if not spans:
        return _tool_timing_from_work_events(db_path, session_id)

    # Filter to tool-related spans
    tool_spans = [s for s in spans if _is_tool_span(cast(str, s["operation_name"]))]

    if not tool_spans:
        return _tool_timing_from_work_events(db_path, session_id)

    timings: list[ToolTimingEntry] = []
    for span in tool_spans:
        timings.append(
            ToolTimingEntry(
                tool_name=_extract_tool_name(cast(str, span["operation_name"])),
                start_time=_unix_ns_to_iso(cast(int, span["start_time_unix_ns"])),
                end_time=_unix_ns_to_iso(cast(int, span["end_time_unix_ns"])),
                duration_ms=cast(int, span["duration_ms"]),
                status=_status_code_to_label(cast(int, span["status_code"])),
                evidence_source="otlp_span",
                span_id=cast("str | None", span.get("span_id")),
            )
        )

    total = len(timings)
    return SessionToolTiming(
        session_id=session_id,
        tool_timings=timings,
        evidence_available=True,
        total_tools_with_otlp=total,
        total_tools_total=total,
    )


def _tool_timing_from_work_events(db_path: str, session_id: str) -> SessionToolTiming:
    """Fallback: estimate tool timing from work events (message gaps)."""
    rows = _query_work_events(db_path, session_id)

    timings: list[ToolTimingEntry] = []
    for d in rows:
        tools_used: list[str] = []
        tools_json = d.get("tools_used_json")
        if tools_json and isinstance(tools_json, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                tools_used = json.loads(tools_json)

        if not tools_used:
            continue

        for tool_name in tools_used:
            timings.append(
                ToolTimingEntry(
                    tool_name=tool_name,
                    start_time=cast("str | None", d.get("start_time")),
                    end_time=cast("str | None", d.get("end_time")),
                    duration_ms=cast(int, d.get("duration_ms", 0)),
                    evidence_source="message_gap_estimate",
                )
            )

    return SessionToolTiming(
        session_id=session_id,
        tool_timings=timings,
        evidence_available=False,
        total_tools_with_otlp=0,
        total_tools_total=len(timings),
    )


# ── LLM timing ───────────────────────────────────────────────────────────


def get_session_llm_timing(db_path: str, session_id: str) -> SessionLLMTiming:
    """Return model thinking time from OTLP llm_request spans.

    Falls back to inter-message gap estimates when OTLP data is absent.
    """
    if not _has_otlp_data(db_path):
        return _llm_timing_from_message_gaps(db_path, session_id)

    spans = _query_spans_for_session(db_path, session_id)
    llm_spans = [s for s in spans if _is_llm_span(cast(str, s["operation_name"]))]

    if not llm_spans:
        return _llm_timing_from_message_gaps(db_path, session_id)

    timings: list[LLMTimingEntry] = []
    for i, span in enumerate(llm_spans):
        attrs: dict[str, Any] = {}
        attrs_json = span.get("attributes_json")
        if attrs_json and isinstance(attrs_json, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                attrs = json.loads(attrs_json)

        model_name_raw = attrs.get("model_name")
        model_name: str | None = str(model_name_raw) if isinstance(model_name_raw, str) else None

        timings.append(
            LLMTimingEntry(
                turn_index=i,
                start_time=_unix_ns_to_iso(cast(int, span["start_time_unix_ns"])),
                end_time=_unix_ns_to_iso(cast(int, span["end_time_unix_ns"])),
                duration_ms=cast(int, span["duration_ms"]),
                model_name=model_name,
                evidence_source="otlp_span",
                span_id=cast("str | None", span.get("span_id")),
            )
        )

    return SessionLLMTiming(
        session_id=session_id,
        llm_timings=timings,
        evidence_available=True,
    )


def _llm_timing_from_message_gaps(db_path: str, session_id: str) -> SessionLLMTiming:
    """Fallback: estimate LLM timing from inter-message gaps."""
    rows = _query_messages(db_path, session_id)

    timings: list[LLMTimingEntry] = []
    prev_sort_key: float | None = None
    prev_iso: str | None = None
    turn_index = 0

    for d in rows:
        role = cast(str, d.get("role", ""))
        current_sk_raw = d.get("sort_key")
        try:
            current_sk = float(current_sk_raw) if current_sk_raw is not None else None
        except (TypeError, ValueError):
            current_sk = None
        current_iso = (
            datetime.fromtimestamp(current_sk, tz=timezone.utc).isoformat() if current_sk is not None else None
        )

        # Assistant messages represent LLM turns
        if role in ("assistant", "Assistant"):
            start = prev_iso or current_iso
            duration_ms = 0
            if current_sk is not None and prev_sort_key is not None:
                duration_ms = max(0, int((current_sk - prev_sort_key) * 1000))
            timings.append(
                LLMTimingEntry(
                    turn_index=turn_index,
                    start_time=start,
                    end_time=current_iso,
                    duration_ms=duration_ms,
                    evidence_source="message_gap_estimate",
                )
            )
            turn_index += 1

        prev_sort_key = current_sk
        prev_iso = current_iso

    return SessionLLMTiming(
        session_id=session_id,
        llm_timings=timings,
        evidence_available=False,
    )


# ── Span classification helpers ──────────────────────────────────────────


_TOOL_SPAN_PREFIXES = (
    "tool.",
    "tool_",
    "agent.",
    "agent_",
    "mcp.",
    "mcp_",
)


def _is_tool_span(operation_name: str) -> bool:
    """Determine whether a span operation represents a tool call."""
    name_lower = operation_name.lower()
    return any(name_lower.startswith(p) for p in _TOOL_SPAN_PREFIXES)


_LLM_SPAN_PREFIXES = (
    "llm.",
    "llm_",
    "model.",
    "model_",
    "generate.",
    "generate_",
    "chat.",
    "chat_",
    "complete.",
    "complete_",
    "inference.",
    "inference_",
)


def _is_llm_span(operation_name: str) -> bool:
    """Determine whether a span operation represents an LLM call."""
    name_lower = operation_name.lower()
    return any(name_lower.startswith(p) for p in _LLM_SPAN_PREFIXES)


def _extract_tool_name(operation_name: str) -> str:
    """Extract a human-readable tool name from the operation name."""
    # Strip common prefixes
    for prefix in _TOOL_SPAN_PREFIXES:
        if operation_name.lower().startswith(prefix):
            return operation_name[len(prefix) :]
    return operation_name


_STATUS_CODE_MAP: dict[int, str] = {0: "ok", 1: "ok", 2: "error"}


def _status_code_to_label(status_code: int) -> str:
    """Map OTLP status codes to human labels."""
    return _STATUS_CODE_MAP.get(status_code, "unknown")


__all__ = [
    "LLMTimingEntry",
    "SessionLLMTiming",
    "SessionToolTiming",
    "ToolTimingEntry",
    "correlate_spans_to_work_events",
    "get_session_llm_timing",
    "get_session_tool_timing",
]
