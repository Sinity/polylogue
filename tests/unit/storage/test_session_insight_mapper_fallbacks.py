from __future__ import annotations

import json
import sqlite3

import pytest

from polylogue.storage.sqlite.queries.mappers_insight_profiles import _row_to_session_profile_record
from polylogue.storage.sqlite.queries.mappers_insight_timelines import (
    _row_to_session_phase_record,
    _row_to_session_work_event_record,
)


def _make_row(columns: dict[str, object]) -> sqlite3.Row:
    names = list(columns)
    values = [columns[name] for name in names]

    def _sqlite_type(value: object) -> str:
        if isinstance(value, (bool, int)):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        return "TEXT"

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    column_defs = ", ".join(f'"{name}" {_sqlite_type(columns[name])}' for name in names)
    placeholders = ", ".join("?" for _ in names)
    conn.execute(f"CREATE TABLE t ({column_defs})")
    conn.execute(f"INSERT INTO t VALUES ({placeholders})", values)
    row = conn.execute("SELECT * FROM t").fetchone()
    conn.close()
    assert row is not None
    assert isinstance(row, sqlite3.Row)
    return row


def _typed_payload_columns(mode: str, *names: str) -> dict[str, object]:
    if mode == "missing":
        return {}
    return dict.fromkeys(names, "{}")


@pytest.mark.parametrize("typed_payload_mode", ["missing", "empty"], ids=["missing-columns", "empty-columns"])
def test_row_to_session_profile_record_falls_back_to_fallback_payloads(typed_payload_mode: str) -> None:
    fallback_payload = {
        "created_at": "2026-04-10T09:00:00+00:00",
        "updated_at": "2026-04-10T09:05:00+00:00",
        "first_message_at": "2026-04-10T09:00:00+00:00",
        "last_message_at": "2026-04-10T09:05:00+00:00",
        "canonical_session_date": "2026-04-10",
        "message_count": 12,
        "substantive_count": 9,
        "attachment_count": 2,
        "tool_use_count": 3,
        "thinking_count": 1,
        "word_count": 120,
        "total_cost_usd": 1.25,
        "total_duration_ms": 300000,
        "wall_duration_ms": 305000,
        "cost_is_estimated": True,
        "tool_categories": {"read": 2, "edit": 1},
        "repo_paths": ["/workspace/polylogue/app.py", "/workspace/polylogue/tests.py"],
        "cwd_paths": ["/workspace/polylogue"],
        "branch_names": ["feature/refactor/storage-lib-product-runtime-renewal"],
        "file_paths_touched": ["/workspace/polylogue/app.py"],
        "languages_detected": ["python"],
        "tags": ["urgent"],
        "is_continuation": True,
        "parent_id": "conv-parent",
        "repo_names": ["polylogue"],
        "work_event_count": 4,
        "phase_count": 2,
        "engaged_duration_ms": 240000,
        "engaged_minutes": 4.0,
        "support_level": "strong",
        "support_signals": ["user_turns", "action_events"],
        "engaged_duration_source": "timeline",
        "repo_inference_strength": "strong",
        "auto_tags": ["bugfix"],
        "work_events": [{"heuristic_label": "file_edit"}],
        "phases": [{"kind": "implementation"}],
    }
    row = _make_row(
        {
            "session_id": "conv-profile-fallback",
            "materialized_at": "2026-04-10T09:06:00+00:00",
            "source_name": "claude-code",
            "title": "Fallback Profile",
            "message_count": 12,
            "repo_paths_json": json.dumps(["/workspace/polylogue/app.py", "/workspace/polylogue/tests.py"]),
            "repo_names_json": json.dumps(["polylogue"]),
            "tags_json": json.dumps(["urgent"]),
            "auto_tags_json": json.dumps(["bugfix"]),
            "search_text": "fallback profile search text",
            "payload_json": json.dumps(fallback_payload),
            **_typed_payload_columns(
                typed_payload_mode,
                "evidence_payload_json",
                "inference_payload_json",
                "enrichment_payload_json",
            ),
        }
    )

    record = _row_to_session_profile_record(row)

    assert str(record.session_id) == "conv-profile-fallback"
    assert record.source_name == "claude-code"
    assert record.title == "Fallback Profile"
    assert record.repo_paths == ("/workspace/polylogue/app.py", "/workspace/polylogue/tests.py")
    assert record.search_text == "fallback profile search text"

    assert record.evidence_payload.created_at == "2026-04-10T09:00:00+00:00"
    assert record.evidence_payload.tool_categories == {"read": 2, "edit": 1}
    assert record.inference_payload.support_level == "strong"
    assert record.inference_payload.auto_tags == ("bugfix",)
    assert record.enrichment_payload.intent_summary == "Fallback Profile"
    assert record.enrichment_payload.support_signals == ("user_turns", "action_events")
    assert record.enrichment_payload.input_band_summary == {
        "user_turns": 0,
        "assistant_turns": 0,
        "action_events": 0,
        "touched_paths": 2,
        "repo_names": 1,
    }


def test_row_to_session_profile_record_normalizes_fallback_typed_inference_events() -> None:
    inference_payload = {
        "repo_names": [],
        "work_event_count": 1,
        "phase_count": 0,
        "engaged_duration_ms": 0,
        "engaged_minutes": 0.0,
        "support_level": "weak",
        "support_signals": ["work_events"],
        "engaged_duration_source": "session_total_fallback",
        "repo_inference_strength": "none",
        "auto_tags": ["provider:codex"],
        "work_events": [
            {
                "heuristic_label": "planning",
                "start_index": 0,
                "end_index": 2,
                "start_time": None,
                "end_time": None,
                "canonical_session_date": "2026-05-07",
                "duration_ms": 0,
                "confidence": 0.2,
                "evidence": ["fallback event"],
                "file_paths": [],
                "tools_used": [],
                "summary": "fallback planning",
            }
        ],
        "phases": [],
    }
    row = _make_row(
        {
            "session_id": "conv-profile-typed-fallback",
            "materialized_at": "2026-05-07T09:06:00+00:00",
            "source_name": "codex",
            "title": "Typed fallback profile",
            "message_count": 3,
            "repo_paths_json": "[]",
            "repo_names_json": "[]",
            "tags_json": "[]",
            "auto_tags_json": json.dumps(["provider:codex"]),
            "search_text": "typed fallback profile search text",
            "payload_json": "{}",
            "evidence_payload_json": "{}",
            "inference_payload_json": json.dumps(inference_payload),
            "enrichment_payload_json": "{}",
        }
    )

    record = _row_to_session_profile_record(row)

    assert record.inference_payload.work_events[0]["timing_provenance"] == "untimestamped"
    assert record.inference_payload.work_events[0]["date_provenance"] == "date_only"


@pytest.mark.parametrize("typed_payload_mode", ["missing", "empty"], ids=["missing-columns", "empty-columns"])
def test_row_to_session_work_event_record_falls_back_to_fallback_payloads(typed_payload_mode: str) -> None:
    fallback_payload = {
        "start_index": 3,
        "end_index": 7,
        "start_time": "2026-04-11T10:00:00+00:00",
        "end_time": "2026-04-11T10:04:00+00:00",
        "canonical_session_date": "2026-04-11",
        "duration_ms": 240000,
        "file_paths": ["/workspace/polylogue/app.py"],
        "tools_used": ["read"],
        "confidence": 0.91,
        "evidence": ["read app.py before patching"],
    }
    row = _make_row(
        {
            "event_id": "event-fallback",
            "session_id": "conv-event-fallback",
            "materialized_at": "2026-04-11T10:05:00+00:00",
            "source_name": "claude-code",
            "event_index": 1,
            "heuristic_label": "file_edit",
            "summary": "Patched the failing test path",
            "search_text": "fallback work event search text",
            "payload_json": json.dumps(fallback_payload),
            **_typed_payload_columns(typed_payload_mode, "evidence_payload_json", "inference_payload_json"),
        }
    )

    record = _row_to_session_work_event_record(row)

    assert record.event_id == "event-fallback"
    assert str(record.session_id) == "conv-event-fallback"
    assert record.heuristic_label == "file_edit"
    assert record.summary == "Patched the failing test path"
    assert record.search_text == "fallback work event search text"

    assert record.evidence_payload.start_index == 3
    assert record.evidence_payload.duration_ms == 240000
    assert record.evidence_payload.file_paths == ("/workspace/polylogue/app.py",)
    assert record.evidence_payload.timing_provenance == "timestamped_range"
    assert record.evidence_payload.date_provenance == "event_timestamp"
    assert record.inference_payload.heuristic_label == "file_edit"
    assert record.inference_payload.summary == "Patched the failing test path"
    assert record.inference_payload.confidence == pytest.approx(0.91)
    assert record.inference_payload.evidence == ("read app.py before patching",)


@pytest.mark.parametrize("typed_payload_mode", ["missing", "empty"], ids=["missing-columns", "empty-columns"])
def test_row_to_session_phase_record_falls_back_to_fallback_payloads(typed_payload_mode: str) -> None:
    fallback_payload = {
        "start_time": "2026-04-12T11:00:00+00:00",
        "end_time": "2026-04-12T11:06:00+00:00",
        "canonical_session_date": "2026-04-12",
        "start_index": 2,
        "end_index": 8,
        "duration_ms": 360000,
        "tool_counts": {"read": 1, "edit": 1},
        "word_count": 180,
        "confidence": 0.73,
        "evidence": ["tool-use burst", "assistant outcome"],
    }
    row = _make_row(
        {
            "phase_id": "phase-fallback",
            "session_id": "conv-phase-fallback",
            "materialized_at": "2026-04-12T11:07:00+00:00",
            "source_name": "claude-code",
            "phase_index": 0,
            "kind": "implementation",
            "search_text": "fallback phase search text",
            "payload_json": json.dumps(fallback_payload),
            **_typed_payload_columns(typed_payload_mode, "evidence_payload_json", "inference_payload_json"),
        }
    )

    record = _row_to_session_phase_record(row)

    assert record.phase_id == "phase-fallback"
    assert str(record.session_id) == "conv-phase-fallback"
    assert record.kind == "implementation"
    assert record.search_text == "fallback phase search text"

    assert record.evidence_payload.message_range == (2, 8)
    assert record.evidence_payload.tool_counts == {"read": 1, "edit": 1}
    assert record.evidence_payload.word_count == 180
    assert record.inference_payload.confidence == pytest.approx(0.73)
    assert record.inference_payload.evidence == ("tool-use burst", "assistant outcome")
