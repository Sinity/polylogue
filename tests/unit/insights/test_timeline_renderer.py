"""Contract tests for the session-timeline renderer (#1135).

These tests pin the fidelity-tag invariant — only ``timestamped_range``
provenance maps to the user-visible ``hook`` tag; every other recognized
provenance string maps to ``sort_key``. They also pin the renderer's
chronological ordering and the plain / markdown output contracts.
"""

from __future__ import annotations

import json

import pytest

from polylogue.insights.archive import SessionPhaseInsight, SessionWorkEventInsight
from polylogue.insights.archive_models import (
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.timeline_renderer import (
    SessionTimeline,
    build_session_timeline,
    build_timeline_entries,
    fidelity_for,
    render_markdown,
    render_plain,
)


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
    )


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        inference_version=1,
        inference_family="heuristic_session_semantics",
    )


def _work_event(
    *,
    event_id: str = "evt-1",
    session_id: str = "conv-1",
    event_index: int = 0,
    kind: str = "implementation",
    summary: str = "editing files",
    confidence: float = 0.8,
    start_time: str | None = "2026-03-24T10:00:00+00:00",
    end_time: str | None = "2026-03-24T10:05:00+00:00",
    duration_ms: int = 300_000,
    timing_provenance: str = "timestamped_range",
    tools_used: tuple[str, ...] = ("edit",),
    file_paths: tuple[str, ...] = ("a.py",),
) -> SessionWorkEventInsight:
    return SessionWorkEventInsight(
        event_id=event_id,
        session_id=session_id,
        source_name="claude-code",
        event_index=event_index,
        provenance=_provenance(),
        inference_provenance=_inference_provenance(),
        evidence=WorkEventEvidencePayload(
            start_index=0,
            end_index=1,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            timing_provenance=timing_provenance,
            tools_used=tools_used,
            file_paths=file_paths,
        ),
        inference=WorkEventInferencePayload(
            heuristic_label=kind,
            summary=summary,
            confidence=confidence,
        ),
    )


def _phase(
    *,
    phase_id: str = "phase-1",
    session_id: str = "conv-1",
    phase_index: int = 0,
    start_time: str | None = "2026-03-24T10:06:00+00:00",
    end_time: str | None = "2026-03-24T10:09:00+00:00",
    duration_ms: int = 180_000,
    timing_provenance: str = "start_timestamp_only",
    word_count: int = 42,
) -> SessionPhaseInsight:
    return SessionPhaseInsight(
        phase_id=phase_id,
        session_id=session_id,
        source_name="claude-code",
        phase_index=phase_index,
        provenance=_provenance(),
        inference_provenance=_inference_provenance(),
        evidence=SessionPhaseEvidencePayload(
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            timing_provenance=timing_provenance,
            message_range=(0, 2),
            word_count=word_count,
        ),
        inference=SessionPhaseInferencePayload(confidence=0.5),
    )


class TestFidelityForInvariant:
    """The fidelity tag is the user-visible contract."""

    def test_timestamped_range_is_hook(self) -> None:
        assert fidelity_for("timestamped_range") == "hook"

    @pytest.mark.parametrize(
        "provenance",
        [
            "start_timestamp_only",
            "end_timestamp_only",
            "untimestamped",
            "sort_key_estimated",
            "unknown_value",
            "",
            None,
        ],
    )
    def test_non_hook_provenances_are_sort_key(self, provenance: str | None) -> None:
        assert fidelity_for(provenance) == "sort_key"


class TestBuildTimeline:
    def test_work_event_with_timestamped_range_tagged_hook(self) -> None:
        entries = build_timeline_entries([_work_event()], [])
        assert len(entries) == 1
        assert entries[0].fidelity == "hook"
        assert entries[0].source == "work_event"
        assert entries[0].timing_provenance == "timestamped_range"

    def test_phase_with_partial_timestamp_tagged_sort_key(self) -> None:
        entries = build_timeline_entries([], [_phase()])
        assert len(entries) == 1
        assert entries[0].fidelity == "sort_key"
        assert entries[0].source == "phase"

    def test_chronological_ordering_across_sources(self) -> None:
        we_late = _work_event(event_id="we-late", start_time="2026-03-24T11:00:00+00:00")
        we_early = _work_event(event_id="we-early", start_time="2026-03-24T09:00:00+00:00")
        ph_mid = _phase(phase_id="ph-mid", start_time="2026-03-24T10:00:00+00:00")
        entries = build_timeline_entries([we_late, we_early], [ph_mid])
        assert [e.entry_id for e in entries] == ["we-early", "ph-mid", "we-late"]

    def test_untimed_entries_sort_last_preserving_input_order(self) -> None:
        we_timed = _work_event(event_id="timed", start_time="2026-03-24T10:00:00+00:00")
        we_untimed_a = _work_event(
            event_id="untimed-a", start_time=None, end_time=None, timing_provenance="untimestamped"
        )
        we_untimed_b = _work_event(
            event_id="untimed-b", start_time=None, end_time=None, timing_provenance="untimestamped"
        )
        entries = build_timeline_entries([we_untimed_a, we_timed, we_untimed_b], [])
        assert [e.entry_id for e in entries] == ["timed", "untimed-a", "untimed-b"]

    def test_fidelity_counts_aggregate(self) -> None:
        timeline = build_session_timeline(
            "conv-1",
            [
                _work_event(event_id="hook-1"),
                _work_event(
                    event_id="sortkey-1",
                    timing_provenance="end_timestamp_only",
                    start_time=None,
                ),
            ],
            [_phase(phase_id="ph-sortkey")],
        )
        assert timeline.fidelity_counts == {"hook": 1, "sort_key": 2}


class TestHookPrecedence:
    """If two entries describe overlapping time, the hook-tagged one wins
    precedence in any caller that has to pick one. We pin this as a
    classification invariant on the per-entry tag — collision-resolution
    happens at the consumer, but the tag itself must be unambiguous.
    """

    def test_hook_and_sort_key_remain_distinct_on_same_session(self) -> None:
        timeline = build_session_timeline(
            "conv-1",
            [
                _work_event(event_id="precise", timing_provenance="timestamped_range"),
                _work_event(
                    event_id="approx",
                    timing_provenance="start_timestamp_only",
                    end_time=None,
                ),
            ],
            [],
        )
        tags = {e.entry_id: e.fidelity for e in timeline.entries}
        assert tags == {"precise": "hook", "approx": "sort_key"}


class TestRenderPlain:
    def test_includes_fidelity_legend_and_marker(self) -> None:
        timeline = build_session_timeline(
            "conv-xyz",
            [_work_event()],
            [_phase()],
        )
        output = render_plain(timeline)
        assert "Timeline for conv-xyz" in output
        assert "hook=1" in output
        assert "sort_key=1" in output
        # Marker column appears for both entries
        lines = output.splitlines()
        data_lines = [line for line in lines if line.startswith("H ") or line.startswith("S ")]
        assert any(line.startswith("H ") for line in data_lines)
        assert any(line.startswith("S ") for line in data_lines)
        assert "legend:" in output

    def test_empty_timeline_message(self) -> None:
        timeline = build_session_timeline("conv-empty", [], [])
        output = render_plain(timeline)
        assert "(no timeline rows)" in output


class TestRenderMarkdown:
    def test_markdown_table_includes_fidelity_column(self) -> None:
        timeline = build_session_timeline(
            "conv-md",
            [_work_event()],
            [_phase()],
        )
        output = render_markdown(timeline)
        assert "# Timeline — `conv-md`" in output
        assert "| Fidelity | Source | Kind |" in output
        assert "| hook |" in output
        assert "| sort_key |" in output

    def test_escapes_pipes_in_summary(self) -> None:
        timeline = build_session_timeline(
            "conv-md",
            [_work_event(summary="a | b | c")],
            [],
        )
        output = render_markdown(timeline)
        # Summary cell contains escaped pipes, table structure preserved
        assert "a \\| b \\| c" in output

    def test_empty_timeline_message_markdown(self) -> None:
        timeline = build_session_timeline("conv-empty", [], [])
        assert "_No timeline rows._" in render_markdown(timeline)


class TestJsonContract:
    def test_to_dict_preserves_timing_provenance_alongside_fidelity(self) -> None:
        timeline = build_session_timeline(
            "conv-json",
            [_work_event(timing_provenance="timestamped_range")],
            [_phase(timing_provenance="end_timestamp_only", start_time=None)],
        )
        payload = timeline.to_dict()
        # The contract surfaces both the user-visible tag and the raw
        # provenance value so downstream consumers can recover precision.
        assert payload["session_id"] == "conv-json"
        assert payload["fidelity_counts"] == {"hook": 1, "sort_key": 1}
        raw_entries = payload["entries"]
        assert isinstance(raw_entries, list)
        entries: list[dict[str, object]] = raw_entries
        we = next(e for e in entries if e["source"] == "work_event")
        assert we["fidelity"] == "hook"
        assert we["timing_provenance"] == "timestamped_range"
        ph = next(e for e in entries if e["source"] == "phase")
        assert ph["fidelity"] == "sort_key"
        assert ph["timing_provenance"] == "end_timestamp_only"
        # Payload is JSON-serializable as-is
        json.dumps(payload)


class TestSessionTimelineDataclass:
    def test_frozen(self) -> None:
        timeline = SessionTimeline(session_id="x", entries=())
        with pytest.raises((AttributeError, TypeError)):
            timeline.session_id = "y"  # type: ignore[misc]
