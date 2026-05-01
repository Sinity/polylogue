from __future__ import annotations

from datetime import date, datetime

from polylogue.archive.conversation.extraction import WorkEvent, WorkEventKind
from polylogue.lib.session.session_profile import SessionProfile
from polylogue.lib.session.session_summaries import summarize_day, summarize_days, summarize_week, summarize_weeks


def _profile(
    conversation_id: str,
    *,
    provider: str,
    created_at: datetime | None,
    first_message_at: datetime | None,
    canonical_session_date: date | None,
    repo_names: tuple[str, ...] = (),
    repo_paths: tuple[str, ...] = (),
    work_events: tuple[WorkEvent, ...] = (),
    total_cost_usd: float = 0.0,
    total_duration_ms: int = 0,
    wall_duration_ms: int = 0,
    message_count: int = 0,
    word_count: int = 0,
) -> SessionProfile:
    return SessionProfile(
        conversation_id=conversation_id,
        provider=provider,
        title=None,
        created_at=created_at,
        updated_at=created_at,
        message_count=message_count,
        substantive_count=message_count,
        tool_use_count=0,
        thinking_count=0,
        attachment_count=0,
        word_count=word_count,
        total_cost_usd=total_cost_usd,
        total_duration_ms=total_duration_ms,
        tool_categories={},
        repo_paths=repo_paths,
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=(),
        languages_detected=(),
        repo_names=repo_names,
        work_events=work_events,
        phases=(),
        first_message_at=first_message_at,
        canonical_session_date=canonical_session_date,
        wall_duration_ms=wall_duration_ms,
    )


def _work_event(kind: WorkEventKind, index: int) -> WorkEvent:
    return WorkEvent(
        kind=kind,
        start_index=index,
        end_index=index,
        confidence=1.0,
        evidence=(kind.value,),
        file_paths=(),
        tools_used=(kind.value,),
        summary=f"{kind.value} event",
        start_time=datetime(2026, 4, 23, 12, index),
        end_time=datetime(2026, 4, 23, 12, index, 1),
    )


def test_summarize_day_aggregates_cost_duration_words_and_repos() -> None:
    profile = _profile(
        "conv-1",
        provider="claude-code",
        created_at=datetime(2026, 4, 23, 8, 0),
        first_message_at=None,
        canonical_session_date=date(2026, 4, 23),
        repo_names=("polylogue",),
        work_events=(_work_event(WorkEventKind.TESTING, 0), _work_event(WorkEventKind.RESEARCH, 1)),
        total_cost_usd=0.125,
        total_duration_ms=900,
        wall_duration_ms=1200,
        message_count=6,
        word_count=42,
    )

    summary = summarize_day([profile], date(2026, 4, 23))

    assert summary.to_dict()["total_cost_usd"] == 0.125
    assert summary.work_event_breakdown == {"testing": 1, "research": 1}
    assert summary.repos_active == ("polylogue",)
    assert summary.providers == {"claude-code": 1}


def test_summarize_days_and_weeks_use_profile_date_fallbacks_and_sort_descending() -> None:
    first = _profile(
        "conv-first",
        provider="chatgpt",
        created_at=datetime(2026, 4, 21, 9, 0),
        first_message_at=datetime(2026, 4, 22, 9, 0),
        canonical_session_date=None,
        repo_names=("polylogue",),
        message_count=3,
        word_count=10,
    )
    second = _profile(
        "conv-second",
        provider="codex",
        created_at=datetime(2026, 4, 23, 9, 0),
        first_message_at=None,
        canonical_session_date=None,
        repo_names=("sinex",),
        message_count=4,
        word_count=20,
    )
    ignored = _profile(
        "conv-ignored",
        provider="claude-ai",
        created_at=None,
        first_message_at=None,
        canonical_session_date=None,
    )

    day_summaries = summarize_days([first, second, ignored])
    week_summary = summarize_week(day_summaries)
    week_summaries = summarize_weeks(day_summaries)

    assert [summary.date.isoformat() for summary in day_summaries] == ["2026-04-23", "2026-04-22"]
    assert week_summary.iso_week == "2026-W17"
    assert week_summary.session_count == 2
    assert week_summary.total_messages == 7
    assert week_summary.to_dict()["iso_week"] == "2026-W17"
    assert [summary.iso_week for summary in week_summaries] == ["2026-W17"]
    assert summarize_week(()).iso_week == ""
