from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.conversation.threads import WorkThread, build_session_threads
from polylogue.lib.session.session_profile import SessionProfile


def _profile(
    conversation_id: str,
    *,
    parent_id: str | None = None,
    repo_names: tuple[str, ...] = ("polylogue",),
    first_message_at: datetime | None = None,
    last_message_at: datetime | None = None,
) -> SessionProfile:
    return SessionProfile(
        conversation_id=conversation_id,
        provider="claude-code",
        title=conversation_id,
        created_at=first_message_at,
        updated_at=last_message_at or first_message_at,
        message_count=2,
        substantive_count=2,
        tool_use_count=0,
        thinking_count=0,
        attachment_count=0,
        word_count=12,
        total_cost_usd=0.0,
        total_duration_ms=0,
        tool_categories={},
        repo_paths=(),
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=(),
        languages_detected=(),
        repo_names=repo_names,
        work_events=(),
        phases=(),
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        timestamped_message_count=2 if first_message_at and last_message_at else 0,
        timestamp_coverage="complete" if first_message_at and last_message_at else "none",
        parent_id=parent_id,
        is_continuation=parent_id is not None,
    )


def test_build_session_threads_explains_explicit_continuation_membership() -> None:
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc)

    [thread] = build_session_threads(
        [
            _profile("root", first_message_at=start, last_message_at=start),
            _profile("child-a", parent_id="root", first_message_at=end, last_message_at=end),
            _profile("child-b", parent_id="root", first_message_at=end, last_message_at=end),
        ]
    )

    assert thread.thread_id == "root"
    assert thread.session_ids == ("root", "child-a", "child-b")
    assert thread.support_level == "strong"
    assert thread.confidence == 1.0
    assert "explicit_lineage" in thread.support_signals
    assert "branching_continuations" in thread.support_signals

    evidence_by_id = {member.conversation_id: member for member in thread.member_evidence}
    assert evidence_by_id["root"].role == "root"
    assert evidence_by_id["root"].depth == 0
    assert evidence_by_id["child-a"].role == "parent_continuation"
    assert evidence_by_id["child-a"].parent_id == "root"
    assert evidence_by_id["child-a"].depth == 1
    assert "parent_conversation_id" in evidence_by_id["child-a"].support_signals


def test_build_session_threads_keeps_obvious_non_matches_separate() -> None:
    threads = build_session_threads(
        [
            _profile("task-a", repo_names=("polylogue",)),
            _profile("task-b", repo_names=("polylogue",)),
        ]
    )

    assert [thread.thread_id for thread in threads] == ["task-a", "task-b"]
    assert all(len(thread.session_ids) == 1 for thread in threads)
    assert all(thread.support_level == "moderate" for thread in threads)
    assert all(thread.confidence == 0.85 for thread in threads)


def test_work_thread_payload_round_trips_membership_evidence() -> None:
    [thread] = build_session_threads([_profile("root"), _profile("child", parent_id="root")])

    rehydrated = WorkThread.from_dict(thread.to_dict())

    assert rehydrated == thread
    assert rehydrated.member_evidence[1].evidence == ("parent_id=root", "root_id=root")
