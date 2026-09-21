"""Unit tests for the insight fallback marker taxonomy (#1278)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.analysis.fallback import FallbackReason
from polylogue.analysis.readiness import InsightReadinessQuery, InsightReadinessReport
from polylogue.api import Polylogue
from polylogue.storage.derived.session.profiles import (
    enrichment_fallback_reasons,
    session_enrichment_payload,
)
from tests.infra.storage_records import SessionBuilder

# ---------------------------------------------------------------------------
# Unit-level: helpers return the typed enum on the documented heuristics
# ---------------------------------------------------------------------------


def _stub_profile_no_events() -> object:
    class _P:
        provider: str = "codex"
        title: str | None = None
        updated_at: object | None = None
        work_events: tuple[object, ...] = ()
        phases: tuple[object, ...] = ()
        repo_paths: tuple[str, ...] = ()
        repo_names: tuple[str, ...] = ()
        file_paths_touched: tuple[str, ...] = ()
        cwd_paths: tuple[str, ...] = ()
        engaged_duration_ms: int = 0
        tool_active_duration_ms: int = 0
        workflow_shape: str = "unknown"
        workflow_shape_confidence: float = 0.0
        terminal_state: str = "unknown"
        terminal_state_confidence: float = 0.0
        terminal_state_evidence: dict[str, int | float | str | None] = {}
        terminal_state_method: str = "unknown"
        inferred_topic: str | None = None
        inferred_topic_source: str = "absent"
        auto_tags: tuple[str, ...] = ()

    return _P()


def test_enrichment_fallback_reasons_flags_missing_analysis_and_no_user_turns() -> None:
    reasons = enrichment_fallback_reasons(None, user_turns=())

    assert FallbackReason.MISSING_SESSION_ANALYSIS in reasons
    assert FallbackReason.NO_USER_TURNS in reasons


def test_session_enrichment_payload_serializes_fallback_reasons() -> None:
    profile = _stub_profile_no_events()
    payload = session_enrichment_payload(profile, None)  # type: ignore[arg-type]

    assert FallbackReason.MISSING_SESSION_ANALYSIS in payload.fallback_reasons
    assert FallbackReason.NO_USER_TURNS in payload.fallback_reasons


# ---------------------------------------------------------------------------
# Integration-shaped: readiness report classifies degraded rows
# ---------------------------------------------------------------------------


def _seed_degraded_session(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "degraded-session")
        .provider("codex")
        .title("Degraded Session")
        .created_at("2026-05-19T09:00:00+00:00")
        .updated_at("2026-05-19T09:05:00+00:00")
        # No user turn at all: enrichment has nothing to summarize intent
        # from, so it records NO_USER_TURNS and the readiness report must
        # classify the row degraded (polylogue-cuxz.7 retired the work-event
        # and phase markers this used to lean on).
        .add_message(
            "a1",
            role="assistant",
            text="Reply with no user turn preceding it.",
            timestamp="2026-05-19T09:05:00+00:00",
        )
        .save()
    )


@pytest.mark.asyncio
async def test_readiness_report_classifies_fallback_rows_as_degraded(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    _seed_degraded_session(db_path)
    # Insight rebuilds are daemon-owned and require a sealed accepted machine
    # part.  Seed the derived read model through the materializer harness
    # directly so this readiness test does not bypass that operation contract.
    from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as connection:
        rebuild_session_insights_sync(connection)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report: InsightReadinessReport = await archive.insight_readiness_report(
        InsightReadinessQuery(insights=("session_profiles",))
    )

    profile = next(entry for entry in report.insights if entry.insight_name == "session_profiles")
    assert profile.degraded_count == 1
    # The seeded session has no user turn, so enrichment records
    # NO_USER_TURNS; the taxonomy surfaces that reason explicitly.
    assert profile.fallback_reason_counts
    # The evidence channel must surface the same markers consumers see.
    assert any("degraded=1" in line for line in profile.evidence)
    assert any("fallback_reason=" in line for line in profile.evidence)

    # Fallback markers describe row quality, not divergence: the rows still
    # reflect the sources they were built from.
    assert not profile.diverged
