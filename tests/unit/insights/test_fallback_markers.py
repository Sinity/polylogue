"""Unit tests for the insight fallback marker taxonomy (#1278)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.insights.fallback import FallbackReason
from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
from polylogue.storage.insights.session.profiles import (
    enrichment_fallback_reasons,
    profile_inference_fallback_reasons,
    profile_inference_payload,
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
        inferred_topic: str | None = None
        inferred_topic_source: str = "absent"
        auto_tags: tuple[str, ...] = ()

    return _P()


def test_profile_inference_fallback_reasons_flags_empty_work_and_phases() -> None:
    profile = _stub_profile_no_events()

    reasons = profile_inference_fallback_reasons(profile)  # type: ignore[arg-type]

    assert FallbackReason.NO_WORK_EVENTS_AND_NO_PHASES in reasons
    assert FallbackReason.ENGAGED_DURATION_SESSION_TOTAL in reasons


def test_enrichment_fallback_reasons_flags_missing_analysis_and_no_user_turns() -> None:
    reasons = enrichment_fallback_reasons(None, user_turns=())

    assert FallbackReason.MISSING_SESSION_ANALYSIS in reasons
    assert FallbackReason.NO_USER_TURNS in reasons


def test_profile_inference_payload_serializes_fallback_reasons() -> None:
    profile = _stub_profile_no_events()
    payload = profile_inference_payload(profile)  # type: ignore[arg-type]

    assert FallbackReason.NO_WORK_EVENTS_AND_NO_PHASES in payload.fallback_reasons
    assert FallbackReason.ENGAGED_DURATION_SESSION_TOTAL in payload.fallback_reasons


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
        .add_message(
            "u1",
            role="user",
            text="Single user turn with no tool use.",
            timestamp="2026-05-19T09:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Reply with no tools either.",
            timestamp="2026-05-19T09:05:00+00:00",
        )
        .save()
    )


@pytest.mark.xfail(
    reason=(
        "Readiness gap (#1782): the archive readiness builder does not yet "
        "compute the #1278 degraded/fallback taxonomy from the stored "
        "enrichment fallback reasons; degraded_count is hardcoded to 0."
    ),
    strict=False,
)
@pytest.mark.asyncio
async def test_readiness_report_classifies_fallback_rows_as_degraded(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    _seed_degraded_session(db_path)
    archive_seed = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive_seed.rebuild_insights()
    await archive_seed.close()

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report: InsightReadinessReport = await archive.insight_readiness_report(
        InsightReadinessQuery(insights=("session_profiles",))
    )

    profile = next(entry for entry in report.insights if entry.insight_name == "session_profiles")
    assert profile.verdict == "degraded"
    assert profile.degraded_count == 1
    # The seeded session materializes weak work-events and tool-less
    # phases; the taxonomy surfaces those reasons explicitly.
    assert profile.fallback_reason_counts
    # The evidence channel must surface the same markers consumers see.
    assert any("degraded=1" in line for line in profile.evidence)
    assert any("fallback_reason=" in line for line in profile.evidence)

    # Aggregate verdict promotes the worst entry; here both profiles and
    # enrichments are degraded but nothing is incompatible/stale/partial.
    assert report.aggregate_verdict == "degraded"


@pytest.mark.xfail(
    reason=(
        "Readiness gap (#1782): the archive readiness builder does not yet "
        "derive incompatible/degraded verdicts (no materializer_version incompatibility "
        "detection, degraded_count hardcoded to 0)."
    ),
    strict=False,
)
@pytest.mark.asyncio
async def test_readiness_verdict_precedence_incompatible_outranks_degraded(
    cli_workspace: dict[str, Path],
) -> None:
    import sqlite3

    db_path = cli_workspace["db_path"]
    _seed_degraded_session(db_path)
    archive_seed = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    await archive_seed.rebuild_insights()
    await archive_seed.close()
    # Force an incompatible materializer version on the archive provenance row
    # so the row is both degraded and incompatible.
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE insight_materialization SET materializer_version = 0")
        conn.commit()

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    profile = next(entry for entry in report.insights if entry.insight_name == "session_profiles")
    assert profile.verdict == "incompatible"
    assert profile.degraded_count == 1  # marker still recorded
