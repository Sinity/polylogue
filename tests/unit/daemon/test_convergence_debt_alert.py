"""Tests for per-source-family convergence-debt alerts (#1226)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon.convergence_debt_alert import (
    DEFAULT_DEDUP_WINDOW_S,
    AlertDedupState,
    ConvergenceDebtThresholds,
    FamilyThreshold,
    aggregate_debt_by_family,
    evaluate_convergence_debt,
    load_thresholds_from_config,
    source_family_for_path,
    source_family_for_subject,
    watchsource_name_to_family,
)
from polylogue.daemon.convergence_debt_status import (
    ConvergenceDebtItem,
    ConvergenceDebtSummary,
    convergence_debt_summary_info,
)
from polylogue.daemon.health import HealthSeverity, HealthTier


def _item(
    *, stage: str = "session_insights", subject_type: str = "source_path", subject_id: str
) -> ConvergenceDebtItem:
    return ConvergenceDebtItem(
        stage=stage,
        subject_type=subject_type,
        subject_id=subject_id,
        status="failed",
        failure_count=1,
        last_failed_at="2026-05-15T00:00:00+00:00",
        next_retry_at=None,
        retry_due=True,
        last_error="stalled",
    )


def _summary(items: list[ConvergenceDebtItem]) -> ConvergenceDebtSummary:
    return ConvergenceDebtSummary(
        failed_count=len(items),
        retry_due_count=sum(1 for it in items if it.retry_due),
        recent=items,
    )


def test_convergence_debt_summary_does_not_count_deferred_as_failed(tmp_path: Path) -> None:
    db_path = tmp_path / "debt.sqlite"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE live_convergence_debt (
                stage TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                status TEXT NOT NULL,
                failure_count INTEGER NOT NULL DEFAULT 0,
                first_failed_at TEXT NOT NULL,
                last_failed_at TEXT NOT NULL,
                next_retry_at TEXT,
                materializer_version TEXT,
                last_error TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO live_convergence_debt (
                stage, subject_type, subject_id, status, failure_count,
                first_failed_at, last_failed_at, next_retry_at, last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "insights",
                    "conversation_id",
                    "conv-deferred",
                    "deferred",
                    1,
                    "2026-05-24T00:00:00+00:00",
                    "2026-05-24T00:00:00+00:00",
                    "2026-05-24T00:01:00+00:00",
                    "insights deferred until source quiet",
                ),
                (
                    "fts",
                    "conversation_id",
                    "conv-failed",
                    "failed",
                    1,
                    "2026-05-24T00:00:00+00:00",
                    "2026-05-24T00:00:00+00:00",
                    None,
                    "boom",
                ),
            ],
        )

    summary = convergence_debt_summary_info(db_path)

    assert summary.failed_count == 1
    assert [item.subject_id for item in summary.recent] == ["conv-failed"]


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def test_for_family_falls_back_to_global_defaults_when_no_override() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=2, default_error=20)
    assert thresholds.for_family("claude-code-session") == (2, 20)


def test_for_family_applies_family_specific_overrides() -> None:
    thresholds = ConvergenceDebtThresholds(
        default_warning=10,
        default_error=100,
        families={"claude-code-session": FamilyThreshold(warning=1, error=5)},
    )
    assert thresholds.for_family("claude-code-session") == (1, 5)
    assert thresholds.for_family("chatgpt-export") == (10, 100)


def test_for_family_partial_override_inherits_unset_field() -> None:
    thresholds = ConvergenceDebtThresholds(
        default_warning=3,
        default_error=30,
        families={"codex-session": FamilyThreshold(error=50)},  # warning unset
    )
    assert thresholds.for_family("codex-session") == (3, 50)


def test_for_family_clamps_error_below_warning_to_warning() -> None:
    # If an operator misconfigures error < warning, the warning band would
    # collapse; we clamp so the alert ladder remains monotonic.
    thresholds = ConvergenceDebtThresholds(
        default_warning=10,
        default_error=5,
    )
    assert thresholds.for_family("anything") == (10, 10)


# ---------------------------------------------------------------------------
# Source-family inference
# ---------------------------------------------------------------------------


def test_watchsource_name_to_family_known_and_unknown() -> None:
    assert watchsource_name_to_family("claude-code") == "claude-code-session"
    assert watchsource_name_to_family("codex") == "codex-session"
    assert watchsource_name_to_family("never-heard-of-it") == "unknown"


def test_source_family_for_subject_returns_unknown_for_conversation_id() -> None:
    # Conversation-id subject attribution requires a DB lookup; for now we
    # group those under "unknown" and they fall under the global default.
    assert source_family_for_subject("conversation_id", "abc-123") == "unknown"


def test_source_family_for_path_returns_unknown_for_arbitrary_path() -> None:
    # A path that does not live under any configured watch root.
    assert source_family_for_path("/nonexistent/random/file.jsonl") == "unknown"


# ---------------------------------------------------------------------------
# Per-family aggregation
# ---------------------------------------------------------------------------


def test_aggregate_debt_by_family_buckets_subjects() -> None:
    summary = _summary(
        [
            _item(subject_id="/x/y/a.jsonl"),
            _item(subject_id="/x/y/b.jsonl"),
            _item(subject_type="conversation_id", subject_id="conv-1"),
        ]
    )
    counts = aggregate_debt_by_family(summary)
    # Both paths and the conversation-id all bucket to "unknown" in this
    # synthetic test because the paths don't live under any real watch root.
    assert counts["unknown"] == 3


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------


def test_evaluate_emits_no_alerts_when_below_warning() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=5, default_error=10)
    state = AlertDedupState()
    summary = _summary([_item(subject_id="/x/a.jsonl")])  # only one item

    alerts = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=0.0)

    assert alerts == []


def test_evaluate_emits_warning_when_count_crosses_warning_threshold() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=2, default_error=10)
    state = AlertDedupState()
    summary = _summary([_item(subject_id=f"/x/a{i}.jsonl") for i in range(2)])

    alerts = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=0.0)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == HealthSeverity.WARNING
    assert alert.tier == HealthTier.MEDIUM
    assert alert.check_name == "convergence_debt[unknown]"


def test_evaluate_emits_error_when_count_crosses_error_threshold() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=1, default_error=3)
    state = AlertDedupState()
    summary = _summary([_item(subject_id=f"/x/a{i}.jsonl") for i in range(3)])

    alerts = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=0.0)

    assert len(alerts) == 1
    assert alerts[0].severity == HealthSeverity.ERROR


def test_evaluate_dedups_repeated_alerts_within_window() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=1, default_error=10, dedup_window_s=600)
    state = AlertDedupState()
    summary = _summary([_item(subject_id="/x/a.jsonl")])

    first = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=1000.0)
    second = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=1100.0)

    assert len(first) == 1
    assert second == []  # within the 600s dedup window


def test_evaluate_reemits_after_dedup_window_expires() -> None:
    thresholds = ConvergenceDebtThresholds(default_warning=1, default_error=10, dedup_window_s=600)
    state = AlertDedupState()
    summary = _summary([_item(subject_id="/x/a.jsonl")])

    first = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=1000.0)
    second = evaluate_convergence_debt(summary, thresholds=thresholds, state=state, now=1700.0)

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.WARNING


def test_evaluate_escalation_fires_immediately_through_dedup_window() -> None:
    # An escalation from WARNING to ERROR must surface even within the
    # dedup window: severity change overrides dedup.
    thresholds = ConvergenceDebtThresholds(default_warning=1, default_error=3, dedup_window_s=3600)
    state = AlertDedupState()

    first = evaluate_convergence_debt(
        _summary([_item(subject_id="/x/a.jsonl")]),
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_convergence_debt(
        _summary([_item(subject_id=f"/x/a{i}.jsonl") for i in range(3)]),
        thresholds=thresholds,
        state=state,
        now=10.0,
    )

    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.ERROR


def test_evaluate_emits_resolution_alert_when_debt_clears() -> None:
    # After firing a warning, a subsequent run with no debt should emit a
    # single OK resolution alert so notification backends see the clear.
    thresholds = ConvergenceDebtThresholds(default_warning=1, default_error=10, dedup_window_s=3600)
    state = AlertDedupState()

    first = evaluate_convergence_debt(
        _summary([_item(subject_id="/x/a.jsonl")]),
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_convergence_debt(_summary([]), thresholds=thresholds, state=state, now=10.0)
    third = evaluate_convergence_debt(_summary([]), thresholds=thresholds, state=state, now=20.0)

    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.OK
    # Once cleared, no further OK alerts fire on subsequent quiet runs.
    assert third == []


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_thresholds_from_config_reads_polylogue_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(
        """
[health.convergence_debt]
default_warning = 2
default_error = 25
dedup_window_s = 900

[health.convergence_debt.families.claude-code-session]
warning = 1
error = 5

[health.convergence_debt.families.chatgpt-export]
warning = 50
error = 500
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_thresholds_from_config(cfg)

    assert thresholds.default_warning == 2
    assert thresholds.default_error == 25
    assert thresholds.dedup_window_s == 900
    assert thresholds.families["claude-code-session"] == FamilyThreshold(warning=1, error=5)
    assert thresholds.families["chatgpt-export"] == FamilyThreshold(warning=50, error=500)
    # Sanity: per-family resolution respects overrides
    assert thresholds.for_family("claude-code-session") == (1, 5)
    assert thresholds.for_family("chatgpt-export") == (50, 500)
    assert thresholds.for_family("codex-session") == (2, 25)


def test_load_thresholds_defaults_when_no_config_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text("[archive]\nroot = '/tmp/archive'\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_thresholds_from_config(cfg)

    assert thresholds.dedup_window_s == DEFAULT_DEDUP_WINDOW_S
    assert thresholds.families == {}
