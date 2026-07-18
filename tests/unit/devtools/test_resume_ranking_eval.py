"""Contracts for the offline resume-ranking quality evaluator."""

from __future__ import annotations

import json

import pytest

from devtools import resume_ranking_eval


def test_resume_ranking_eval_improves_lineage_grounded_metrics() -> None:
    fixture = resume_ranking_eval.load_fixture(resume_ranking_eval.DEFAULT_FIXTURE)

    report = resume_ranking_eval.evaluate_fixture(fixture)

    overall = report.metrics["overall"]
    assert overall.before == resume_ranking_eval.RankingMetrics(
        scenarios=5,
        hit_at_1=0.2,
        hit_at_3=0.2,
        mrr=0.39,
    )
    assert overall.after == resume_ranking_eval.RankingMetrics(
        scenarios=5,
        hit_at_1=1.0,
        hit_at_3=1.0,
        mrr=1.0,
    )
    assert report.metrics["synthetic"].before.hit_at_1 == 0.333333
    assert report.metrics["snapshot-derived"].before.mrr == 0.25
    assert report.verdict.non_regressing is True
    assert report.verdict.strict_overall_improvement is True
    assert report.verdict.all_fixed_targets_hit_at_1 is True


def test_resume_ranking_eval_catches_legacy_dead_path_anti_selection() -> None:
    """Replacing refactor-aware mode with legacy mode makes every asserted repair fail."""
    report = resume_ranking_eval.evaluate_fixture(resume_ranking_eval.load_fixture(resume_ranking_eval.DEFAULT_FIXTURE))
    by_id = {row.scenario_id: row for row in report.scenarios}

    assert by_id["synthetic-dead-shared-directory"].before_rank == 4
    assert by_id["synthetic-dead-shared-directory"].after_rank == 1
    assert by_id["synthetic-dead-union-deflation"].before_rank == 5
    assert by_id["synthetic-dead-union-deflation"].after_rank == 1
    assert by_id["snapshot-storage-repository-file-to-package"].before_rank == 4
    assert by_id["snapshot-storage-repository-file-to-package"].after_rank == 1
    assert by_id["snapshot-pipeline-runner-directory-recovery"].before_rank == 4
    assert by_id["snapshot-pipeline-runner-directory-recovery"].after_rank == 1
    assert by_id["synthetic-live-exact-control"].before_rank == 1
    assert by_id["synthetic-live-exact-control"].after_rank == 1


def test_resume_ranking_eval_reproduces_seeded_evidence_recovery() -> None:
    report = resume_ranking_eval.evaluate_fixture(resume_ranking_eval.load_fixture(resume_ranking_eval.DEFAULT_FIXTURE))

    assert len(report.evidence_recovery) == 1
    evidence = report.evidence_recovery[0]
    assert evidence.path_count == 1000
    assert evidence.resolvable_count == 570
    assert evidence.dead_count == 430
    assert evidence.directory_recovered_count == 254
    assert evidence.dead_excluded_count == 176
    assert evidence.usable_share_before == 0.57
    assert evidence.usable_share_after == 0.824
    assert evidence.dead_recovery_rate == pytest.approx(254 / 430, abs=1e-6)


def test_resume_ranking_fixture_has_no_manual_relevance_labels() -> None:
    raw = json.loads(resume_ranking_eval.DEFAULT_FIXTURE.read_text(encoding="utf-8"))

    for scenario in raw["scenarios"]:
        assert "relevant" not in scenario
        assert "expected_target" not in scenario
        assert scenario["current"]["logical_session_id"]
        assert scenario["current"]["parent_session_id"]


def test_resume_ranking_metrics_assign_zero_reciprocal_rank_to_a_miss() -> None:
    row = resume_ranking_eval.ScenarioEvaluation(
        scenario_id="missing-target",
        source="synthetic",
        target_logical_session_ids=("target",),
        before_rank=None,
        after_rank=1,
        before_order=("distractor",),
        after_order=("target",),
        after_overlap_basis={},
    )

    before = resume_ranking_eval._ranking_metrics((row,), use_after=False)

    assert before.hit_at_1 == 0
    assert before.hit_at_3 == 0
    assert before.mrr == 0


def test_resume_ranking_eval_main_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = resume_ranking_eval.main(["--fixture", str(resume_ranking_eval.DEFAULT_FIXTURE), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["overall"]["before"]["hit_at_1"] == 0.2
    assert payload["metrics"]["overall"]["after"]["hit_at_1"] == 1.0
    assert payload["evidence_recovery"][0]["usable_share_after"] == 0.824
