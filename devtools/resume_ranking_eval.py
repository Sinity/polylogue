"""Offline before/after evaluation for resume-candidate ranking changes.

The fixture format carries a current-work context, a candidate profile pool,
and lineage identifiers. Ground truth is derived from that lineage rather than
from an evaluator-only relevance label. Both ranking variants call the
production scorer in :mod:`polylogue.insights.resume`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polylogue.insights.archive import ArchiveInsightProvenance, SessionProfileInsight
from polylogue.insights.archive_models import SessionEvidencePayload, SessionInferencePayload
from polylogue.insights.resume import (
    ResumeCandidate,
    _PathResolutionContext,
    _rank_resume_profiles,
    _score_file_overlap,
)

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "resume-ranking-eval-v1.json"
FixtureSource = Literal["synthetic", "snapshot-derived"]


class _FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CurrentWorkFixture(_FixtureModel):
    session_id: str
    logical_session_id: str
    parent_session_id: str | None = None
    recent_files: tuple[str, ...]
    cwd: str | None = None


class CandidateFixture(_FixtureModel):
    session_id: str
    logical_session_id: str
    parent_session_id: str | None = None
    title: str
    last_message_at: str
    file_paths_touched: tuple[str, ...]
    repo_root_alias: str | None = None
    cwd_paths: tuple[str, ...] = ()
    terminal_state: str = "unknown"
    workflow_shape: str = "unknown"


class RankingScenarioFixture(_FixtureModel):
    id: str
    source: FixtureSource
    present_paths: tuple[str, ...]
    current: CurrentWorkFixture
    candidates: tuple[CandidateFixture, ...]

    @model_validator(mode="after")
    def _require_candidate_pool(self) -> RankingScenarioFixture:
        if not self.candidates:
            raise ValueError("ranking scenario must include at least one candidate")
        return self


class EvidenceSampleFixture(_FixtureModel):
    id: str
    source: FixtureSource
    resolvable_count: int = Field(ge=0)
    recoverable_dead_count: int = Field(ge=0)
    unrecoverable_dead_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_evidence(self) -> EvidenceSampleFixture:
        if self.resolvable_count + self.recoverable_dead_count + self.unrecoverable_dead_count == 0:
            raise ValueError("evidence sample must include at least one path")
        return self


class RankingEvaluationFixture(_FixtureModel):
    version: int
    description: str
    scenarios: tuple[RankingScenarioFixture, ...]
    evidence_samples: tuple[EvidenceSampleFixture, ...] = ()

    @model_validator(mode="after")
    def _validate_version_and_scenarios(self) -> RankingEvaluationFixture:
        if self.version != 1:
            raise ValueError(f"unsupported resume ranking fixture version: {self.version}")
        if not self.scenarios:
            raise ValueError("fixture must include at least one ranking scenario")
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("ranking scenario ids must be unique")
        return self


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    scenarios: int
    hit_at_1: float
    hit_at_3: float
    mrr: float


@dataclass(frozen=True, slots=True)
class BeforeAfterMetrics:
    before: RankingMetrics
    after: RankingMetrics


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    scenario_id: str
    source: FixtureSource
    target_logical_session_ids: tuple[str, ...]
    before_rank: int | None
    after_rank: int | None
    before_order: tuple[str, ...]
    after_order: tuple[str, ...]
    after_overlap_basis: dict[str, object]


@dataclass(frozen=True, slots=True)
class EvidenceRecoveryEvaluation:
    sample_id: str
    source: FixtureSource
    path_count: int
    resolvable_count: int
    dead_count: int
    directory_recovered_count: int
    dead_excluded_count: int
    usable_share_before: float
    usable_share_after: float
    dead_recovery_rate: float


@dataclass(frozen=True, slots=True)
class EvaluationVerdict:
    non_regressing: bool
    strict_overall_improvement: bool
    all_fixed_targets_hit_at_1: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    fixture_version: int
    fixture_description: str
    metrics: dict[str, BeforeAfterMetrics]
    scenarios: tuple[ScenarioEvaluation, ...]
    evidence_recovery: tuple[EvidenceRecoveryEvaluation, ...]
    verdict: EvaluationVerdict

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_fixture(path: Path) -> RankingEvaluationFixture:
    """Load and strictly validate one ranking-evaluation fixture."""

    return RankingEvaluationFixture.model_validate_json(path.read_text(encoding="utf-8"))


def _safe_fixture_path(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"present_paths entries must be repo-relative: {relative_path}")
    resolved = (repo_root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"fixture path escapes repo root: {relative_path}") from exc
    return resolved


def _materialize_present_paths(repo_root: Path, paths: tuple[str, ...]) -> None:
    for relative_path in paths:
        path = _safe_fixture_path(repo_root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture: {relative_path}\n", encoding="utf-8")


def _profile_from_fixture(candidate: CandidateFixture, repo_root: Path) -> SessionProfileInsight:
    profile_repo_root = candidate.repo_root_alias or str(repo_root)
    return SessionProfileInsight(
        session_id=candidate.session_id,
        logical_session_id=candidate.logical_session_id,
        origin="claude-code",
        title=candidate.title,
        provenance=ArchiveInsightProvenance(
            materializer_version=1,
            materialized_at=candidate.last_message_at,
            source_updated_at=candidate.last_message_at,
        ),
        evidence=SessionEvidencePayload(
            last_message_at=candidate.last_message_at,
            repo_paths=(profile_repo_root,),
            cwd_paths=candidate.cwd_paths,
            file_paths_touched=candidate.file_paths_touched,
            parent_id=candidate.parent_session_id,
            logical_session_id=candidate.logical_session_id,
        ),
        inference=SessionInferencePayload(
            terminal_state=candidate.terminal_state,
            workflow_shape=candidate.workflow_shape,
        ),
    )


def _lineage_targets(scenario: RankingScenarioFixture) -> tuple[str, ...]:
    current = scenario.current
    target_logical_ids: set[str] = set()
    for candidate in scenario.candidates:
        same_logical_family = candidate.logical_session_id == current.logical_session_id
        true_parent = current.parent_session_id is not None and candidate.session_id == current.parent_session_id
        true_sibling = (
            current.parent_session_id is not None
            and candidate.parent_session_id is not None
            and candidate.parent_session_id == current.parent_session_id
        )
        if same_logical_family or true_parent or true_sibling:
            target_logical_ids.add(candidate.logical_session_id)
    if not target_logical_ids:
        raise ValueError(f"scenario {scenario.id!r} has no lineage-derived resume target")
    return tuple(sorted(target_logical_ids))


def _first_relevant_rank(candidates: tuple[ResumeCandidate, ...], targets: tuple[str, ...]) -> int | None:
    target_set = set(targets)
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.logical_session_id in target_set:
            return rank
    return None


def _target_basis(candidates: tuple[ResumeCandidate, ...], targets: tuple[str, ...]) -> dict[str, object]:
    target_set = set(targets)
    for candidate in candidates:
        if candidate.logical_session_id in target_set:
            return candidate.overlap_basis.model_dump(mode="json")
    return {}


def evaluate_scenario(scenario: RankingScenarioFixture) -> ScenarioEvaluation:
    """Run one fixture through the production legacy and fixed rankers."""

    with tempfile.TemporaryDirectory(prefix=f"polylogue-resume-eval-{scenario.id}-") as temporary:
        repo_root = Path(temporary).resolve()
        _materialize_present_paths(repo_root, scenario.present_paths)
        profiles = [_profile_from_fixture(candidate, repo_root) for candidate in scenario.candidates]
        target_ids = _lineage_targets(scenario)
        logical_pool_size = len({candidate.logical_session_id for candidate in scenario.candidates})
        before = _rank_resume_profiles(
            profiles,
            repo_path=str(repo_root),
            cwd=scenario.current.cwd,
            recent_files=scenario.current.recent_files,
            limit=logical_pool_size,
            overlap_mode="legacy",
        )
        after = _rank_resume_profiles(
            profiles,
            repo_path=str(repo_root),
            cwd=scenario.current.cwd,
            recent_files=scenario.current.recent_files,
            limit=logical_pool_size,
            overlap_mode="refactor-aware",
        )

    return ScenarioEvaluation(
        scenario_id=scenario.id,
        source=scenario.source,
        target_logical_session_ids=target_ids,
        before_rank=_first_relevant_rank(before, target_ids),
        after_rank=_first_relevant_rank(after, target_ids),
        before_order=tuple(candidate.logical_session_id for candidate in before),
        after_order=tuple(candidate.logical_session_id for candidate in after),
        after_overlap_basis=_target_basis(after, target_ids),
    )


def _ranking_metrics(rows: tuple[ScenarioEvaluation, ...], *, use_after: bool) -> RankingMetrics:
    if not rows:
        return RankingMetrics(scenarios=0, hit_at_1=0.0, hit_at_3=0.0, mrr=0.0)
    ranks = [row.after_rank if use_after else row.before_rank for row in rows]
    count = len(ranks)
    return RankingMetrics(
        scenarios=count,
        hit_at_1=round(sum(rank is not None and rank <= 1 for rank in ranks) / count, 6),
        hit_at_3=round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 6),
        mrr=round(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count, 6),
    )


def _evaluate_evidence_sample(sample: EvidenceSampleFixture) -> EvidenceRecoveryEvaluation:
    with tempfile.TemporaryDirectory(prefix=f"polylogue-resume-evidence-{sample.id}-") as temporary:
        repo_root = Path(temporary).resolve()
        resolvable = tuple(f"mass/live/file_{index:03d}.py" for index in range(sample.resolvable_count))
        recoverable_dead = tuple(
            f"mass/refactor/retired_{index:03d}.py" for index in range(sample.recoverable_dead_count)
        )
        replacement_files = tuple(
            f"mass/refactor/current_{index:03d}.py" for index in range(sample.recoverable_dead_count)
        )
        unrecoverable_dead = tuple(
            f"retired/area_{index:03d}/ghost.py" for index in range(sample.unrecoverable_dead_count)
        )
        _materialize_present_paths(repo_root, (*resolvable, *replacement_files))
        score = _score_file_overlap(
            context=_PathResolutionContext.from_repo_path(str(repo_root)),
            recent_files=set(replacement_files),
            candidate_paths={*resolvable, *recoverable_dead, *unrecoverable_dead},
        )

    path_count = sample.resolvable_count + sample.recoverable_dead_count + sample.unrecoverable_dead_count
    resolvable_count = len(score.resolvable_paths)
    dead_count = len(score.dead_paths)
    recovered_count = len(score.basis.dir)
    dead_excluded_count = len(score.basis.dead_excluded)
    if not score.resolution_available:
        raise RuntimeError(f"evidence sample {sample.id!r} could not resolve its temporary repo root")
    expected_dead = sample.recoverable_dead_count + sample.unrecoverable_dead_count
    if (resolvable_count, dead_count, recovered_count, dead_excluded_count) != (
        sample.resolvable_count,
        expected_dead,
        sample.recoverable_dead_count,
        sample.unrecoverable_dead_count,
    ):
        raise RuntimeError(
            f"evidence sample {sample.id!r} did not exercise the intended partition: "
            f"resolved={resolvable_count}, dead={dead_count}, recovered={recovered_count}, "
            f"excluded={dead_excluded_count}"
        )
    return EvidenceRecoveryEvaluation(
        sample_id=sample.id,
        source=sample.source,
        path_count=path_count,
        resolvable_count=resolvable_count,
        dead_count=dead_count,
        directory_recovered_count=recovered_count,
        dead_excluded_count=dead_excluded_count,
        usable_share_before=round(resolvable_count / path_count, 6),
        usable_share_after=round((resolvable_count + recovered_count) / path_count, 6),
        dead_recovery_rate=round(recovered_count / dead_count, 6) if dead_count else 0.0,
    )


def evaluate_fixture(fixture: RankingEvaluationFixture) -> EvaluationReport:
    """Evaluate every cohort and evidence-recovery sample in one fixture."""

    scenario_rows = tuple(evaluate_scenario(scenario) for scenario in fixture.scenarios)
    cohorts: dict[str, list[ScenarioEvaluation]] = defaultdict(list)
    cohorts["overall"].extend(scenario_rows)
    for row in scenario_rows:
        cohorts[row.source].append(row)
    metrics = {
        cohort: BeforeAfterMetrics(
            before=_ranking_metrics(tuple(rows), use_after=False),
            after=_ranking_metrics(tuple(rows), use_after=True),
        )
        for cohort, rows in sorted(cohorts.items())
    }
    evidence_rows = tuple(_evaluate_evidence_sample(sample) for sample in fixture.evidence_samples)
    non_regressing = all(
        metric.after.hit_at_1 >= metric.before.hit_at_1
        and metric.after.hit_at_3 >= metric.before.hit_at_3
        and metric.after.mrr >= metric.before.mrr
        for metric in metrics.values()
    )
    overall = metrics["overall"]
    strict_improvement = (
        overall.after.hit_at_1 > overall.before.hit_at_1
        or overall.after.hit_at_3 > overall.before.hit_at_3
        or overall.after.mrr > overall.before.mrr
    )
    return EvaluationReport(
        fixture_version=fixture.version,
        fixture_description=fixture.description,
        metrics=metrics,
        scenarios=scenario_rows,
        evidence_recovery=evidence_rows,
        verdict=EvaluationVerdict(
            non_regressing=non_regressing,
            strict_overall_improvement=strict_improvement,
            all_fixed_targets_hit_at_1=all(row.after_rank == 1 for row in scenario_rows),
        ),
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_report(report: EvaluationReport) -> str:
    lines = [
        "Resume ranking evaluation",
        f"Fixture v{report.fixture_version}: {report.fixture_description}",
        "",
        "Ranking quality (before -> after)",
    ]
    for cohort, metrics in report.metrics.items():
        lines.append(
            f"  {cohort:16} n={metrics.before.scenarios:<2d} "
            f"hit@1 {_pct(metrics.before.hit_at_1)} -> {_pct(metrics.after.hit_at_1)}; "
            f"hit@3 {_pct(metrics.before.hit_at_3)} -> {_pct(metrics.after.hit_at_3)}; "
            f"MRR {metrics.before.mrr:.3f} -> {metrics.after.mrr:.3f}"
        )
    lines.extend(("", "Scenario ranks (before -> after)"))
    for row in report.scenarios:
        before_rank = str(row.before_rank) if row.before_rank is not None else "miss"
        after_rank = str(row.after_rank) if row.after_rank is not None else "miss"
        lines.append(f"  {row.scenario_id:48} {before_rank} -> {after_rank}")
    if report.evidence_recovery:
        lines.extend(("", "Evidence usability"))
        for evidence_row in report.evidence_recovery:
            lines.append(
                f"  {evidence_row.sample_id}: "
                f"{_pct(evidence_row.usable_share_before)} -> {_pct(evidence_row.usable_share_after)} usable; "
                f"{_pct(evidence_row.dead_recovery_rate)} of dead paths directory-recovered "
                f"({evidence_row.directory_recovered_count}/{evidence_row.dead_count})"
            )
    lines.extend(
        (
            "",
            "Verdict: "
            f"non-regressing={str(report.verdict.non_regressing).lower()}, "
            f"strict-overall-improvement={str(report.verdict.strict_overall_improvement).lower()}, "
            f"all-fixed-targets-hit@1={str(report.verdict.all_fixed_targets_hit_at_1).lower()}",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devtools.resume_ranking_eval",
        description="Compare legacy and refactor-aware resume ranking over lineage-grounded offline fixtures.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Path to a v1 evaluation fixture.")
    parser.add_argument("--json", action="store_true", help="Emit the complete machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate_fixture(load_fixture(args.fixture))
    except (OSError, ValueError, ValidationError, RuntimeError) as exc:
        print(f"resume-ranking-eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report.verdict.non_regressing else 1


if __name__ == "__main__":
    raise SystemExit(main())
