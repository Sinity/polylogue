"""Independent black-box continuity scenarios and known-answer oracles.

The oracle consumes fixture evidence directly. It deliberately does not call a
query route, so a replay cannot pass by reproducing the production bug.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

ContinuityFailureClass = Literal[
    "source_coverage",
    "discovery",
    "formulation",
    "plan",
    "execution",
    "projection",
    "reasoning",
]


@dataclass(frozen=True, slots=True)
class ContinuityBudget:
    max_calls: int = 10
    max_page_bytes: int = 25_000
    max_cancel_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class ContinuityScenarioSpec:
    """One sparse operator job with an independently computed answer."""

    scenario_id: str
    title: str
    sparse_prompt: str
    required_facts: tuple[str, ...]
    expected_evidence_refs: tuple[str, ...]
    canonical_plan_families: tuple[str, ...]
    result_semantics: str
    allowed_discovery: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    failure_taxonomy: tuple[ContinuityFailureClass, ...]
    mutation_cases: tuple[str, ...] = ()
    budget: ContinuityBudget = ContinuityBudget()

    def oracle_record(self, fixture: Mapping[str, object]) -> Mapping[str, object]:
        """Return the fixture-owned answer record without invoking production code."""
        raw = fixture.get(self.scenario_id)
        if isinstance(raw, Mapping):
            refs = raw.get("refs", ())
            if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
                raise ValueError(f"fixture answer for {self.scenario_id} has no refs list")
            return {**raw, "refs": tuple(str(ref) for ref in refs)}
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return {"refs": tuple(str(ref) for ref in raw)}
        raise ValueError(f"fixture has no independent answer population for {self.scenario_id}")

    def oracle(self, fixture: Mapping[str, object]) -> tuple[str, ...]:
        """Build target refs from fixture evidence, not production query output."""
        return tuple(str(ref) for ref in self.oracle_record(fixture)["refs"])

    def classify(
        self,
        fixture: Mapping[str, object],
        *,
        observed_refs: Sequence[str],
        calls: int,
        observed_failure: ContinuityFailureClass | None = None,
        page_bytes: int | None = None,
        cancel_seconds: float | None = None,
        continuation_state_lost: bool = False,
        non_progressing_continuation: bool = False,
    ) -> str:
        expected = self.oracle(fixture)
        if calls > self.budget.max_calls:
            return "discovery"
        if page_bytes is not None and page_bytes > self.budget.max_page_bytes:
            return "projection"
        if cancel_seconds is not None and cancel_seconds > self.budget.max_cancel_seconds:
            return "execution"
        if continuation_state_lost or non_progressing_continuation:
            return "execution"
        if observed_failure is not None:
            return observed_failure
        observed = tuple(observed_refs)
        if len(observed) != len(set(observed)):
            return "projection"
        return "pass" if observed == expected else "projection"


def _scenario(
    scenario_id: str,
    title: str,
    prompt: str,
    facts: tuple[str, ...],
    refs: tuple[str, ...],
    plan: tuple[str, ...],
    mutation_cases: tuple[str, ...],
) -> ContinuityScenarioSpec:
    return ContinuityScenarioSpec(
        scenario_id=scenario_id,
        title=title,
        sparse_prompt=prompt,
        required_facts=facts,
        expected_evidence_refs=refs,
        canonical_plan_families=plan,
        result_semantics="exhaustive_page",
        allowed_discovery=("polylogue://capabilities/query", "query_units", "search"),
        stop_conditions=("all expected refs enumerated exactly once", "no advancing continuation remains"),
        failure_taxonomy=(
            "source_coverage",
            "discovery",
            "formulation",
            "plan",
            "execution",
            "projection",
            "reasoning",
        ),
        mutation_cases=mutation_cases,
    )


CONTINUITY_SCENARIOS: tuple[ContinuityScenarioSpec, ...] = (
    _scenario(
        "resume",
        "Resume current work",
        "What was I doing in this repo?",
        ("session", "repo", "continuation"),
        ("session:resume-1",),
        ("session-query", "read-context"),
        ("lost-continuation-state",),
    ),
    _scenario(
        "forensic-debug",
        "Find a forensic file/session",
        "Where did this failure happen?",
        ("file", "action", "failure"),
        ("session:debug-1", "file:src/app.py"),
        ("file-query", "action-query"),
        ("global-action-materialization",),
    ),
    _scenario(
        "prior-art",
        "Retrieve prior art",
        "Have we solved this before?",
        ("similar-session", "decision"),
        ("session:prior-1",),
        ("near-query", "text-query"),
        ("wrong-ranking-scope",),
    ),
    _scenario(
        "decision",
        "Recover a decision",
        "What did we decide about this?",
        ("assertion", "decision", "evidence"),
        ("assertion:decision-1",),
        ("assertion-query", "read-ref"),
        ("unresolved-ref-broadening",),
    ),
    _scenario(
        "postmortem",
        "Explain a failure",
        "Why did the last run fail?",
        ("run", "action", "outcome"),
        ("run:failed-1", "action:failed-1"),
        ("run-query", "action-query"),
        ("capped-pseudo-total",),
    ),
    _scenario(
        "cost",
        "Audit cost and usage",
        "How much did this work cost?",
        ("usage", "reported-tokens", "price"),
        ("usage:run-1",),
        ("usage-query", "read-ref"),
        ("reported-vs-estimated-collapse",),
    ),
    _scenario(
        "self-inspection",
        "Inspect the archive itself",
        "What can you answer about agent work?",
        ("capability", "coverage", "freshness"),
        ("capability:query",),
        ("capability-resource", "status-query"),
        ("unpaged-capability-catalog",),
    ),
    _scenario(
        "parallel-claude-incident",
        "Reconstruct parallel Claude work",
        "Which agents handled the concerns and what changed?",
        ("coordinator", "attempt", "claim", "git-effect", "beads-effect"),
        ("session:coordinator", "run:wf-1", "commit:abc123", "bead:xyz"),
        ("capability-resource", "delegation-query", "effect-query"),
        ("lost-continuation-state", "global-delegation-materialization", "wrong-workflow-membership"),
    ),
)

CONTINUITY_SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in CONTINUITY_SCENARIOS}


def continuity_scenario(name: str) -> ContinuityScenarioSpec:
    try:
        return CONTINUITY_SCENARIO_BY_ID[name]
    except KeyError as exc:
        raise KeyError(f"unknown continuity scenario {name!r}") from exc


__all__ = [
    "CONTINUITY_SCENARIOS",
    "CONTINUITY_SCENARIO_BY_ID",
    "ContinuityBudget",
    "ContinuityFailureClass",
    "ContinuityScenarioSpec",
    "continuity_scenario",
]
