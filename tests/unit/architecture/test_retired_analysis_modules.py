"""Retired analysis modules must not grow back without a route.

``polylogue/analysis`` was audited against the declared operation surface
(daemon operation specs, CLI verbs, MCP dispatcher tools, insight
descriptors in ``polylogue/analysis/registry.py``, and HTTP routes). A
module that no declared route reached, and whose only importers were its
own tests, was deleted rather than kept as a Python-only affordance.

This test is the ratchet that keeps the file deleted and keeps prose from
pointing at it. It is deliberately name-based: a route-totality gate is
separate work, but a resurrected file reintroduces exactly the condition
the audit removed, and a lingering reference is documentation that lies.

Anti-vacuity: recreate any file in :data:`RETIRED_MODULES` (even empty),
or write any name in :data:`RETIRED_NAMES` into a tracked Python file,
Markdown document, or the repository contract, and this test goes red
naming the file. Emptying either constant turns
``test_the_ratchet_still_has_something_to_protect`` red, so the test
cannot pass by having nothing left to check.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules deleted by the analysis-package audit, relative to the repo root.
RETIRED_MODULES = (
    "polylogue/analysis/archive_summaries.py",
    "polylogue/analysis/claude_todo_projection.py",
    "polylogue/analysis/hermes_topology_projection.py",
    "polylogue/analysis/hermes_verification_coverage.py",
    "polylogue/analysis/judgment/cascades.py",
    "polylogue/analysis/judgment/elicitation.py",
    "polylogue/analysis/judgment/experiments.py",
    "polylogue/analysis/judgment/rankers.py",
    "polylogue/analysis/measurement/alert_budget.py",
    "polylogue/analysis/measurement/evidence_ancestry.py",
    "polylogue/analysis/measurement/ratio.py",
    "polylogue/analysis/measurement/registration.py",
    "polylogue/analysis/measurement/uncertainty.py",
)

#: Import paths and public names the retired modules owned. Each token is
#: specific enough that a match is a real reference, not an English word:
#: ``ratio`` and ``registration`` appear all over the tree as prose, and
#: ``judgment.elicitation`` is a prefix of the surviving attribute access
#: ``judgment.elicitation_ref``, so the fully qualified module path is what
#: is forbidden, never the bare word.
RETIRED_NAMES = (
    "analysis.archive_summaries",
    "analysis/archive_summaries",
    "aggregate_day_session_summary_insights",
    "analysis.claude_todo_projection",
    "analysis/claude_todo_projection",
    "load_claude_todo_plan_states",
    "ClaudeTodoPlanState",
    "analysis.hermes_topology_projection",
    "analysis/hermes_topology_projection",
    "insights.hermes_topology_projection",
    "project_hermes_topology",
    "HermesTopologyConflict",
    "HermesSubagentEvidenceRef",
    "analysis.hermes_verification_coverage",
    "analysis/hermes_verification_coverage",
    "correlate_verification_coverage",
    "analysis.judgment.cascades",
    "analysis/judgment/cascades",
    "route_judgment",
    "analysis.judgment.elicitation",
    "analysis/judgment/elicitation",
    "ElicitationSession",
    "ExplorationQuota",
    "analysis.judgment.experiments",
    "analysis/judgment/experiments",
    "analyze_experiment",
    "analysis.judgment.rankers",
    "analysis/judgment/rankers",
    "analysis.measurement.alert_budget",
    "analysis/measurement/alert_budget",
    "analysis.measurement.evidence_ancestry",
    "analysis/measurement/evidence_ancestry",
    "analysis.measurement.ratio",
    "analysis/measurement/ratio",
    "analysis.measurement.registration",
    "analysis/measurement/registration",
    "analysis.measurement.uncertainty",
    "analysis/measurement/uncertainty",
    "PLAN_COMPLETION_RATE_METRIC",
    "PLAN_COMPLETION_MEASURE",
)

#: Where a reference would be a lie: tracked source, tests, tooling, docs
#: and the repository contract.
SEARCHED_PATHSPECS = (
    "polylogue",
    "devtools",
    "tests",
    "docs",
    "CLAUDE.md",
    "TESTING.md",
    "CONTRIBUTING.md",
    "README.md",
)


def _tracked_hits(token: str) -> list[str]:
    """Tracked lines naming ``token``, excluding this ratchet itself."""
    result = subprocess.run(
        ["git", "grep", "-n", "--fixed-strings", token, "--", *SEARCHED_PATHSPECS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):  # 1 == no matches
        raise AssertionError(f"git grep failed: {result.stderr.strip()}")
    own_path = str(Path(__file__).relative_to(REPO_ROOT))
    return [line for line in result.stdout.splitlines() if not line.startswith(f"{own_path}:")]


@pytest.mark.parametrize("relative_path", RETIRED_MODULES)
def test_retired_analysis_module_stays_deleted(relative_path: str) -> None:
    """A retired module file must not reappear under polylogue/analysis."""
    assert not (REPO_ROOT / relative_path).exists(), (
        f"{relative_path} was deleted by the analysis-package audit because no declared "
        "operation, CLI verb, MCP tool, insight descriptor or HTTP route reached it. "
        "Re-adding it requires adding the route that reaches it and a behaviour test "
        "through that route -- and then removing it from RETIRED_MODULES here."
    )


@pytest.mark.parametrize("token", RETIRED_NAMES)
def test_no_tracked_reference_to_a_retired_analysis_name(token: str) -> None:
    """Source, tooling and prose must not name something that no longer exists."""
    hits = _tracked_hits(token)
    assert not hits, (
        f"{token!r} names a construct the analysis-package audit deleted, but is still "
        "referenced by:\n  " + "\n  ".join(hits) + "\nUpdate the reference or restore the "
        "construct together with the declared route that reaches it."
    )


def test_the_ratchet_still_has_something_to_protect() -> None:
    """The ratchet must not pass by holding an empty list."""
    assert len(RETIRED_MODULES) >= 13
    assert len(RETIRED_NAMES) >= 38
    # The searched paths must really contain tracked files, or every grep
    # above would be vacuously empty.
    assert _tracked_hits("polylogue.analysis.registry")
