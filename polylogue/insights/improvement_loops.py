"""Declarative registry of improvement-loop specs (polylogue-rxdo.11).

Every closed-loop mechanism in the epic's design is the same 5-tuple: watch
(a standing query or other signal source) -> measure (a content-addressed
metric) -> propose (a recipe emitting candidates, never auto-applying) ->
judge (the existing assertion judgment lifecycle) -> bump (a content-addressed
artifact version). This module is the "declare loops like insight
descriptors in one LOOP_REGISTRY" contract: a single place naming every loop
instance and its five parts, so loop health becomes a queryable fact instead
of scattered prose across beads.

Scope note (read before adding an ``active`` entry): the corrective
acceptance criteria for polylogue-rxdo.11 require the first two pilots (L1
recall-relevance, L2 classifier-residue) to *execute* through one shared
scheduler/state contract, not merely be declared. That scheduler does not
exist yet, and L1's own signal source (polylogue-37t.17's read-access log)
is itself unimplemented. Every entry below is therefore ``status="horizon"``
-- this registry is the declaration surface the eventual scheduler will read,
not a claim that any loop is running. Flipping an entry to ``"active"``
requires the shared scheduler/state module plus that loop's own watch/measure/
propose/judge/bump wiring to exist and be tested; do not flip it to make this
registry look more complete than it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LoopStatus = Literal["horizon", "active"]


@dataclass(frozen=True, slots=True)
class ImprovementLoopSpec:
    """One declared instance of the watch/measure/propose/judge/bump 5-tuple."""

    loop_id: str
    title: str
    watch: str
    """What signal source the loop observes (a standing query ref, event stream, etc.)."""
    measure: str
    """The content-addressed metric/definition the loop reduces its signal to."""
    propose: str
    """What produces candidates from the measurement. Never auto-applies."""
    judge: str
    """The gate that turns a candidate into an accepted change (existing assertion lifecycle unless noted)."""
    artifact: str
    """What content-addressed artifact the loop bumps a version of when judged."""
    status: LoopStatus
    implementation_ref: str | None = None
    """Bead id (or module) that owns this loop's concrete wiring, if any."""


LOOP_REGISTRY: dict[str, ImprovementLoopSpec] = {
    spec.loop_id: spec
    for spec in (
        ImprovementLoopSpec(
            loop_id="L1",
            title="Recall relevance",
            watch="context-delivery receipts (injected refs) + downstream usage (cite/quote/re-read)",
            measure="per-item usage rate",
            propose="retrieval ranker reweighting from implicit relevance feedback",
            judge="operator/agent judgment lifecycle (assertion candidate -> accepted)",
            artifact="ranker:<hash>",
            status="horizon",
            implementation_ref="polylogue-37t.17",
        ),
        ImprovementLoopSpec(
            loop_id="L2",
            title="Classifier residue",
            watch="PACK-A/B classifier 'unclassified residue' confessions",
            measure="residue volume per command shape (standing query)",
            propose="agent-drafted rules for the top residue clusters",
            judge="operator/agent judgment lifecycle",
            artifact="classifier:<hash>",
            status="horizon",
        ),
        ImprovementLoopSpec(
            loop_id="L3",
            title="Judge calibration",
            watch="agent-judge vs operator-gold overlap",
            measure="per-judge per-dimension agreement",
            propose="weight/routing updates",
            judge="operator confirmation of routing policy changes",
            artifact="judge-routing-policy:<hash>",
            status="horizon",
            implementation_ref="polylogue-rxdo.9.12",
        ),
        ImprovementLoopSpec(
            loop_id="L4",
            title="Orchestration prompt outcomes",
            watch="stored lane-prompt artifacts + outcomes (PR merged, review iterations, cost, time)",
            measure="prompt-feature x outcome correlations",
            propose="findings + prompt template diffs",
            judge="operator adoption of a template version",
            artifact="prompt-template:<hash>",
            status="horizon",
        ),
        ImprovementLoopSpec(
            loop_id="L5",
            title="Detector precision",
            watch="pathology/finding detector candidates + their judgments",
            measure="per-detector precision (standing query)",
            propose="threshold/rule adjustments",
            judge="operator/agent judgment lifecycle",
            artifact="detector:<hash>",
            status="horizon",
        ),
        ImprovementLoopSpec(
            loop_id="L6",
            title="Title/summary CTR",
            watch="query-run telemetry + subsequent read events (implicit click-through)",
            measure="per-title-source CTR, rank-at-click recorded to avoid position bias",
            propose="title-generation strategy ranking",
            judge="operator strategy flag flip",
            artifact="title-strategy:<hash>",
            status="horizon",
            implementation_ref="polylogue-rxdo.3",
        ),
        ImprovementLoopSpec(
            loop_id="L7",
            title="Compaction regret",
            watch="compaction boundaries + later agent re-derivations of discarded prefix content",
            measure="regret = re-derived mass that was discarded (embedding match)",
            propose="compaction policy tuning (what to preserve)",
            judge="operator adoption of a policy version",
            artifact="compaction-policy:<hash>",
            status="horizon",
            implementation_ref="polylogue-gjg.3",
        ),
        ImprovementLoopSpec(
            loop_id="L8",
            title="Cost routing",
            watch="routing decisions + judged outcomes",
            measure="tier efficiency frontier",
            propose="routing advisor updates",
            judge="operator adoption",
            artifact="routing-advisor:<hash>",
            status="horizon",
        ),
        ImprovementLoopSpec(
            loop_id="L9",
            title="Ontology drift",
            watch="taxonomy/ontology usage drift signals",
            measure="drift metric (per polylogue-dve1's design)",
            propose="ontology revision candidates",
            judge="operator adoption",
            artifact="ontology:<hash>",
            status="horizon",
            implementation_ref="polylogue-dve1",
        ),
        ImprovementLoopSpec(
            loop_id="L10",
            title="Elicitation value (meta-loop)",
            watch="every recorded judgment's downstream decision impact (retrospective re-derivation diff)",
            measure="decision-impact per judgment type",
            propose="asking-policy update (solicit highest-expected-impact next)",
            judge="operator adoption of a policy version",
            artifact="asking-policy:<hash>",
            status="horizon",
        ),
        ImprovementLoopSpec(
            loop_id="L11",
            title="Declaration recall",
            watch="retrospective PACK-D detection of undeclared corrections/claims vs declared markers",
            measure="per-agent recall score",
            propose="skill/preamble revisions",
            judge="operator adoption (explicit, revocable policy assertion; no blocking enforcement)",
            artifact="skill:<hash>",
            status="horizon",
            implementation_ref="polylogue-37t.2",
        ),
        ImprovementLoopSpec(
            loop_id="L12",
            title="Curriculum",
            watch="query-run telemetry",
            measure="recipe value",
            propose="curriculum diff",
            judge="operator gate",
            artifact="skill:<hash>",
            status="horizon",
            implementation_ref="polylogue-xv1u",
        ),
        ImprovementLoopSpec(
            loop_id="L13",
            title="Capture-coverage error",
            watch="sessions-known-to-exist vs archived, per origin",
            measure="coverage gap volume",
            propose="budgeted alerts / remediation candidates",
            judge="operator/agent judgment lifecycle",
            artifact="capture-coverage-policy:<hash>",
            status="horizon",
            implementation_ref="polylogue-3uw",
        ),
    )
}


def active_loops() -> tuple[ImprovementLoopSpec, ...]:
    """Loops the shared scheduler is actually executing today (empty until it exists)."""
    return tuple(spec for spec in LOOP_REGISTRY.values() if spec.status == "active")


def horizon_loops() -> tuple[ImprovementLoopSpec, ...]:
    """Loops declared but not yet wired to a real scheduler."""
    return tuple(spec for spec in LOOP_REGISTRY.values() if spec.status == "horizon")


__all__ = [
    "LOOP_REGISTRY",
    "ImprovementLoopSpec",
    "LoopStatus",
    "active_loops",
    "horizon_loops",
]
