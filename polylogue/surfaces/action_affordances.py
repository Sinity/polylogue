"""Shared action-affordance payload models for public surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

ActionEffect = Literal["read", "write", "destructive", "stream", "ops", "config", "import"]
InputUnit = Literal["none", "query_result_set", "session", "path", "config", "runtime", "assertion_candidate"]
ActionCardinality = Literal["any", "singleton", "explicit_multi", "destructive_multi"]
ActionFormat = Literal["human", "json", "ndjson"]
MachineEnvelope = Literal["result_set", "item", "mutation", "error", "stream_item"]
ActionTarget = Literal["none", "selection", "session", "path", "config", "runtime", "candidate"]
ActionSafetyLevel = Literal["safe", "mutating", "destructive", "operational"]
ActionDestination = Literal["terminal", "stdout", "browser", "clipboard", "file", "api", "mcp"]
CompletionContext = Literal[
    "config_key",
    "filesystem_path",
    "query_expression",
    "session_id",
]


class ActionInputPayload(BaseModel):
    """Input side of an action affordance. Keeps selection/query input explicit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: InputUnit
    completion_context: CompletionContext | None = None


class ActionExecutionPayload(BaseModel):
    """Execution constraints for an affordance before the action runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cardinality_state: ActionCardinality
    guards: tuple[str, ...] = ()
    requires_daemon: bool


class ActionOutputPayload(BaseModel):
    """Output contract for an affordance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    destination_support: tuple[ActionDestination, ...]
    format_support: tuple[ActionFormat, ...]
    default_format: ActionFormat
    machine_envelope: MachineEnvelope


class ActionSafetyPayload(BaseModel):
    """Safety and confirmation posture for an affordance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    safety_level: ActionSafetyLevel
    confirmation_command: str | None = None
    selection_command: str | None = None


class ActionAvailabilityPayload(BaseModel):
    """Availability and next-action posture for an affordance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disabled_reason: str | None = None
    estimated_cost: str | None = None
    next_actions: tuple[str, ...] = ()


class ActionAffordancePayload(BaseModel):
    """Shared static action-affordance contract for public action surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    path: tuple[str, ...]
    effect: ActionEffect
    target: ActionTarget
    input: ActionInputPayload
    execution: ActionExecutionPayload
    output: ActionOutputPayload
    safety: ActionSafetyPayload
    availability: ActionAvailabilityPayload


class ActionAffordanceListPayload(BaseModel):
    """Machine-readable inventory for public query/action affordances."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionAffordancePayload, ...]


CandidateReviewDecision = Literal["accept", "reject", "defer", "supersede"]


def assertion_candidate_review_affordances(
    *,
    candidate_ref: str | None = None,
    disabled_reasons: Mapping[CandidateReviewDecision, str | None] | None = None,
) -> tuple[ActionAffordancePayload, ...]:
    """Return shared review affordances for derived assertion candidates.

    Candidate review is a contextual surface rather than a top-level query
    action.  It still uses the same DTO so JSON/Markdown pack consumers see
    stable machine-readable ids and disabled reasons instead of display-only
    prose.
    """

    reasons = dict(disabled_reasons or {})
    base_disabled_reason = "candidate_ref_required" if not candidate_ref else None

    payloads: list[ActionAffordancePayload] = []
    for decision in ("accept", "reject", "defer", "supersede"):
        disabled_reason = reasons.get(decision) or base_disabled_reason
        payloads.append(
            ActionAffordancePayload(
                id=f"assertion_candidate.{decision}",
                path=("assertion-candidate", decision),
                effect="write",
                target="candidate",
                input=ActionInputPayload(unit="assertion_candidate"),
                execution=ActionExecutionPayload(
                    cardinality_state="singleton",
                    guards=("candidate_ref_required",),
                    requires_daemon=False,
                ),
                output=ActionOutputPayload(
                    destination_support=("terminal", "api", "mcp"),
                    format_support=("json",),
                    default_format="json",
                    machine_envelope="mutation",
                ),
                safety=ActionSafetyPayload(
                    safety_level="mutating",
                    confirmation_command=None,
                    selection_command=None,
                ),
                availability=ActionAvailabilityPayload(
                    disabled_reason=disabled_reason,
                    next_actions=("read", "context-image"),
                ),
            )
        )
    return tuple(payloads)


__all__ = [
    "ActionAffordanceListPayload",
    "ActionAffordancePayload",
    "ActionAvailabilityPayload",
    "ActionCardinality",
    "ActionExecutionPayload",
    "ActionInputPayload",
    "ActionOutputPayload",
    "ActionSafetyPayload",
    "ActionDestination",
    "ActionEffect",
    "ActionFormat",
    "ActionSafetyLevel",
    "ActionTarget",
    "CandidateReviewDecision",
    "CompletionContext",
    "InputUnit",
    "MachineEnvelope",
    "assertion_candidate_review_affordances",
]
