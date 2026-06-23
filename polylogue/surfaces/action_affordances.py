"""Shared action-affordance payload models for public surfaces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ActionEffect = Literal["read", "write", "destructive", "stream", "ops", "config", "import"]
InputUnit = Literal["none", "query_result_set", "session", "path", "config", "runtime"]
ActionCardinality = Literal["any", "singleton", "explicit_multi", "destructive_multi"]
ActionFormat = Literal["human", "json", "ndjson"]
MachineEnvelope = Literal["result_set", "item", "mutation", "error", "stream_item"]
ActionTarget = Literal["none", "selection", "session", "path", "config", "runtime"]
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
    input_unit: InputUnit
    cardinality_state: ActionCardinality
    safety_level: ActionSafetyLevel
    confirmation_command: str | None = None
    selection_command: str | None = None
    destination_support: tuple[ActionDestination, ...]
    format_support: tuple[ActionFormat, ...]
    default_format: ActionFormat
    machine_envelope: MachineEnvelope
    disabled_reason: str | None = None
    estimated_cost: str | None = None
    next_actions: tuple[str, ...] = ()
    guards: tuple[str, ...] = ()
    completion_context: CompletionContext | None = None
    requires_daemon: bool


class ActionAffordanceListPayload(BaseModel):
    """Machine-readable inventory for public query/action affordances."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionAffordancePayload, ...]


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
    "CompletionContext",
    "InputUnit",
    "MachineEnvelope",
]
