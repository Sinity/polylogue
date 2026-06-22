"""Executable CLI action contracts for the public command floor (#1816)."""

from __future__ import annotations

from dataclasses import dataclass
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


class ActionAffordancePayload(BaseModel):
    """Shared static action-affordance contract for public action surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    path: tuple[str, ...]
    effect: ActionEffect
    target: ActionTarget
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


@dataclass(frozen=True, slots=True)
class CliActionContract:
    """Behavioral contract for one public CLI action path.

    The Click command tree remains the source of truth for which commands
    exist. These contracts describe the public behavior that generated tests,
    docs, and future machine-envelope checks can bind to.
    """

    path: tuple[str, ...]
    effect: ActionEffect
    input_unit: InputUnit
    cardinality: ActionCardinality
    formats: frozenset[ActionFormat]
    default_format: ActionFormat
    machine_envelope: MachineEnvelope
    requires_daemon: bool
    guards: tuple[str, ...] = ()
    completion_context: CompletionContext | None = None
    target: ActionTarget = "selection"
    safety_level: ActionSafetyLevel = "safe"
    confirmation_command: str | None = None
    selection_command: str | None = None
    destination_support: tuple[ActionDestination, ...] = ("terminal",)
    disabled_reason: str | None = None
    estimated_cost: str | None = None
    next_actions: tuple[str, ...] = ()

    @property
    def action_id(self) -> str:
        """Return the stable dot-path id used by affordance consumers."""

        return ".".join(self.path)

    def to_affordance_payload(self) -> ActionAffordancePayload:
        """Return the shared query-action affordance payload.

        This is intentionally typed and narrow: CLI, web, MCP, docs, and
        completion adapters can consume the same metadata without importing
        Click. Runtime availability remains a separate per-target concern; this
        payload records the static contract and any known disabled reason.
        """

        return ActionAffordancePayload(
            id=self.action_id,
            path=self.path,
            effect=self.effect,
            target=self.target,
            input_unit=self.input_unit,
            cardinality_state=self.cardinality,
            safety_level=self.safety_level,
            confirmation_command=self.confirmation_command,
            selection_command=self.selection_command,
            destination_support=self.destination_support,
            format_support=tuple(sorted(self.formats, key=_format_order)),
            default_format=self.default_format,
            machine_envelope=self.machine_envelope,
            disabled_reason=self.disabled_reason,
            estimated_cost=self.estimated_cost,
            next_actions=self.next_actions,
            guards=self.guards,
            completion_context=self.completion_context,
            requires_daemon=self.requires_daemon,
        )


VIRTUAL_ACTION_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        # `find` is the #1842 query-intent keyword consumed by QueryFirstGroup,
        # not a registered Click command.
        ("find",),
    }
)


PUBLIC_ACTION_FLOOR: tuple[tuple[str, ...], ...] = (
    ("find",),
    ("select",),
    ("read",),
    ("continue",),
    ("mark",),
    ("analyze",),
    ("delete",),
    ("import",),
    ("config",),
    ("ops",),
)


ACTION_CONTRACTS: tuple[CliActionContract, ...] = (
    CliActionContract(
        path=("find",),
        effect="read",
        input_unit="none",
        cardinality="any",
        formats=frozenset({"human", "json", "ndjson"}),
        default_format="human",
        machine_envelope="result_set",
        requires_daemon=False,
        guards=("explicit_query_intent",),
        completion_context="query_expression",
        target="none",
        destination_support=("terminal", "stdout", "file", "api", "mcp"),
        next_actions=("select", "read", "analyze", "continue", "mark", "delete"),
    ),
    CliActionContract(
        path=("read",),
        effect="read",
        input_unit="query_result_set",
        cardinality="explicit_multi",
        formats=frozenset({"human", "json", "ndjson"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        guards=("single_match_unless_all", "file_destination_requires_out"),
        completion_context="session_id",
        confirmation_command=None,
        selection_command="polylogue find QUERY then select",
        destination_support=("terminal", "stdout", "browser", "clipboard", "file", "api", "mcp"),
        next_actions=("continue", "mark", "delete"),
    ),
    CliActionContract(
        path=("continue",),
        effect="read",
        input_unit="query_result_set",
        cardinality="singleton",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        completion_context="session_id",
        confirmation_command=None,
        selection_command="polylogue find QUERY then select",
        destination_support=("terminal", "stdout", "clipboard", "file", "api", "mcp"),
        next_actions=("read", "mark"),
    ),
    CliActionContract(
        path=("select",),
        effect="read",
        input_unit="query_result_set",
        cardinality="singleton",
        formats=frozenset({"human", "json"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        completion_context="session_id",
        destination_support=("terminal", "stdout", "api", "mcp"),
        next_actions=("read", "continue", "analyze", "mark", "delete"),
    ),
    CliActionContract(
        path=("mark",),
        effect="write",
        input_unit="query_result_set",
        cardinality="explicit_multi",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="mutation",
        requires_daemon=False,
        guards=("single_match_unless_all_or_first",),
        completion_context="session_id",
        safety_level="mutating",
        selection_command="polylogue find QUERY then select",
        destination_support=("terminal", "stdout", "api", "mcp"),
        next_actions=("read", "analyze"),
    ),
    CliActionContract(
        path=("analyze",),
        effect="read",
        input_unit="query_result_set",
        cardinality="any",
        formats=frozenset({"human", "json", "ndjson"}),
        default_format="human",
        machine_envelope="result_set",
        requires_daemon=False,
        completion_context="query_expression",
        destination_support=("terminal", "stdout", "file", "api", "mcp"),
        next_actions=("select", "read"),
    ),
    CliActionContract(
        path=("delete",),
        effect="destructive",
        input_unit="query_result_set",
        cardinality="destructive_multi",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="mutation",
        requires_daemon=False,
        guards=("dry_run_or_yes_required", "single_match_unless_all"),
        completion_context="session_id",
        safety_level="destructive",
        confirmation_command="polylogue find QUERY then delete --dry-run",
        selection_command="polylogue find QUERY then select",
        destination_support=("terminal", "stdout", "api", "mcp"),
        next_actions=("find",),
    ),
    CliActionContract(
        path=("import",),
        effect="import",
        input_unit="path",
        cardinality="singleton",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="mutation",
        requires_daemon=True,
        guards=("path_exists_or_demo", "daemon_accepts_schedule"),
        completion_context="filesystem_path",
        target="path",
        safety_level="mutating",
        destination_support=("terminal", "stdout", "api"),
        estimated_cost="archive write and downstream convergence work",
        next_actions=("ops", "read", "analyze"),
    ),
    CliActionContract(
        path=("config",),
        effect="config",
        input_unit="config",
        cardinality="any",
        formats=frozenset({"human", "json"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        guards=("secret_values_redacted",),
        completion_context="config_key",
        target="config",
        safety_level="operational",
        destination_support=("terminal", "stdout"),
        next_actions=("ops",),
    ),
    CliActionContract(
        path=("ops",),
        effect="ops",
        input_unit="runtime",
        cardinality="any",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        target="runtime",
        safety_level="operational",
        destination_support=("terminal", "stdout", "api"),
        next_actions=("find", "read"),
    ),
)


ACTION_CONTRACT_BY_PATH: dict[tuple[str, ...], CliActionContract] = {entry.path: entry for entry in ACTION_CONTRACTS}


def contract_for_path(path: tuple[str, ...]) -> CliActionContract | None:
    """Return the public action contract for ``path`` if one is declared."""
    return ACTION_CONTRACT_BY_PATH.get(path)


def action_completion_contexts() -> tuple[CompletionContext, ...]:
    """Return completion contexts declared by public action contracts."""
    return tuple(
        dict.fromkeys(entry.completion_context for entry in ACTION_CONTRACTS if entry.completion_context is not None)
    )


def action_affordance_payloads() -> list[ActionAffordancePayload]:
    """Return typed action affordance metadata for public contracts."""

    return [contract.to_affordance_payload() for contract in ACTION_CONTRACTS]


def query_result_action_affordance_payloads() -> list[ActionAffordancePayload]:
    """Return action affordances that operate on a query-result selection."""

    return [payload for payload in action_affordance_payloads() if payload.target == "selection"]


def action_affordance_list_payload() -> ActionAffordanceListPayload:
    """Return the public action-affordance inventory as one payload model."""

    return ActionAffordanceListPayload(actions=tuple(action_affordance_payloads()))


def _format_order(value: str) -> int:
    order = {"human": 0, "json": 1, "ndjson": 2}
    return order[value]


def _validate_contracts() -> None:
    duplicate_paths = len(ACTION_CONTRACT_BY_PATH) != len(ACTION_CONTRACTS)
    if duplicate_paths:
        raise ValueError("ACTION_CONTRACTS contains duplicate paths")
    for entry in ACTION_CONTRACTS:
        if entry.default_format not in entry.formats:
            raise ValueError(f"{entry.path!r} default_format is not declared in formats")
        if entry.confirmation_command is not None and entry.safety_level not in {"destructive", "mutating"}:
            raise ValueError(f"{entry.path!r} confirmation_command requires mutating/destructive safety")
        if (
            entry.cardinality in {"explicit_multi", "destructive_multi", "singleton"}
            and entry.input_unit == "query_result_set"
            and entry.path not in {("select",)}
            and entry.selection_command is None
        ):
            raise ValueError(f"{entry.path!r} query-result action must declare selection_command")
        if entry.safety_level == "destructive" and entry.confirmation_command is None:
            raise ValueError(f"{entry.path!r} destructive action must declare confirmation_command")


_validate_contracts()


__all__ = [
    "ACTION_CONTRACTS",
    "ACTION_CONTRACT_BY_PATH",
    "PUBLIC_ACTION_FLOOR",
    "VIRTUAL_ACTION_PATHS",
    "ActionCardinality",
    "ActionAffordanceListPayload",
    "ActionAffordancePayload",
    "ActionEffect",
    "ActionFormat",
    "CliActionContract",
    "CompletionContext",
    "InputUnit",
    "MachineEnvelope",
    "ActionDestination",
    "ActionSafetyLevel",
    "ActionTarget",
    "action_affordance_list_payload",
    "action_affordance_payloads",
    "action_completion_contexts",
    "contract_for_path",
    "query_result_action_affordance_payloads",
]
