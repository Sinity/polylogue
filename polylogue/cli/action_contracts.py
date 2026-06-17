"""Executable CLI action contracts for the public command floor (#1816)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionEffect = Literal["read", "write", "destructive", "stream", "ops", "config", "import"]
InputUnit = Literal["none", "query_result_set", "session", "path", "config", "runtime"]
ActionCardinality = Literal["any", "singleton", "explicit_multi", "destructive_multi"]
ActionFormat = Literal["human", "json", "ndjson"]
MachineEnvelope = Literal["result_set", "item", "mutation", "error", "stream_item"]
CompletionContext = Literal[
    "archive_path",
    "config_key",
    "daemon_url",
    "filesystem_path",
    "maintenance_operation",
    "query_expression",
    "session_id",
]


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
    ("mark",),
    ("analyze",),
    ("delete",),
    ("import",),
    ("config",),
    # Current ops reality is split across concrete root commands; there is no
    # `polylogue ops` group yet.
    ("status",),
    ("doctor",),
    ("maintenance",),
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
    ),
    CliActionContract(
        path=("status",),
        effect="ops",
        input_unit="runtime",
        cardinality="any",
        formats=frozenset({"human", "json"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        completion_context="daemon_url",
    ),
    CliActionContract(
        path=("doctor",),
        effect="ops",
        input_unit="runtime",
        cardinality="any",
        formats=frozenset({"human", "json"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        completion_context="archive_path",
    ),
    CliActionContract(
        path=("maintenance",),
        effect="ops",
        input_unit="runtime",
        cardinality="any",
        formats=frozenset({"human"}),
        default_format="human",
        machine_envelope="item",
        requires_daemon=False,
        completion_context="maintenance_operation",
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


def _validate_contracts() -> None:
    duplicate_paths = len(ACTION_CONTRACT_BY_PATH) != len(ACTION_CONTRACTS)
    if duplicate_paths:
        raise ValueError("ACTION_CONTRACTS contains duplicate paths")
    for entry in ACTION_CONTRACTS:
        if entry.default_format not in entry.formats:
            raise ValueError(f"{entry.path!r} default_format is not declared in formats")


_validate_contracts()


__all__ = [
    "ACTION_CONTRACTS",
    "ACTION_CONTRACT_BY_PATH",
    "PUBLIC_ACTION_FLOOR",
    "VIRTUAL_ACTION_PATHS",
    "ActionCardinality",
    "ActionEffect",
    "ActionFormat",
    "CliActionContract",
    "CompletionContext",
    "InputUnit",
    "MachineEnvelope",
    "action_completion_contexts",
    "contract_for_path",
]
