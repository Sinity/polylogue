"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class OperationKind(str, Enum):
    """High-level operation class over runtime artifacts."""

    PLANNING = "planning"
    MATERIALIZATION = "materialization"
    INDEXING = "indexing"
    PROJECTION = "projection"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One named runtime operation over declared artifact nodes."""

    name: str
    kind: OperationKind
    description: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    mutates_state: bool = False
    previewable: bool = False
    idempotent: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


RUNTIME_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="plan-validation-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that still require validation before normal parse planning.",
        consumes=("raw_validation_state",),
        produces=("validation_backlog",),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.validation_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
    ),
    OperationSpec(
        name="plan-parse-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that are eligible for parse planning under ordinary or force-reparse rules.",
        consumes=("raw_validation_state",),
        produces=("parse_backlog", "parse_quarantine"),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.parse_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
    ),
    OperationSpec(
        name="materialize-action-events",
        kind=OperationKind.MATERIALIZATION,
        description="Build the action-event read model and trigger-maintained FTS projection from tool-use source blocks.",
        consumes=("tool_use_source_blocks",),
        produces=("action_event_rows", "action_event_fts"),
        code_refs=(
            "polylogue.storage.action_event_rebuild_runtime.rebuild_action_event_read_model_sync",
            "polylogue.storage.backends.schema_ddl_actions.ACTION_FTS_DDL",
        ),
        surfaces=("index", "doctor", "repair", "retrieval_evidence"),
        mutates_state=True,
    ),
    OperationSpec(
        name="project-action-event-health",
        kind=OperationKind.PROJECTION,
        description="Project health, debt, and repair semantics from action-event rows and FTS state.",
        consumes=("action_event_rows", "action_event_fts"),
        produces=("action_event_health",),
        code_refs=(
            "polylogue.storage.derived_status",
            "polylogue.storage.repair",
            "polylogue.storage.embedding_stats_support",
        ),
        surfaces=("doctor", "archive_debt", "repair"),
        previewable=True,
    ),
)


def build_runtime_operation_specs() -> tuple[OperationSpec, ...]:
    """Return the authored runtime operation metadata."""

    return RUNTIME_OPERATION_SPECS


__all__ = [
    "OperationKind",
    "OperationSpec",
    "RUNTIME_OPERATION_SPECS",
    "build_runtime_operation_specs",
]
