"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Any


class OperationKind(str, Enum):
    """High-level operation class over runtime artifacts."""

    PLANNING = "planning"
    MATERIALIZATION = "materialization"
    INDEXING = "indexing"
    PROJECTION = "projection"
    CLI = "cli"
    BENCHMARK = "benchmark"
    QUERY = "query"
    HEALTHCHECK = "healthcheck"


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


@dataclass(frozen=True, slots=True)
class OperationCatalog:
    """Canonical operation registry with stable lookup and resolution helpers."""

    specs: tuple[OperationSpec, ...]

    def by_name(self) -> dict[str, OperationSpec]:
        return {spec.name: spec for spec in self.specs}

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def resolve(self, names: tuple[str, ...]) -> tuple[OperationSpec, ...]:
        by_name = self.by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def to_dict(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs]


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
    OperationSpec(
        name="materialize-session-products",
        kind=OperationKind.MATERIALIZATION,
        description="Build durable session-product rows and their trigger-maintained FTS projections from archive conversations.",
        consumes=("session_product_source_conversations",),
        produces=("session_product_rows", "session_product_fts"),
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_refresh.refresh_session_products_for_conversation_async",
        ),
        surfaces=("products", "doctor", "repair", "run.materialize"),
        mutates_state=True,
    ),
    OperationSpec(
        name="project-session-product-health",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and stale-surface semantics from durable session-product rows and FTS state.",
        consumes=("session_product_rows", "session_product_fts"),
        produces=("session_product_health",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "doctor", "archive_debt", "repair"),
        previewable=True,
    ),
)

DECLARED_CONTROL_PLANE_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="cli.help",
        kind=OperationKind.CLI,
        description="Render Click help for one command path without mutating archive state.",
        surfaces=("help", "showcase"),
        previewable=True,
    ),
    OperationSpec(
        name="cli.json-contract",
        kind=OperationKind.CLI,
        description="Exercise a machine-readable CLI JSON surface and verify its contract envelope.",
        surfaces=("doctor", "audit", "schema", "tags", "showcase"),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.query.search-filters",
        kind=OperationKind.BENCHMARK,
        description="Measure the canonical FTS and ConversationFilter query benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.storage.crud",
        kind=OperationKind.BENCHMARK,
        description="Measure repository and backend CRUD latency for the storage benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.pipeline.index-and-helpers",
        kind=OperationKind.BENCHMARK,
        description="Measure indexing and hot pipeline-helper throughput in the benchmark campaign domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.repair.action-events",
        kind=OperationKind.BENCHMARK,
        description="Measure action-event repair throughput in focused benchmark scenarios.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="index.message-fts-rebuild",
        kind=OperationKind.INDEXING,
        description="Benchmark full message FTS rebuild over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="index.message-fts-incremental",
        kind=OperationKind.INDEXING,
        description="Benchmark incremental message FTS updates over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="query.filters.synthetic-scan",
        kind=OperationKind.QUERY,
        description="Benchmark common synthetic filter-query scans over generated archives.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="health.startup.synthetic",
        kind=OperationKind.HEALTHCHECK,
        description="Benchmark startup health checks over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
)

DECLARED_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    *RUNTIME_OPERATION_SPECS,
    *DECLARED_CONTROL_PLANE_OPERATION_SPECS,
)


@lru_cache(maxsize=1)
def build_runtime_operation_catalog() -> OperationCatalog:
    """Return the authored runtime operation catalog."""

    return OperationCatalog(specs=RUNTIME_OPERATION_SPECS)


@lru_cache(maxsize=1)
def build_declared_operation_catalog() -> OperationCatalog:
    """Return every authored operation target referenced across verification surfaces."""

    return OperationCatalog(specs=DECLARED_OPERATION_SPECS)


__all__ = [
    "DECLARED_CONTROL_PLANE_OPERATION_SPECS",
    "DECLARED_OPERATION_SPECS",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "RUNTIME_OPERATION_SPECS",
]
