"""Explicit artifact/dependency map for Polylogue runtime semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ArtifactLayer(str, Enum):
    SOURCE = "source"
    DURABLE = "durable"
    DERIVED = "derived"
    INDEX = "index"
    PROJECTION = "projection"


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    """One named artifact or projection in the Polylogue runtime graph."""

    name: str
    layer: ArtifactLayer
    description: str
    depends_on: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    repair_targets: tuple[str, ...] = ()
    health_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer.value
        return data


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    """One curated path through the artifact graph."""

    name: str
    description: str
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactOperation:
    """One named runtime operation over declared artifact nodes."""

    name: str
    description: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Named artifact nodes plus curated high-value paths through them."""

    nodes: tuple[ArtifactNode, ...]
    paths: tuple[ArtifactPath, ...]
    operations: tuple[ArtifactOperation, ...]

    def by_name(self) -> dict[str, ArtifactNode]:
        return {node.name: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "paths": [path.to_dict() for path in self.paths],
            "operations": [operation.to_dict() for operation in self.operations],
        }


def build_artifact_graph() -> ArtifactGraph:
    nodes = (
        ArtifactNode(
            name="raw_validation_state",
            layer=ArtifactLayer.DURABLE,
            description="Persisted raw-conversation validation and parse state in raw_conversations.",
            code_refs=(
                "polylogue.storage.raw_ingest_artifacts.RawIngestArtifactState",
                "polylogue.storage.backends.queries.raw_state",
            ),
        ),
        ArtifactNode(
            name="validation_backlog",
            layer=ArtifactLayer.PROJECTION,
            description="Raw records that still require validation before ordinary parse planning can trust them.",
            depends_on=("raw_validation_state",),
            code_refs=(
                "polylogue.storage.raw_ingest_artifacts.validation_backlog_query_spec",
                "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
            ),
        ),
        ArtifactNode(
            name="parse_backlog",
            layer=ArtifactLayer.PROJECTION,
            description="Raw records eligible for parse planning, including force-reparse simulation.",
            depends_on=("raw_validation_state",),
            code_refs=(
                "polylogue.storage.raw_ingest_artifacts.parse_backlog_query_spec",
                "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
                "polylogue.pipeline.services.planning_runtime.build_ingest_plan",
            ),
        ),
        ArtifactNode(
            name="parse_quarantine",
            layer=ArtifactLayer.PROJECTION,
            description="Validation-failed unparsed raws that stay out of ordinary parse backlog but return under force-reparse.",
            depends_on=("raw_validation_state",),
            code_refs=("polylogue.storage.raw_ingest_artifacts.RawIngestArtifactState",),
        ),
        ArtifactNode(
            name="tool_use_source_blocks",
            layer=ArtifactLayer.SOURCE,
            description="Tool-use content blocks anchored to valid conversations.",
            code_refs=("polylogue.storage.action_event_status",),
        ),
        ArtifactNode(
            name="action_event_rows",
            layer=ArtifactLayer.DERIVED,
            description="Materialized action-event read model derived from tool-use source blocks.",
            depends_on=("tool_use_source_blocks",),
            code_refs=(
                "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
                "polylogue.storage.action_event_status",
            ),
            repair_targets=("action_event_read_model",),
            health_surfaces=("doctor", "archive_debt"),
        ),
        ArtifactNode(
            name="action_event_fts",
            layer=ArtifactLayer.INDEX,
            description="FTS projection over the action-event read model.",
            depends_on=("action_event_rows",),
            code_refs=(
                "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
                "polylogue.storage.derived_status_products.build_action_statuses",
            ),
            repair_targets=("action_event_read_model",),
            health_surfaces=("doctor", "archive_debt", "retrieval_evidence"),
        ),
        ArtifactNode(
            name="action_event_health",
            layer=ArtifactLayer.PROJECTION,
            description="Projected health, debt, and repair semantics over action-event rows and FTS.",
            depends_on=("action_event_rows", "action_event_fts"),
            code_refs=(
                "polylogue.storage.derived_status",
                "polylogue.storage.repair",
                "polylogue.storage.embedding_stats_support",
            ),
            repair_targets=("action_event_read_model",),
            health_surfaces=("doctor", "archive_debt", "retrieval_evidence"),
        ),
    )
    paths = (
        ArtifactPath(
            name="raw-reparse-loop",
            description="Raw validation state to validation/parse backlog and quarantine projections.",
            nodes=(
                "raw_validation_state",
                "validation_backlog",
                "parse_backlog",
                "parse_quarantine",
            ),
        ),
        ArtifactPath(
            name="action-event-repair-loop",
            description="Tool-use source blocks through action-event rows, FTS, and projected repair semantics.",
            nodes=(
                "tool_use_source_blocks",
                "action_event_rows",
                "action_event_fts",
                "action_event_health",
            ),
        ),
    )
    operations = (
        ArtifactOperation(
            name="plan-validation-backlog",
            description="Select raw records that still require validation before normal parse planning.",
            consumes=("raw_validation_state",),
            produces=("validation_backlog",),
            code_refs=(
                "polylogue.storage.raw_ingest_artifacts.validation_backlog_query_spec",
                "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
            ),
            surfaces=("run.parse", "reparse"),
        ),
        ArtifactOperation(
            name="plan-parse-backlog",
            description="Select raw records that are eligible for parse planning under ordinary or force-reparse rules.",
            consumes=("raw_validation_state",),
            produces=("parse_backlog", "parse_quarantine"),
            code_refs=(
                "polylogue.storage.raw_ingest_artifacts.parse_backlog_query_spec",
                "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
            ),
            surfaces=("run.parse", "reparse"),
        ),
        ArtifactOperation(
            name="materialize-action-events",
            description="Build the action-event read model from tool-use source blocks.",
            consumes=("tool_use_source_blocks",),
            produces=("action_event_rows",),
            code_refs=(
                "polylogue.storage.action_event_status",
                "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
            ),
            surfaces=("doctor", "repair"),
        ),
        ArtifactOperation(
            name="index-action-events",
            description="Refresh the action-event FTS projection from the materialized action-event rows.",
            consumes=("action_event_rows",),
            produces=("action_event_fts",),
            code_refs=(
                "polylogue.storage.derived_status_products.build_action_statuses",
                "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
            ),
            surfaces=("doctor", "repair", "retrieval_evidence"),
        ),
        ArtifactOperation(
            name="project-action-event-health",
            description="Project health, debt, and repair semantics from action-event rows and FTS state.",
            consumes=("action_event_rows", "action_event_fts"),
            produces=("action_event_health",),
            code_refs=(
                "polylogue.storage.derived_status",
                "polylogue.storage.repair",
                "polylogue.storage.embedding_stats_support",
            ),
            surfaces=("doctor", "archive_debt", "repair"),
        ),
    )
    return ArtifactGraph(nodes=nodes, paths=paths, operations=operations)


__all__ = [
    "ArtifactGraph",
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactOperation",
    "ArtifactPath",
    "build_artifact_graph",
]
