"""Typed runtime artifact specifications shared across control-plane surfaces."""

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


RUNTIME_ARTIFACT_NODES: tuple[ArtifactNode, ...] = (
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
        name="message_source_rows",
        layer=ArtifactLayer.SOURCE,
        description="Persisted message rows that feed lexical FTS indexing and archive search.",
        code_refs=(
            "polylogue.storage.fts_lifecycle.FTS_INDEXABLE_MESSAGE_COUNT_SQL",
            "polylogue.storage.fts_lifecycle.repair_fts_index_sync",
        ),
    ),
    ArtifactNode(
        name="message_fts",
        layer=ArtifactLayer.INDEX,
        description="Lexical FTS projection over persisted message rows.",
        depends_on=("message_source_rows",),
        code_refs=(
            "polylogue.storage.fts_lifecycle.message_fts_readiness_sync",
            "polylogue.storage.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts_lifecycle.repair_fts_index_sync",
        ),
        repair_targets=("fts",),
        health_surfaces=("doctor", "archive_debt", "query"),
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
    ArtifactNode(
        name="session_product_source_conversations",
        layer=ArtifactLayer.SOURCE,
        description="Hydratable conversation/message/attachment/block rows that feed durable session-product rebuilds.",
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_refresh.refresh_session_products_for_conversation_async",
        ),
    ),
    ArtifactNode(
        name="session_product_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session-profile, timeline, thread, and aggregate rows derived from archive conversations.",
        depends_on=("session_product_source_conversations",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_status",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_product_fts",
        layer=ArtifactLayer.INDEX,
        description="The session-product FTS family over profiles, work events, and threads.",
        depends_on=("session_product_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.derived_status_products",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_product_health",
        layer=ArtifactLayer.PROJECTION,
        description="Projected readiness, debt, and stale-surface semantics for durable session products.",
        depends_on=("session_product_rows", "session_product_fts"),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.products",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="conversation_query_results",
        layer=ArtifactLayer.PROJECTION,
        description="Conversation-level query and search results resolved from lexical retrieval over the archive.",
        depends_on=("message_fts",),
        code_refs=(
            "polylogue.operations.archive.ArchiveSearchMixin.query_conversations",
            "polylogue.operations.archive.ArchiveSearchMixin.search",
            "polylogue.lib.query_plan_execution",
        ),
        health_surfaces=("query", "mcp", "facade"),
    ),
    ArtifactNode(
        name="archive_health",
        layer=ArtifactLayer.PROJECTION,
        description="Projected archive-wide health and maintenance view over message FTS and durable derived-model readiness.",
        depends_on=("message_fts", "action_event_health", "session_product_health"),
        code_refs=(
            "polylogue.health.run_archive_health",
            "polylogue.storage.derived_status.collect_derived_model_statuses_sync",
            "polylogue.storage.repair.collect_archive_debt_statuses_sync",
        ),
        repair_targets=("fts", "action_event_read_model", "session_products"),
        health_surfaces=("doctor", "archive_debt", "maintenance"),
    ),
)

RUNTIME_ARTIFACT_PATHS: tuple[ArtifactPath, ...] = (
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
    ArtifactPath(
        name="session-product-repair-loop",
        description="Archive conversations through durable session-product rows, FTS, and projected repair semantics.",
        nodes=(
            "session_product_source_conversations",
            "session_product_rows",
            "session_product_fts",
            "session_product_health",
        ),
    ),
    ArtifactPath(
        name="message-fts-health-loop",
        description="Persisted messages through lexical FTS and the archive-wide health projection.",
        nodes=(
            "message_source_rows",
            "message_fts",
            "action_event_health",
            "session_product_health",
            "archive_health",
        ),
    ),
    ArtifactPath(
        name="conversation-query-loop",
        description="Lexical message FTS through conversation-level query and search result projections.",
        nodes=(
            "message_fts",
            "conversation_query_results",
        ),
    ),
)


def build_runtime_artifact_nodes() -> tuple[ArtifactNode, ...]:
    return RUNTIME_ARTIFACT_NODES


def build_runtime_artifact_paths() -> tuple[ArtifactPath, ...]:
    return RUNTIME_ARTIFACT_PATHS


__all__ = [
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "RUNTIME_ARTIFACT_NODES",
    "RUNTIME_ARTIFACT_PATHS",
    "build_runtime_artifact_nodes",
    "build_runtime_artifact_paths",
]
