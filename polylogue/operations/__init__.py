"""Canonical archive operations shared across facade, CLI, and MCP surfaces."""

from .archive import (
    ArchiveDebtInsight,
    ArchiveOperations,
    ArchiveStats,
    CompletionAggregate,
    get_provider_counts,
    list_provider_analytics_insights,
)
from .import_contracts import (
    ImportOperation,
    RawFailureSample,
    bounded_failure_samples,
)
from .specs import (
    OperationCatalog,
    OperationKind,
    OperationSpec,
    build_declared_operation_catalog,
    build_runtime_operation_catalog,
)

__all__ = [
    "ArchiveDebtInsight",
    "ArchiveOperations",
    "ArchiveStats",
    "CompletionAggregate",
    "ImportOperation",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "RawFailureSample",
    "bounded_failure_samples",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "get_provider_counts",
    "list_provider_analytics_insights",
]
