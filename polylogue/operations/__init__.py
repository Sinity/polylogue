"""Canonical archive operations shared across facade, CLI, and MCP surfaces."""

from .archive import (
    ArchiveDebtProduct,
    ArchiveOperations,
    ArchiveStats,
    get_provider_counts,
    list_provider_analytics_products,
)
from .specs import (
    OperationKind,
    OperationSpec,
    build_declared_operation_specs,
    build_runtime_operation_specs,
)

__all__ = [
    "ArchiveDebtProduct",
    "ArchiveOperations",
    "ArchiveStats",
    "OperationKind",
    "OperationSpec",
    "build_declared_operation_specs",
    "build_runtime_operation_specs",
    "get_provider_counts",
    "list_provider_analytics_products",
]
