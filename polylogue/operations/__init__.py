"""Canonical archive operations shared across facade, CLI, and MCP surfaces."""

from .archive import (
    ArchiveDebtProduct,
    ArchiveOperations,
    ArchiveStats,
    get_provider_counts,
    list_provider_analytics_products,
)
from .specs import (
    OperationCatalog,
    OperationKind,
    OperationSpec,
    build_declared_operation_catalog,
    build_runtime_operation_catalog,
)

__all__ = [
    "ArchiveDebtProduct",
    "ArchiveOperations",
    "ArchiveStats",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "get_provider_counts",
    "list_provider_analytics_products",
]
