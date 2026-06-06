"""Canonical archive operations shared across facade, CLI, and MCP surfaces."""

from .archive import (
    ArchiveDebtInsight,
    ArchiveStats,
    CompletionAggregate,
)
from .import_contracts import (
    ImportOperation,
    RawFailureSample,
    bounded_failure_samples,
)
from .import_operations import (
    ImportAck,
    ImportRequest,
)
from .operation_contract import (
    OperationAck,
    OperationFollowUp,
    OperationRequest,
    OperationStatus,
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
    "ArchiveStats",
    "CompletionAggregate",
    "ImportAck",
    "ImportOperation",
    "ImportRequest",
    "OperationAck",
    "OperationCatalog",
    "OperationFollowUp",
    "OperationKind",
    "OperationRequest",
    "OperationSpec",
    "OperationStatus",
    "RawFailureSample",
    "bounded_failure_samples",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
]
