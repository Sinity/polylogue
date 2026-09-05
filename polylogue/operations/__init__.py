"""Canonical archive operations shared across facade, CLI, and MCP surfaces.

Re-exports are lazy (PEP 562 module ``__getattr__``): ``.archive`` alone pulls
in the whole insights registry (``insights.archive`` -> ``storage.raw_convergence`` and
friends), so a caller that only needs e.g. ``OperationStatus`` from
``.operation_contract`` -- reached simply by importing a *submodule* of this
package, which Python resolves by running this ``__init__`` first -- used to
pay for every sibling submodule's full import weight too (polylogue-8s70).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .archive import (
        ArchiveDebtInsight,
        ArchiveStats,
        CompletionAggregate,
    )
    from .candidate_build import (
        CandidateBuildBudget,
        CandidateBuildError,
        CandidateBuildGeneration,
        CandidateBuildObligation,
        CandidateBuildPlan,
        CandidateBuildPlanningContext,
        CandidateBuildProgress,
        CandidateBuildReceipt,
        CandidateBuildRequest,
        CandidateBuildResult,
        CandidateBuildWireRequest,
        SourceSeal,
        lower_candidate_build_wire,
        plan_candidate_build,
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
    from .operation_contract import OperationFollowUp
    from .operation_status import OperationStatus
    from .specs import (
        OperationCatalog,
        OperationKind,
        OperationSpec,
        SafetyGuard,
        build_declared_operation_catalog,
        build_runtime_operation_catalog,
    )


def __getattr__(name: str) -> object:
    lazy_exports = {
        "ArchiveDebtInsight": (".archive", "ArchiveDebtInsight"),
        "ArchiveStats": (".archive", "ArchiveStats"),
        "CompletionAggregate": (".archive", "CompletionAggregate"),
        "ImportOperation": (".import_contracts", "ImportOperation"),
        "RawFailureSample": (".import_contracts", "RawFailureSample"),
        "bounded_failure_samples": (".import_contracts", "bounded_failure_samples"),
        "ImportAck": (".import_operations", "ImportAck"),
        "ImportRequest": (".import_operations", "ImportRequest"),
        "OperationFollowUp": (".operation_contract", "OperationFollowUp"),
        "CandidateBuildBudget": (".candidate_build", "CandidateBuildBudget"),
        "CandidateBuildError": (".candidate_build", "CandidateBuildError"),
        "CandidateBuildGeneration": (".candidate_build", "CandidateBuildGeneration"),
        "CandidateBuildObligation": (".candidate_build", "CandidateBuildObligation"),
        "CandidateBuildPlan": (".candidate_build", "CandidateBuildPlan"),
        "CandidateBuildPlanningContext": (".candidate_build", "CandidateBuildPlanningContext"),
        "CandidateBuildProgress": (".candidate_build", "CandidateBuildProgress"),
        "CandidateBuildReceipt": (".candidate_build", "CandidateBuildReceipt"),
        "CandidateBuildRequest": (".candidate_build", "CandidateBuildRequest"),
        "CandidateBuildResult": (".candidate_build", "CandidateBuildResult"),
        "CandidateBuildWireRequest": (".candidate_build", "CandidateBuildWireRequest"),
        "SourceSeal": (".candidate_build", "SourceSeal"),
        "lower_candidate_build_wire": (".candidate_build", "lower_candidate_build_wire"),
        "plan_candidate_build": (".candidate_build", "plan_candidate_build"),
        "OperationStatus": (".operation_status", "OperationStatus"),
        "OperationCatalog": (".specs", "OperationCatalog"),
        "OperationKind": (".specs", "OperationKind"),
        "OperationSpec": (".specs", "OperationSpec"),
        "SafetyGuard": (".specs", "SafetyGuard"),
        "build_declared_operation_catalog": (".specs", "build_declared_operation_catalog"),
        "build_runtime_operation_catalog": (".specs", "build_runtime_operation_catalog"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArchiveDebtInsight",
    "ArchiveStats",
    "CandidateBuildBudget",
    "CandidateBuildError",
    "CandidateBuildGeneration",
    "CandidateBuildObligation",
    "CandidateBuildPlan",
    "CandidateBuildPlanningContext",
    "CandidateBuildProgress",
    "CandidateBuildReceipt",
    "CandidateBuildRequest",
    "CandidateBuildResult",
    "CandidateBuildWireRequest",
    "CompletionAggregate",
    "ImportAck",
    "ImportOperation",
    "ImportRequest",
    "OperationCatalog",
    "OperationFollowUp",
    "OperationKind",
    "OperationSpec",
    "OperationStatus",
    "RawFailureSample",
    "SafetyGuard",
    "SourceSeal",
    "bounded_failure_samples",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "lower_candidate_build_wire",
    "plan_candidate_build",
]
