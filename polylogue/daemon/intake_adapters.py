"""Public daemon seam for production intake adapters.

The domain-specific implementation lives in ``polylogue.operations`` so the
daemon ring does not acquire a second dependency on source/storage packages.
This module keeps the stable import surface used by daemon composition and
focused tests.
"""

from polylogue.operations.intake_adapters import (
    CallbackIntakeAdapter,
    ColdBuildCoverageError,
    ColdBuildGeneration,
    ColdBuildSettlement,
    DaemonIntakeContext,
    DaemonIntakeService,
    FileIntakeAdapter,
    RawMaterializationDiscovery,
    RawMaterializationIntakeAdapter,
    SubUnitHaltPolicy,
    active_cold_build_generation,
    active_index_generation_is_empty,
    build_intake_adapters,
    classify_cold_build_settlement_failure,
    clear_cold_build_generation,
    discover_pending_raw_ids,
    promote_cold_build_covering_active_index,
    register_cold_build_generation,
)

__all__ = [
    "ColdBuildCoverageError",
    "ColdBuildGeneration",
    "ColdBuildSettlement",
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "RawMaterializationDiscovery",
    "RawMaterializationIntakeAdapter",
    "SubUnitHaltPolicy",
    "active_cold_build_generation",
    "active_index_generation_is_empty",
    "discover_pending_raw_ids",
    "promote_cold_build_covering_active_index",
    "CallbackIntakeAdapter",
    "build_intake_adapters",
    "classify_cold_build_settlement_failure",
    "clear_cold_build_generation",
    "register_cold_build_generation",
]
