"""Public daemon seam for production intake adapters.

The domain-specific implementation lives in ``polylogue.operations`` so the
daemon ring does not acquire a second dependency on source/storage packages.
This module keeps the stable import surface used by daemon composition and
focused tests.
"""

from polylogue.operations.intake_adapters import (
    CallbackIntakeAdapter,
    ColdBuildGeneration,
    DaemonIntakeContext,
    DaemonIntakeService,
    FileIntakeAdapter,
    RawMaterializationDiscovery,
    RawMaterializationIntakeAdapter,
    active_index_generation_is_empty,
    build_intake_adapters,
    clear_cold_build_generation,
    discover_pending_raw_ids,
    register_cold_build_generation,
)

__all__ = [
    "ColdBuildGeneration",
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "RawMaterializationDiscovery",
    "RawMaterializationIntakeAdapter",
    "active_index_generation_is_empty",
    "discover_pending_raw_ids",
    "CallbackIntakeAdapter",
    "build_intake_adapters",
    "clear_cold_build_generation",
    "register_cold_build_generation",
]
