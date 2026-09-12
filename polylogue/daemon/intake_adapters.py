"""Public daemon seam for production intake adapters.

The domain-specific implementation lives in ``polylogue.operations`` so the
daemon ring does not acquire a second dependency on source/storage packages.
This module keeps the stable import surface used by daemon composition and
focused tests.
"""

from polylogue.operations.intake_adapters import (
    CallbackIntakeAdapter,
    DaemonIntakeContext,
    DaemonIntakeService,
    FileIntakeAdapter,
    HookSpoolIntakeAdapter,
    build_intake_adapters,
)

__all__ = [
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "HookSpoolIntakeAdapter",
    "CallbackIntakeAdapter",
    "build_intake_adapters",
]
