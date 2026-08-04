"""Compatibility import for the canonical ``OperationStatus`` enum.

Split out of :mod:`polylogue.operations.operation_contract` (polylogue-8s70):
The canonical enum lives in :mod:`polylogue.core.enums` so storage DDL and
operation surfaces share one vocabulary without a dependency cycle. This
module remains the stable import path for callers that only need the status
type.
"""

from __future__ import annotations

from polylogue.core.enums import OperationStatus

__all__ = ["OperationStatus"]
