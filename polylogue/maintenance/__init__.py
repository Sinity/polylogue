"""Fail-closed integrity verification and operator-supervised repair workflows."""

# Public maintenance primitive used by candidate acceptance and offline
# verification.  Keeping the import here makes the owner visible to callers.
from polylogue.maintenance.assertion_transition import reconcile_object_refs

__all__ = ["reconcile_object_refs"]
