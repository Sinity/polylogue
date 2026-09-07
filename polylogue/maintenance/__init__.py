"""Fail-closed integrity verification and guarded recovery workflows."""

# The durable-reference transition's production entry points. Keeping the
# imports here makes the owner of the rebuild's rebind step visible to callers.
from polylogue.maintenance.assertion_transition import reconcile_object_refs
from polylogue.maintenance.durable_reference_transition import (
    apply_durable_reference_transition,
    plan_durable_reference_transition,
)

__all__ = [
    "apply_durable_reference_transition",
    "plan_durable_reference_transition",
    "reconcile_object_refs",
]
