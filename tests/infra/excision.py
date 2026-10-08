"""Test adapters that call excision planning through an owned read snapshot."""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.operation_context import open_operation_read
from polylogue.security.excision import (
    ExcisionPlan,
    ExcisionTarget,
    plan_session_excision,
    resolve_session_excision_target,
)


def plan_session_excision_from_root(
    archive_root: Path, session_id: str, *, cascade_lineage: bool = False
) -> ExcisionPlan:
    with open_operation_read(archive_root) as pinned:
        return plan_session_excision(pinned.archive, session_id, cascade_lineage=cascade_lineage)


def resolve_session_excision_target_from_root(archive_root: Path, session_id: str) -> ExcisionTarget:
    with open_operation_read(archive_root) as pinned:
        return resolve_session_excision_target(pinned.archive, session_id)


def find_lineage_dependents_from_root(archive_root: Path, session_id: str) -> tuple[str, ...]:
    with open_operation_read(archive_root) as pinned:
        from polylogue.security.excision import find_lineage_dependents

        return find_lineage_dependents(pinned.archive, session_id)
