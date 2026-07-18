"""Product boundary for durable raw-authority maintenance.

The storage implementation deliberately owns the durable receipts and replay
algorithms.  CLI and daemon surfaces use this module so that they share one
typed product operation rather than importing storage internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.core.json import JSONDocument


@dataclass(frozen=True, slots=True)
class RawMaterializationCounts:
    """Separate units produced by one bounded maintenance pass."""

    repaired_sessions: int = 0
    executed_plans: int = 0

    @property
    def made_progress(self) -> bool:
        return self.repaired_sessions > 0 or self.executed_plans > 0


def inspect_frontier(config: Config) -> Any:
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    return inspect_raw_authority_frontier(config)


def apply_frontier(config: Config, *, preview_census_id: str, selected_plan_ids: tuple[str, ...]) -> Any:
    from polylogue.storage.raw_reconciler import apply_raw_authority_frontier

    return apply_raw_authority_frontier(
        config,
        preview_census_id=preview_census_id,
        selected_plan_ids=selected_plan_ids,
    )


def recover_interrupted_frontier(config: Config) -> tuple[str, ...]:
    from polylogue.storage.raw_reconciler import recover_interrupted_raw_authority_frontier

    return recover_interrupted_raw_authority_frontier(config)


def repair_materialization(
    config: Config,
    *,
    dry_run: bool,
    raw_artifact_limit: int,
    max_payload_bytes: int,
) -> Any:
    from polylogue.storage.repair import repair_raw_materialization

    return repair_raw_materialization(
        config,
        dry_run=dry_run,
        raw_artifact_limit=raw_artifact_limit,
        max_payload_bytes=max_payload_bytes,
    )


def read_census(archive_root: Path, query_handle: str, *, limit: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_census

    return read_raw_authority_census(archive_root, query_handle, limit=limit, offset=offset)


def read_detail(archive_root: Path, query_handle: str, *, chunk_chars: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_detail

    return read_raw_authority_detail(archive_root, query_handle, chunk_chars=chunk_chars, offset=offset)


def resolve_blocker(
    archive_root: Path,
    blocker_id: str,
    *,
    resolution: str,
    assertion_id: str | None,
    judgment_disposition: str | None,
) -> JSONDocument:
    from polylogue.storage.raw_authority import resolve_raw_authority_blocker

    return resolve_raw_authority_blocker(
        archive_root,
        blocker_id,
        resolution=resolution,
        assertion_id=assertion_id,
        judgment_disposition=judgment_disposition,
    )


__all__ = [
    "RawMaterializationCounts",
    "apply_frontier",
    "inspect_frontier",
    "read_census",
    "read_detail",
    "recover_interrupted_frontier",
    "repair_materialization",
    "resolve_blocker",
]
