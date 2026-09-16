"""Product seam for hook-event materialization through the derivation kernel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.derivation import (
    Budget,
    DerivationFrame,
    DerivationRegistry,
    DerivationReport,
    converge,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.derived.hook_events import (
    HOOK_EVENTS_DOMAIN as _HOOK_EVENTS_DOMAIN,
)
from polylogue.storage.derived.hook_events import (
    HookEventsDerivation,
    HookEventsScope,
)

HOOK_EVENTS_DOMAIN = _HOOK_EVENTS_DOMAIN
_HOOK_EVENTS_RECIPE_VERSION = HookEventsDerivation.recipe_version


def make_hook_events_derivation(archive_root: Path) -> HookEventsDerivation:
    """Construct the storage-owned hook adapter from the operations boundary."""

    return HookEventsDerivation(archive_root)


def hook_events_frame(archive_root: Path, *, raw_ids: Sequence[str] = ()) -> DerivationFrame:
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(ArchiveLocation.resolve(archive_root).active_index_path.resolve()),
        recipe_versions={HOOK_EVENTS_DOMAIN: _HOOK_EVENTS_RECIPE_VERSION},
        scope=HookEventsScope(raw_ids=tuple(raw_ids)),
    )


def discover_pending_hook_carriers(archive_root: Path, limit: int) -> tuple[tuple[str, int], ...]:
    """Page the canonical carrier key space for carriers still unmaterialized.

    Costed in carrier bytes, like every other intake class, so a single large
    compacted carrier cannot claim an unbounded share of the dispatcher's
    budget by looking free.
    """

    if limit < 1:
        return ()
    adapter = make_hook_events_derivation(archive_root)
    frame = hook_events_frame(archive_root)
    try:
        keys, _next_cursor = adapter.required_page(frame, cursor=None, limit=limit)
    except (FileNotFoundError, OSError):
        return ()
    if not keys:
        return ()
    stale = tuple(key for key, status in adapter.inspect(frame, keys).items() if status != "valid")
    if not stale:
        return ()
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(archive_root) as pinned:
        sizes = pinned.archive.raw_payload_sizes(stale)
    return tuple((key, max(1, int(sizes.get(key, 1)))) for key in stale)


def converge_hook_carriers(
    archive_root: Path,
    *,
    raw_ids: Sequence[str] = (),
    limit: int = 32,
) -> DerivationReport:
    """Materialize the named carriers, or one bounded page of pending ones."""

    adapter = make_hook_events_derivation(archive_root)
    selected = tuple(raw_ids)
    return converge(
        DerivationRegistry((adapter,)),
        hook_events_frame(archive_root, raw_ids=selected),
        # Each discovered key needs inspection before compute and again to
        # certify publication. Discovery alone must not exhaust that budget.
        budget=Budget(
            page=min(128, limit),
            discovery=max(limit, len(selected)),
            inspection=2 * max(limit, len(selected)),
            compute=max(limit, len(selected)),
            publication=max(limit, len(selected)),
        ),
    )


__all__ = [
    "HOOK_EVENTS_DOMAIN",
    "converge_hook_carriers",
    "discover_pending_hook_carriers",
    "hook_events_frame",
    "make_hook_events_derivation",
]
