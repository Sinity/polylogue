"""Single owner for archive storage-layout product truth.

The archive file set has a small, repeated classification truth: which tiers
are present, whether the layout is missing/partial/complete, which layout
blockers apply, and which tier role a given database path plays. Before this
module that truth was hand-written twice — in the ``polylogue config paths``
CLI command and in the daemon ``/metrics`` exposition — and the two copies had
already drifted (the CLI never surfaced ``schema_mismatch:*`` blockers and
hardcoded the backup-required tier list).

This module owns the vocabulary and the classification rules. Surfaces resolve
their own tier paths (display order is a surface concern) and delegate the
semantic decisions here. The bounded label tuples are derived from
``ARCHIVE_TIER_SPECS`` so they cannot drift from the canonical tier set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

# Canonical tier order (source -> index -> embeddings -> user -> ops), taken
# straight from the spec registry so it never diverges from the runtime tiers.
ARCHIVE_TIER_ORDER: tuple[str, ...] = tuple(spec.tier.value for spec in ARCHIVE_TIER_SPECS.values())

# Tiers whose absence is itself a layout blocker because they carry
# irreplaceable / expensive-to-rebuild durability classes.
BACKUP_REQUIRED_TIERS: tuple[str, ...] = tuple(
    spec.tier.value for spec in ARCHIVE_TIER_SPECS.values() if spec.backup_required
)

# Bounded label sets for surfaces that enumerate every possible value (e.g. the
# Prometheus exposition emits a 0/1 sample per layout/role/blocker label).
ARCHIVE_STORAGE_LAYOUTS: tuple[str, ...] = (
    "archive_missing",
    "archive_partial",
    "archive_complete",
)

ARCHIVE_ACTIVE_TIER_ROLES: tuple[str, ...] = (*ARCHIVE_TIER_ORDER, "unknown")

ARCHIVE_LAYOUT_BLOCKER_LABELS: tuple[str, ...] = (
    "no_archive_tiers_present",
    "missing_archive_tiers",
    *(f"schema_mismatch:{tier}" for tier in ARCHIVE_TIER_ORDER),
    *(f"missing_backup_required_tier:{tier}" for tier in BACKUP_REQUIRED_TIERS),
)


def classify_storage_layout(*, present_count: int, final_shape_ready: bool) -> str:
    """Classify the archive file set as missing / partial / complete."""
    if final_shape_ready:
        return "archive_complete"
    if present_count:
        return "archive_partial"
    return "archive_missing"


def archive_layout_blockers(
    *,
    present_count: int,
    final_shape_ready: bool,
    schema_mismatches: Iterable[str] = (),
    missing_backup_required: Iterable[str] = (),
) -> list[str]:
    """Return active layout blockers in deterministic, label-ordered form.

    Order matches :data:`ARCHIVE_LAYOUT_BLOCKER_LABELS`: the two presence
    blockers first, then ``schema_mismatch:*`` in canonical tier order, then
    ``missing_backup_required_tier:*`` in backup-tier order. Callers that do not
    inspect schema versions simply pass ``schema_mismatches=()``.
    """
    mismatched = set(schema_mismatches)
    missing_backup = set(missing_backup_required)
    blockers: list[str] = []
    if present_count == 0:
        blockers.append("no_archive_tiers_present")
    if not final_shape_ready:
        blockers.append("missing_archive_tiers")
    blockers.extend(f"schema_mismatch:{tier}" for tier in ARCHIVE_TIER_ORDER if tier in mismatched)
    blockers.extend(f"missing_backup_required_tier:{tier}" for tier in BACKUP_REQUIRED_TIERS if tier in missing_backup)
    return blockers


def active_tier_role(active_path: Path, tier_paths: Mapping[str, Path] | Sequence[tuple[str, Path]]) -> str:
    """Return the tier role of ``active_path`` among the known tier paths.

    Accepts either a ``{tier: path}`` mapping or a sequence of ``(tier, path)``
    pairs. Paths are resolved (non-strict) before comparison so callers may pass
    either resolved or unresolved paths.
    """
    pairs = tier_paths.items() if isinstance(tier_paths, Mapping) else tier_paths
    resolved_active = active_path.resolve(strict=False)
    for tier, path in pairs:
        if resolved_active == path.resolve(strict=False):
            return tier
    return "unknown"
