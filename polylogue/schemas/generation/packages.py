"""Package-candidate assembly helpers for schema generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from polylogue.schemas.generation.models import (
    PackageAssemblyResult,
    _ClusterAccumulator,
    _PackageAccumulator,
    _UnitMembership,
)

_ANCHOR_ELEMENT_KINDS = {
    "conversation_document",
    "conversation_record_stream",
}
_PROFILE_MAX_TOKENS = 128


def _parse_observed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _merge_representative_paths(target: list[str], source: list[str]) -> None:
    for path in source:
        if path not in target and len(target) < 5:
            target.append(path)


def _update_observed_window(acc: _ClusterAccumulator | _PackageAccumulator, observed_at: str | None) -> None:
    parsed = _parse_observed_at(observed_at)
    if parsed is None:
        return
    iso = parsed.astimezone(timezone.utc).isoformat()
    first_seen = _parse_observed_at(acc.first_seen)
    if acc.first_seen is None or first_seen is None or parsed < first_seen:
        acc.first_seen = iso
    last_seen = _parse_observed_at(acc.last_seen)
    if acc.last_seen is None or last_seen is None or parsed > last_seen:
        acc.last_seen = iso


def _membership_observed_window(memberships: list[_UnitMembership]) -> tuple[str | None, str | None]:
    first_seen: str | None = None
    last_seen: str | None = None
    for membership in memberships:
        parsed = _parse_observed_at(membership.unit.observed_at)
        if parsed is None:
            continue
        iso = parsed.astimezone(timezone.utc).isoformat()
        parsed_first_seen = _parse_observed_at(first_seen)
        if first_seen is None or parsed_first_seen is None or parsed < parsed_first_seen:
            first_seen = iso
        parsed_last_seen = _parse_observed_at(last_seen)
        if last_seen is None or parsed_last_seen is None or parsed > parsed_last_seen:
            last_seen = iso
    return first_seen, last_seen


def _membership_scope_key(membership: _UnitMembership) -> str:
    unit = membership.unit
    return (
        unit.bundle_scope
        or unit.raw_id
        or unit.source_path
        or f"{membership.profile_family_id}:{unit.artifact_kind}:{unit.exact_structure_id}"
    )


def _dedupe_bundle_memberships(memberships: list[_UnitMembership]) -> dict[str, list[_UnitMembership]]:
    scoped: dict[str, list[_UnitMembership]] = {}
    for membership in memberships:
        scoped.setdefault(_membership_scope_key(membership), []).append(membership)

    deduped: dict[str, list[_UnitMembership]] = {}
    for scope, items in scoped.items():
        items = sorted(
            items,
            key=lambda item: (
                item.unit.observed_at or "",
                item.unit.source_path or "",
                item.profile_family_id,
            ),
        )
        seen: set[tuple[str, str]] = set()
        retained: list[_UnitMembership] = []
        for membership in items:
            dedupe_key = (membership.unit.artifact_kind, membership.unit.exact_structure_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            retained.append(membership)
        deduped[scope] = retained
    return deduped


def _attach_package_membership(package: _PackageAccumulator, membership: _UnitMembership, *, scope: str) -> None:
    package.memberships.append(membership)
    package.bundle_scopes.add(scope)
    package.profile_family_ids.add(membership.profile_family_id)
    if membership.unit.source_path:
        _merge_representative_paths(package.representative_paths, [membership.unit.source_path])
    _update_observed_window(package, membership.unit.observed_at)


def _build_package_candidates(
    provider: str,
    *,
    memberships: list[_UnitMembership],
    clusters: dict[str, _ClusterAccumulator],
) -> tuple[list[_PackageAccumulator], dict[str, int]]:
    result = assemble_package_candidates(
        provider,
        memberships=memberships,
        clusters=clusters,
    )
    return result.packages, result.orphan_adjunct_counts


def assemble_package_candidates(
    provider: str,
    *,
    memberships: list[_UnitMembership],
    clusters: dict[str, _ClusterAccumulator],
) -> PackageAssemblyResult:
    scoped = _dedupe_bundle_memberships(memberships)
    packages: dict[str, _PackageAccumulator] = {}
    orphan_adjunct_counts: Counter[str] = Counter()

    for scope, items in scoped.items():
        anchor_families = sorted(
            {
                membership.profile_family_id
                for membership in items
                if membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS
            }
        )

        if not anchor_families:
            for membership in items:
                orphan_adjunct_counts[membership.unit.artifact_kind] += 1
            continue

        for family_id in anchor_families:
            acc = packages.get(family_id)
            if acc is None:
                cluster = clusters[family_id]
                acc = _PackageAccumulator(
                    provider=provider,
                    anchor_family_id=family_id,
                    anchor_kind=cluster.artifact_kind,
                )
                packages[family_id] = acc
            for membership in items:
                if membership.profile_family_id == family_id and membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(acc, membership, scope=scope)

        if len(anchor_families) == 1:
            target = packages[anchor_families[0]]
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(target, membership, scope=scope)
        else:
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    orphan_adjunct_counts[membership.unit.artifact_kind] += 1

    ordered = sorted(
        packages.values(),
        key=lambda item: (
            _parse_observed_at(item.first_seen) or datetime.max.replace(tzinfo=timezone.utc),
            -len(item.bundle_scopes),
            item.anchor_family_id,
        ),
    )
    return PackageAssemblyResult(
        packages=ordered,
        orphan_adjunct_counts=dict(orphan_adjunct_counts),
    )


def _element_profile_tokens(memberships: list[_UnitMembership]) -> list[str]:
    token_counts: Counter[str] = Counter()
    for membership in memberships:
        token_counts.update(membership.unit.profile_tokens)
    if not token_counts:
        return []
    min_count = max(1, len(memberships) // 2)
    tokens = sorted(token for token, count in token_counts.items() if count >= min_count)
    if not tokens:
        tokens = [
            token
            for token, _count in sorted(
                token_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_PROFILE_MAX_TOKENS]
        ]
    return tokens[:_PROFILE_MAX_TOKENS]


__all__ = [
    "assemble_package_candidates",
    "_build_package_candidates",
    "_element_profile_tokens",
    "_membership_observed_window",
    "_merge_representative_paths",
    "_parse_observed_at",
]
