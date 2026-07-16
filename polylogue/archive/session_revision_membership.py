"""Authority classification for sessions extracted from multi-session raws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.timestamps import parse_timestamp
from polylogue.pipeline.ids import SessionRevisionProjection


@dataclass(frozen=True, slots=True)
class MembershipRevision:
    raw_id: str
    projection: SessionRevisionProjection
    provider_updated_at: str | None = None
    observed_at_ms: int | None = None
    browser_snapshot_fidelity: Literal["dom", "native"] | None = None
    provider_message_ids: frozenset[str] = frozenset()
    provider_attachment_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MembershipClassification:
    accepted_raw_ids: tuple[str, ...]
    equivalent_raw_ids: tuple[str, ...]
    ambiguous_raw_ids: tuple[str, ...]


def classify_membership_revisions(revisions: list[MembershipRevision]) -> MembershipClassification:
    """Accept one total strict-growth chain; never choose between branches."""
    if not revisions:
        return MembershipClassification((), (), ())
    by_content: dict[tuple[tuple[bytes, ...], tuple[bytes, ...], frozenset[bytes]], list[MembershipRevision]] = {}
    for revision in revisions:
        projection = revision.projection
        key = (projection.message_hashes, projection.event_hashes, projection.attachment_hashes)
        by_content.setdefault(key, []).append(revision)
    representatives: list[MembershipRevision] = []
    equivalents: list[str] = []
    for group in by_content.values():
        by_session_hash: dict[bytes, list[MembershipRevision]] = {}
        for item in group:
            by_session_hash.setdefault(item.projection.session_hash, []).append(item)
        metadata_variants: list[MembershipRevision] = []
        for hash_group in by_session_hash.values():
            representative = min(hash_group, key=lambda item: item.raw_id)
            metadata_variants.append(representative)
            equivalents.extend(item.raw_id for item in hash_group if item.raw_id != representative.raw_id)
        if len(metadata_variants) == 1:
            representatives.extend(metadata_variants)
            continue
        timestamped = [
            (parsed.timestamp(), item)
            for item in metadata_variants
            if (parsed := parse_timestamp(item.provider_updated_at)) is not None
        ]
        timestamps = [timestamp for timestamp, _item in timestamped]
        if len(timestamped) == len(metadata_variants) and len(set(timestamps)) == len(timestamps):
            representative = max(timestamped, key=lambda pair: pair[0])[1]
            representatives.append(representative)
            equivalents.extend(item.raw_id for item in metadata_variants if item.raw_id != representative.raw_id)
        else:
            representatives.extend(metadata_variants)
    representatives.sort(key=lambda item: (_frontier(item.projection), item.raw_id))
    if any(
        not _strictly_dominates(older.projection, newer.projection)
        for older, newer in zip(representatives, representatives[1:], strict=False)
    ):
        browser_order = _provider_ordered_browser_snapshots(representatives)
        if browser_order is None:
            return MembershipClassification(
                (),
                tuple(sorted(equivalents)),
                tuple(sorted(item.raw_id for item in representatives)),
            )
        representatives = browser_order
    return MembershipClassification(
        tuple(item.raw_id for item in representatives),
        tuple(sorted(equivalents)),
        (),
    )


def _provider_ordered_browser_snapshots(
    revisions: list[MembershipRevision],
) -> list[MembershipRevision] | None:
    """Order compatible mutable browser snapshots by provider authority.

    Browser-native payloads are complete snapshots, not append logs.  ChatGPT
    can move an already-present context/tool node when later work appears, and
    can complete text in place under the same provider message id.  A strict
    serialized-content prefix therefore rejects ordinary provider progress.
    Provider timestamps may resolve that progress only when stable message and
    attachment identities are preserved.  DOM-to-native is the sole fidelity
    upgrade and may use different synthetic ids; a native-to-DOM downgrade is
    never selected.
    """

    if not revisions or any(item.browser_snapshot_fidelity is None for item in revisions):
        return None
    timestamped: list[tuple[int, float, int, str, MembershipRevision]] = []
    for item in revisions:
        parsed = parse_timestamp(item.provider_updated_at)
        if parsed is None:
            return None
        fidelity_rank = 1 if item.browser_snapshot_fidelity == "native" else 0
        timestamped.append((fidelity_rank, parsed.timestamp(), item.observed_at_ms or -1, item.raw_id, item))
    timestamped.sort(key=lambda entry: entry[:4])
    ordered = [entry[4] for entry in timestamped]
    for older, newer in zip(ordered, ordered[1:], strict=False):
        if not _browser_snapshot_dominates(older, newer):
            return None
    return ordered


def _browser_snapshot_dominates(older: MembershipRevision, newer: MembershipRevision) -> bool:
    older_time = parse_timestamp(older.provider_updated_at)
    newer_time = parse_timestamp(newer.provider_updated_at)
    if older_time is None or newer_time is None:
        return False
    if older.browser_snapshot_fidelity == "dom" and newer.browser_snapshot_fidelity == "native":
        older_frontier = _frontier(older.projection)
        newer_frontier = _frontier(newer.projection)
        return all(
            newer_count >= older_count for older_count, newer_count in zip(older_frontier, newer_frontier, strict=True)
        )
    if older.browser_snapshot_fidelity != newer.browser_snapshot_fidelity:
        return False
    identities_preserved = (
        bool(older.provider_message_ids)
        and older.provider_message_ids <= newer.provider_message_ids
        and older.provider_attachment_ids <= newer.provider_attachment_ids
    )
    if not identities_preserved:
        return False
    if newer_time.timestamp() > older_time.timestamp():
        return True
    return (
        newer_time.timestamp() == older_time.timestamp()
        and older.observed_at_ms is not None
        and newer.observed_at_ms is not None
        and newer.observed_at_ms > older.observed_at_ms
        and older.projection.message_hashes == newer.projection.message_hashes
        and older.projection.event_hashes == newer.projection.event_hashes
    )


def _frontier(projection: SessionRevisionProjection) -> tuple[int, int, int]:
    return len(projection.message_hashes), len(projection.event_hashes), len(projection.attachment_hashes)


def _strictly_dominates(older: SessionRevisionProjection, newer: SessionRevisionProjection) -> bool:
    content_grew = (
        len(newer.message_hashes) > len(older.message_hashes)
        or len(newer.event_hashes) > len(older.event_hashes)
        or newer.attachment_hashes > older.attachment_hashes
    )
    return (
        content_grew
        and older.message_hashes == newer.message_hashes[: len(older.message_hashes)]
        and older.event_hashes == newer.event_hashes[: len(older.event_hashes)]
        and older.attachment_hashes <= newer.attachment_hashes
    )


__all__ = ["MembershipClassification", "MembershipRevision", "classify_membership_revisions"]
