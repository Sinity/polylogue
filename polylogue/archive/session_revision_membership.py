"""Authority classification for sessions extracted from multi-session raws."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.timestamps import parse_timestamp
from polylogue.pipeline.ids import SessionRevisionProjection


@dataclass(frozen=True, slots=True)
class MembershipRevision:
    raw_id: str
    projection: SessionRevisionProjection
    provider_updated_at: str | None = None


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
        return MembershipClassification(
            (),
            tuple(sorted(equivalents)),
            tuple(sorted(item.raw_id for item in representatives)),
        )
    return MembershipClassification(
        tuple(item.raw_id for item in representatives),
        tuple(sorted(equivalents)),
        (),
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
