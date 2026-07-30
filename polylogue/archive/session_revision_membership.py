"""Authority classification for sessions extracted from multi-session raws."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.core.timestamps import parse_timestamp
from polylogue.pipeline.ids import AttachmentRecord, SessionRevisionProjection

#: Everything that must agree for two revisions to be the same content: which
#: messages exist and what they say (order-insensitive -- polylogue-c429),
#: the event chain with provider-reported measurement excluded
#: (polylogue-nuec), which attachments exist, and which of their bytes have
#: been read. The attachment component is the strict (id-bearing) identity --
#: two groups differing only in attachment id *presence* do not share a key
#: here and need the separate correlation-based merge step in
#: ``classify_membership_revisions`` (polylogue-d8al); this key alone would
#: under-merge for that case, never over-merge, so it stays a safe first pass.
_ContentKey: TypeAlias = tuple[
    frozenset[tuple[bytes, bytes]], tuple[bytes, ...], frozenset[bytes], frozenset[tuple[bytes, bytes]]
]


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
    # Keyed on acquisition state as well as identity: two revisions are the same
    # content only when they also agree on which attachment bytes are in hand,
    # otherwise collapsing them as equivalents could discard the one that
    # actually carries the bytes.
    by_content: dict[_ContentKey, list[MembershipRevision]] = {}
    for revision in revisions:
        projection = revision.projection
        key = (
            projection.message_contents,
            projection.event_identity_hashes,
            projection.attachment_identities,
            projection.attachment_contents,
        )
        by_content.setdefault(key, []).append(revision)
    representatives: list[MembershipRevision] = []
    equivalents: list[str] = []
    for group in _merge_attachment_id_presence_variants(by_content):
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
            direct_export = _direct_export_precedence(representatives)
            if direct_export is None:
                return MembershipClassification(
                    (),
                    tuple(sorted(equivalents)),
                    tuple(sorted(item.raw_id for item in representatives)),
                )
            accepted, browser_capture_raw_ids = direct_export
            return MembershipClassification(
                (accepted.raw_id,),
                tuple(sorted((*equivalents, *browser_capture_raw_ids))),
                (),
            )
        representatives = browser_order
    return MembershipClassification(
        tuple(item.raw_id for item in representatives),
        tuple(sorted(equivalents)),
        (),
    )


def _merge_attachment_id_presence_variants(
    by_content: dict[_ContentKey, list[MembershipRevision]],
) -> list[list[MembershipRevision]]:
    """Merge content groups that differ only in attachment id presence.

    The strict content key built in ``classify_membership_revisions`` never
    merges a real-id/synthetic-id pair for the same physical attachment into
    one group: a provider's export can omit a stable id for the same
    attachment on a different export request of the same conversation, so an
    otherwise byte-identical pair's attachment sets hash to disjoint strict
    identities and the two groups never meet in ``by_content`` (polylogue-d8al).
    This pass merges any two groups whose message/event portion of the key
    already matches exactly and whose attachments correlate as fully
    equivalent (``_attachments_equivalent``): same cardinality, every
    attachment pairwise-matched by strict id or, when unambiguous on both
    sides, by the id-independent key, and no content contradiction. Only ever
    merges groups the strict key under-merged -- it can never combine two
    groups that genuinely differ in message or event content, so this cannot
    introduce a false equivalence on those axes.
    """
    entries = list(by_content.items())
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(entries)):
        key_i, revisions_i = entries[i]
        for j in range(i + 1, len(entries)):
            key_j, revisions_j = entries[j]
            if key_i[0] != key_j[0] or key_i[1] != key_j[1]:
                continue
            if _attachments_equivalent(revisions_i[0].projection, revisions_j[0].projection):
                union(i, j)

    merged: dict[int, list[MembershipRevision]] = {}
    for i, (_key, revisions) in enumerate(entries):
        merged.setdefault(find(i), []).extend(revisions)
    return list(merged.values())


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


def _direct_export_precedence(
    revisions: list[MembershipRevision],
) -> tuple[MembershipRevision, tuple[str, ...]] | None:
    """A genuine non-browser-capture revision always outranks browser-capture siblings.

    Browser capture exists to backfill a session before its direct/native
    provider export shows up, never to compete with or shadow that export
    once it arrives -- a materially different transcript (e.g. a browser
    snapshot's own message ids) is expected and does not make the group
    ambiguous. When exactly one candidate in an otherwise-unresolved
    membership group carries no browser-capture provenance at all
    (``browser_snapshot_fidelity is None``) and at least one sibling is
    browser-capture-sourced, the non-browser candidate is authoritative for
    session content regardless of byte-growth or provider-timestamp
    comparisons; the browser-capture siblings are retained as raw
    provenance only, not indexed content (polylogue-z1c6). Returns ``None``
    when this specific shape does not apply (zero or multiple non-browser
    candidates), leaving the caller's existing ambiguous-quarantine
    behavior untouched.
    """
    direct = [item for item in revisions if item.browser_snapshot_fidelity is None]
    browser_sourced = [item for item in revisions if item.browser_snapshot_fidelity is not None]
    if len(direct) != 1 or not browser_sourced:
        return None
    return direct[0], tuple(item.raw_id for item in browser_sourced)


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


def _frontier(projection: SessionRevisionProjection) -> tuple[int, int, int, int]:
    """Rank a revision along every axis on which it can only grow.

    Attachment acquisition is its own axis. Without it, a revision that added
    nothing but the bytes of attachments it already referenced tied with its
    predecessor on every dimension, the sort fell through to ``raw_id``, and the
    dominance test was then run in whichever direction the hex happened to
    order -- half the time backwards, against a revision that genuinely does
    dominate (polylogue-bu1i).
    """
    return (
        len(projection.message_contents),
        len(projection.event_identity_hashes),
        len(projection.attachment_identities),
        len(projection.attachment_contents),
    )


def _message_evidence_preserved(
    older: SessionRevisionProjection,
    newer: SessionRevisionProjection,
) -> bool:
    """True when ``newer`` loses no message identity and contradicts no shared content.

    A provider's export ordering is not guaranteed stable across separate
    export requests for the SAME conversation -- Claude.ai's own tree
    flattening can interleave edited-message siblings differently from one
    export to the next even though every message's id, role, text, and
    timestamp are byte-identical (polylogue-c429). Array position is
    therefore not treated as identity here: this checks only that every
    ``(identity, content)`` pair present in ``older`` is still present in
    ``newer``. Unlike attachments, a message is never lazily fetched -- its
    content is always known when it exists -- so there is no separate
    identity-only membership to check: an id that disappeared from ``newer``
    fails this lookup on its own (``.get()`` returns ``None``, which never
    equals a real content hash), and an id whose content actually changed
    fails it too. Both are real divergence and are refused.
    """
    newer_contents = dict(newer.message_contents)
    return all(newer_contents.get(identity) == content for identity, content in older.message_contents)


def _correlate_attachments(
    older: SessionRevisionProjection, newer: SessionRevisionProjection
) -> tuple[list[tuple[AttachmentRecord, AttachmentRecord]], list[AttachmentRecord], list[AttachmentRecord]]:
    """Pair each attachment in ``older`` with its counterpart in ``newer``.

    Matches by strict identity first (provider id, anchoring message, name,
    media type) -- the exact bu1i behavior when both revisions agree on a
    provider id. When a provider omits a stable id for the same attachment on
    a different export request, the two revisions never share a strict
    identity for it (polylogue-d8al): Claude.ai does not consistently emit an
    id field for the same attachment across separate export requests of the
    same conversation. In that case matching falls back to the
    id-independent (anchoring message, name, media type) key, but ONLY when
    that looser key is unambiguous on BOTH sides being compared (exactly one
    attachment carries it in ``older`` and exactly one in ``newer``): if
    either revision has two attachments sharing the same anchor/name/media
    type, there is no way to tell which is which without an id, and guessing
    by array position is exactly the bug this replaces -- that ambiguous case
    is left uncorrelated rather than resolved by a guess.

    Returns matched ``(older, newer)`` record pairs, ``older`` records with no
    counterpart in ``newer`` (a potential loss), and ``newer`` records with no
    counterpart in ``older`` (growth).
    """
    newer_by_identity = {record[0]: record for record in newer.attachment_records}
    newer_loose_counts = Counter(record[1] for record in newer.attachment_records)
    newer_by_loose = {record[1]: record for record in newer.attachment_records}
    older_loose_counts = Counter(record[1] for record in older.attachment_records)

    matched: list[tuple[AttachmentRecord, AttachmentRecord]] = []
    unmatched_older: list[AttachmentRecord] = []
    matched_newer_identities: set[bytes] = set()
    for older_record in older.attachment_records:
        identity, loose_identity, _content = older_record
        newer_record = newer_by_identity.get(identity)
        if (
            newer_record is None
            and older_loose_counts[loose_identity] == 1
            and newer_loose_counts.get(loose_identity) == 1
        ):
            newer_record = newer_by_loose.get(loose_identity)
        if newer_record is None:
            unmatched_older.append(older_record)
            continue
        matched.append((older_record, newer_record))
        matched_newer_identities.add(newer_record[0])
    unmatched_newer = [record for record in newer.attachment_records if record[0] not in matched_newer_identities]
    return matched, unmatched_older, unmatched_newer


def _attachments_equivalent(a: SessionRevisionProjection, b: SessionRevisionProjection) -> bool:
    """True when every attachment in ``a`` and ``b`` correlates 1:1 with identical content.

    Used only to decide whether two otherwise-identical content groups may
    merge as the same content (``_merge_attachment_id_presence_variants``);
    unlike ``_attachment_evidence_preserved`` this is symmetric and requires
    an exact match on both sides, not merely no loss in one direction.
    """
    matched, unmatched_a, unmatched_b = _correlate_attachments(a, b)
    if unmatched_a or unmatched_b:
        return False
    return all(a_record[2] == b_record[2] for a_record, b_record in matched)


def _attachment_evidence_preserved(
    older: SessionRevisionProjection,
    newer: SessionRevisionProjection,
) -> bool:
    """True when ``newer`` loses no attachment and contradicts no fetched bytes.

    Two distinct regressions are refused here. An attachment whose bytes were
    read and are no longer present in the newer revision is a fidelity
    *downgrade*: the newer revision knows strictly less, so accepting it would
    silently mark an acquired attachment unfetched. An attachment present in
    both but carrying *different* bytes is a genuine conflict -- two sources
    disagree about content under one identity -- and no ordering rule can
    resolve that, so the cohort stays ambiguous. Correlation
    (``_correlate_attachments``) is what lets this hold across an id-presence
    mismatch the same way it always did for a plain matching id
    (polylogue-d8al).
    """
    matched, unmatched_older, _unmatched_newer = _correlate_attachments(older, newer)
    if unmatched_older:
        return False
    return all(older_record[2] is None or older_record[2] == newer_record[2] for older_record, newer_record in matched)


def _attachment_axis_grew(older: SessionRevisionProjection, newer: SessionRevisionProjection) -> bool:
    """True when ``newer`` has a genuinely new attachment or newly-read bytes.

    An uncorrelated attachment in ``newer`` (no older counterpart, by strict
    id or unambiguous loose key) is growth. So is resolving the bytes of an
    already-referenced attachment -- growth in evidence even when the
    transcript is untouched, the shape a lazily-fetched attachment produces
    on its second acquisition (polylogue-bu1i).
    """
    matched, _unmatched_older, unmatched_newer = _correlate_attachments(older, newer)
    if unmatched_newer:
        return True
    return any(older_record[2] is None and newer_record[2] is not None for older_record, newer_record in matched)


def _strictly_dominates(older: SessionRevisionProjection, newer: SessionRevisionProjection) -> bool:
    content_grew = (
        # Order-insensitive: a permuted-but-otherwise-equal message set is not
        # growth (equal count), but a genuinely appended or resurfaced message
        # id is (polylogue-c429).
        len(newer.message_contents) > len(older.message_contents)
        or len(newer.event_identity_hashes) > len(older.event_identity_hashes)
        or _attachment_axis_grew(older, newer)
    )
    return (
        content_grew
        and _message_evidence_preserved(older, newer)
        and older.event_identity_hashes == newer.event_identity_hashes[: len(older.event_identity_hashes)]
        and _attachment_evidence_preserved(older, newer)
    )


__all__ = ["MembershipClassification", "MembershipRevision", "classify_membership_revisions"]
