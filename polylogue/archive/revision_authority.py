"""Durable source-revision evidence and conservative legacy classification."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import BinaryIO, Literal

from polylogue.core.enums import Origin, PolylogueStrEnum, Provider
from polylogue.core.sources import origin_from_provider


class RawRevisionKind(StrEnum):
    FULL = "full"
    APPEND = "append"
    UNKNOWN = "unknown"


class RawRevisionAuthority(PolylogueStrEnum):
    """Closed vocabulary for ``revision_authority``/``previous_revision_authority``.

    ``PolylogueStrEnum`` (not a plain ``StrEnum``) so this generates its SQL
    CHECK via ``archive_tiers/common.py``'s ``check()``/``nullable_check()``
    at every column sharing the full 3-value domain, instead of each site
    hand-copying the literal list (polylogue-h57ic). ``raw_session_
    memberships.revision_authority`` is a genuinely narrower 2-value domain
    (``ASSERTED`` never applies there) and uses the separate
    ``ProvenRevisionAuthority`` literal alias in
    ``storage/sqlite/archive_tiers/types.py`` instead of this enum.
    """

    ASSERTED = "asserted"
    BYTE_PROVEN = "byte_proven"
    QUARANTINED = "quarantined"


BYTE_AUTHORITY_CENSUS_DETAIL = "append fragments are governed by byte revision authority"

#: Current parser semantics for source-tier membership and parser receipts.
#: Both admission writers and replay readers depend on revision authority, so
#: this token belongs with that shared contract rather than either storage
#: projection that consumes it.
RAW_AUTHORITY_PARSER_FINGERPRINT = "revision-membership-v4"


def canonical_authority_logical_key(logical_key: str) -> str:
    """Normalize a provider or public-origin authority key to public origin form."""
    prefix, separator, native_id = logical_key.partition(":")
    if not separator or not native_id:
        raise ValueError(f"invalid logical source key: {logical_key!r}")
    try:
        origin = Origin(prefix)
    except ValueError:
        try:
            origin = origin_from_provider(Provider(prefix))
        except ValueError as exc:
            raise ValueError(f"unknown logical source key prefix: {prefix!r}") from exc
    return f"{origin.value}:{native_id}"


def logical_head_cohort_sql(conn: sqlite3.Connection, *, raw_alias: str, has_memberships: bool) -> str:
    """Return the canonical, membership-aware SQL partition key for raw heads."""

    def _canonical_or_original(value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        try:
            return canonical_authority_logical_key(text)
        except ValueError:
            # A malformed legacy key must remain observable as its own cohort,
            # never make a read-only verifier fail while grouping heads.
            return text

    conn.create_function("canonical_authority_logical_key", 1, _canonical_or_original, deterministic=True)
    membership_key = "NULL"
    if has_memberships:
        membership_key = f"""
            canonical_authority_logical_key(
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT canonical_authority_logical_key(m.logical_source_key)) = 1
                        THEN MIN(canonical_authority_logical_key(m.logical_source_key))
                    END
                    FROM raw_session_memberships AS m
                    WHERE m.raw_id = {raw_alias}.raw_id
                )
            )
        """
    return (
        f"COALESCE(canonical_authority_logical_key({raw_alias}.logical_source_key), "
        f"{membership_key}, {raw_alias}.native_id, {raw_alias}.source_path)"
    )


def durable_authority_logical_keys(
    *,
    raw_logical_key: object,
    revision_kind: object,
    membership_logical_keys: Iterable[object],
) -> tuple[str, ...] | None:
    """Return the canonical durable identity set a parser receipt must prove.

    A ``pending-raw:`` envelope preserves bytes before parsing; it is not a
    parser identity and must never leak into the durable parser receipt.
    """
    values = [str(value) for value in membership_logical_keys if value is not None]
    if raw_logical_key is not None and str(revision_kind) != RawRevisionKind.UNKNOWN.value:
        typed_key = str(raw_logical_key)
        if not typed_key.startswith("pending-raw:"):
            values.append(typed_key)
    try:
        return tuple(sorted({canonical_authority_logical_key(value) for value in values}))
    except ValueError:
        return None


#: ``raw_membership_census.detail`` marker written when a full-only,
#: non-prefix-chain cohort is retired from byte-revision governance to
#: membership governance (``ArchiveStore.replace_raw_membership_census``,
#: ``retire_full_revision_governance=True``). Shared between the writer and
#: ``revision_governance._classify_raw_revision_cohort``'s polylogue-52l2 guard,
#: which uses it to detect that a logical identity already has retired,
#: previously-ambiguous sibling evidence before ever accepting a
#: later-discovered raw as an unconditional singleton byte-proven baseline.
HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL = "historical non-prefix full revision governance"

#: All ``raw_membership_census.detail`` marker literals that mean "this raw
#: was retired from full-revision byte governance to membership governance"
#: (polylogue-hm2f, residual of polylogue-52l2). Durable ``raw_membership_
#: census`` rows written before the #3234 fix used the literal
#: ``"cross-route full revision governance"`` at the live-watcher call site
#: (``sources/live/batch.py``, pre-fix) -- a DIFFERENT string from
#: ``HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL``, which only the offline
#: backfill call site used at the time. Those pre-fix rows are durable
#: (``source.db``) and were never rewritten in place (durable-tier changes
#: need an explicit additive migration, not a silent detail-string rewrite),
#: so identities retired under the legacy literal remain invisible to a guard
#: query keyed on the new marker alone. ``ArchiveStore.raw_membership_
#: retired_full_revision_siblings`` matches against every literal in this
#: tuple so old and new retirements are both recognized. The legacy literal
#: is frozen here for read-compatibility only -- it must never be written by
#: new code (both current call sites use ``HISTORICAL_NON_PREFIX_GOVERNANCE_
#: DETAIL`` exclusively).
RETIRED_FULL_REVISION_GOVERNANCE_DETAILS: tuple[str, ...] = (
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    "cross-route full revision governance",  # legacy pre-#3234 literal -- read-only, never write.
)


@dataclass(frozen=True)
class RawRevisionEnvelope:
    """Evidence captured with raw bytes, before any derived write occurs."""

    logical_source_key: str
    kind: RawRevisionKind
    source_revision: str
    acquisition_generation: int
    predecessor_source_revision: str | None = None
    predecessor_raw_id: str | None = None
    baseline_raw_id: str | None = None
    append_start_offset: int | None = None
    append_end_offset: int | None = None
    authority: RawRevisionAuthority = RawRevisionAuthority.ASSERTED

    def __post_init__(self) -> None:
        if not self.logical_source_key or not self.source_revision:
            raise ValueError("revision envelope identity must be non-empty")
        if self.acquisition_generation < 0:
            raise ValueError("acquisition_generation must be non-negative")
        offsets = (self.append_start_offset, self.append_end_offset)
        if self.kind is RawRevisionKind.APPEND:
            if self.predecessor_source_revision is None or None in offsets:
                raise ValueError("append evidence requires predecessor revision and offsets")
            assert self.append_start_offset is not None and self.append_end_offset is not None
            if self.append_start_offset < 0 or self.append_end_offset <= self.append_start_offset:
                raise ValueError("append offsets must describe a non-empty forward range")
            raw_predecessors = (self.predecessor_raw_id, self.baseline_raw_id)
            if self.authority is RawRevisionAuthority.QUARANTINED:
                if any(value is not None for value in raw_predecessors):
                    raise ValueError("quarantined append evidence may not claim raw predecessors")
            elif any(value is None for value in raw_predecessors):
                raise ValueError("replay-eligible append evidence requires baseline and raw predecessor")
        elif self.predecessor_source_revision is not None or any(value is not None for value in offsets):
            raise ValueError("only append evidence may carry predecessor revision or byte offsets")


@dataclass(frozen=True)
class HistoricalRawRevision:
    raw_id: str
    payload: bytes


@dataclass(frozen=True)
class HistoricalRawRevisionStream:
    """A retained full revision whose bytes can be compared without loading it."""

    raw_id: str
    payload_size: int
    open_payload: Callable[[], BinaryIO]


@dataclass(frozen=True)
class HistoricalRevisionDecision:
    raw_id: str
    authority: RawRevisionAuthority
    relation: Literal["baseline", "predecessor", "ambiguous", "duplicate"]
    predecessor_raw_id: str | None = None
    #: For ``relation="duplicate"`` only: the representative raw_id whose
    #: verdict (and, for callers that derive chain position such as
    #: ``revision_governance.classify_raw_revision_cohort``, generation
    #: number) this duplicate mirrors. Deliberately separate from
    #: ``predecessor_raw_id`` -- a duplicate is NOT a chain-continuing child
    #: of anything; it is a second copy of its representative's own bytes.
    #: Mirroring the representative's ``predecessor_raw_id`` onto the
    #: duplicate instead (the naive fix CodeRabbit flagged on #3574) would
    #: make the duplicate and its representative collide on the same
    #: predecessor-keyed dict entry in a plain-dict generation walk, risking
    #: silently dropping the real chain-continuing representative. This
    #: field lets a caller copy the representative's already-computed
    #: generation in a separate post-pass instead of participating in the
    #: predecessor-keyed walk at all (polylogue-5unky).
    duplicate_of_raw_id: str | None = None


def append_source_revision(predecessor_revision: str, payload_hash: str) -> str:
    """Return the exact content fingerprint committed by the append cursor."""
    return sha256(f"{predecessor_revision}\0{payload_hash}".encode()).hexdigest()


def _classify_deduped_nodes(
    node_ids: list[str],
    sizes: dict[str, int],
    is_prefix: Callable[[str, str], bool],
) -> dict[str, HistoricalRevisionDecision]:
    """Prove a byte-prefix DAG over already-deduplicated (distinct-content) nodes.

    Implements I4/I5 (polylogue-lb39z): a divergence quarantines only the
    genuinely-divergent suffix, never a proven prefix chain or an unrelated
    sibling that happens to share a cohort with it.

    ``is_prefix(parent, child)`` must only be evaluated for pairs where
    ``sizes[parent] < sizes[child]`` (the caller guarantees this so streamed
    implementations can skip same-size/reverse-size comparisons entirely).

    A node is only ever classified ``BYTE_PROVEN`` if it sits on a single,
    unbranched path back to the cohort's one true root: every ancestor,
    including the node itself, has exactly one maximal parent candidate and
    is the sole child of that parent. The moment a node's parent has more
    than one child (a fork) or a node has more than one incomparable maximal
    parent (an ambiguous parent set), that node -- and everything built on
    top of it -- is quarantined, while everything already proven up to that
    point (including the fork point itself) is left alone. If the cohort has
    zero or more than one root (no shared ancestor at all, e.g. two
    unrelated same-size captures), there is no anchor to localize from and
    the whole cohort is quarantined, matching prior behavior for that case.
    """
    ordered = sorted(node_ids, key=lambda raw_id: (sizes[raw_id], raw_id))
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {raw_id: [] for raw_id in ordered}
    for child in ordered:
        candidates = [
            parent
            for parent in ordered
            if parent != child and sizes[parent] < sizes[child] and is_prefix(parent, child)
        ]
        # Prefix is transitive: if parent1 and parent2 are both prefixes of
        # child and sizes[parent1] < sizes[parent2], then parent1 is
        # necessarily a prefix of parent2 (parent1's bytes equal child's
        # first N bytes, which equal parent2's first N bytes since parent2
        # is itself a prefix of child of length >= N). Candidates therefore
        # form a totally ordered chain and the unique maximal element is
        # simply the largest one -- no further is_prefix comparisons (and
        # their streamed blob re-reads) are needed here.
        maximal = [max(candidates, key=lambda raw_id: (sizes[raw_id], raw_id))] if candidates else []
        parents[child] = maximal
        for parent in maximal:
            children[parent].append(child)

    roots = [raw_id for raw_id in ordered if not parents[raw_id]]
    if len(roots) != 1:
        return {
            raw_id: HistoricalRevisionDecision(
                raw_id=raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
            for raw_id in ordered
        }
    root = roots[0]
    clean: dict[str, bool] = {root: True}
    for raw_id in ordered:
        if raw_id == root:
            continue
        parent_set = parents[raw_id]
        if len(parent_set) != 1:
            clean[raw_id] = False
            continue
        (parent,) = parent_set
        clean[raw_id] = clean.get(parent, False) and len(children[parent]) == 1

    decisions: dict[str, HistoricalRevisionDecision] = {}
    for raw_id in ordered:
        if clean.get(raw_id, False):
            parent_set = parents[raw_id]
            predecessor = parent_set[0] if parent_set else None
            relation: Literal["baseline", "predecessor"] = "baseline" if predecessor is None else "predecessor"
            decisions[raw_id] = HistoricalRevisionDecision(
                raw_id=raw_id,
                authority=RawRevisionAuthority.BYTE_PROVEN,
                relation=relation,
                predecessor_raw_id=predecessor,
            )
        else:
            decisions[raw_id] = HistoricalRevisionDecision(
                raw_id=raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
    return decisions


def classify_historical_full_revisions(
    revisions: list[HistoricalRawRevision],
) -> list[HistoricalRevisionDecision]:
    """Prove a unique byte-prefix chain; quarantine only the divergent suffix.

    Acquisition time, source path, provider timestamps, and raw-id ordering are
    intentionally absent. Equal or divergent payloads do not establish which
    capture is newer.

    I4 (polylogue-lb39z): byte-identical captures of one source are one
    evidence node observed twice, not competing revisions -- they are
    collapsed onto a single representative (the lexicographically smallest
    ``raw_id``) before the chain proof runs, and every non-representative
    duplicate mirrors its representative's verdict with ``relation=
    "duplicate"``. Output order matches input size order (ascending, ties
    broken by ``raw_id``), preserving the existing "first is oldest, last is
    head" contract relied on by callers such as
    ``classify_untyped_full_revision_groups``.
    """
    if not revisions:
        return []
    groups: dict[bytes, list[HistoricalRawRevision]] = {}
    for revision in revisions:
        groups.setdefault(revision.payload, []).append(revision)
    representative_payload: dict[str, bytes] = {}
    members_of: dict[str, list[str]] = {}
    for payload, members in groups.items():
        representative = min(members, key=lambda revision: revision.raw_id)
        representative_payload[representative.raw_id] = payload
        members_of[representative.raw_id] = sorted(member.raw_id for member in members)

    def is_prefix(parent: str, child: str) -> bool:
        return representative_payload[child].startswith(representative_payload[parent])

    sizes = {raw_id: len(payload) for raw_id, payload in representative_payload.items()}
    rep_decisions = _classify_deduped_nodes(list(representative_payload), sizes, is_prefix)
    return _expand_duplicate_decisions(rep_decisions, members_of, sizes)


def _expand_duplicate_decisions(
    rep_decisions: dict[str, HistoricalRevisionDecision],
    members_of: dict[str, list[str]],
    rep_sizes: dict[str, int],
) -> list[HistoricalRevisionDecision]:
    size_by_raw_id: dict[str, int] = {}
    decision_by_raw_id: dict[str, HistoricalRevisionDecision] = {}
    for representative_id, member_ids in members_of.items():
        rep_decision = rep_decisions[representative_id]
        decision_by_raw_id[representative_id] = rep_decision
        size_by_raw_id[representative_id] = rep_sizes[representative_id]
        for member_id in member_ids:
            if member_id == representative_id:
                continue
            decision_by_raw_id[member_id] = HistoricalRevisionDecision(
                raw_id=member_id,
                authority=rep_decision.authority,
                relation="duplicate",
                predecessor_raw_id=None,
                duplicate_of_raw_id=representative_id,
            )
            size_by_raw_id[member_id] = rep_sizes[representative_id]
    ordered_ids = sorted(decision_by_raw_id, key=lambda raw_id: (size_by_raw_id[raw_id], raw_id))
    return [decision_by_raw_id[raw_id] for raw_id in ordered_ids]


def _stream_size_and_hash(revision: HistoricalRawRevisionStream) -> tuple[int, str]:
    size = 0
    digest = sha256()
    with revision.open_payload() as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _stream_is_prefix(
    parent: HistoricalRawRevisionStream,
    child: HistoricalRawRevisionStream,
    *,
    parent_size: int,
    child_size: int,
) -> bool:
    """Return whether *parent* is an exact proper byte prefix of *child*."""
    if parent_size >= child_size:
        return False
    remaining = parent_size
    with parent.open_payload() as parent_handle, child.open_payload() as child_handle:
        while remaining:
            chunk = parent_handle.read(min(1024 * 1024, remaining))
            if not chunk or child_handle.read(len(chunk)) != chunk:
                return False
            remaining -= len(chunk)
        return parent_handle.read(1) == b""


def classify_historical_full_revision_streams(
    revisions: list[HistoricalRawRevisionStream],
) -> list[HistoricalRevisionDecision]:
    """Stream the same dedup-first, localized-ambiguity proof, without eager payloads.

    Each stream is read exactly once to learn its true size and content hash
    (never trusting the caller-supplied ``payload_size`` alone); streams whose
    hash matches are byte-identical duplicates and are collapsed per I4 before
    any prefix comparison runs. Prefix comparisons between distinct-content
    representatives are themselves streamed (``_stream_is_prefix``), so no
    full payload is ever held in memory at once.
    """
    if not revisions:
        return []
    size_and_hash = {revision.raw_id: _stream_size_and_hash(revision) for revision in revisions}
    by_hash: dict[str, list[HistoricalRawRevisionStream]] = {}
    for revision in revisions:
        _, digest = size_and_hash[revision.raw_id]
        by_hash.setdefault(digest, []).append(revision)

    representative_stream: dict[str, HistoricalRawRevisionStream] = {}
    representative_size: dict[str, int] = {}
    members_of: dict[str, list[str]] = {}
    for members in by_hash.values():
        representative = min(members, key=lambda revision: revision.raw_id)
        representative_stream[representative.raw_id] = representative
        representative_size[representative.raw_id] = size_and_hash[representative.raw_id][0]
        members_of[representative.raw_id] = sorted(member.raw_id for member in members)

    def is_prefix(parent: str, child: str) -> bool:
        return _stream_is_prefix(
            representative_stream[parent],
            representative_stream[child],
            parent_size=representative_size[parent],
            child_size=representative_size[child],
        )

    rep_decisions = _classify_deduped_nodes(list(representative_stream), representative_size, is_prefix)
    return _expand_duplicate_decisions(rep_decisions, members_of, representative_size)


__all__ = [
    "HistoricalRawRevision",
    "HistoricalRawRevisionStream",
    "HistoricalRevisionDecision",
    "RawRevisionAuthority",
    "RawRevisionEnvelope",
    "RawRevisionKind",
    "append_source_revision",
    "classify_historical_full_revisions",
    "classify_historical_full_revision_streams",
]
