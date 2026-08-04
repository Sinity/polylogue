"""Authority classification for sessions extracted from multi-session raws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.core.enums import PolylogueStrEnum
from polylogue.core.timestamps import parse_timestamp
from polylogue.pipeline.ids import MessageContent, SessionRevisionProjection


class MembershipDecision(PolylogueStrEnum):
    """Closed vocabulary for ``raw_session_memberships.decision`` (source.db).

    The membership-classification pipeline's own verdict on a raw revision's
    fate within its logical-source cohort, persisted in the durable source
    tier. Distinct from :class:`polylogue.archive.revision_replay.
    ApplicationDecision` (the index-tier replay outcome for the same
    underlying event) -- the two vocabularies model related but genuinely
    different axes (pipeline verdict vs. replay-time application) and are
    kept separate rather than merged; see polylogue-z22ml.
    """

    APPLIED = "applied"
    SUPERSEDED_EQUIVALENT = "superseded_equivalent"
    SUPERSEDED_PREFIX = "superseded_prefix"
    AMBIGUOUS = "ambiguous"
    DEFERRED = "deferred"


#: One content axis's relation between two revisions: total and decidable,
#: no residual category (polylogue-aggz). ``equal`` is the same identity set
#: with equal content per identity. ``a_contains_b``/``b_contains_a`` is one
#: side's identity set containing the other's with equal content on the
#: intersection -- set containment, not sequence prefix, so this subsumes
#: append-only growth without caring about array order. ``conflict`` is
#: either side disagreeing about content under a shared identity, or each
#: side holding an identity the other lacks (a genuine fork).
_Relation: TypeAlias = Literal["equal", "a_contains_b", "b_contains_a", "conflict"]


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
    #: Materialized head(s): an append-only growth chain (oldest to newest,
    #: by set containment). Empty ONLY when the cohort is a genuine,
    #: irreducible conflict (see ``ambiguous_raw_ids``) AND a head already
    #: exists for this cohort under any authority -- see
    #: ``classify_membership_revisions``'s ``existing_accepted_raw_id``
    #: parameter for the guard that decides this. A cohort with NO prior
    #: head resolves to a single-element tuple via the presence-guarantee
    #: fallback (``_maximal_evidence_fallback``) even on irreducible
    #: conflict.
    accepted_raw_ids: tuple[str, ...]
    #: Raws proven to carry the SAME content as an accepted head (``equal``
    #: per ``_relation``) -- legitimately superseded, not debt.
    equivalent_raw_ids: tuple[str, ...]
    #: Raws NOT reflected in ``accepted_raw_ids`` because no containment
    #: relation could order them against it: unresolved conflict debt
    #: attached to the session, retained as raw provenance only (never
    #: deleted, never silently discarded).
    ambiguous_raw_ids: tuple[str, ...]


def _identities(contents: frozenset[tuple[bytes, bytes]]) -> frozenset[bytes]:
    """Project the identity half of a (identity, content) pair set.

    Messages and events always carry content for every identity they have --
    unlike attachments, which project identity and content as two separate
    fields because an attachment can be referenced before its bytes are
    acquired (polylogue-bu1i) -- so their identity set is always exactly
    this projection of their content set.
    """
    return frozenset(identity for identity, _content in contents)


def _message_identities(contents: frozenset[MessageContent]) -> frozenset[bytes]:
    return frozenset(identity for identity, _content, _multiplicity in contents)


def _message_axis_relation(contents_a: frozenset[MessageContent], contents_b: frozenset[MessageContent]) -> _Relation:
    """Compare message content as an unordered multiset.

    Reordering remains equivalent, while a repeated id-less message is an
    additional persisted turn rather than a duplicate that set semantics can
    erase. A shared identity carrying different content remains a conflict.
    """
    counts_a = {(identity, content): multiplicity for identity, content, multiplicity in contents_a}
    counts_b = {(identity, content): multiplicity for identity, content, multiplicity in contents_b}
    identities_a = _message_identities(contents_a)
    identities_b = _message_identities(contents_b)
    for identity in identities_a & identities_b:
        content_values_a = {content for candidate_identity, content in counts_a if candidate_identity == identity}
        content_values_b = {content for candidate_identity, content in counts_b if candidate_identity == identity}
        if content_values_a != content_values_b:
            return "conflict"
    a_richer = bool(identities_a - identities_b) or any(count > counts_b.get(key, 0) for key, count in counts_a.items())
    b_richer = bool(identities_b - identities_a) or any(count > counts_a.get(key, 0) for key, count in counts_b.items())
    if a_richer and b_richer:
        return "conflict"
    if a_richer:
        return "a_contains_b"
    if b_richer:
        return "b_contains_a"
    return "equal"


def _content_by_identity(contents: frozenset[tuple[bytes, bytes]]) -> dict[bytes, frozenset[bytes]]:
    """Group content hashes by identity, retaining EVERY value under a colliding identity.

    Identity is not always injective: two acquired attachments on one
    message can share one identity (same ``message_id``/``name``/
    ``mime_type``, different bytes -- the accepted limit documented on
    ``attachment_identity_hash``, not a new one). Collapsing such a
    collision to a single arbitrary content hash (a plain ``dict``, last
    value wins) would let a real content conflict compare as equal instead.
    Grouping into a set per identity means a collision always degrades to
    ``conflict`` when compared against a revision that disagrees, never to
    ``equal``.
    """
    grouped: dict[bytes, set[bytes]] = {}
    for identity, content in contents:
        grouped.setdefault(identity, set()).add(content)
    return {identity: frozenset(values) for identity, values in grouped.items()}


def _axis_relation(
    identities_a: frozenset[bytes],
    contents_a: frozenset[tuple[bytes, bytes]],
    identities_b: frozenset[bytes],
    contents_b: frozenset[tuple[bytes, bytes]],
) -> _Relation:
    """Compare one content axis (messages, attachments, or events) between two revisions.

    An identity present without content on either side (an attachment
    referenced but not yet acquired) is compatible with, never in conflict
    with, the SAME identity carrying content on the other side -- unknown is
    less evidence, not a disagreement (polylogue-bu1i). For messages and
    events every identity always carries content, so this degrades to plain
    set equality/containment for those two axes.
    """
    content_a = _content_by_identity(contents_a)
    content_b = _content_by_identity(contents_b)
    shared = identities_a & identities_b
    for identity in shared:
        value_a, value_b = content_a.get(identity), content_b.get(identity)
        if value_a is not None and value_b is not None and value_a != value_b:
            return "conflict"
    a_richer = bool(identities_a - identities_b) or any(
        content_a.get(identity) is not None and content_b.get(identity) is None for identity in shared
    )
    b_richer = bool(identities_b - identities_a) or any(
        content_b.get(identity) is not None and content_a.get(identity) is None for identity in shared
    )
    if a_richer and b_richer:
        return "conflict"
    if a_richer:
        return "a_contains_b"
    if b_richer:
        return "b_contains_a"
    return "equal"


def _relation(a: SessionRevisionProjection, b: SessionRevisionProjection) -> _Relation:
    """Combine the message/attachment/event axes into one overall relation.

    Each axis is independently equal/contains/conflict; the whole document is
    ``equal`` only if every axis is, ``a_contains_b``/``b_contains_a`` only if
    every non-equal axis agrees on the SAME direction (a mix -- one axis grew
    forward, another backward -- is exactly a fork: each side holds
    something the other lacks, so it is a conflict too), and ``conflict`` if
    any single axis already is.
    """
    axes = (
        _message_axis_relation(a.message_contents, b.message_contents),
        _axis_relation(a.attachment_identities, a.attachment_contents, b.attachment_identities, b.attachment_contents),
        _axis_relation(
            _identities(a.event_contents), a.event_contents, _identities(b.event_contents), b.event_contents
        ),
    )
    if "conflict" in axes:
        return "conflict"
    directions = {axis for axis in axes if axis != "equal"}
    if not directions:
        return "equal"
    if directions == {"a_contains_b"}:
        return "a_contains_b"
    if directions == {"b_contains_a"}:
        return "b_contains_a"
    return "conflict"


def _frontier(projection: SessionRevisionProjection) -> tuple[int, int, int, int]:
    """Rank a revision along every axis on which it can only grow.

    Used only to order candidates for the pairwise ``_relation`` chain check
    and to break ties deterministically in the presence-guarantee fallback
    (stable regardless of processing order -- a rebuild must always resolve
    the same cohort to the same head).
    """
    return (
        sum(multiplicity for _identity, _content, multiplicity in projection.message_contents),
        len(projection.event_contents),
        len(projection.attachment_identities),
        len(projection.attachment_contents),
    )


def _equal_content_representative(
    incumbent: MembershipRevision, candidate: MembershipRevision
) -> tuple[MembershipRevision, MembershipRevision]:
    """Pick (winner, loser) between two revisions already proven ``equal`` in content.

    Equal content is not equal authority: a direct/native export and a
    browser DOM scrape can project to byte-identical messages while one is
    first-class provenance and the other is a lossy capture, and which one
    survives as the representative determines what future re-acquisitions
    are compared against. Source authority is therefore decided BEFORE any
    timestamp tiebreak -- mirroring, for the equal case, the same
    ``_direct_export_precedence`` / browser-fidelity ordering
    (``_browser_snapshot_dominates``) already applied to the non-equal
    growth-chain case below: a direct export always outranks a
    browser-capture sibling, and a native browser snapshot always outranks a
    DOM snapshot of the same underlying session. Only when neither
    revision's provenance outranks the other's does provider timestamp (and
    finally a stable raw_id) decide, exactly as before this function
    existed.
    """
    incumbent_direct = incumbent.browser_snapshot_fidelity is None
    candidate_direct = candidate.browser_snapshot_fidelity is None
    if incumbent_direct != candidate_direct:
        return (incumbent, candidate) if incumbent_direct else (candidate, incumbent)
    if (
        not incumbent_direct
        and not candidate_direct
        and incumbent.browser_snapshot_fidelity != candidate.browser_snapshot_fidelity
    ):
        return (incumbent, candidate) if incumbent.browser_snapshot_fidelity == "native" else (candidate, incumbent)
    incumbent_time = parse_timestamp(incumbent.provider_updated_at)
    candidate_time = parse_timestamp(candidate.provider_updated_at)
    if (
        incumbent_time is not None
        and candidate_time is not None
        and candidate_time.timestamp() != incumbent_time.timestamp()
    ):
        return (
            (candidate, incumbent)
            if candidate_time.timestamp() > incumbent_time.timestamp()
            else (incumbent, candidate)
        )
    # No distinguishing provenance or provider timestamp -- these two are
    # already proven identical content, so which raw_id represents them does
    # not matter for correctness; pick deterministically rather than
    # requiring a timestamp that a re-export of an untouched conversation
    # may never carry (its own provider updated_at legitimately does not
    # move when nothing provider-visible changed).
    winner, loser = sorted((incumbent, candidate), key=lambda item: item.raw_id)
    return winner, loser


def classify_membership_revisions(
    revisions: list[MembershipRevision], *, existing_accepted_raw_id: str | None = None
) -> MembershipClassification:
    """Accept one total growth chain by set containment; never choose a branch silently.

    Two revisions are the same content, contain each other, or conflict --
    total and decidable, with no residual category (polylogue-aggz). This
    replaces what used to be four separate mechanisms (a positional-prefix
    message dominance test, a denylist-stripped ordered event-hash prefix
    test, a strict-vs-loose attachment id correlation pass, and a
    session-hash/metadata-timestamp tiebreak layered on top of all three)
    with one relation, applied uniformly to every axis.

    A genuine, irreducible conflict -- no containment chain exists at all,
    not export-vintage noise -- falls back to the presence-guarantee pick
    (``_maximal_evidence_fallback``, deterministic maximal-evidence
    representative, conflict recorded as debt rather than absence) ONLY when
    ``existing_accepted_raw_id`` is ``None`` -- no head exists yet for this
    cohort under ANY authority, so nothing is retired or downgraded by
    materializing one. Any existing head at all -- even one that happens to
    sit at the exact raw_id the fallback would itself pick -- refuses the
    fallback and quarantines exactly as this cohort always did: every
    representative in ``ambiguous_raw_ids``, nothing accepted. This is
    deliberately narrower than "only refuse a DIFFERENT raw_id": re-accepting
    the SAME raw_id through membership governance still overwrites that
    head's ``accepted_frontier_kind``/generation metadata (e.g. downgrading
    a byte-governed head to "semantic"), a real authority downgrade even
    though the pointed-to raw_id never changed -- proven by a real
    integration-test regression during development, not a theoretical
    concern. The guard is a structural guarantee, not a runtime check that
    could be bypassed: a later membership pass can never silently touch an
    already-established head just because arbitration could not order the
    new cohort (polylogue-miwv, PR #3211's write-back invariant). Callers
    are responsible for passing the CURRENT accepted head's raw_id (via
    ``archive.raw_revision_head_raw_id(logical_source_key)`` or equivalent)
    -- omitting it when a head does exist defeats the guard.
    """
    if not revisions:
        return MembershipClassification((), (), ())

    representatives: list[MembershipRevision] = []
    equivalents: list[str] = []
    for revision in revisions:
        match_index = next(
            (i for i, rep in enumerate(representatives) if _relation(rep.projection, revision.projection) == "equal"),
            None,
        )
        if match_index is None:
            representatives.append(revision)
            continue
        incumbent = representatives[match_index]
        winner, loser = _equal_content_representative(incumbent, revision)
        representatives[match_index] = winner
        equivalents.append(loser.raw_id)

    representatives.sort(key=lambda item: (_frontier(item.projection), item.raw_id))
    conflict = any(
        _relation(older.projection, newer.projection) != "b_contains_a"
        for older, newer in zip(representatives, representatives[1:], strict=False)
    )
    if not conflict:
        return MembershipClassification(
            tuple(item.raw_id for item in representatives),
            tuple(sorted(equivalents)),
            (),
        )

    browser_order = _provider_ordered_browser_snapshots(representatives)
    if browser_order is not None:
        return MembershipClassification(
            tuple(item.raw_id for item in browser_order),
            tuple(sorted(equivalents)),
            (),
        )
    direct_export = _direct_export_precedence(representatives)
    if direct_export is not None:
        accepted, browser_capture_raw_ids = direct_export
        return MembershipClassification(
            (accepted.raw_id,),
            tuple(sorted((*equivalents, *browser_capture_raw_ids))),
            (),
        )
    # Presence-guarantee fallback, guarded against ANY interference with an
    # already-established head -- see this function's own docstring for why
    # this is deliberately narrower than "only refuse when the raw_id
    # differs". Applying the fallback even when its pick happens to be the
    # SAME raw_id as the existing head is still unsafe: verified directly
    # during development (a real integration-test regression) that
    # ``apply_raw_membership_classification`` re-accepting that raw_id
    # through MEMBERSHIP governance downgrades a byte-governed head's own
    # ``accepted_frontier_kind``/generation metadata from "byte" to
    # "semantic", even though the pointed-to raw_id never changed -- a
    # genuine authority downgrade, not a no-op. So the fallback applies
    # ONLY when no head exists at all yet.
    fallback = _maximal_evidence_fallback(representatives)
    if existing_accepted_raw_id is None:
        return MembershipClassification(
            (fallback.raw_id,),
            tuple(sorted(equivalents)),
            tuple(sorted(item.raw_id for item in representatives if item.raw_id != fallback.raw_id)),
        )
    return MembershipClassification(
        (),
        tuple(sorted(equivalents)),
        tuple(sorted(item.raw_id for item in representatives)),
    )


def _maximal_evidence_fallback(representatives: list[MembershipRevision]) -> MembershipRevision:
    """Deterministically select the maximal-evidence revision for an irreducible conflict.

    Designed and unit-tested as the presence-guarantee alternative to
    quarantining every representative with nothing accepted: a document
    should never simply vanish from the archive just because arbitration
    could not order its revisions. Frontier order (most messages, most
    events, most attachments, most acquired bytes) with a stable raw_id
    tiebreak means the same cohort always resolves to the same head
    regardless of processing order -- verified directly by
    ``test_maximal_evidence_fallback_is_order_independent``.

    Called by ``classify_membership_revisions`` for a genuine, irreducible
    conflict, but GUARDED there: the caller passes the cohort's current
    accepted head's raw_id (if any) as ``existing_accepted_raw_id``, and the
    fallback pick is applied ONLY when no head exists yet. ANY existing head
    -- even one that happens to sit at the exact raw_id this function would
    itself pick -- refuses the fallback and preserves plain quarantine
    (nothing accepted) exactly as before this function was wired in. This
    guard exists because, unguarded (or guarded only against a DIFFERENT
    raw_id), a fallback pick can silently damage an already-established
    head -- proven by real integration-test regressions during development:
    ``test_divergent_bundle_member_preserves_last_accepted_session`` and its
    sibling ``test_divergent_bundle_member_does_not_block_safe_members``
    exercise the DIFFERENT-raw_id case against a real, carefully-designed
    invariant in ``archive.py``'s ``apply_raw_membership_classification``
    write-back (polylogue-miwv, PR #3211): once a logical source has an
    accepted head, a later membership pass may not silently retire it in
    favor of an unrelated raw. A second, independently-discovered case ruled
    out even the SAME-raw_id "re-affirmation" shape: re-accepting that exact
    raw_id through MEMBERSHIP governance still overwrites the head's own
    ``accepted_frontier_kind``/generation metadata (e.g. downgrading a
    byte-governed head to "semantic"), a real authority downgrade despite
    the pointed-to raw_id never changing (``test_live_multi_session_divergence_reopens_raw_authority``).
    """
    return max(representatives, key=lambda item: (_frontier(item.projection), item.raw_id))


def _provider_ordered_browser_snapshots(
    revisions: list[MembershipRevision],
) -> list[MembershipRevision] | None:
    """Order compatible mutable browser snapshots by provider authority.

    Kept as a genuine exception to the content-only relation above, not a
    dead alternate path: browser-captured DOM and native snapshots synthesize
    their OWN local message/attachment ids from DOM structure, not a stable
    provider identity, so the same underlying message can carry a different
    synthetic id in a DOM capture than in a native one -- the content-only
    ``_relation`` genuinely cannot correlate them (it would read a
    DOM-to-native fidelity upgrade as two disjoint, conflicting message id
    sets). ``provider_message_ids``/``provider_attachment_ids`` on
    ``MembershipRevision`` carry the browser-capture layer's own identity
    evidence for exactly this reason. ChatGPT can also move an already-present
    context/tool node when later work appears, and can complete text in place
    under the same provider message id, so a strict serialized-content
    comparison would reject ordinary provider progress; provider timestamps
    resolve that progress only when stable message and attachment identities
    are preserved. DOM-to-native is the sole fidelity upgrade and may use
    different synthetic ids; a native-to-DOM downgrade is never selected.
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
    session content regardless of set-containment or provider-timestamp
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


__all__ = [
    "MembershipClassification",
    "MembershipDecision",
    "MembershipRevision",
    "classify_membership_revisions",
]
