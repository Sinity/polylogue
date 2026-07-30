from __future__ import annotations

from itertools import permutations

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import (
    MembershipRevision,
    _maximal_evidence_fallback,
    _relation,
    classify_membership_revisions,
)
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent


def _revision(raw_id: str, *texts: str) -> MembershipRevision:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id=str(i), role=Role.USER, text=text) for i, text in enumerate(texts)],
    )
    return MembershipRevision(raw_id, session_revision_projection(session))


def _ordered_message_revision(raw_id: str, *id_text_pairs: tuple[str, str]) -> MembershipRevision:
    """A session whose messages carry explicit provider ids in a given array order."""
    session = ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id="session",
        messages=[
            ParsedMessage(provider_message_id=provider_id, role=Role.USER, text=text)
            for provider_id, text in id_text_pairs
        ],
    )
    return MembershipRevision(raw_id, session_revision_projection(session))


# ---------------------------------------------------------------------------
# Core equal/contains/conflict semantics (polylogue-aggz)
# ---------------------------------------------------------------------------


def test_classifies_strict_growth_and_semantic_equivalence() -> None:
    result = classify_membership_revisions(
        [_revision("raw-b", "one", "two"), _revision("raw-z", "one"), _revision("raw-a", "one")]
    )
    assert result.accepted_raw_ids == ("raw-a", "raw-b")
    assert result.equivalent_raw_ids == ("raw-z",)
    assert result.ambiguous_raw_ids == ()


def test_refuses_divergent_maxima() -> None:
    """Genuine divergence stays quarantined ambiguous.

    raw-b and raw-c both grew from raw-a in incompatible directions (each
    holds a message the other lacks) -- a genuine fork, "conflict" under
    ``_relation``. See ``_maximal_evidence_fallback`` for the designed,
    unit-tested presence-guarantee alternative to this quarantine, and its
    docstring for why it is not wired into ``classify_membership_revisions``
    yet.
    """
    result = classify_membership_revisions(
        [_revision("raw-a", "one"), _revision("raw-b", "one", "left"), _revision("raw-c", "one", "right")]
    )
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b", "raw-c")


def test_maximal_evidence_fallback_picks_deterministically() -> None:
    """The presence-guarantee fallback (not yet wired live) resolves a fork deterministically.

    Frontier tie among raw-b/raw-c falls through to the raw_id tiebreak
    ("raw-c" > "raw-b"), same as the frontier-sort used elsewhere in this
    module.
    """
    representatives = [_revision("raw-a", "one"), _revision("raw-b", "one", "left"), _revision("raw-c", "one", "right")]
    assert _maximal_evidence_fallback(representatives).raw_id == "raw-c"


def test_maximal_evidence_fallback_is_order_independent() -> None:
    """The same genuinely-ambiguous cohort always resolves to the same head.

    A rebuild that processes raws in a different order must not produce a
    different archive: every permutation of the same three divergent
    revisions must pick the identical fallback head. ``_frontier`` plus the
    stable raw_id tiebreak guarantees this -- it depends only on each
    revision's own projected content, never on input order.
    """
    revisions = [_revision("raw-a", "one"), _revision("raw-b", "one", "left"), _revision("raw-c", "one", "right")]
    results = {_maximal_evidence_fallback(list(ordering)).raw_id for ordering in permutations(revisions)}
    assert results == {"raw-c"}


def test_containment_chain_resolves_without_needing_the_fallback() -> None:
    """A clean append-only growth chain resolves via set containment alone --
    accepting the WHOLE chain (both raws), not a single frontier-max pick,
    which would silently discard the older revision's own head-of-chain
    role.
    """
    result = classify_membership_revisions([_revision("raw-old", "one"), _revision("raw-new", "one", "two")])
    assert result.accepted_raw_ids == ("raw-old", "raw-new")
    assert result.ambiguous_raw_ids == ()


def test_metadata_only_revision_uses_latest_provider_timestamp() -> None:
    def revision(raw_id: str, updated_at: str | None, *, title: str = "title") -> MembershipRevision:
        session = ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="session",
            title=title,
            updated_at=updated_at,
            messages=[ParsedMessage(provider_message_id="0", role=Role.USER, text="one")],
        )
        return MembershipRevision(raw_id, session_revision_projection(session), updated_at)

    older = revision("raw-old", "2026-01-01T00:00:00Z")
    newer = revision("raw-new", "2026-01-02T00:00:00Z")
    assert older.projection.session_hash != newer.projection.session_hash

    result = classify_membership_revisions([older, newer])

    assert result.accepted_raw_ids == ("raw-new",)
    assert result.equivalent_raw_ids == ("raw-old",)
    assert result.ambiguous_raw_ids == ()


def test_metadata_revision_without_complete_provider_time_still_resolves_by_content() -> None:
    """A title difference with an unresolvable timestamp no longer blocks presence.

    Both revisions carry the identical single message -- content equality
    already proves these are the same revision (title is not part of
    ``_relation`` at all); a missing timestamp is no longer a reason to
    withhold a revision this comparison has already proven is the same
    conversation. Deterministic tiebreak (min raw_id) picks a representative
    when no timestamp can settle it.
    """
    with_timestamp = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        title="new title",
        updated_at="2026-01-02T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="0", role=Role.USER, text="one")],
    )
    missing_timestamp = with_timestamp.model_copy(update={"title": "old title", "updated_at": None})

    result = classify_membership_revisions(
        [
            MembershipRevision("raw-new", session_revision_projection(with_timestamp), with_timestamp.updated_at),
            MembershipRevision(
                "raw-old",
                session_revision_projection(missing_timestamp),
                missing_timestamp.updated_at,
            ),
        ]
    )

    assert result.accepted_raw_ids == ("raw-new",)
    assert result.equivalent_raw_ids == ("raw-old",)
    assert result.ambiguous_raw_ids == ()


def test_metadata_revisions_with_equal_provider_time_still_resolve_by_content() -> None:
    """A title difference under a TIED timestamp also resolves by content."""
    timestamp = "2026-01-02T00:00:00Z"
    first = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        title="first",
        updated_at=timestamp,
        messages=[ParsedMessage(provider_message_id="0", role=Role.USER, text="one")],
    )
    second = first.model_copy(update={"title": "second"})

    result = classify_membership_revisions(
        [
            MembershipRevision("raw-a", session_revision_projection(first), timestamp),
            MembershipRevision("raw-b", session_revision_projection(second), timestamp),
        ]
    )

    assert result.accepted_raw_ids == ("raw-a",)
    assert result.equivalent_raw_ids == ("raw-b",)
    assert result.ambiguous_raw_ids == ()


# ---------------------------------------------------------------------------
# Messages: order-insensitive set containment (polylogue-c429)
# ---------------------------------------------------------------------------


def test_accepts_reordered_message_array_with_identical_content_as_equivalent() -> None:
    """Same message ids, byte-identical content per id, different array order.

    Reproduces the exact shape measured on the live archive: Claude.ai's own
    export ordering for a conversation is not guaranteed stable across
    separate export requests -- 34 of 35 sampled claude-ai-export ambiguous
    cohorts had identical message-id sets with zero per-id content
    differences (polylogue-c429). A set has no order to violate; this must
    resolve as equivalent content.
    """
    chronological = _ordered_message_revision("raw-chrono", ("a", "one"), ("b", "two"), ("c", "three"))
    resequenced = _ordered_message_revision("raw-resequenced", ("b", "two"), ("a", "one"), ("c", "three"))

    assert chronological.projection.message_hashes != resequenced.projection.message_hashes
    assert chronological.projection.message_contents == resequenced.projection.message_contents
    assert _relation(chronological.projection, resequenced.projection) == "equal"

    result = classify_membership_revisions([chronological, resequenced])

    assert result.accepted_raw_ids == ("raw-chrono",)
    assert result.equivalent_raw_ids == ("raw-resequenced",)
    assert result.ambiguous_raw_ids == ()


def test_accepts_message_growth_across_a_reordered_prefix() -> None:
    """A genuinely appended message is still growth even when the shared ids reorder."""
    older = _ordered_message_revision("raw-old", ("b", "two"), ("a", "one"))
    newer = _ordered_message_revision("raw-new", ("a", "one"), ("c", "three"), ("b", "two"))

    assert _relation(older.projection, newer.projection) == "b_contains_a"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ("raw-old", "raw-new")
    assert result.ambiguous_raw_ids == ()


def test_refuses_message_content_change_under_a_shared_id_despite_reorder() -> None:
    """A real edit under a shared id is a conflict, not laundered by the set model."""
    older = _ordered_message_revision("raw-old", ("a", "one"), ("b", "two"))
    newer = _ordered_message_revision("raw-new", ("b", "two"), ("a", "EDITED"))

    assert _relation(older.projection, newer.projection) == "conflict"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_refuses_message_id_disappearing_despite_reorder() -> None:
    """A message id present in older but absent from newer is a real loss (a fork)."""
    older = _ordered_message_revision("raw-old", ("a", "one"), ("b", "two"))
    newer = _ordered_message_revision("raw-new", ("b", "two"), ("c", "three"))

    assert _relation(older.projection, newer.projection) == "conflict"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_message_reorder_does_change_the_session_content_hash() -> None:
    """Order tolerance must not leak into idempotency.

    ``session_hash`` must stay order-sensitive: if two array orderings of the
    same messages hashed identically, a real reorder written by the
    archive's own writer path would be silently skipped as unchanged on
    re-ingest.
    """
    chronological = _ordered_message_revision("raw-chrono", ("a", "one"), ("b", "two"))
    resequenced = _ordered_message_revision("raw-resequenced", ("b", "two"), ("a", "one"))

    assert chronological.projection.session_hash != resequenced.projection.session_hash


# ---------------------------------------------------------------------------
# Attachments: content-derived identity, acquisition growth (polylogue-bu1i,
# polylogue-d8al, polylogue-hith)
# ---------------------------------------------------------------------------


def _attachment_revision(
    raw_id: str,
    *,
    acquired: bool,
    provider_attachment_id: str = "drive-file-1",
    inline: bytes = b"attachment bytes",
    name: str = "screenshot.png",
    mime_type: str = "image/png",
) -> MembershipRevision:
    """A session whose single attachment is referenced, optionally with bytes read."""
    attachment = ParsedAttachment(
        provider_attachment_id=provider_attachment_id,
        message_provider_id="0",
        name=name,
        mime_type=mime_type,
        size_bytes=len(inline) if acquired else None,
        inline_bytes=inline if acquired else None,
    )
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.USER, text="one")],
        attachments=[attachment],
    )
    return MembershipRevision(raw_id, session_revision_projection(session))


def test_accepts_attachment_byte_acquisition_as_growth_not_a_branch() -> None:
    """Resolving an already-referenced attachment's bytes is a fidelity upgrade.

    A Drive/Gemini document acquired twice -- once before its attachment
    bytes were fetched, once after -- has an identical transcript and
    attachment reference. All 157 two-member aistudio-drive ambiguous
    cohorts in the live archive had exactly this shape (polylogue-bu1i).
    """
    bare = _attachment_revision("raw-bbbb", acquired=False)
    enriched = _attachment_revision("raw-aaaa", acquired=True)

    assert _relation(bare.projection, enriched.projection) == "b_contains_a"

    result = classify_membership_revisions([enriched, bare])
    # Ordering matters as much as acceptance: the revision holding the bytes
    # must be the head of the chain, or the archive indexes the emptier one.
    assert result.accepted_raw_ids == ("raw-bbbb", "raw-aaaa")
    assert result.ambiguous_raw_ids == ()


def test_refuses_attachment_byte_loss_as_a_fidelity_downgrade() -> None:
    """Dropping bytes already in hand is never an upgrade, in either direction."""
    enriched = _attachment_revision("raw-a", acquired=True)
    bare = _attachment_revision("raw-b", acquired=False)

    assert _relation(enriched.projection, bare.projection) == "a_contains_b"
    assert _relation(bare.projection, enriched.projection) == "b_contains_a"

    result = classify_membership_revisions([enriched, bare])
    assert result.accepted_raw_ids == ("raw-b", "raw-a")


def test_refuses_conflicting_bytes_under_one_attachment_identity() -> None:
    """Two sources disagreeing about one attachment's content is a conflict."""
    left = _attachment_revision("raw-a", acquired=True, inline=b"left bytes")
    right = _attachment_revision("raw-b", acquired=True, inline=b"right bytes")

    assert _relation(left.projection, right.projection) == "conflict"

    result = classify_membership_revisions([left, right])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b")


def test_attachment_acquisition_does_change_the_session_content_hash() -> None:
    """Acquisition is invisible to *revision comparison*, not to *idempotency*."""
    bare = _attachment_revision("raw-a", acquired=False)
    enriched = _attachment_revision("raw-b", acquired=True)

    assert bare.projection.session_hash != enriched.projection.session_hash
    assert bare.projection.attachment_identities == enriched.projection.attachment_identities
    assert bare.projection.attachment_contents == frozenset()
    assert len(enriched.projection.attachment_contents) == 1


def test_accepts_real_and_synthetic_id_variants_of_the_same_attachment_as_equivalent() -> None:
    """Same physical attachment, a real id on one export vintage, none on the other.

    Attachment identity is content-derived (anchoring message, name, media
    type) and never the provider's own attachment id (polylogue-d8al,
    polylogue-hith): Claude.ai does not consistently emit a real attachment
    id across separate export requests of the same conversation. No
    id-minting scheme can make a real id and a synthetic hash collide, so
    the id was never part of identity to begin with.
    """
    real_id = _attachment_revision("raw-real", acquired=False, provider_attachment_id="e950263f-51d1-4b7a-9c2e-0")
    synthetic_id = _attachment_revision("raw-synthetic", acquired=False, provider_attachment_id="att-ce21cd12d650")

    assert real_id.projection.attachment_identities == synthetic_id.projection.attachment_identities
    assert _relation(real_id.projection, synthetic_id.projection) == "equal"

    result = classify_membership_revisions([real_id, synthetic_id])

    assert result.accepted_raw_ids == ("raw-real",)
    assert result.equivalent_raw_ids == ("raw-synthetic",)
    assert result.ambiguous_raw_ids == ()


def test_refuses_to_guess_when_two_distinct_attachments_share_message_name_and_type() -> None:
    """Two distinct attachments sharing (message, name, media type) are indistinguishable.

    This is the documented, accepted limit (polylogue-d8al, polylogue-aggz):
    with no bytes on either side, there is no signal left to tell two
    same-metadata attachments apart, so they collapse to ONE identity by
    construction. Stated plainly rather than engineered around.
    """
    a = _attachment_revision("raw-a", acquired=False, provider_attachment_id="uuid-1")
    b = _attachment_revision("raw-b", acquired=False, provider_attachment_id="uuid-2")
    assert a.projection.attachment_identities == b.projection.attachment_identities


# ---------------------------------------------------------------------------
# Events: order-insensitive, measurement-excluded set containment
# (polylogue-nuec)
# ---------------------------------------------------------------------------


def _generation_lifecycle_revision(raw_id: str, elapsed_duration_ms: int) -> MembershipRevision:
    """A ChatGPT-shaped session with one `generation_lifecycle` event."""
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
        session_events=[
            ParsedSessionEvent(
                event_type="generation_lifecycle",
                timestamp=str(elapsed_duration_ms / 1000),
                source_message_provider_id="0",
                payload={
                    "state": "completed",
                    "evidence_source": "provider_native",
                    "fidelity": "exact",
                    "duration_semantics": "provider_reported_elapsed",
                    "elapsed_duration_ms": elapsed_duration_ms,
                },
            )
        ],
    )
    return MembershipRevision(raw_id, session_revision_projection(session))


def test_accepts_generation_lifecycle_duration_change_as_equivalent() -> None:
    """Byte-identical transcript, only a provider-remeasured duration differs.

    Reproduces the shape measured on the live archive: 33 of 35 sampled
    chatgpt-export ambiguous cohorts had identical messages and attachments
    but a `generation_lifecycle` event whose `elapsed_duration_ms` varied
    non-monotonically between export requests for the SAME generation
    (polylogue-nuec). Excluded from ``event_contents`` by an explicit
    per-event-type ALLOWLIST of content-bearing fields, not a denylist of
    fields discovered volatile after the fact.
    """
    first_export = _generation_lifecycle_revision("raw-first", 13000)
    second_export = _generation_lifecycle_revision("raw-second", 21000)

    assert first_export.projection.event_hashes != second_export.projection.event_hashes
    assert first_export.projection.event_contents == second_export.projection.event_contents
    assert first_export.projection.session_hash != second_export.projection.session_hash
    assert _relation(first_export.projection, second_export.projection) == "equal"

    result = classify_membership_revisions([first_export, second_export])

    assert result.accepted_raw_ids == ("raw-first",)
    assert result.equivalent_raw_ids == ("raw-second",)
    assert result.ambiguous_raw_ids == ()


def test_generation_lifecycle_duration_change_does_change_the_session_content_hash() -> None:
    """Measurement tolerance must not leak into idempotency."""
    first_export = _generation_lifecycle_revision("raw-first", 13000)
    second_export = _generation_lifecycle_revision("raw-second", 21000)
    assert first_export.projection.session_hash != second_export.projection.session_hash


def test_refuses_generation_lifecycle_state_change_despite_duration_tolerance() -> None:
    """Duration tolerance must not launder every field of the event as noise.

    Only the allowlisted fields (state, evidence_source, fidelity) are
    content -- a genuinely different ``state`` on the same event is real
    divergence.
    """
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
        session_events=[
            ParsedSessionEvent(
                event_type="generation_lifecycle",
                timestamp="13.0",
                source_message_provider_id="0",
                payload={
                    "state": "completed",
                    "evidence_source": "provider_native",
                    "fidelity": "exact",
                    "duration_semantics": "provider_reported_elapsed",
                    "elapsed_duration_ms": 13000,
                },
            )
        ],
    )
    changed_state = session.model_copy(
        update={
            "session_events": [
                ParsedSessionEvent(
                    event_type="generation_lifecycle",
                    timestamp="13.0",
                    source_message_provider_id="0",
                    payload={
                        "state": "in_progress",
                        "evidence_source": "provider_native",
                        "fidelity": "exact",
                        "duration_semantics": "provider_reported_elapsed",
                        "elapsed_duration_ms": 13000,
                    },
                )
            ]
        }
    )
    older = MembershipRevision("raw-old", session_revision_projection(session))
    newer = MembershipRevision("raw-new", session_revision_projection(changed_state))

    assert older.projection.event_contents != newer.projection.event_contents
    assert _relation(older.projection, newer.projection) == "conflict"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_non_allowlisted_event_type_keeps_its_full_payload_as_content() -> None:
    """The allowlist is scoped to `generation_lifecycle` -- an unrelated event
    type with no registered allowlist compares its FULL payload, so a real
    difference there still counts as divergence.
    """
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
        session_events=[
            ParsedSessionEvent(
                event_type="chatgpt_block_metadata",
                timestamp="13.0",
                source_message_provider_id="0",
                payload={"elapsed_duration_ms": 13000},
            )
        ],
    )
    other_value = session.model_copy(
        update={
            "session_events": [
                ParsedSessionEvent(
                    event_type="chatgpt_block_metadata",
                    timestamp="13.0",
                    source_message_provider_id="0",
                    payload={"elapsed_duration_ms": 21000},
                )
            ]
        }
    )
    older = MembershipRevision("raw-old", session_revision_projection(session))
    newer = MembershipRevision("raw-new", session_revision_projection(other_value))

    assert older.projection.event_contents != newer.projection.event_contents
    assert _relation(older.projection, newer.projection) == "conflict"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_accepts_reordered_events_with_identical_content_as_equivalent() -> None:
    """Same three durations, different array positions -- observed directly on
    the live archive for `generation_lifecycle` events across two ChatGPT
    export requests of the SAME conversation (polylogue-nuec).
    """

    def session_with(order: list[int]) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="session",
            messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
            session_events=[
                ParsedSessionEvent(
                    event_type="generation_lifecycle",
                    timestamp=str(duration / 1000),
                    source_message_provider_id=f"m{duration}",
                    payload={
                        "state": "completed",
                        "evidence_source": "provider_native",
                        "fidelity": "exact",
                        "duration_semantics": "provider_reported_elapsed",
                        "elapsed_duration_ms": duration,
                    },
                )
                for duration in order
            ],
        )

    forward = MembershipRevision("raw-forward", session_revision_projection(session_with([161000, 79000, 115000])))
    shuffled = MembershipRevision("raw-shuffled", session_revision_projection(session_with([79000, 115000, 161000])))

    assert forward.projection.event_hashes != shuffled.projection.event_hashes
    assert forward.projection.event_contents == shuffled.projection.event_contents
    assert _relation(forward.projection, shuffled.projection) == "equal"


def test_refuses_event_growth_that_loses_an_existing_event() -> None:
    """A genuinely appended event alongside a genuinely lost one is a fork."""
    shared_event = ParsedSessionEvent(
        event_type="generation_lifecycle",
        timestamp="13.0",
        source_message_provider_id="0",
        payload={
            "state": "completed",
            "evidence_source": "provider_native",
            "fidelity": "exact",
            "duration_semantics": "provider_reported_elapsed",
            "elapsed_duration_ms": 13000,
        },
    )
    other_event = ParsedSessionEvent(
        event_type="chatgpt_block_metadata",
        timestamp="1.0",
        source_message_provider_id="0",
        payload={"block_index": 0},
    )
    older_session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
        session_events=[shared_event],
    )
    newer_session = older_session.model_copy(update={"session_events": [other_event]})

    older = MembershipRevision("raw-old", session_revision_projection(older_session))
    newer = MembershipRevision("raw-new", session_revision_projection(newer_session))

    assert _relation(older.projection, newer.projection) == "conflict"

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_disambiguates_multiple_same_type_events_on_one_message_by_content() -> None:
    """Multiple same-type events on one message (e.g. one chatgpt_block_metadata
    per block) are distinguished by their OWN content, never by array
    position.
    """

    def block_event(index: int) -> ParsedSessionEvent:
        return ParsedSessionEvent(
            event_type="chatgpt_block_metadata",
            timestamp="1.0",
            source_message_provider_id="0",
            payload={"block_index": index},
        )

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.ASSISTANT, text="answer")],
        session_events=[block_event(0), block_event(1)],
    )
    reordered = session.model_copy(update={"session_events": [block_event(1), block_event(0)]})

    older = MembershipRevision("raw-old", session_revision_projection(session))
    newer = MembershipRevision("raw-new", session_revision_projection(reordered))

    assert len(older.projection.event_contents) == 2
    assert older.projection.event_contents == newer.projection.event_contents
    assert _relation(older.projection, newer.projection) == "equal"


# ---------------------------------------------------------------------------
# Cross-axis fork detection
# ---------------------------------------------------------------------------


def test_conflict_on_one_axis_forces_overall_conflict_even_when_others_agree() -> None:
    """A single-axis conflict (here: attachments) must not be masked by the
    other two axes agreeing.
    """
    left = _attachment_revision("raw-a", acquired=True, inline=b"left bytes")
    right = _attachment_revision("raw-b", acquired=True, inline=b"right bytes")
    # Messages and events are identical (both sessions built the same way);
    # only the attachment bytes conflict.
    assert left.projection.message_contents == right.projection.message_contents
    assert _relation(left.projection, right.projection) == "conflict"


def test_mixed_direction_axes_is_a_conflict_not_a_pick() -> None:
    """One axis growing forward while another grows backward is a fork.

    raw-a has an extra message but is missing an attachment raw-b has --
    neither strictly contains the other overall, even though each
    individual axis alone looks like ordinary growth in ONE direction.
    """
    attachment = ParsedAttachment(
        provider_attachment_id="uuid-1",
        message_provider_id="0",
        name="a.png",
        mime_type="image/png",
    )
    session_a = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[
            ParsedMessage(provider_message_id="0", role=Role.USER, text="one"),
            ParsedMessage(provider_message_id="1", role=Role.ASSISTANT, text="two"),
        ],
        attachments=[],
    )
    session_b = session_a.model_copy(update={"messages": session_a.messages[:1], "attachments": [attachment]})

    a = MembershipRevision("raw-a", session_revision_projection(session_a))
    b = MembershipRevision("raw-b", session_revision_projection(session_b))

    assert _relation(a.projection, b.projection) == "conflict"

    result = classify_membership_revisions([a, b])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b")


# ---------------------------------------------------------------------------
# Browser-capture fidelity ordering and direct-export precedence (unchanged
# escape hatches -- see _provider_ordered_browser_snapshots's docstring for
# why this one survives polylogue-aggz's content-only model)
# ---------------------------------------------------------------------------


def test_browser_native_snapshot_accepts_later_provider_revision_when_messages_reorder() -> None:
    older = _revision("raw-old", "prompt", "attachment context")
    newer = _revision("raw-new", "prompt", "tool result", "attachment context")
    older = MembershipRevision(
        older.raw_id,
        older.projection,
        "2026-01-01T00:00:00Z",
        observed_at_ms=1,
        browser_snapshot_fidelity="native",
        provider_message_ids=frozenset({"prompt", "attachment"}),
    )
    newer = MembershipRevision(
        newer.raw_id,
        newer.projection,
        "2026-01-01T00:01:00Z",
        observed_at_ms=2,
        browser_snapshot_fidelity="native",
        provider_message_ids=frozenset({"prompt", "tool", "attachment"}),
    )

    result = classify_membership_revisions([newer, older])

    assert result.accepted_raw_ids == ("raw-old", "raw-new")
    assert result.ambiguous_raw_ids == ()


def test_browser_native_snapshot_refuses_later_revision_that_loses_message_identity() -> None:
    older = _revision("raw-old", "prompt", "left")
    newer = _revision("raw-new", "prompt", "right")
    revisions = [
        MembershipRevision(
            older.raw_id,
            older.projection,
            "2026-01-01T00:00:00Z",
            observed_at_ms=1,
            browser_snapshot_fidelity="native",
            provider_message_ids=frozenset({"prompt", "left"}),
        ),
        MembershipRevision(
            newer.raw_id,
            newer.projection,
            "2026-01-01T00:01:00Z",
            observed_at_ms=2,
            browser_snapshot_fidelity="native",
            provider_message_ids=frozenset({"prompt", "right"}),
        ),
    ]

    result = classify_membership_revisions(revisions)

    # Genuine divergence (disjoint message identities) stays quarantined
    # ambiguous -- neither the browser-fidelity ordering nor direct-export
    # precedence can order it.
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_browser_native_upgrade_refuses_any_shrinking_frontier_dimension() -> None:
    older = _revision("raw-old", "prompt")
    older_projection = older.projection.__class__(
        session_hash=b"o" * 32,
        message_hashes=older.projection.message_hashes,
        message_contents=older.projection.message_contents,
        event_hashes=older.projection.event_hashes,
        event_contents=older.projection.event_contents,
        attachment_identities=frozenset({b"attachment"}),
        attachment_contents=frozenset(),
    )
    newer = _revision("raw-new", "prompt", "answer")
    revisions = [
        MembershipRevision(
            older.raw_id,
            older_projection,
            "2026-01-01T00:00:00Z",
            observed_at_ms=1,
            browser_snapshot_fidelity="dom",
        ),
        MembershipRevision(
            newer.raw_id,
            newer.projection,
            "2026-01-01T00:01:00Z",
            observed_at_ms=2,
            browser_snapshot_fidelity="native",
        ),
    ]

    result = classify_membership_revisions(revisions)

    # Mixed-direction axes (newer gained a message, older's-only attachment
    # is now missing) is a fork -- neither browser-fidelity ordering (the
    # dom->native frontier check requires every dimension to be >=) nor
    # direct-export precedence can order it, so it stays quarantined
    # ambiguous.
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_direct_export_outranks_browser_capture_siblings_regardless_of_growth() -> None:
    """A genuine non-browser-capture revision always wins over dom/native
    browser-capture siblings, even though its content is neither a
    growth superset of them nor resolvable by provider-timestamp comparison.
    Browser capture exists to backfill a session before its paired direct/
    native provider export shows up, never to compete with or shadow that
    export once it arrives (polylogue-z1c6)."""
    direct = _revision("raw-direct", "real one", "real two", "real three")
    dom = MembershipRevision(
        "raw-dom",
        _revision("raw-dom", "dom-only-turn").projection,
        "2026-07-04T09:55:00Z",
        browser_snapshot_fidelity="dom",
    )
    native = MembershipRevision(
        "raw-native",
        _revision("raw-native", "native-turn-a", "native-turn-b").projection,
        "2026-07-04T09:54:00Z",
        browser_snapshot_fidelity="native",
    )

    result = classify_membership_revisions([dom, native, direct])

    assert result.accepted_raw_ids == ("raw-direct",)
    assert result.equivalent_raw_ids == ("raw-dom", "raw-native")
    assert result.ambiguous_raw_ids == ()


def test_browser_snapshot_accepts_later_attachment_enrichment_without_provider_update() -> None:
    older = _revision("raw-old", "prompt", "answer")
    newer = _revision("raw-new", "prompt", "answer")
    newer_projection = newer.projection.__class__(
        session_hash=b"n" * 32,
        message_hashes=newer.projection.message_hashes,
        message_contents=newer.projection.message_contents,
        event_hashes=newer.projection.event_hashes,
        event_contents=newer.projection.event_contents,
        attachment_identities=frozenset({b"attachment-v2"}),
        attachment_contents=frozenset(),
    )
    revisions = [
        MembershipRevision(
            older.raw_id,
            older.projection,
            "2026-01-01T00:01:00Z",
            observed_at_ms=1,
            browser_snapshot_fidelity="native",
            provider_message_ids=frozenset({"prompt", "answer"}),
            provider_attachment_ids=frozenset({"asset"}),
        ),
        MembershipRevision(
            newer.raw_id,
            newer_projection,
            "2026-01-01T00:01:00Z",
            observed_at_ms=2,
            browser_snapshot_fidelity="native",
            provider_message_ids=frozenset({"prompt", "answer"}),
            provider_attachment_ids=frozenset({"asset"}),
        ),
    ]

    result = classify_membership_revisions(revisions)

    assert result.accepted_raw_ids == ("raw-old", "raw-new")
    assert result.ambiguous_raw_ids == ()
