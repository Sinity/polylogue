from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import (
    MembershipRevision,
    _strictly_dominates,
    classify_membership_revisions,
)
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession


def _revision(raw_id: str, *texts: str) -> MembershipRevision:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id=str(i), role=Role.USER, text=text) for i, text in enumerate(texts)],
    )
    return MembershipRevision(raw_id, session_revision_projection(session))


def test_classifies_strict_growth_and_semantic_equivalence() -> None:
    result = classify_membership_revisions(
        [_revision("raw-b", "one", "two"), _revision("raw-z", "one"), _revision("raw-a", "one")]
    )
    assert result.accepted_raw_ids == ("raw-a", "raw-b")
    assert result.equivalent_raw_ids == ("raw-z",)
    assert result.ambiguous_raw_ids == ()


def test_refuses_divergent_maxima() -> None:
    result = classify_membership_revisions(
        [_revision("raw-a", "one"), _revision("raw-b", "one", "left"), _revision("raw-c", "one", "right")]
    )
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b", "raw-c")


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


def test_metadata_revision_without_complete_provider_time_is_ambiguous() -> None:
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

    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_metadata_revisions_with_equal_provider_time_are_ambiguous() -> None:
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

    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b")


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

    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_browser_native_upgrade_refuses_any_shrinking_frontier_dimension() -> None:
    older = _revision("raw-old", "prompt")
    older_projection = older.projection.__class__(
        session_hash=b"o" * 32,
        message_hashes=older.projection.message_hashes,
        event_hashes=older.projection.event_hashes,
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

    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-new", "raw-old")


def test_direct_export_outranks_browser_capture_siblings_regardless_of_growth() -> None:
    """A genuine non-browser-capture revision always wins over dom/native
    browser-capture siblings, even though its content is neither a byte/hash
    -growth superset of them nor resolvable by provider-timestamp comparison.
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
        event_hashes=newer.projection.event_hashes,
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


def _attachment_revision(
    raw_id: str,
    *,
    acquired: bool,
    provider_attachment_id: str = "drive-file-1",
    inline: bytes = b"attachment bytes",
) -> MembershipRevision:
    """A session whose single attachment is referenced, optionally with bytes read.

    Models the shape a lazily-fetched attachment actually produces: the provider
    emits a bare reference (no size, no bytes), and a later acquisition pass
    resolves the same reference into real bytes. The transcript is untouched
    either way -- only the attachment's acquisition state differs.
    """
    attachment = ParsedAttachment(
        provider_attachment_id=provider_attachment_id,
        message_provider_id="0",
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

    A Drive/Gemini document acquired twice -- once before its attachment bytes
    were fetched, once after -- has an identical transcript and an identical
    attachment reference. Folding acquisition state into the attachment identity
    hash made the two revisions look like equal-sized disjoint branches, so the
    whole cohort was quarantined as ambiguous and neither revision was indexed
    (polylogue-bu1i). Measured on the live archive: all 157 two-member
    aistudio-drive cohorts had exactly this shape.
    """
    bare = _attachment_revision("raw-bbbb", acquired=False)
    enriched = _attachment_revision("raw-aaaa", acquired=True)

    result = classify_membership_revisions([enriched, bare])

    # Ordering matters as much as acceptance: the revision holding the bytes must
    # be the head of the chain, or the archive indexes the emptier one. The raw
    # ids are chosen so plain lexical ordering would put the enriched revision
    # first, which is the direction the old frontier tie fell through to.
    assert result.accepted_raw_ids == ("raw-bbbb", "raw-aaaa")
    assert result.ambiguous_raw_ids == ()


def test_refuses_attachment_byte_loss_as_a_fidelity_downgrade() -> None:
    """Dropping bytes already in hand is never an upgrade, in either direction.

    Without this, an acquisition regression would look like ordinary growth
    running backwards and could be accepted, silently re-marking a fetched
    attachment as unfetched -- the concrete harm observed on five live sessions.
    """
    enriched = _attachment_revision("raw-a", acquired=True)
    bare = _attachment_revision("raw-b", acquired=False)

    result = classify_membership_revisions([enriched, bare])

    assert result.accepted_raw_ids == ("raw-b", "raw-a")

    # And the downgrade direction on its own is refused outright.
    assert not _strictly_dominates(enriched.projection, bare.projection)


def test_refuses_conflicting_bytes_under_one_attachment_identity() -> None:
    """Two sources disagreeing about one attachment's content stays ambiguous.

    Splitting identity from acquisition must not turn a genuine conflict into an
    upgrade: when both revisions have read bytes for the same attachment and the
    bytes differ, no ordering rule can decide which is authoritative.
    """
    left = _attachment_revision("raw-a", acquired=True, inline=b"left bytes")
    right = _attachment_revision("raw-b", acquired=True, inline=b"right bytes")

    result = classify_membership_revisions([left, right])

    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b")


def test_attachment_acquisition_does_change_the_session_content_hash() -> None:
    """Acquisition is invisible to *revision comparison*, not to *idempotency*.

    The split must not leak into ``session_hash``: if acquiring bytes did not
    change the session's content hash, a re-ingest carrying newly-fetched
    attachments would be skipped as unchanged and the bytes would never land.
    """
    bare = _attachment_revision("raw-a", acquired=False)
    enriched = _attachment_revision("raw-b", acquired=True)

    assert bare.projection.session_hash != enriched.projection.session_hash
    assert bare.projection.attachment_identities == enriched.projection.attachment_identities
    assert bare.projection.attachment_contents == frozenset()
    assert len(enriched.projection.attachment_contents) == 1


def test_refuses_contradicted_bytes_even_when_the_revision_otherwise_grew() -> None:
    """Growth elsewhere must not launder a contradiction about bytes already read.

    The equal-cardinality conflict case is already refused by the growth test
    itself, which leaves the byte-agreement check unexercised and therefore
    unproven. This is the shape that genuinely needs it: the newer revision adds
    a second attachment -- real growth on the identity axis -- while changing the
    bytes it reports for the attachment both revisions share. Accepting that
    would overwrite content already in hand with content from a disagreeing
    source, under cover of a legitimate-looking frontier advance.
    """
    shared_id = "drive-file-1"
    older = _attachment_revision("raw-a", acquired=True, provider_attachment_id=shared_id, inline=b"original bytes")

    contradicting = ParsedAttachment(
        provider_attachment_id=shared_id,
        message_provider_id="0",
        size_bytes=len(b"rewritten bytes"),
        inline_bytes=b"rewritten bytes",
    )
    added = ParsedAttachment(
        provider_attachment_id="drive-file-2",
        message_provider_id="0",
        size_bytes=4,
        inline_bytes=b"more",
    )
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="session",
        messages=[ParsedMessage(provider_message_id="0", role=Role.USER, text="one")],
        attachments=[contradicting, added],
    )
    newer = MembershipRevision("raw-b", session_revision_projection(session))

    # The identity axis really did grow, so the refusal cannot come from there.
    assert older.projection.attachment_identities < newer.projection.attachment_identities
    assert not _strictly_dominates(older.projection, newer.projection)

    result = classify_membership_revisions([older, newer])
    assert result.accepted_raw_ids == ()
    assert result.ambiguous_raw_ids == ("raw-a", "raw-b")
