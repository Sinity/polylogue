from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession


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
