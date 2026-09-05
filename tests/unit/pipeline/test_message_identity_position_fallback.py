"""Revision-match identity and private owner coordinates must not read array position.

polylogue-gysk3: the attachment-class bug (a synthetic identity seeded partly
by array index, unstable across export vintages that reorder/insert array
entries) was found and fixed for attachments (polylogue-hith/-d8al) by
excluding the (real-or-synthetic) attachment id from identity entirely and
deriving it from stable content fields instead (message_id, name, mime_type).

The revision-match identity is now separate from mutable message content:
timestamped id-less edits share one axis while their content hash changes.
Private attachment ownership uses the typed coordinate contract and fails
closed for indistinguishable duplicate occurrences.
"""

from __future__ import annotations

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import MembershipRevision, _relation, classify_membership_revisions
from polylogue.core.enums import BlockType, Provider
from polylogue.core.message_owner import MessageOwnerAmbiguityError, MessageOwnerCoordinate
from polylogue.pipeline.ids import (
    attachment_message_owner_key,
    message_owner_resolution,
    session_content_hash,
    session_revision_projection,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession


def _session(messages: list[ParsedMessage], attachments: list[ParsedAttachment] | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=messages,
        attachments=attachments or [],
    )


def _with_id(provider_message_id: str, role: str, text: str, timestamp: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id, role=Role.normalize(role), text=text, timestamp=timestamp
    )


def _id_less(role: str, text: str, timestamp: str | None) -> ParsedMessage:
    """A message the parser could not assign a native id to (empty string)."""
    return ParsedMessage(provider_message_id="", role=Role.normalize(role), text=text, timestamp=timestamp)


def test_id_less_message_content_identity_is_stable_across_reorder() -> None:
    """Same message set, different array order -- an id-less message with a
    provider timestamp must still resolve to the same (identity, content)
    pair in both orderings, not shift with its array position.
    """
    anchored = _id_less("assistant", "hello there", "2024-01-01T00:01:00Z")
    keyed = _with_id("m1", "user", "hi", "2024-01-01T00:00:00Z")

    forward = session_revision_projection(_session([keyed, anchored]))
    reordered = session_revision_projection(_session([anchored, keyed]))

    assert forward.message_contents == reordered.message_contents


def test_id_less_messages_with_distinct_timestamps_do_not_collide() -> None:
    """Two different id-less messages must not collapse to one identity."""
    a = _id_less("assistant", "first", "2024-01-01T00:01:00Z")
    b = _id_less("assistant", "second", "2024-01-01T00:02:00Z")

    projection = session_revision_projection(_session([a, b]))

    assert len(projection.message_contents) == 2


def test_timestamped_idless_edit_shares_revision_identity_but_changes_content() -> None:
    older = _id_less("assistant", "before", "2024-01-01T00:01:00Z")
    edited = _id_less("assistant", "after", "2024-01-01T00:01:00Z")

    old_projection = session_revision_projection(_session([older]))
    edited_projection = session_revision_projection(_session([edited]))

    old_identity, old_content, _ = next(iter(old_projection.message_contents))
    edited_identity, edited_content, _ = next(iter(edited_projection.message_contents))
    assert old_identity == edited_identity
    assert old_content != edited_content
    assert old_projection.session_hash != edited_projection.session_hash


def test_timestamped_idless_sibling_edit_is_not_a_membership_conflict() -> None:
    """A same-role, same-timestamp sibling set has one mutable revision axis.

    The content hash still changes, so the real writer replaces the stored
    payload. Membership must compare the axis cardinality instead of treating
    the sibling's changed text as a contradictory native identity.
    """
    older = _session(
        [
            _id_less("assistant", "first", "2024-01-01T00:01:00Z"),
            _id_less("assistant", "second", "2024-01-01T00:01:00Z"),
        ]
    )
    edited = _session(
        [
            _id_less("assistant", "edited first", "2024-01-01T00:01:00Z"),
            _id_less("assistant", "second", "2024-01-01T00:01:00Z"),
        ]
    )
    older_projection = session_revision_projection(older)
    edited_projection = session_revision_projection(edited)

    assert older_projection.session_hash != edited_projection.session_hash
    assert _relation(older_projection, edited_projection) == "equal"


def test_whitespace_native_id_uses_the_same_revision_axis_as_missing_id() -> None:
    whitespace = _session(
        [ParsedMessage(provider_message_id="  ", role=Role.ASSISTANT, text="same", timestamp="2024-01-01")]
    )
    missing = _session([_id_less("assistant", "same", "2024-01-01")])

    assert (
        session_revision_projection(whitespace).message_contents
        == session_revision_projection(missing).message_contents
    )


def test_duplicate_physical_owner_coordinate_fails_closed_before_hashing_attachment() -> None:
    messages = [
        _id_less("assistant", "first", None).model_copy(update={"position": 3, "variant_index": 0}),
        _id_less("assistant", "second", None).model_copy(update={"position": 3, "variant_index": 0}),
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="attachment-1",
        message_provider_id="",
        message_position=3,
        message_variant_index=0,
        name="note.txt",
        mime_type="text/plain",
    )

    with pytest.raises(MessageOwnerAmbiguityError):
        session_revision_projection(_session(messages, [attachment]))


def test_unique_stable_owner_evidence_precedes_mutable_content() -> None:
    def session(first_text: str) -> ParsedSession:
        messages = [
            _id_less("assistant", first_text, "2024-01-01T00:01:00Z").model_copy(
                update={
                    "position": 0,
                    "owner_coordinate": MessageOwnerCoordinate("owner-first", 0, 0),
                }
            ),
            _id_less("assistant", "second", "2024-01-01T00:01:00Z").model_copy(
                update={
                    "position": 1,
                    "owner_coordinate": MessageOwnerCoordinate("owner-second", 1, 0),
                }
            ),
        ]
        return _session(
            messages,
            [
                ParsedAttachment(
                    provider_attachment_id="owner-first-attachment",
                    message_provider_id="",
                    message_position=0,
                    message_variant_index=0,
                    owner_coordinate=MessageOwnerCoordinate("owner-first", 0, 0),
                    name="first.txt",
                    mime_type="text/plain",
                )
            ],
        )

    older = session("before")
    edited = session("after")
    older_projection = session_revision_projection(older)
    edited_projection = session_revision_projection(edited)

    assert older_projection.attachment_identities == edited_projection.attachment_identities
    assert _relation(older_projection, edited_projection) == "equal"


def test_duplicate_native_id_at_duplicate_physical_coordinate_fails_closed() -> None:
    messages = [
        ParsedMessage(
            provider_message_id="duplicate-native",
            role=Role.ASSISTANT,
            text="first",
            position=3,
            variant_index=0,
        ),
        ParsedMessage(
            provider_message_id="duplicate-native",
            role=Role.ASSISTANT,
            text="second",
            position=3,
            variant_index=0,
        ),
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="attachment-duplicate-native",
        message_provider_id="duplicate-native",
        message_position=3,
        message_variant_index=0,
        name="note.txt",
        mime_type="text/plain",
    )

    with pytest.raises(MessageOwnerAmbiguityError):
        session_revision_projection(_session(messages, [attachment]))


def test_id_less_and_timestamp_less_message_uses_content_anchor() -> None:
    bare = _id_less("user", "x", None)
    keyed = _with_id("m1", "assistant", "y", "2024-01-01T00:00:00Z")

    forward = session_revision_projection(_session([keyed, bare]))
    reordered = session_revision_projection(_session([bare, keyed]))

    assert forward.message_contents == reordered.message_contents


def test_timestamp_less_idless_duplicates_preserve_unordered_multiplicity() -> None:
    repeated = _id_less("assistant", "repeat", None)
    one = session_revision_projection(_session([repeated]))
    two = session_revision_projection(_session([repeated, repeated]))

    assert sum(count for _identity, _content, count in one.message_contents) == 1
    assert sum(count for _identity, _content, count in two.message_contents) == 2

    classification = classify_membership_revisions(
        [MembershipRevision(raw_id="one", projection=one), MembershipRevision(raw_id="two", projection=two)]
    )
    assert classification.accepted_raw_ids == ("one", "two")
    assert not classification.equivalent_raw_ids
    assert not classification.ambiguous_raw_ids


def test_indistinguishable_duplicate_idless_attachment_owner_fails_closed() -> None:
    """A duplicate with no stable evidence cannot receive guessed ownership."""
    repeated = [
        _id_less("assistant", "repeat", "2024-01-01T00:00:00Z").model_copy(update={"position": position})
        for position in (0, 1)
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="drive-doc",
        message_provider_id="",
        message_position=0,
        name="note.txt",
        mime_type="text/plain",
    )
    with pytest.raises(MessageOwnerAmbiguityError):
        session_revision_projection(_session(repeated, [attachment]))


def _document_only(reference_id: str, position: int) -> ParsedMessage:
    """An id-less, timestamp-less, text-less turn whose only content is a document reference."""
    return ParsedMessage(
        provider_message_id="",
        role=Role.USER,
        text=None,
        timestamp=None,
        position=position,
        blocks=[
            ParsedContentBlock(
                type=BlockType.DOCUMENT,
                media_type="text/plain",
                metadata={"driveDocument": {"id": reference_id}},
            )
        ],
    )


def _document_attachment(reference_id: str, position: int) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id=reference_id,
        message_provider_id="",
        message_position=position,
        message_variant_index=0,
        name="shared-name.txt",
        mime_type="text/plain",
    )


def test_duplicate_idless_turns_distinguished_by_referenced_document_own_their_attachments() -> None:
    """Block reference identity is content: turns citing different documents are different turns.

    Anti-vacuity: dropping ``metadata`` from
    ``_HASHED_FIELDS["ParsedContentBlock"]`` makes both attachments raise
    ``MessageOwnerAmbiguityError`` here, because the turns share role,
    timestamp, text, and block type.
    """
    messages = [_document_only("doc-a", 0), _document_only("doc-b", 1)]
    attachments = [_document_attachment("doc-a", 0), _document_attachment("doc-b", 1)]

    resolution = message_owner_resolution(messages)
    assert not resolution.ambiguous_keys
    owner_keys = [attachment_message_owner_key(attachment, resolution) for attachment in attachments]
    assert owner_keys == list(resolution.keys)

    projection = session_revision_projection(_session(messages, attachments))
    assert len(projection.attachment_identities) == 2


def test_duplicate_idless_turns_with_identical_document_reference_still_fail_closed() -> None:
    """Two turns citing the same document with nothing else differing stay ambiguous."""
    messages = [_document_only("doc-a", 0), _document_only("doc-a", 1)]
    attachment = _document_attachment("doc-a", 0)

    with pytest.raises(MessageOwnerAmbiguityError):
        session_revision_projection(_session(messages, [attachment]))


def test_session_content_hash_degrades_ambiguous_attachment_to_unowned() -> None:
    """A parseable session must survive an attachment owner ambiguity.

    The strict revision projection still rejects ambiguous ownership. The
    ingest content hash instead records the attachment without a guessed
    message anchor, allowing the writer to retain the session and typed
    attachment evidence.
    """
    repeated = [
        _id_less("assistant", "repeat", "2024-01-01T00:00:00Z").model_copy(update={"position": position})
        for position in (0, 1)
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="drive-doc",
        message_provider_id="",
        message_position=0,
        name="note.txt",
        mime_type="text/plain",
    )

    assert session_content_hash(_session(repeated, [attachment]))


def test_duplicate_stable_owner_evidence_without_physical_coordinate_fails_closed() -> None:
    messages = [
        _id_less("assistant", "first", "2024-01-01T00:01:00Z").model_copy(
            update={
                "position": 4,
                "owner_coordinate": MessageOwnerCoordinate("same-owner-evidence", 4, 0),
            }
        ),
        _id_less("assistant", "second", "2024-01-01T00:01:00Z").model_copy(
            update={
                "position": 4,
                "owner_coordinate": MessageOwnerCoordinate("same-owner-evidence", 4, 1),
            }
        ),
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="shared-owner-attachment",
        message_provider_id="",
        owner_coordinate=MessageOwnerCoordinate("same-owner-evidence"),
        name="note.txt",
        mime_type="text/plain",
    )

    with pytest.raises(MessageOwnerAmbiguityError):
        session_revision_projection(_session(messages, [attachment]))
