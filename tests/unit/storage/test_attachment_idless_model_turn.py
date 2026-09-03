"""A model turn with no provider id must still produce attachment provenance.

``derive_attachment_provenance`` yields ``("model_output", None)`` when the
owning turn carries no provider-assigned id, because the parser has no
identity to name at parse time. The write boundary rejects exactly that
combination, so one id-less model turn fails the whole session's write.
"""

from __future__ import annotations

from polylogue.core.enums import Role
from polylogue.sources.parsers.base_models import ParsedAttachment, ParsedMessage
from polylogue.sources.parsers.base_support import derive_attachment_provenance
from polylogue.storage.sqlite.archive_tiers.write import _attachment_provenance


def test_parser_derivation_still_has_no_identity_to_name() -> None:
    """Pins the upstream shape this fix exists to absorb.

    Anti-vacuity: if ``derive_attachment_provenance`` ever gains a producer
    for an id-less model turn, this assertion goes red and the write-boundary
    fallback below is no longer the only thing standing between that shape and
    a rejected session.
    """
    assert derive_attachment_provenance(Role.ASSISTANT, None) == ("model_output", None)


def test_declared_model_output_without_producer_uses_the_stored_message_id() -> None:
    """A parser-declared model_output with no producer is completed, not rejected.

    ``attachment.direction`` is already set, so the role-derived path is never
    reached; the stored message id is the identity the parser could not name.

    Anti-vacuity: dropping the ``producer_fallback`` assignment on the
    ``attachment.direction is not None`` branch of ``_attachment_provenance``
    returns ``None`` for the producer, turning the second assertion red -- and
    that value is what ``_write_attachments`` raises ``model_output attachment
    requires producer provenance`` on.
    """
    attachment = ParsedAttachment(provider_attachment_id="att-1", direction="model_output", producer_ref=None)

    direction, producer_ref = _attachment_provenance(attachment, None, resolved_message_id="origin:s1:4.0")

    assert direction == "model_output"
    assert producer_ref == "message:origin:s1:4.0"


def test_role_derived_model_output_without_producer_uses_the_stored_message_id() -> None:
    """The same completion applies to provenance derived from the owning turn.

    ``ParsedMessage.provider_message_id`` is a required ``str``, so the falsy
    id this branch can actually see is the empty one rather than ``None``.

    Anti-vacuity: dropping the ``producer_fallback`` assignment on the derived
    branch returns ``None`` for the producer and turns the second assertion
    red.
    """
    attachment = ParsedAttachment(provider_attachment_id="att-2")
    owning = ParsedMessage(role=Role.ASSISTANT, provider_message_id="")

    direction, producer_ref = _attachment_provenance(attachment, owning, resolved_message_id="origin:s1:7.0")

    assert direction == "model_output"
    assert producer_ref == "message:origin:s1:7.0"


def test_parser_supplied_producer_is_never_overwritten() -> None:
    """The fallback completes missing provenance; it never replaces real evidence.

    Anti-vacuity: making the fallback unconditional (assigning it regardless of
    ``producer_ref``) overwrites the parser's own producer and turns this
    assertion red.
    """
    attachment = ParsedAttachment(
        provider_attachment_id="att-3",
        direction="model_output",
        producer_ref="message:parser-supplied",
    )

    _direction, producer_ref = _attachment_provenance(attachment, None, resolved_message_id="origin:s1:9.0")

    assert producer_ref == "message:parser-supplied"


def test_user_input_attachments_keep_their_absent_producer() -> None:
    """A user_input attachment has no producer by contract and gains none here.

    Anti-vacuity: applying the fallback to every direction rather than only
    ``model_output`` produces a producer for a user upload and turns this
    assertion red.
    """
    attachment = ParsedAttachment(provider_attachment_id="att-4", direction="user_input")

    direction, producer_ref = _attachment_provenance(attachment, None, resolved_message_id="origin:s1:2.0")

    assert direction == "user_input"
    assert producer_ref is None
