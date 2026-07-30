"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from polylogue.core.enums import Origin, Provider
from polylogue.core.hashing import hash_bytes, hash_payload
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import ContentHash, MessageId, SessionEventId, SessionId

# ParsedMessage/ParsedSession/ParsedAttachment/ParsedContentBlock are used only
# as parameter/return type annotations below (never constructed or
# isinstance-checked here). Importing them eagerly forces the whole
# `polylogue.sources` package init -- including the Drive download subsystem
# -- onto every caller of this pure hashing/id module (polylogue-8s70: this
# was ~395ms of `polylogue.storage.repair`'s ~670ms import cost, the single
# largest contributor). TYPE_CHECKING-only keeps static typing intact while
# deferring the real import to whichever caller actually needs `sources`.
if TYPE_CHECKING:
    from polylogue.sources import ParsedMessage, ParsedSession
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedSessionEvent

# Sentinel values to distinguish None from empty in hash computations
_NULL_SENTINEL = "__POLYLOGUE_NULL__"
_EMPTY_SENTINEL = "__POLYLOGUE_EMPTY__"
HashScalar: TypeAlias = str | int | float | bool | None

#: One attachment's (strict identity, loose/id-independent identity, content
#: hash or ``None`` if unacquired) triple -- see ``SessionRevisionProjection``
#: for why the comparison layer needs both identities (polylogue-d8al).
AttachmentRecord: TypeAlias = tuple[bytes, bytes, bytes | None]


@dataclass(frozen=True, slots=True)
class SessionRevisionProjection:
    """Canonical content hashes used to prove append-only session growth.

    Attachments are projected on two axes deliberately, because folding them
    into one hash makes acquisition look like divergence. ``attachment_identities``
    answers *which attachment is this* (provider id, anchoring message, name,
    media type) and never changes once a provider has emitted the reference.
    ``attachment_contents`` answers *have we read its bytes yet, and which bytes*
    -- it carries an ``(identity, content)`` pair only for attachments whose
    bytes are actually in hand.

    A single hash over both axes made an ordinary lazy fetch indistinguishable
    from a branch: a Drive/Gemini document acquired before and after its
    attachment bytes were resolved produced equal-cardinality *disjoint* hash
    sets, so the later revision was neither a superset nor a prefix of the
    earlier one and the whole cohort was quarantined as ambiguous. Splitting the
    axes lets a dominance test say what is actually true -- same attachments,
    strictly more of their bytes now known (polylogue-bu1i).

    ``attachment_records`` carries the same per-attachment data as a
    ``(strict identity, loose identity, content)`` triple instead of two
    separate sets, so the *comparison layer* (``session_revision_membership.py``)
    can correlate attachments across two revisions even when a provider's
    export omits a stable id for the same physical attachment on a different
    export request of the same conversation -- one vintage has a real UUID,
    the other has none and a parser synthesizes one, and no synthetic-id
    scheme can make a real id and a synthetic hash collide by construction
    (polylogue-d8al). The "loose" identity drops the provider id and keeps
    only the anchoring message, name, and media type;
    ``attachment_identities``/``attachment_contents`` stay strict (id
    included) so this projection's own equality/hashing behavior is
    unchanged -- only the membership module's pairwise correlation consults
    the loose key, and only as a fallback when the strict identity does not
    match and the loose key is unambiguous on both sides being compared.
    Known, accepted limit stated explicitly rather than engineered around:
    two genuinely distinct attachments that share one message/name/media-type
    and carry no bytes on either side of a comparison are indistinguishable
    by any signal this projection can offer.

    Messages get an analogous split, for the same reason applied to a
    different volatility source: a provider's own export can replay an
    unchanged message set in a different array sequence across separate export
    requests (Claude.ai's own tree flattening is not guaranteed to serialize
    the same way twice). ``message_contents`` pairs each message's identity
    hash (its provider message id) with a hash of its content
    (role/text/timestamp/blocks) -- unlike an attachment, a message is never
    lazily fetched, so there is no separate identity-only axis to track:
    ``message_contents`` alone answers both *which messages exist* and *what
    do they say*. Order lives only in ``message_hashes`` (kept for
    ``session_hash`` and diagnostics) -- ``message_contents`` is
    order-insensitive by construction, so a bare permutation of the same
    id-to-content mapping is not read as divergence (polylogue-c429).

    ``event_identity_hashes`` strips designated provider-reported-measurement
    fields (see ``_PROVIDER_REPORTED_ELAPSED_VOLATILE_PAYLOAD_KEYS``) out of
    events whose own payload declares them non-durable
    (``duration_semantics == "provider_reported_elapsed"``) before hashing.
    ChatGPT's ``generation_lifecycle`` event re-derives its
    ``elapsed_duration_ms`` from the raw export's own timing metadata on every
    export request, and that value is not stable across requests for the same
    generation -- folding it into revision identity made byte-identical
    conversations look like divergent branches on every re-export
    (polylogue-nuec). ``event_hashes`` (unstripped, order-preserving) remains
    unchanged for ``session_hash`` and diagnostics.

    ``metadata_hash`` covers exactly the ``session_hash`` payload fields
    OTHER than messages/attachments/session_events -- title, created_at,
    updated_at -- normalized the same way. Membership classification uses it
    to tell apart two reasons a same-content-key group can still carry
    different ``session_hash`` values: a genuine title/timestamp edit (needs
    the existing provider-timestamp tie-break), versus pure serialization
    noise along an axis this projection already tolerates (message order,
    attachment id presence, event-duration measurement) with the provider's
    own metadata otherwise unchanged -- the common shape for a re-export of
    an untouched conversation, where ``updated_at`` legitimately never moves
    and a distinct-timestamp tie-break can never fire (polylogue-c429,
    polylogue-nuec, polylogue-d8al).
    """

    session_hash: bytes
    message_hashes: tuple[bytes, ...]
    message_contents: frozenset[tuple[bytes, bytes]]
    attachment_identities: frozenset[bytes]
    attachment_contents: frozenset[tuple[bytes, bytes]]
    attachment_records: tuple[AttachmentRecord, ...]
    event_hashes: tuple[bytes, ...]
    event_identity_hashes: tuple[bytes, ...]
    metadata_hash: bytes


def _normalize_nested_for_hash(value: object) -> object:
    """NFC-normalize every string inside a nested payload, recursively.

    ``_normalize_for_hash`` covers scalar fields, but two nested payloads --
    ``ParsedContentBlock.tool_input`` and ``ParsedSessionEvent.payload`` --
    were passed straight to ``hash_payload``, which by its own docstring does
    NOT normalize the strings it serializes. So a tool_use block or session
    event whose nested content differed only in Unicode normalization form
    hashed as two distinct logical identities and could never dedupe, while
    the very same text in ``message.text`` hashed identically.

    Measured before fixing: 0 of 20,000 sampled ``tool_use.tool_input`` rows
    in the live archive carry non-NFC content, so no stored hash changes and
    nothing needs re-hashing -- this closes a latent trap rather than
    repairing active corruption. Plausible future sources are macOS-originated
    exports (HFS+ historically stored NFD) and browser-capture DOM extraction.

    Dict keys are normalized as well as values: a key is just as capable of
    carrying an NFD form, and an un-normalized key would split the hash the
    same way.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", key) if isinstance(key, str) else key: _normalize_nested_for_hash(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_nested_for_hash(item) for item in value]
    return value


def _normalize_for_hash(value: HashScalar) -> JSONValue:
    """Normalize a value for hashing, distinguishing None from empty.

    Args:
        value: Hash-compatible scalar value to normalize.

    Returns:
        Normalized JSON value with None → _NULL_SENTINEL and "" → _EMPTY_SENTINEL.
    """
    if value is None:
        return _NULL_SENTINEL
    if value == "":
        return _EMPTY_SENTINEL
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def session_id(source_name: Provider | Origin | str, provider_session_id: str) -> SessionId:
    """Generate the archive session ID from source/provider input.

    Args:
        provider_session_id: Provider's session identifier.

    Returns:
        Formatted session ID.

    Raises:
        ValueError: If source_name or provider_session_id is empty.
    """
    source_text = str(source_name).strip()
    if source_text == "":
        raise ValueError("source_name cannot be empty")
    if not provider_session_id or not provider_session_id.strip():
        raise ValueError("provider_session_id cannot be empty")
    origin = origin_from_provider(source_name)
    return SessionId(f"{origin.value}:{provider_session_id}")


def message_id(session_id: SessionId, provider_message_id: str) -> MessageId:
    return MessageId(f"{session_id}:{provider_message_id}")


def session_event_id(session_id: SessionId, event_index: int) -> SessionEventId:
    return SessionEventId(f"{session_id}:session-event:{event_index:06d}")


def _content_block_payload(block: ParsedContentBlock) -> dict[str, JSONValue]:
    """Build a hash-stable payload for a single content block."""
    payload: dict[str, JSONValue] = {
        "type": str(block.type),
        "text": _normalize_for_hash(block.text),
    }
    if block.tool_name:
        payload["tool_name"] = _normalize_for_hash(block.tool_name)
    if block.tool_id:
        payload["tool_id"] = _normalize_for_hash(block.tool_id)
    if block.tool_input is not None:
        payload["tool_input"] = hash_payload(_normalize_nested_for_hash(dict(block.tool_input)))
    if block.media_type:
        payload["media_type"] = _normalize_for_hash(block.media_type)
    return payload


def _message_hash_payload(message: ParsedMessage, message_id: str) -> dict[str, JSONValue]:
    """Build the hash-stable payload for a single message."""
    payload: dict[str, JSONValue] = {
        "id": message_id,
        "role": str(message.role),
        "text": _normalize_for_hash(message.text),
        "timestamp": _normalize_for_hash(message.timestamp),
    }
    if message.blocks:
        payload["content_blocks"] = [_content_block_payload(b) for b in message.blocks]
    return payload


#: The one field of a message hash payload that answers *which message is
#: this*, as opposed to *what does it currently say*. A provider's own
#: message id is stable across re-exports even when the export's array
#: ordering is not (polylogue-c429).
_MESSAGE_IDENTITY_FIELDS = ("id",)


def _message_identity_payload(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project the order-independent identity of one message payload.

    Reads the already-normalized value out of ``_message_hash_payload`` rather
    than re-deriving it, mirroring ``_attachment_identity_payload``'s single
    normalization site.
    """
    return {field: payload[field] for field in _MESSAGE_IDENTITY_FIELDS}


#: Fields of an attachment hash payload that answer *which attachment is this*,
#: as opposed to *what have we managed to read about it*. ``size_bytes`` is
#: excluded on purpose: for lazily-fetched attachments (Drive/Gemini references,
#: browser capture) the provider states no size until the bytes are actually
#: read, so treating it as identity makes acquisition look like a different
#: attachment. ``inline_content_hash`` is excluded for the same reason and is
#: recovered separately as acquisition evidence.
_ATTACHMENT_IDENTITY_FIELDS = ("id", "message_id", "name", "mime_type")


def _attachment_hash_payload(attachment: ParsedAttachment) -> dict[str, JSONValue]:
    """Build attachment identity without perturbing legacy metadata-only hashes."""
    payload: dict[str, JSONValue] = {
        "id": _normalize_for_hash(attachment.provider_attachment_id),
        "message_id": _normalize_for_hash(attachment.message_provider_id),
        "name": _normalize_for_hash(attachment.name),
        "mime_type": _normalize_for_hash(attachment.mime_type),
        "size_bytes": _normalize_for_hash(attachment.size_bytes),
    }
    if attachment.inline_bytes is not None:
        payload["inline_content_hash"] = hash_bytes(attachment.inline_bytes)
    return payload


def _attachment_identity_payload(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project the acquisition-independent identity of one attachment payload.

    Reads the already-normalized values out of ``_attachment_hash_payload``
    rather than re-deriving them, so there is exactly one normalization site and
    identity can never drift from the content hash it is paired with.
    """
    return {field: payload[field] for field in _ATTACHMENT_IDENTITY_FIELDS}


#: The subset of ``_ATTACHMENT_IDENTITY_FIELDS`` that survives even when a
#: provider omits a stable id for the same physical attachment on a different
#: export request (polylogue-d8al): Claude.ai does not consistently emit
#: ``id``/``file_id``/``fileId``/``uuid``/``file_uuid`` for the same
#: attachment across separate export requests of the same conversation -- one
#: vintage carries a real UUID-shaped id, the other has none and a parser
#: synthesizes one instead. No synthetic-id scheme can make a real id and a
#: synthetic hash collide by construction, so revision comparison correlates
#: by this looser, id-independent key when the strict identity does not
#: match (see ``session_revision_projection``'s canonicalization step).
#: ``size_bytes`` stays excluded for the same lazy-fetch reason it is excluded
#: from ``_ATTACHMENT_IDENTITY_FIELDS``.
_ATTACHMENT_LOOSE_IDENTITY_FIELDS = ("message_id", "name", "mime_type")


def _attachment_loose_identity_payload(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project the id-independent correlation key of one attachment payload."""
    return {field: payload[field] for field in _ATTACHMENT_LOOSE_IDENTITY_FIELDS}


#: `generation_lifecycle` payload keys that are provider-reported measurement,
#: not identity, when the event's own payload declares them non-durable via
#: ``duration_semantics == "provider_reported_elapsed"``. ChatGPT re-derives
#: these from the raw export's own ``finished_duration_sec`` /
#: ``reasoning_start_time``/``reasoning_end_time`` metadata on every export
#: request, and the value is not stable across requests for the SAME
#: generation (observed varying non-monotonically, e.g. 13000 vs 21000ms;
#: 123000 vs 33000ms) even when the transcript is byte-identical
#: (polylogue-nuec).
_PROVIDER_REPORTED_ELAPSED_VOLATILE_PAYLOAD_KEYS = frozenset({"elapsed_duration_ms", "started_at_ms", "ended_at_ms"})
_PROVIDER_REPORTED_ELAPSED_MARKER_KEY = "duration_semantics"
_PROVIDER_REPORTED_ELAPSED_MARKER_VALUE = "provider_reported_elapsed"


def _event_identity_hash_payload(event: ParsedSessionEvent, event_index: int) -> dict[str, JSONValue]:
    """Build an event hash payload with provider-reported-elapsed measurement excluded.

    Mirrors ``_attachment_identity_payload``'s identity/acquisition split for a
    different volatility source: an event that labels itself
    ``duration_semantics: "provider_reported_elapsed"`` carries a measurement,
    not content, in ``_PROVIDER_REPORTED_ELAPSED_VOLATILE_PAYLOAD_KEYS`` --
    stripped here before hashing. The event's own ``timestamp`` is stripped
    alongside it for the same events: ChatGPT sets it from the same
    ``reasoning_end_time`` value the duration is derived from, so it varies in
    tandem and is measurement too, not identity. ``session_hash`` still covers
    the full, unstripped payload and timestamp (see ``session_hash_payload``),
    so a real change in reported duration still triggers a re-write; only the
    *revision comparison* axis (``event_identity_hashes``) is tolerant.
    """
    payload = event.payload
    timestamp = event.timestamp
    if payload.get(_PROVIDER_REPORTED_ELAPSED_MARKER_KEY) == _PROVIDER_REPORTED_ELAPSED_MARKER_VALUE:
        payload = {
            key: value for key, value in payload.items() if key not in _PROVIDER_REPORTED_ELAPSED_VOLATILE_PAYLOAD_KEYS
        }
        timestamp = None
    return {
        "event_index": event_index,
        "event_type": _normalize_for_hash(event.event_type),
        "timestamp": _normalize_for_hash(timestamp),
        "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
        "payload": hash_payload(_normalize_nested_for_hash(payload)),
    }


def _session_hash_payload(
    *,
    title: str | None,
    created_at: str | None,
    updated_at: str | None,
    messages: list[dict[str, JSONValue]],
    attachments: list[dict[str, JSONValue]],
    session_events: list[dict[str, JSONValue]],
) -> dict[str, object]:
    """Build the content-hash payload dict shared by pipeline and async write paths."""
    return {
        "title": _normalize_for_hash(title),
        "created_at": _normalize_for_hash(created_at),
        "updated_at": _normalize_for_hash(updated_at),
        "messages": messages,
        "session_events": session_events,
        "attachments": sorted(
            attachments,
            key=lambda item: (
                str(item.get("message_id") or ""),
                str(item.get("id") or ""),
                str(item.get("name") or ""),
            ),
        ),
    }


def _session_hash_components(
    convo: ParsedSession,
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    """Build the hash-stable message/attachment/event payloads once.

    ``session_content_hash`` and ``session_revision_projection`` both need
    these payloads (the former to hash the whole tree, the latter to also
    hash each item individually); building them once and sharing halves the
    payload-construction and nested tool_input hashing volume versus each
    caller re-deriving its own copy. Byte-identical to computing each
    payload independently -- pure sharing of an already-pure computation.
    """
    messages_payload = [
        _message_hash_payload(msg, msg.provider_message_id or f"msg-{idx}")
        for idx, msg in enumerate(convo.messages, start=1)
    ]
    attachments_payload = [_attachment_hash_payload(attachment) for attachment in convo.attachments]
    session_events_payload = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": hash_payload(_normalize_nested_for_hash(event.payload)),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    return messages_payload, attachments_payload, session_events_payload


def _session_tree_hash(
    convo: ParsedSession,
    *,
    messages_payload: list[dict[str, JSONValue]],
    attachments_payload: list[dict[str, JSONValue]],
    session_events_payload: list[dict[str, JSONValue]],
) -> str:
    return hash_payload(
        _session_hash_payload(
            title=convo.title,
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            messages=messages_payload,
            attachments=attachments_payload,
            session_events=session_events_payload,
        )
    )


def session_content_hash(convo: ParsedSession) -> ContentHash:
    """Generate the content hash for a session.

    Uses sentinel values to distinguish None from empty/missing fields. The
    hash incorporates the per-message payload (id, role, text, timestamp,
    content blocks), attachments, and session events, so any change to a
    message also changes the session hash.
    """
    messages_payload, attachments_payload, session_events_payload = _session_hash_components(convo)
    return ContentHash(
        _session_tree_hash(
            convo,
            messages_payload=messages_payload,
            attachments_payload=attachments_payload,
            session_events_payload=session_events_payload,
        )
    )


def session_revision_projection(convo: ParsedSession) -> SessionRevisionProjection:
    """Project canonical content hashes used to prove append-only session growth.

    Builds message/attachment/event payloads once (``_session_hash_components``)
    and reuses them for both the whole-tree ``session_hash`` and the per-item
    hashes below, instead of recomputing each payload from scratch a second
    time. Output is byte-identical to the previous double-computation --
    this is a pure elimination of redundant work, not an identity-hash
    change (polylogue-fqp0).

    ``session_hash`` still covers the full attachment payload, acquisition state
    included, so acquiring an attachment's bytes does change the session's
    content hash and does trigger a re-write. Only the *revision comparison*
    axes separate identity from acquisition (polylogue-bu1i).

    The same holds for message order (polylogue-c429) and provider-reported
    generation-duration measurement (polylogue-nuec): ``session_hash`` still
    covers the full, order-sensitive message array and the full,
    unstripped event payload/timestamp, so a real reorder or a real duration
    change still triggers a re-write. Only ``message_contents`` /
    ``event_identity_hashes`` -- the *revision comparison* axes -- are
    tolerant of the volatility each bug describes.
    """
    messages_payload, attachments_payload, session_events_payload = _session_hash_components(convo)
    session_hash_hex = _session_tree_hash(
        convo,
        messages_payload=messages_payload,
        attachments_payload=attachments_payload,
        session_events_payload=session_events_payload,
    )
    message_contents: set[tuple[bytes, bytes]] = set()
    message_hashes: list[bytes] = []
    for payload in messages_payload:
        identity = bytes.fromhex(hash_payload(_message_identity_payload(payload)))
        content = bytes.fromhex(hash_payload(payload))
        message_contents.add((identity, content))
        message_hashes.append(content)
    attachment_records: list[AttachmentRecord] = []
    attachment_identities: set[bytes] = set()
    attachment_contents: set[tuple[bytes, bytes]] = set()
    for payload in attachments_payload:
        identity = bytes.fromhex(hash_payload(_attachment_identity_payload(payload)))
        loose_identity = bytes.fromhex(hash_payload(_attachment_loose_identity_payload(payload)))
        inline_content_hash = payload.get("inline_content_hash")
        attachment_content = bytes.fromhex(inline_content_hash) if isinstance(inline_content_hash, str) else None
        attachment_records.append((identity, loose_identity, attachment_content))
        attachment_identities.add(identity)
        if attachment_content is not None:
            attachment_contents.add((identity, attachment_content))
    event_hashes: list[bytes] = []
    event_identity_hashes: list[bytes] = []
    for event_index, (payload, event) in enumerate(zip(session_events_payload, convo.session_events, strict=True)):
        event_hashes.append(bytes.fromhex(hash_payload(payload)))
        event_identity_hashes.append(bytes.fromhex(hash_payload(_event_identity_hash_payload(event, event_index))))
    metadata_hash = bytes.fromhex(
        hash_payload(
            {
                "title": _normalize_for_hash(convo.title),
                "created_at": _normalize_for_hash(convo.created_at),
                "updated_at": _normalize_for_hash(convo.updated_at),
            }
        )
    )
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(message_hashes),
        message_contents=frozenset(message_contents),
        attachment_identities=frozenset(attachment_identities),
        attachment_contents=frozenset(attachment_contents),
        attachment_records=tuple(attachment_records),
        event_hashes=tuple(event_hashes),
        event_identity_hashes=tuple(event_identity_hashes),
        metadata_hash=metadata_hash,
    )
