"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.core.hashing import hash_bytes, hash_payload
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import ContentHash, MessageId, SessionId

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
MessageContent: TypeAlias = tuple[bytes, bytes, int]


@dataclass(frozen=True, slots=True)
class SessionRevisionProjection:
    """Canonical content-only comparison value for a session revision.

    One invariant governs every field below: *a conversation is an unordered
    collection of items keyed by content-derived identity, each carrying only
    content-bearing fields -- nothing else may enter the value used to
    compare two acquisitions of it* (polylogue-aggz). Concretely:

    - Identity is never array position. ``message_contents``,
      ``attachment_contents``, and ``event_contents`` are unordered, not
      ordered tuples -- a provider's export can replay an unchanged item set
      in a different array sequence across separate export requests (proven
      for Claude.ai messages and ChatGPT ``generation_lifecycle`` events:
      same items, different sequence, every re-export), and a set has no
      order to violate (polylogue-c429, polylogue-nuec).
    - Identity is derived from content, never a provider id whose PRESENCE
      is itself unstable. ``attachment_contents``' key is
      ``(anchoring message, name, media type)`` -- never the provider's own
      attachment id, which Claude.ai does not consistently emit for the same
      attachment across export vintages (one vintage carries a real UUID,
      the other has none) -- no id-minting scheme can make a real id and a
      synthetic one collide, so the id is simply not part of identity
      (polylogue-d8al, polylogue-hith). The same reasoning selects
      ``(event type, anchoring message)`` for events, folding in the event's
      own content hash only when that pair is ambiguous within one revision
      (e.g. more than one ``chatgpt_block_metadata`` event on one message,
      one per block) -- still content-derived, never the array index.
    - Acquisition state and provider-reported measurement are not content.
      An attachment's bytes may be known or not (``attachment_contents``
      omits an identity until its bytes are read, while
      ``attachment_identities`` already knows the reference exists) --
      resolving them is evidence *growing*, not the attachment becoming a
      different one (polylogue-bu1i). ChatGPT's ``generation_lifecycle``
      event re-derives ``elapsed_duration_ms`` from the raw export's own
      timing metadata on every export request, and the value is not stable
      across requests for the SAME generation even when the transcript is
      byte-identical -- excluded from ``event_contents`` by
      ``_EVENT_CONTENT_PAYLOAD_ALLOWLIST``, an explicit per-event-type
      ALLOWLIST of content-bearing payload fields rather than a denylist of
      fields discovered volatile after the fact (three volatility axes were
      each found only after shipping: acquisition state, array order,
      provider-reported duration -- a denylist means the next provider
      quirk is silently invisible to comparison until someone notices and
      files a bead; an allowlist means a NEW field a parser adds later
      cannot silently enter identity without an explicit decision to add it).

    ``session_hash``, ``message_hashes`` (ordered), and ``event_hashes``
    (ordered, unstripped) are UNCHANGED by any of this: they still cover the
    full, order-sensitive, unstripped payload, so a real reorder, a real
    duration change, or newly-acquired bytes still change the session's
    content hash and still trigger a re-write (idempotency is a different
    question from revision *comparison*, and only the latter is
    content-only). ``attachment_identities`` is kept as a plain
    ``frozenset[bytes]`` of the same content-derived keys, read by
    ``storage/repair.py``/``archive.py`` only via ``len()`` for a frontier
    count.
    """

    session_hash: bytes
    message_hashes: tuple[bytes, ...]
    # Each item carries its unordered multiplicity. Repeated timestamp-less
    # id-less messages can have the same content-derived identity and content;
    # collapsing them into a set would falsely make one and two turns equal.
    message_contents: frozenset[MessageContent]
    attachment_identities: frozenset[bytes]
    attachment_contents: frozenset[tuple[bytes, bytes]]
    event_hashes: tuple[bytes, ...]
    event_contents: frozenset[tuple[bytes, bytes]]


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


def _is_redundant_text_only_block(message: ParsedMessage) -> bool:
    """True when ``message.blocks`` is a single TEXT block that just repeats ``message.text``.

    polylogue-0qfy: some claude-ai-export vintages parse an otherwise
    identical message with an empty ``blocks`` list, others with exactly one
    ``[{"type":"text","text": message.text}]`` block carrying no other
    field (tool_name/tool_id/tool_input/media_type all unset). This is a
    parser-shape artifact, not a second content axis -- both vintages carry
    the same real content -- so it must not make the hash-stable payload
    (and therefore the raw-authority membership comparison, which reuses
    this same payload) see the two vintages as a genuine conflict.
    """
    if len(message.blocks) != 1:
        return False
    block = message.blocks[0]
    return (
        block.type is BlockType.TEXT
        and block.text == message.text
        and not block.tool_name
        and not block.tool_id
        and block.tool_input is None
        and not block.media_type
    )


def _message_hash_payload(message: ParsedMessage, message_id: str) -> dict[str, JSONValue]:
    """Build the hash-stable payload for a single message."""
    payload: dict[str, JSONValue] = {"id": message_id}
    payload.update(_message_comparison_payload(message))
    return payload


def _message_comparison_payload(message: ParsedMessage) -> dict[str, JSONValue]:
    """Build the content payload that distinguishes an idless message."""
    payload: dict[str, JSONValue] = {
        "role": str(message.role),
        "text": _normalize_for_hash(message.text),
        "timestamp": _normalize_for_hash(message.timestamp),
    }
    if message.blocks and not _is_redundant_text_only_block(message):
        payload["content_blocks"] = [_content_block_payload(b) for b in message.blocks]
    return payload


#: Marker prefix for a content-derived message identity anchor, used only
#: when a message carries no native ``provider_message_id``. Namespaced so it
#: can never collide with a real provider id string (a provider id never
#: contains this literal token by construction).
_CONTENT_ANCHOR_PREFIX = "__polylogue_msg_content_anchor__"


def _message_comparison_id(message: ParsedMessage) -> str:
    """Resolve the id used as both a message's content-payload id and its
    comparison identity (``message_identity_hash``'s sole input).

    Prefers the provider's own id -- stable across reordering even when the
    export's array ordering is not (polylogue-c429). When a parser could not
    populate one (``provider_message_id`` is empty), this used to fall back
    to ``f"msg-{index}"`` -- positionally-derived, exactly the same failure
    class the attachment identity fix (polylogue-hith/-d8al) removed for a
    different field: two parses of the same id-less message in a different
    array position would get two different fallback "ids" and compare as a
    conflict instead of the same message (polylogue-gysk3).

    The fix: fall back to one content-derived anchor over role, timestamp,
    and message text instead of array position. Including text even when a
    timestamp exists keeps two same-timestamp id-less messages distinct and
    gives attachment ownership a stable non-positional anchor when a Drive
    attachment moves between them.

    Parser normalization maintains the complementary invariant: a missing
    native id is never replaced with an array-position-derived value before
    reaching this function. Parser-local occurrence keys may use position,
    but they are not persisted as ``provider_message_id``.
    """
    if message.provider_message_id:
        return message.provider_message_id
    return f"{_CONTENT_ANCHOR_PREFIX}:{hash_payload(_message_comparison_payload(message))}"


def _message_owner_anchors(messages: list[ParsedMessage], comparison_ids: list[str]) -> dict[int, str]:
    """Resolve attachment owners without changing public message identity.

    A content anchor is intentionally shared by duplicate id-less messages.
    That is correct for message revision comparison, but it is not enough to
    identify which occurrence owns an attachment. When an anchor occurs more
    than once, add a private discriminator derived from the parser-normalized
    message position. The position is a transport coordinate already used by
    ``ParsedAttachment.message_position``; it never enters the public
    ``provider_message_id`` or the message comparison payload.

    Messages without a position cannot be addressed by a position-linked
    attachment either, so they retain the content anchor in this owner map.
    """
    counts = Counter(comparison_ids)
    anchors: dict[int, str] = {}
    for message, comparison_id in zip(messages, comparison_ids, strict=True):
        if message.position is None:
            continue
        if counts[comparison_id] == 1:
            anchors[message.position] = comparison_id
            continue
        occurrence = hash_payload(
            {
                "position": message.position,
                "variant_index": _normalize_for_hash(message.variant_index),
            }
        )
        anchors[message.position] = f"{comparison_id}:occurrence:{occurrence}"
    return anchors


def message_identity_hash(*, id: str) -> bytes:
    """The sole constructor of a message's comparison identity (polylogue-aggz).

    A provider's own message id is stable across re-exports even when the
    export's array ordering is not (polylogue-c429) -- it is the only
    content field that answers *which message is this*, as opposed to *what
    does it currently say*.

    This is a fixed keyword-only signature, not a dict projected by a list
    of field names: passing ``role``/``text``/``timestamp``/anything else is
    a ``TypeError`` at the call boundary, not a value that has to be
    remembered and stripped. Extending what a message's comparison identity
    covers requires editing this signature -- an explicit, reviewable
    decision, never a side effect of a parser gaining a new field.
    """
    return bytes.fromhex(hash_payload({"id": id}))


#: Fields of an attachment hash payload that answer *which attachment is
#: this*, as opposed to *what have we managed to read about it* -- the
#: anchoring message plus content descriptors, content-derived and never the
#: provider's own attachment id. Claude.ai does not consistently emit
#: ``id``/``file_id``/``fileId``/``uuid``/``file_uuid`` for the same
#: attachment across separate export requests of the same conversation: one
#: vintage carries a real UUID-shaped id, the other has none (a positionally
#: -seeded synthetic id is not identity either -- polylogue-hith). No
#: id-minting scheme can make a real id and any synthetic value collide by
#: construction, so the id is excluded from identity altogether rather than
#: used when present (polylogue-d8al). ``size_bytes`` is excluded because for
#: lazily-fetched attachments (Drive/Gemini references, browser capture) the
#: provider states no size until the bytes are actually read, so treating it
#: as identity would make acquisition look like a different attachment
#: (polylogue-bu1i). ``inline_content_hash`` is excluded for the same reason
#: and is recovered separately as acquisition evidence in
#: ``attachment_contents``.
#:
#: Known, accepted limit stated explicitly rather than engineered around: two
#: genuinely distinct attachments that share one message/name/media-type and
#: carry no bytes on either side of a comparison are indistinguishable by any
#: signal this projection can offer.


def attachment_identity_hash(*, message_id: JSONValue, name: JSONValue, mime_type: JSONValue) -> bytes:
    """The sole constructor of an attachment's comparison identity (polylogue-aggz).

    Fixed to (anchoring message, name, media type) -- content-derived and
    never the provider's own attachment id (polylogue-d8al, polylogue-hith)
    or acquisition state such as ``size_bytes``/inline bytes
    (polylogue-bu1i). Those fields are not parameters here; passing them
    (e.g. spreading a full attachment payload dict as ``**kwargs``) is a
    ``TypeError``, not a value this function has to remember to strip.
    """
    return bytes.fromhex(hash_payload({"message_id": message_id, "name": name, "mime_type": mime_type}))


def _attachment_hash_payload(
    attachment: ParsedAttachment, *, message_owner_anchor: str | None = None
) -> dict[str, JSONValue]:
    """Build attachment identity without perturbing legacy metadata-only hashes."""
    owner_id = (
        message_owner_anchor
        if not attachment.message_provider_id and message_owner_anchor
        else attachment.message_provider_id
    )
    payload: dict[str, JSONValue] = {
        "id": _normalize_for_hash(attachment.provider_attachment_id),
        "message_id": _normalize_for_hash(owner_id),
        "name": _normalize_for_hash(attachment.name),
        "mime_type": _normalize_for_hash(attachment.mime_type),
        "size_bytes": _normalize_for_hash(attachment.size_bytes),
    }
    if attachment.inline_bytes is not None:
        payload["inline_content_hash"] = hash_bytes(attachment.inline_bytes)
    elif attachment.precomputed_blob is not None:
        # polylogue-8ac0: bytes already streamed into the blob store during
        # sidecar discovery carry a known hash without needing to re-read
        # them here (mirrors the ``inline_bytes`` branch above, whose content
        # hash marks re-ingest content-changed once bytes newly arrive).
        payload["inline_content_hash"] = attachment.precomputed_blob[0]
    return payload


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

#: Event payload ALLOWLIST by event type: only these fields, per type, ever
#: enter ``event_contents``. This is deliberately an allowlist, not a
#: denylist of fields discovered volatile after the fact -- three separate
#: volatility axes (attachment acquisition state, message array order,
#: provider-reported generation duration) were each found only after
#: shipping, one bead and one branch at a time, because a denylist design
#: means a NEW field silently enters identity the moment a parser starts
#: emitting it, until someone notices and adds it to the strip-list. Under
#: an allowlist, a field a parser adds later is excluded from comparison by
#: construction -- it takes an explicit decision to add it here before it
#: can affect identity, not an explicit decision to exclude it.
#:
#: Event types with no entry here compare their FULL payload (today's
#: behavior for every type except ``generation_lifecycle``, which is the
#: only one with proven volatility -- polylogue-nuec; the raw export's own
#: ``finished_duration_sec``/``reasoning_start_time``/``reasoning_end_time``
#: metadata is not stable across separate export requests for the SAME
#: generation, so ``elapsed_duration_ms``/``started_at_ms``/``ended_at_ms``
#: and the derived ``timestamp`` are excluded; ``duration_semantics`` merely
#: documents that fact and carries no content of its own).
_EVENT_CONTENT_PAYLOAD_ALLOWLIST: dict[str, frozenset[str]] = {
    "generation_lifecycle": frozenset({"state", "evidence_source", "fidelity"}),
}


def _event_content_payload(event: ParsedSessionEvent) -> dict[str, JSONValue]:
    """Build the position- and measurement-independent CONTENT payload for one event.

    Array position is not identity for events any more than for messages:
    ChatGPT's ``generation_lifecycle`` events were independently observed to
    reorder alongside their duration values across separate export requests
    of the SAME conversation (same three durations, different array
    positions each time) -- the same volatility polylogue-c429 found for
    messages, on a different axis (polylogue-nuec). This never includes
    ``event_index``; ``session_revision_projection`` builds identity purely
    from this content plus the event's own type and anchoring message.

    The allowlist strip is narrowed to the specific provider-remeasured
    shape it targets: it applies only when the event's own payload declares
    itself non-durable via ``duration_semantics ==
    "provider_reported_elapsed"`` (set by the ChatGPT parser). The
    browser-capture parser emits the SAME ``event_type`` for its own DOM/UI
    generation observations, tagged with a different ``duration_semantics``
    (e.g. ``dom_observed_wall``, ``provider_ui_elapsed``) -- those are a real
    first-party measurement this projection has no other record of, not the
    re-derived-on-every-export ChatGPT value nuec exists for, so their
    observation id, timestamp, and duration/label/trigger fields remain
    content rather than being silently stripped.
    """
    allowlist = _EVENT_CONTENT_PAYLOAD_ALLOWLIST.get(event.event_type)
    provider_reported_elapsed = (
        allowlist is not None
        and event.payload.get(_PROVIDER_REPORTED_ELAPSED_MARKER_KEY) == _PROVIDER_REPORTED_ELAPSED_MARKER_VALUE
    )
    if allowlist is None or not provider_reported_elapsed:
        payload = event.payload
        timestamp = event.timestamp
    else:
        payload = {key: value for key, value in event.payload.items() if key in allowlist}
        # An event type with a registered allowlist also has its own
        # provider-remeasured timestamp excluded: for generation_lifecycle,
        # ChatGPT sets it from the same reasoning_end_time value the
        # duration is derived from, so it varies in tandem and is
        # measurement too, not content.
        timestamp = None
    return {
        "event_type": _normalize_for_hash(event.event_type),
        "timestamp": _normalize_for_hash(timestamp),
        "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
        "payload": hash_payload(_normalize_nested_for_hash(payload)),
    }


#: The subset of an event content payload that answers *which event slot is
#: this*, as opposed to *what does it say*: anchoring message plus event
#: type, content-derived and never the array index. Can be shared by more
#: than one event within one revision (e.g. multiple
#: ``chatgpt_block_metadata`` events on the same message, one per block), so
#: ``session_revision_projection`` always folds the event's own content hash
#: into the FINAL identity on top of this base -- unconditionally, not only
#: when a sibling is present in that particular revision, so identity never
#: depends on what else happens to be in the set. Still content-derived
#: (each block's own content, including any content-intrinsic field such as
#: ``block_index``, already differs), never the array position.
def event_base_identity_hash(*, event_type: JSONValue, source_message_provider_id: JSONValue) -> bytes:
    """The sole constructor of an event's position-independent base identity.

    Anchoring message plus event type only -- content-derived, never the
    array index and never provider-reported measurement (polylogue-nuec),
    which is not a parameter here. Fixed keyword-only signature: passing a
    whole event content payload as ``**kwargs`` (which also carries
    ``timestamp``/``payload``) is a ``TypeError``.
    """
    return bytes.fromhex(
        hash_payload({"event_type": event_type, "source_message_provider_id": source_message_provider_id})
    )


def event_canonical_identity_hash(*, base_identity: bytes, content_hash: bytes) -> bytes:
    """Fold an event's base identity with its own content hash.

    Used only when a base identity (event type + anchoring message) may be
    shared by more than one event within one revision (e.g. multiple
    ``chatgpt_block_metadata`` events on the same message, one per block) --
    still content-derived, never the array index (polylogue-aggz).
    """
    return bytes.fromhex(hash_payload({"base_identity": base_identity.hex(), "content": content_hash.hex()}))


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
    message_comparison_ids = [_message_comparison_id(msg) for msg in convo.messages]
    messages_payload = [
        _message_hash_payload(message, comparison_id)
        for message, comparison_id in zip(convo.messages, message_comparison_ids, strict=True)
    ]
    owner_anchor_by_position = _message_owner_anchors(convo.messages, message_comparison_ids)
    attachments_payload = [
        _attachment_hash_payload(
            attachment,
            message_owner_anchor=(
                owner_anchor_by_position.get(attachment.message_position)
                if attachment.message_position is not None
                else None
            ),
        )
        for attachment in convo.attachments
    ]
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

    The same holds for message order (polylogue-c429), attachment identity
    presence (polylogue-d8al), and provider-reported generation-duration
    measurement (polylogue-nuec): ``session_hash`` still covers the full,
    order-sensitive message array, the full attachment payload including
    whatever id the provider did or didn't emit, and the full, unstripped
    event payload/timestamp, so a real reorder, a real id change, or a real
    duration change still triggers a re-write. Only ``message_contents`` /
    ``attachment_identities`` / ``attachment_contents`` / ``event_contents``
    -- the *revision comparison* axes -- are content-only (polylogue-aggz).
    """
    messages_payload, attachments_payload, session_events_payload = _session_hash_components(convo)
    session_hash_hex = _session_tree_hash(
        convo,
        messages_payload=messages_payload,
        attachments_payload=attachments_payload,
        session_events_payload=session_events_payload,
    )
    message_content_counts: Counter[tuple[bytes, bytes]] = Counter()
    message_hashes: list[bytes] = []
    for payload in messages_payload:
        message_native_id = payload["id"]
        assert isinstance(message_native_id, str)  # built as str above, never anything else
        identity = message_identity_hash(id=message_native_id)
        content = bytes.fromhex(hash_payload(payload))
        message_content_counts[(identity, content)] += 1
        message_hashes.append(content)
    attachment_identities: set[bytes] = set()
    attachment_contents: set[tuple[bytes, bytes]] = set()
    for payload in attachments_payload:
        identity = attachment_identity_hash(
            message_id=payload["message_id"], name=payload["name"], mime_type=payload["mime_type"]
        )
        inline_content_hash = payload.get("inline_content_hash")
        attachment_identities.add(identity)
        if isinstance(inline_content_hash, str):
            attachment_contents.add((identity, bytes.fromhex(inline_content_hash)))
    event_hashes: list[bytes] = []
    event_base_identities: list[bytes] = []
    event_content_hashes: list[bytes] = []
    for payload, event in zip(session_events_payload, convo.session_events, strict=True):
        event_hashes.append(bytes.fromhex(hash_payload(payload)))
        content_payload = _event_content_payload(event)
        event_base_identities.append(
            event_base_identity_hash(
                event_type=content_payload["event_type"],
                source_message_provider_id=content_payload["source_message_provider_id"],
            )
        )
        event_content_hashes.append(bytes.fromhex(hash_payload(content_payload)))
    event_contents: set[tuple[bytes, bytes]] = set()
    for base_identity, content_hash in zip(event_base_identities, event_content_hashes, strict=True):
        # A base identity (event type + anchoring message) is ambiguous
        # whenever it is EVER possible for more than one event to share it
        # (e.g. one chatgpt_block_metadata event per block on a message), so
        # the event's own content is always folded into identity here --
        # unconditionally, not only when a sibling happens to be present in
        # THIS revision. An item's identity must not depend on what else is
        # in the set: computing it from this revision's own sibling count
        # made the same event's identity shift between `base_identity` (one
        # instance) and `hash(base_identity, content)` (two or more) purely
        # because a sibling appeared in a later revision, which made an
        # ordinary event-growth revision compare as a disjoint conflict
        # instead of containment. Folding content in always keeps identity
        # intrinsic to the event itself: still content-derived, never the
        # array index (distinct blocks already differ in content, e.g. a
        # content-intrinsic block_index), and true duplicates (same base
        # identity, same content, whether or not any sibling exists)
        # correctly collapse to one set entry either way.
        canonical_identity = event_canonical_identity_hash(base_identity=base_identity, content_hash=content_hash)
        event_contents.add((canonical_identity, content_hash))
    return SessionRevisionProjection(
        session_hash=bytes.fromhex(session_hash_hex),
        message_hashes=tuple(message_hashes),
        message_contents=frozenset(
            (identity, content, multiplicity) for (identity, content), multiplicity in message_content_counts.items()
        ),
        attachment_identities=frozenset(attachment_identities),
        attachment_contents=frozenset(attachment_contents),
        event_hashes=tuple(event_hashes),
        event_contents=frozenset(event_contents),
    )
