"""Shared parser for xAI Grok account exports and app-chat endpoint bundles.

Wire shape (no official xAI schema publication exists; this is reconstructed
from independent evidence and cross-checked across sources rather than taken
from a single blog post):

* https://github.com/beejaksharam/grok-export-viewer — an open-source tool
  whose ``grok_export_viewer/core.py`` parses the real export file (named
  ``prod-grok-backend.json`` by xAI) with executable, tested code:
  ``data["conversations"]`` -> each item has ``item["conversation"]["title"]``
  and ``item["responses"]``, each response nested as
  ``r["response"]["sender"]`` / ``r["response"]["message"]``.
* https://ai-chat-importer.com/blog/how-to-export-grok-conversations — shows
  the same top-level ``conversations`` / ``conversation`` / ``responses``
  shape (with a flatter per-response layout: ``sender``/``message`` directly
  on the response entry rather than nested under ``response``) and documents
  ``create_time`` as MongoDB extended JSON (``{"$date": {"$numberLong":
  "<epoch-ms>"}}``) plus the explicit absence of a native conversation id or
  attachment/image data in the export.
* A live grok.com API-scraping userscript
  (https://greasyfork.org/en/scripts/571847) independently confirms
  ``sender``/``message``/``createTime`` field names and ``sender == "human"``
  for the user turn.

This parser accepts both the nested (``{"response": {...}}``) and flat
per-response shapes, and both MongoDB extended-JSON and plain (ISO/epoch)
timestamps, since the shape is reconstructed from secondary sources rather
than one authoritative spec.

In the account-export evidence, neither conversation nor responses carry a native id in the
confirmed shapes above, so both ``provider_session_id`` and
``provider_message_id`` are content-derived (``pipeline.ids``'s declared
``idless_session_identity`` vocabulary, and ``synthetic_message_id`` under a
constant namespace). The resulting ids are not provider-native, but they are
intrinsic to the conversation: stable when a re-export reorders responses or
conversations or lands under a different filename, and distinct between two
conversations exported from files that happen to share a stem
(polylogue-31zag).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence

from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, TitleSource, ToolResultUnknownReason
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.timestamps import canonical_timestamp_text
from polylogue.pipeline.ids import idless_session_identity
from polylogue.sources.detection_projection import DetectorProjection

from .base import (
    AdmissionLedger,
    AdmissionRefusalReason,
    AdmissionUnit,
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    human_authored_override,
    parser_admission,
    synthetic_message_id,
)

#: Namespace for this parser's synthetic message ids. It is deliberately a
#: constant and not the acquisition fallback id: a message id is already
#: scoped by its session id (``pipeline.ids.message_id``), so seeding it with
#: the export filename adds no disambiguation and only makes the id move when
#: the same conversation is re-exported under a different name.
_MESSAGE_ID_NAMESPACE = "grok"

_SENDER_ROLE: dict[str, Role] = {
    "human": Role.USER,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "grok": Role.ASSISTANT,
}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _response_fields(entry: object) -> Mapping[str, object]:
    """Unwrap the ``{"response": {...}}`` nesting used by the real export.

    Some secondary documentation shows responses flattened one level
    (``{"sender": ..., "message": ...}`` directly on the entry); both shapes
    are accepted.
    """
    entry_map = _mapping(entry)
    nested = entry_map.get("response")
    return _mapping(nested) if isinstance(nested, Mapping) else entry_map


def _timestamp_text(value: object) -> str | None:
    if isinstance(value, Mapping):
        date_value = value.get("$date")
        if isinstance(date_value, Mapping):
            number_long = date_value.get("$numberLong")
            if number_long is None:
                return None
            try:
                epoch_ms = int(number_long)
            except (TypeError, ValueError):
                return None
            return canonical_timestamp_text(epoch_ms / 1000)
        if isinstance(date_value, (str, int, float)):
            return canonical_timestamp_text(date_value)
        return None
    if isinstance(value, (str, int, float)):
        return canonical_timestamp_text(value)
    return None


def _role_for_sender(sender: object) -> Role:
    if not isinstance(sender, str) or not sender:
        return Role.UNKNOWN
    role = _SENDER_ROLE.get(sender.strip().lower())
    return role if role is not None else Role.normalize(sender)


def looks_like_conversation(payload: object) -> bool:
    """Return whether ``payload`` is a single Grok export conversation entry."""
    if not isinstance(payload, Mapping):
        return False
    conversation = payload.get("conversation")
    responses = payload.get("responses")
    return isinstance(conversation, Mapping) and isinstance(responses, list)


def looks_like_export(payload: object) -> bool:
    """Return whether ``payload`` is a top-level Grok account-data export document."""
    if not isinstance(payload, Mapping):
        return False
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return False
    if not conversations:
        return True
    return any(looks_like_conversation(item) for item in conversations)


def _session_identity(messages: Iterable[ParsedMessage], created_at: str | None, fallback_id: str) -> str:
    """Derive this conversation's content-derived provider session id.

    The one narrow exception is a conversation that genuinely carries nothing
    to hash: no admitted response and no ``create_time``. Such an entry has no
    intrinsic content at all, so there is no content-derived identity to
    compute and the acquisition fallback id is the only value left. It is not
    a second identity route for ordinary conversations -- a conversation with
    a single blank-text response but a ``create_time``, or with responses but
    no timestamps, still hashes.
    """
    # Reorder-stable selection of the opening turn: earliest declared
    # timestamp, ties broken by the (content-derived) message id. Taking
    # ``messages[0]`` would reintroduce exactly the array-order sensitivity
    # this identity exists to remove.
    opening = min(messages, key=lambda m: (m.timestamp or "", m.provider_message_id), default=None)
    if opening is None and created_at is None:
        return fallback_id
    return idless_session_identity(
        first_message_provider_id=opening.provider_message_id if opening is not None else None,
        first_message_text=opening.text if opening is not None else None,
        created_at=created_at,
    )


@parser_admission("grok")
def parse_conversation(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    """Parse a single Grok export conversation entry into a session."""
    if looks_like_native_bundle(payload):
        return _parse_native_session(payload, fallback_id)
    conversation = _mapping(payload.get("conversation"))
    responses_raw = payload.get("responses")
    responses = responses_raw if isinstance(responses_raw, list) else []
    return parse_conversation_stream(conversation, responses, fallback_id, messages=[])


def parse_conversation_stream(
    conversation: Mapping[str, object],
    responses: Iterable[object],
    fallback_id: str,
    *,
    messages: MutableSequence[ParsedMessage],
) -> ParsedSession:
    """Normalize responses one at a time into a list or a disk-backed sink."""
    for entry in responses:
        append_conversation_response(messages, entry)
    return finish_conversation(conversation, fallback_id, messages)


def append_conversation_response(messages: MutableSequence[ParsedMessage], entry: object) -> None:
    """Normalize one Grok response with a stable admitted-message position."""
    fields = _response_fields(entry)
    text_raw = fields.get("message")
    text = text_raw if isinstance(text_raw, str) else None
    if not text:
        return
    grok_role = _role_for_sender(fields.get("sender"))
    timestamp = _timestamp_text(fields.get("create_time"))
    provider_message_id = synthetic_message_id(
        namespace=_MESSAGE_ID_NAMESPACE,
        role=grok_role,
        text=text,
        timestamp=timestamp,
        kind="grok-response",
    )
    messages.append(
        ParsedMessage(
            provider_message_id=provider_message_id,
            role=grok_role,
            text=text,
            timestamp=timestamp,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            position=len(messages),
            variant_index=0,
            is_active_path=True,
            is_active_leaf=False,
            material_origin=human_authored_override(
                grok_role,
                MessageType.MESSAGE,
                classify_material_origin(role=grok_role, message_type=MessageType.MESSAGE, text=text),
            ),
        )
    )


def finish_conversation(
    conversation: Mapping[str, object], fallback_id: str, messages: MutableSequence[ParsedMessage]
) -> ParsedSession:
    """Derive session fields from a complete list or scratch-backed response sink."""
    title_raw = conversation.get("title")
    provider_title = title_raw if isinstance(title_raw, str) and title_raw else None
    title = provider_title or fallback_id
    created_at = _timestamp_text(conversation.get("create_time"))

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if messages:
        messages[-1] = messages[-1].model_copy(update={"is_active_leaf": True})
    updated_at = messages[-1].timestamp if messages and messages[-1].timestamp else created_at

    provider_session_id = _session_identity(messages, created_at, fallback_id)

    return ParsedSession(
        source_name=Provider.GROK,
        provider_session_id=provider_session_id,
        title=title,
        title_source=TitleSource.ORIGIN if provider_title else None,
        created_at=created_at,
        updated_at=updated_at,
        messages=[],
        active_leaf_message_provider_id=active_leaf_message_provider_id,
    ).model_copy(update={"messages": messages})


__all__ = [
    "looks_like_conversation",
    "looks_like_export",
    "parse_conversation",
    "looks_like_native_bundle",
    "parse_native_bundle",
]


def detection_projection() -> DetectorProjection:
    """Consume every conversation and retain an exact existential shape witness."""
    item = DetectorProjection(fields={"conversation": DetectorProjection(), "responses": DetectorProjection()})
    return DetectorProjection(
        fields={
            "conversations": DetectorProjection(item=item, array_fold="any", array_predicate=looks_like_conversation),
        }
    )


# These fields are emitted by the retained app-chat /responses reply and
# were previously projected only by the browser adapter. Outcome-free search
# evidence is not a successful tool execution.
_RESULT_FIELDS = {
    "webSearchResults": "web_search",
    "citedWebSearchResults": "web_search",
    "xposts": "x_search",
    "citedXposts": "x_search",
    "ragResults": "rag_search",
    "citedRagResults": "rag_search",
    "searchProductResults": "product_search",
    "connectorSearchResults": "connector_search",
    "citedConnectorSearchResults": "connector_search",
    "collectionSearchResults": "collection_search",
    "citedCollectionSearchResults": "collection_search",
}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _native_conversation(payload: Mapping[str, object]) -> Mapping[str, object]:
    reply = _mapping(payload.get("conversation"))
    return _mapping(reply["conversation"]) if isinstance(reply.get("conversation"), Mapping) else reply


def looks_like_native_bundle(payload: object) -> bool:
    """Identify original Grok endpoint replies, independently of acquisition metadata."""
    if not isinstance(payload, Mapping):
        return False
    conversation = _native_conversation(payload)
    responses = payload.get("responses")
    return bool(_string(conversation.get("conversationId"))) and (
        isinstance(responses, list) or isinstance(_mapping(responses).get("responses"), list)
    )


def _native_result(
    value: object, own_id: str | None, *, metadata: dict[str, object], name: str | None = None
) -> ParsedContentBlock:
    fields = _mapping(value)
    # Only explicitly retained structural signals decide outcome. The
    # provider's unknown tool/search payloads remain verbatim evidence.
    error = fields.get("is_error")
    if not isinstance(error, bool):
        error = fields.get("isError")
    error = error if isinstance(error, bool) else None
    exit_code = fields.get("exitCode", fields.get("exit_code"))
    exit_code = exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None
    unknown_reason = (
        ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT.value
        if any(key in fields for key in ("is_error", "isError", "exitCode", "exit_code"))
        else ToolResultUnknownReason.NOT_REPORTED.value
    )
    return ParsedContentBlock(
        type=BlockType.TOOL_RESULT,
        tool_id=own_id or None,
        tool_name=name,
        text=_string(fields.get("text")),
        metadata=metadata,
        is_error=error,
        exit_code=exit_code,
        outcome_unknown_reason=unknown_reason if error is None and exit_code is None else None,
    )


def _native_blocks(fields: Mapping[str, object], own_id: str) -> list[ParsedContentBlock]:
    blocks: list[ParsedContentBlock] = []
    text = _string(fields.get("message"))
    if text:
        blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=text))
    for index, step in enumerate(_list(fields.get("steps"))):
        step_fields = _mapping(step)
        lines = step_fields.get("text")
        thinking = (
            "\n".join(line for line in _list(lines) if isinstance(line, str))
            if isinstance(lines, list)
            else _string(lines)
        )
        if thinking is not None:
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=thinking,
                    metadata={"step_index": index, "tags": step_fields.get("tags")},
                )
            )
        for field in ("toolUsageResults", "toolUsageCards"):
            for usage in _list(step_fields.get(field)):
                usage_fields = _mapping(usage)
                name = _string(
                    usage_fields.get("toolName")
                    or usage_fields.get("tool_name")
                    or usage_fields.get("name")
                    or usage_fields.get("type")
                )
                blocks.append(
                    _native_result(
                        usage, own_id, name=name, metadata={"step_index": index, "source": field, "raw": usage}
                    )
                )
        if thinking is None and not step_fields.get("toolUsageResults") and not step_fields.get("toolUsageCards"):
            blocks.append(
                _native_result(
                    step,
                    own_id,
                    metadata={"source": "steps", "step_index": index, "unrecognized_shape": True, "raw": step},
                )
            )
    if fields.get("query") is not None:
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_id=own_id or None,
                tool_name="web_search",
                tool_input={"query": fields["query"], "query_type": fields.get("queryType")},
            )
        )
    for field, name in _RESULT_FIELDS.items():
        results = _list(fields.get(field))
        if results:
            blocks.append(_native_result({}, own_id, name=name, metadata={"field": field, "results": results}))
    for index, entry in enumerate(_list(fields.get("toolResponses"))):
        tool = _mapping(entry)
        name = _string(tool.get("toolName") or tool.get("tool_name") or tool.get("name"))
        tool_id = _string(tool.get("toolId")) or (f"{own_id}:tool_response:{index}" if own_id else None)
        if name:
            inputs = tool.get("input", tool.get("tool_input"))
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_id=tool_id,
                    tool_name=name,
                    tool_input=inputs if isinstance(inputs, Mapping) else None,
                    metadata={"source": "toolResponses", "index": index, "raw": entry},
                )
            )
        # Unlike the retired projection, a named response's output/outcome
        # must survive alongside its invocation.
        if not name or any(
            key in tool for key in ("text", "output", "result", "is_error", "isError", "exitCode", "exit_code")
        ):
            blocks.append(
                _native_result(
                    entry, tool_id, name=name, metadata={"source": "toolResponses", "index": index, "raw": entry}
                )
            )
    if fields.get("imageAttachments"):
        blocks.append(
            ParsedContentBlock(
                type=BlockType.IMAGE, metadata={"field": "imageAttachments", "raw": fields["imageAttachments"]}
            )
        )
    return blocks


def _native_attachments(fields: Mapping[str, object], own_id: str) -> list[ParsedAttachment]:
    attachments: list[ParsedAttachment] = []
    assets = {
        _string(_mapping(asset).get("assetId")): _mapping(asset)
        for asset in _list(fields.get("fileAttachmentAssetMetadata"))
    }
    seen: set[str] = set()
    for entry in _list(fields.get("fileAttachmentsMetadata")):
        meta = _mapping(entry)
        attachment_id = _string(meta.get("fileMetadataId"))
        if not attachment_id or attachment_id in seen:
            continue
        seen.add(attachment_id)
        asset = assets.get(attachment_id, {})
        size = asset.get("sizeBytes")
        attachments.append(
            ParsedAttachment(
                provider_attachment_id=attachment_id,
                message_provider_id=own_id,
                name=_string(meta.get("fileName") or asset.get("name")),
                mime_type=_string(meta.get("fileMimeType") or asset.get("mimeType")),
                size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
                source_url=_string(meta.get("fileUri") or asset.get("key")),
            )
        )
    for attachment_id, asset in assets.items():
        if not attachment_id or attachment_id in seen:
            continue
        seen.add(attachment_id)
        size = asset.get("sizeBytes")
        attachments.append(
            ParsedAttachment(
                provider_attachment_id=attachment_id,
                message_provider_id=own_id,
                name=_string(asset.get("name")),
                mime_type=_string(asset.get("mimeType")),
                size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
                source_url=_string(asset.get("key")),
            )
        )
    for field in ("generatedImageUrls", "imageEditUris"):
        for url in _list(fields.get(field)):
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            # The provider URL itself is the durable locator. No filename or
            # ordinal is promoted into a provider-issued asset identity.
            attachments.append(ParsedAttachment(provider_attachment_id=url, message_provider_id=own_id, source_url=url))
    return attachments


def _parse_native_session(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    conversation = _native_conversation(payload)
    responses_reply = payload.get("responses")
    responses = (
        responses_reply if isinstance(responses_reply, list) else _list(_mapping(responses_reply).get("responses"))
    )
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    events: list[ParsedSessionEvent] = []
    variants: dict[str, int] = {}
    ledger = AdmissionLedger()
    ledger.expect(AdmissionUnit.OUTER_RECORD, 1)
    ledger.materialized(AdmissionUnit.OUTER_RECORD, 0, "bundle")
    ledger.expect(AdmissionUnit.MESSAGE, len(responses))
    # Match the acquired browser record ordering, independent of pagination
    # or the endpoint's array ordering; identity still uses native IDs.
    ordered = sorted(
        responses,
        key=lambda entry: (
            _timestamp_text(_response_fields(entry).get("createTime")) or "",
            _string(_response_fields(entry).get("responseId")) or "",
        ),
    )
    for ordinal, entry in enumerate(ordered):
        if not isinstance(entry, Mapping):
            ledger.refusal(AdmissionUnit.MESSAGE, ordinal, str(ordinal), AdmissionRefusalReason.MALFORMED)
            events.append(
                ParsedSessionEvent(event_type="grok_response_refusal", payload={"reason": "malformed", "raw": entry})
            )
            continue
        ledger.materialized(AdmissionUnit.MESSAGE, ordinal, str(ordinal))
        fields = _response_fields(entry)
        own_id = _string(fields.get("responseId"))
        if not own_id:
            # Native absence remains absence; the shared archive identity
            # owner derives an intrinsic ID rather than a positional one.
            own_id = ""
        text = _string(fields.get("message"))
        timestamp = _timestamp_text(fields.get("createTime"))
        role = _role_for_sender(fields.get("sender"))
        blocks = _native_blocks(fields, own_id)
        variant = variants.get(own_id, 0)
        variants[own_id] = variant + 1
        coordinate = MessageOwnerCoordinate(stable_key=own_id or None, position=len(messages), variant_index=variant)
        messages.append(
            ParsedMessage(
                provider_message_id=own_id,
                role=role,
                text=text,
                timestamp=timestamp,
                blocks=blocks,
                parent_message_provider_id=_string(fields.get("parentResponseId")),
                position=len(messages),
                variant_index=variant,
                owner_coordinate=coordinate,
                model_name=_string(fields.get("model")),
                material_origin=human_authored_override(
                    role,
                    MessageType.MESSAGE,
                    classify_material_origin(role=role, message_type=MessageType.MESSAGE, text=text),
                ),
            )
        )
        for attachment in _native_attachments(fields, own_id):
            attachment.owner_coordinate = coordinate
            attachment.message_position = coordinate.position
            attachment.message_variant_index = variant
            attachment.message_provider_id = own_id or None
            attachments.append(attachment)
        facts = {key: fields[key] for key in ("partial", "manual", "shared", "streamErrors") if key in fields}
        if facts:
            events.append(
                ParsedSessionEvent(
                    event_type="grok_response_state",
                    timestamp=timestamp,
                    source_message_provider_id=own_id or None,
                    payload=facts,
                    boundary_message_position=len(messages) - 1,
                )
            )
    # The acquisition order is chronological. Parent edges carry
    # branches; absence of a selected leaf must not invent one on a fork.
    children = {message.parent_message_provider_id for message in messages if message.parent_message_provider_id}
    leaves = [message for message in messages if message.provider_message_id not in children]
    leaf = leaves[0] if len(leaves) == 1 else None
    active: set[str] = set()
    by_id = {message.provider_message_id: message for message in messages if message.provider_message_id}
    current = leaf
    while current is not None and current.provider_message_id not in active:
        active.add(current.provider_message_id)
        current = by_id.get(current.parent_message_provider_id or "")
    for message in messages:
        message.is_active_leaf = message is leaf if leaf is not None else None
        message.is_active_path = message.provider_message_id in active if leaf is not None else None
    created_at = _timestamp_text(conversation.get("createTime"))
    updated_at = _timestamp_text(conversation.get("modifyTime"))
    events.append(
        ParsedSessionEvent(event_type="grok_conversation_state", timestamp=updated_at, payload=dict(conversation))
    )
    if payload.get("response_nodes") is not None:
        events.append(
            ParsedSessionEvent(
                event_type="grok_response_nodes", timestamp=updated_at, payload={"reply": payload["response_nodes"]}
            )
        )
    title = _string(conversation.get("title"))
    return ParsedSession(
        source_name=Provider.GROK,
        provider_session_id=str(conversation["conversationId"]),
        title=title or fallback_id,
        title_source=TitleSource.ORIGIN if title else None,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        unit_accounting=ledger.close(),
        attachments=attachments,
        session_events=events,
        active_leaf_message_provider_id=leaf.provider_message_id if leaf is not None else None,
    )


def parse_native_bundle(payload: Mapping[str, object], fallback_id: str) -> list[ParsedSession]:
    """Parse one acquired endpoint bundle through the ordinary Grok owner.

    This ordinary model route materializes Python strings and block lists.
    It does not establish scalar-independent capture or lowering memory.
    """
    if not looks_like_native_bundle(payload):
        raise ValueError("invalid Grok endpoint bundle")
    return [parse_conversation(payload, fallback_id)]


def native_detection_projection() -> DetectorProjection:
    """Preserve the native predicate's exact nested identity and list shapes."""
    identity = DetectorProjection(fields={"conversationId": DetectorProjection()})
    conversation = DetectorProjection(
        fields={
            "conversationId": DetectorProjection(),
            "conversation": identity,
        }
    )
    return DetectorProjection(
        fields={
            "conversation": conversation,
            "responses": DetectorProjection(fields={"responses": DetectorProjection()}),
        }
    )
