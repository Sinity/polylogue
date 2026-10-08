"""Serialize canonical native parsing into an unfinished capture envelope.

Authority and durable custody belong to CaptureJobRegistry. This module only
lowers the ordinary parser's scratch-backed rows into the existing envelope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from typing import BinaryIO, cast

from polylogue.browser_capture.models import BrowserCaptureProvenance
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base_models import ParsedAttachment, ParsedSession
from polylogue.sources.prepared_message_sink import ScratchSessionSpill


def json_chunks(value: object, *, sort_keys: bool = False) -> Iterator[bytes]:
    """Yield compact ASCII JSON without the C encoder's dict-subclass shortcut.

    ``StreamedJSONDocument`` mappings store values outside ``dict``'s inherited
    storage. Its Python ``items()`` view keeps arbitrary provider metadata
    visible while each scalar is encoded.
    """
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"))
    if not sort_keys:
        for piece in encoder.iterencode(value):
            if piece:
                yield piece.encode("ascii")
        return

    from polylogue.schemas.observation_spill import SpilledObject

    def ordered(item: object) -> Iterator[bytes]:
        if isinstance(item, dict):
            yield b"{"
            keys = item.sorted_keys() if isinstance(item, SpilledObject) else sorted(item)
            for index, key in enumerate(keys):
                if not isinstance(key, str):
                    raise TypeError("native JSON object keys must be strings")
                if index:
                    yield b","
                yield from ordered(key)
                yield b":"
                yield from ordered(item[key])
            yield b"}"
        elif isinstance(item, (list, tuple)):
            yield b"["
            for index, child in enumerate(item):
                if index:
                    yield b","
                yield from ordered(child)
            yield b"]"
        else:
            for piece in encoder.iterencode(item):
                if piece:
                    yield piece.encode("ascii")

    yield from ordered(value)


def json_bytes(value: object) -> bytes:
    # ASCII escapes preserve provider lone surrogates and exact scalar values.
    return b"".join(json_chunks(value))


def raw_chunks(handle: BinaryIO, progress: Callable[[], None]) -> Iterator[bytes]:
    handle.seek(0)
    while chunk := handle.read(64 * 1024):
        progress()
        yield chunk


def envelope_prefix(
    parsed: ParsedSession,
    spill: ScratchSessionSpill,
    members: Mapping[str, BinaryIO],
    provenance: BrowserCaptureProvenance,
    metadata: dict[str, object],
    progress: Callable[[], None],
) -> Iterator[bytes]:
    """Keep literal raw and all canonical turns, leaving asset suffix unsealed."""
    yield b'{"polylogue_capture_kind":"browser_llm_session","schema_version":1,"source":"browser-extension","provenance":'
    yield json_bytes(provenance.model_dump(mode="json"))
    yield b',"provider_meta":'
    yield from json_chunks(metadata)
    yield b',"raw_provider_payload":'
    if parsed.source_name is Provider.GROK:
        yield b'{"conversation":'
        yield from raw_chunks(members["conversation"], progress)
        yield b',"responses":'
        yield from raw_chunks(members["responses"], progress)
        if "response_nodes" in members:
            yield b',"response_nodes":'
            yield from raw_chunks(members["response_nodes"], progress)
        yield b"}"
    else:
        yield from raw_chunks(members["conversation"], progress)
    header = {
        "provider": parsed.source_name.value,
        "provider_session_id": parsed.provider_session_id,
        "session_kind": "temporary" if parsed.session_kind.value == "temporary" else "standard",
        "title": parsed.title,
        "created_at": parsed.created_at,
        "updated_at": parsed.updated_at,
    }
    yield b',"session":'
    yield json_bytes(header)[:-1]
    yield b',"turns":['
    owner_ordinals = spill.string_map()
    for ordinal, message in enumerate(parsed.messages):
        progress()
        if message.position is not None:
            owner_ordinals[str(message.position)] = str(ordinal)
        if ordinal:
            yield b","
        yield json_bytes(
            {
                "provider_turn_id": message.provider_message_id,
                "role": message.role.value,
                "text": message.text,
                "timestamp": message.timestamp,
                "ordinal": ordinal,
                "parent_turn_id": message.parent_message_provider_id,
                "blocks": [block.model_dump(mode="json") for block in message.blocks],
            }
        )
    if not parsed.messages:
        raise ValueError("native capture has no canonical messages")
    for attachment_ordinal, attachment in enumerate(parsed.attachments):
        progress()
        raw_position = attachment.message_position
        if parsed.source_name is Provider.CHATGPT:
            raw_position = spill.attachment_record_origin(attachment_ordinal)
        owner = owner_ordinals.get(str(raw_position)) if raw_position is not None else None
        owner_ordinal = int(owner) if owner is not None else None
        descriptor = attachment_descriptor(attachment, owner_ordinal)
        descriptor_metadata = cast(dict[str, object], descriptor["provider_meta"])
        descriptor_metadata["native_attachment_ordinal"] = attachment_ordinal
        descriptor_metadata["native_raw_position"] = raw_position
        if parsed.source_name is Provider.CHATGPT:
            descriptor["original_record_ordinal"] = raw_position
        elif parsed.source_name is Provider.GROK and owner_ordinal is not None:
            descriptor["original_record_ordinal"] = int(spill.record_origin(owner_ordinal))
        if parsed.source_name is Provider.CHATGPT:
            row = spill.store.conn.execute(
                "SELECT node_key FROM chatgpt_node WHERE ordinal=?", (descriptor["original_record_ordinal"],)
            ).fetchone()
            if row is None:
                raise ValueError("native attachment raw occurrence is unavailable")
            descriptor["original_record_key"] = row[0]
        spill.store.conn.execute(
            "INSERT INTO capture_preparation_plan VALUES (?, ?)",
            (attachment_ordinal, json_bytes(descriptor).decode("ascii")),
        )
    yield b'],"attachments":['


def attachment_descriptor(attachment: ParsedAttachment, owner_ordinal: int | None) -> dict[str, object]:
    """Keep one occurrence for every canonical parser attachment."""
    descriptor_metadata: dict[str, object] = {
        "native_turn_ordinal": owner_ordinal,
        "provider_file_id": attachment.provider_file_id,
        "provider_drive_id": attachment.provider_drive_id,
        "path": attachment.path,
    }
    descriptor: dict[str, object] = {
        "provider_attachment_id": attachment.provider_attachment_id,
        "message_provider_id": attachment.message_provider_id,
        "attachment_kind": attachment.attachment_kind,
        "name": attachment.name,
        "mime_type": attachment.mime_type,
        "size_bytes": attachment.size_bytes,
        "url": attachment.source_url,
        "provider_meta": descriptor_metadata,
    }
    if attachment.inline_bytes is not None:
        descriptor_metadata["native_inline_sha256"] = hashlib.sha256(attachment.inline_bytes).hexdigest()
        descriptor_metadata["native_inline_size_bytes"] = len(attachment.inline_bytes)
    return descriptor
