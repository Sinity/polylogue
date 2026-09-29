"""Streamed JSON primitives shared by capture admission and capture ingest.

The ingest worker decodes a retained browser capture through
:func:`load_capture_for_ingest`; the receiver's admission fold
(``capture_stream``) is built on the same event helpers. Keeping them here
keeps the receiver out of the ingest route's import closure.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import IO

import ijson
from ijson.common import ObjectBuilder

from polylogue.browser_capture.models import ATTACHMENT_CARRIER_FIELDS, SpilledCarrier

_START = frozenset({"start_map", "start_array"})
_END = frozenset({"end_map", "end_array"})

_Event = tuple[str, object]


#: Base64 characters decoded per step when hashing a carrier (a multiple of 4).
_CARRIER_DECODE_CHUNK_CHARS = 1 << 22


def iter_carrier_bytes(value: str) -> Iterator[bytes]:
    """Decode a ``content_base64``-style carrier in pieces.

    Decoded in 4-aligned chunks, so the decoded payload never exists as one
    buffer beside the encoded string. Padding may only end the carrier; a
    carrier with ``=`` elsewhere is decoded whole so it is accepted or
    refused exactly as a one-shot ``b64decode(validate=True)`` would. Raises
    ``ValueError`` (``binascii.Error``) on a malformed carrier.
    """
    start = 0
    if value.startswith("data:"):
        marker = value.find(";base64,")
        if marker >= 0:
            start = marker + len(";base64,")
    if value.find("=", start, max(start, len(value) - 2)) >= 0:
        yield base64.b64decode(value[start:], validate=True)
        return
    step = _CARRIER_DECODE_CHUNK_CHARS
    for offset in range(start, len(value), step):
        yield base64.b64decode(value[offset : offset + step], validate=True)


def _json_events(handle: IO[bytes]) -> Iterator[_Event]:
    """Parse events with numbers as the stdlib decoder reads them.

    ijson's float mode goes through a native double and can overflow on a
    valid integer token wider than 64 bits. The default mode yields exact
    ``int`` for integer tokens and ``Decimal`` for the rest; each ``Decimal``
    becomes the ``float`` ``json.loads`` would produce, so hashing and model
    validation see the same values as before streaming.
    """
    for event, value in ijson.basic_parse(handle):
        yield event, float(value) if isinstance(value, Decimal) else value


def _build(events: Iterator[_Event], event: str, value: object) -> object:
    """Materialize one value (a turn, an attachment, a small field)."""
    if event not in _START:
        return value
    builder = ObjectBuilder()
    builder.event(event, value)
    depth = 1
    while depth:
        event, value = next(events)
        builder.event(event, value)
        if event in _START:
            depth += 1
        elif event in _END:
            depth -= 1
    return builder.value


def _skip(events: Iterator[_Event], event: str) -> None:
    if event not in _START:
        return
    depth = 1
    while depth:
        event, _ = next(events)
        if event in _START:
            depth += 1
        elif event in _END:
            depth -= 1


#: Decodes one attachment carrier into the blob store; ``None`` keeps the
#: carrier inline (it is malformed and the parser refuses it as before).
CarrierSpill = Callable[[str, str], SpilledCarrier | None]


def _load_attachment(events: Iterator[_Event], event: str, value: object, spill: CarrierSpill) -> object:
    if event != "start_map":
        return _build(events, event, value)
    attachment: dict[str, object] = {}
    while True:
        event, value = next(events)
        if event == "end_map":
            return attachment
        key = str(value)
        event, value = next(events)
        if key in ATTACHMENT_CARRIER_FIELDS and event == "string":
            assert isinstance(value, str)
            spilled = spill(key, value)
            attachment[key] = spilled if spilled is not None else value
        else:
            attachment[key] = _build(events, event, value)


def _load_attachments(events: Iterator[_Event], event: str, value: object, spill: CarrierSpill) -> object:
    if event != "start_array":
        return _build(events, event, value)
    attachments: list[object] = []
    while True:
        event, value = next(events)
        if event == "end_array":
            return attachments
        attachments.append(_load_attachment(events, event, value, spill))


def _load_members(
    events: Iterator[_Event],
    event: str,
    value: object,
    loaders: dict[str, Callable[[Iterator[_Event], str, object], object]],
) -> object:
    """Build an object, loading the members named in ``loaders`` with them."""
    if event != "start_map":
        return _build(events, event, value)
    document: dict[str, object] = {}
    while True:
        event, value = next(events)
        if event == "end_map":
            return document
        key = str(value)
        event, value = next(events)
        loader = loaders.get(key)
        document[key] = loader(events, event, value) if loader is not None else _build(events, event, value)


def _load_items(
    events: Iterator[_Event], event: str, value: object, load: Callable[[Iterator[_Event], str, object], object]
) -> object:
    if event != "start_array":
        return _build(events, event, value)
    items: list[object] = []
    while True:
        event, value = next(events)
        if event == "end_array":
            return items
        items.append(load(events, event, value))


def load_capture_for_ingest(handle: IO[bytes], spill: CarrierSpill) -> object:
    """Decode a retained capture document from a stream for parsing.

    The file is never read whole. Each attachment byte carrier under
    ``session.attachments`` and ``session.turns[].attachments`` is handed to
    ``spill`` as soon as it is read and replaced by the returned
    :class:`SpilledCarrier`, so no carrier is held beside the rest of the
    document; the decoded tree holds the conversation itself. Numbers decode
    as the stdlib decoder reads them, and a later duplicate key wins. Raises
    ``ValueError`` when the bytes are not one JSON document.
    """

    def attachments(events: Iterator[_Event], event: str, value: object) -> object:
        return _load_attachments(events, event, value, spill)

    def turn(events: Iterator[_Event], event: str, value: object) -> object:
        return _load_members(events, event, value, {"attachments": attachments})

    def turns(events: Iterator[_Event], event: str, value: object) -> object:
        return _load_items(events, event, value, turn)

    def session(events: Iterator[_Event], event: str, value: object) -> object:
        return _load_members(events, event, value, {"turns": turns, "attachments": attachments})

    events: Iterator[_Event] = _json_events(handle)
    try:
        event, value = next(events)
        document = _load_members(events, event, value, {"session": session})
        for _ in events:
            raise ValueError("content after the JSON document")
    except ijson.JSONError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    except StopIteration as exc:
        raise ValueError("truncated JSON document") from exc
    return document


__all__ = ["CarrierSpill", "iter_carrier_bytes", "load_capture_for_ingest"]
