"""Root-field envelopes of JSON documents, read without holding the document.

Structural signatures (source-class recognition, sidecar dispatch identity)
decide from a document's root fields: their presence, type and short leading
text. :func:`top_level_envelopes` streams a document and keeps exactly that,
so a signature gives the same answer for a document of any size.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO

#: Characters of a top-level string an envelope keeps. The source-class
#: signatures read only the type and short leading text of root fields.
ENVELOPE_TEXT_PREFIX_CHARS = 4096


def _envelope_scalar(value: object) -> object:
    if isinstance(value, str) and len(value) > ENVELOPE_TEXT_PREFIX_CHARS:
        return value[:ENVELOPE_TEXT_PREFIX_CHARS]
    return value


def top_level_envelopes(handle: IO[bytes], *, multiple_values: bool, expand_arrays: bool) -> Iterator[object]:
    """Stream each top-level JSON value's envelope without holding the value.

    An object's envelope keeps its keys with scalar values (strings as a
    leading prefix) and a typed empty placeholder for container values. With
    ``expand_arrays`` a top-level array yields one envelope per element.
    The source-class signatures read exactly these root fields, so the
    envelope decides every signature the decoded document would, at any size.
    """
    import ijson

    depth = 0
    root: object = None
    element: object = None
    key: str | None = None
    element_key: str | None = None
    expanding = False
    for event, value in ijson.basic_parse(handle, multiple_values=multiple_values, use_float=True):
        if event in ("start_map", "start_array"):
            placeholder: object = {} if event == "start_map" else []
            if depth == 0:
                root = placeholder
                expanding = expand_arrays and event == "start_array"
            elif depth == 1 and expanding:
                element = placeholder
            elif depth == 1 and isinstance(root, dict) and key is not None:
                root[key] = placeholder
            elif depth == 2 and expanding and isinstance(element, dict) and element_key is not None:
                element[element_key] = placeholder
            depth += 1
            continue
        if event in ("end_map", "end_array"):
            depth -= 1
            if depth == 0 and not expanding:
                yield root
            elif depth == 1 and expanding:
                yield element
            continue
        if event == "map_key":
            if depth == 1:
                key = str(value)
            elif depth == 2 and expanding:
                element_key = str(value)
            continue
        scalar = _envelope_scalar(value)
        if depth == 0 or depth == 1 and expanding:
            yield scalar
        elif depth == 1 and isinstance(root, dict) and key is not None:
            root[key] = scalar
        elif depth == 2 and expanding and isinstance(element, dict) and element_key is not None:
            element[element_key] = scalar


__all__ = ["ENVELOPE_TEXT_PREFIX_CHARS", "top_level_envelopes"]
