"""Deterministic canonical-JSON framing shared by every record and the manifest.

Byte-stability (decode/re-encode is byte-identical, checked-in fixture bytes
match a fresh encode) depends on exactly one canonicalization rule applied
everywhere: recursively NFC-normalize every string, then serialize with
sorted object keys and no incidental whitespace. Key order therefore never
carries meaning in this protocol -- consumers must not depend on it.
"""

from __future__ import annotations

import unicodedata

import orjson

from polylogue.core.json import JSONValue, loads

_ORJSON_CANONICAL_OPTIONS = orjson.OPT_SORT_KEYS


def nfc_normalize(value: JSONValue) -> JSONValue:
    """Recursively NFC-normalize every string in a JSON-compatible value."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [nfc_normalize(item) for item in value]
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", key): nfc_normalize(item) for key, item in value.items()}
    return value


def canonical_bytes(value: JSONValue) -> bytes:
    """Serialize *value* to canonical (NFC-normalized, sorted-key) JSON bytes.

    No trailing newline -- callers that frame this as an NDJSON line append
    ``b"\\n"`` themselves so the digest/line-length story stays explicit.
    """
    normalized = nfc_normalize(value)
    return orjson.dumps(normalized, option=_ORJSON_CANONICAL_OPTIONS)


def canonical_line(value: JSONValue) -> bytes:
    """Serialize *value* to one canonical NDJSON line, including the trailing LF."""
    return canonical_bytes(value) + b"\n"


def parse_json_value(data: bytes) -> JSONValue:
    """Parse JSON bytes back into a JSONValue (used by decode/verify)."""
    return loads(data)


__all__ = ["canonical_bytes", "canonical_line", "nfc_normalize", "parse_json_value"]
