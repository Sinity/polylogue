"""Deterministic canonical-JSON framing shared by every record and the manifest.

Byte-stability (decode/re-encode is byte-identical, checked-in fixture bytes
match a fresh encode) depends on exactly one canonicalization rule applied
everywhere: recursively NFC-normalize every string, then serialize with
sorted object keys and no incidental whitespace. Key order therefore never
carries meaning in this protocol -- consumers must not depend on it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from polylogue.core.digest import IDENTITY
from polylogue.core.digest import canonical_bytes as canonical_profile_bytes
from polylogue.core.json import JSONValue, loads

_MATERIAL_PROTOCOL_V1 = replace(IDENTITY, name="material-protocol-v1", strict_object_keys=True)


def nfc_normalize(value: JSONValue) -> JSONValue:
    """Normalize a material value without allowing two keys to become one."""
    return cast(JSONValue, parse_json_value(canonical_bytes(value)))


def canonical_bytes(value: JSONValue) -> bytes:
    """Serialize *value* to canonical (NFC-normalized, sorted-key) JSON bytes.

    No trailing newline -- callers that frame this as an NDJSON line append
    ``b"\\n"`` themselves so the digest/line-length story stays explicit.
    """
    return canonical_profile_bytes(value, _MATERIAL_PROTOCOL_V1)


def canonical_line(value: JSONValue) -> bytes:
    """Serialize *value* to one canonical NDJSON line, including the trailing LF."""
    return canonical_bytes(value) + b"\n"


def parse_json_value(data: bytes) -> JSONValue:
    """Parse JSON bytes back into a JSONValue (used by decode/verify)."""
    return loads(data)


__all__ = ["canonical_bytes", "canonical_line", "nfc_normalize", "parse_json_value"]
