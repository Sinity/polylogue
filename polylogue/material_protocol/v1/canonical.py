"""Deterministic canonical-JSON framing shared by every record and the manifest.

Framing preserves opaque values and mapping keys exactly. Record builders fold
only declared prose fields in NFC before sorted-key JSON serialization.
"""

from __future__ import annotations

from dataclasses import replace

from polylogue.core.digest import IDENTITY, CanonicalizationError
from polylogue.core.digest import canonical_bytes as canonical_profile_bytes
from polylogue.core.json import JSONValue, loads
from polylogue.material_protocol.v1.errors import MaterialValueError

_MATERIAL_PROTOCOL_V1 = replace(
    IDENTITY, name="material-protocol-v1", strict_object_keys=True, normalize_unicode=False, non_finite="reject"
)


def canonical_bytes(value: JSONValue) -> bytes:
    """Serialize *value* to exact-value, sorted-key JSON bytes.

    No trailing newline -- callers that frame this as an NDJSON line append
    ``b"\\n"`` themselves so the digest/line-length story stays explicit.
    """
    try:
        return canonical_profile_bytes(value, _MATERIAL_PROTOCOL_V1)
    except (CanonicalizationError, TypeError, ValueError) as exc:
        raise MaterialValueError(str(exc)) from exc


def canonical_line(value: JSONValue) -> bytes:
    """Serialize *value* to one canonical NDJSON line, including the trailing LF."""
    return canonical_bytes(value) + b"\n"


def parse_json_value(data: bytes) -> JSONValue:
    """Parse JSON bytes back into a JSONValue (used by decode/verify)."""
    try:
        return loads(data)
    except (ValueError, TypeError) as exc:
        raise MaterialValueError("invalid material JSON") from exc


__all__ = ["canonical_bytes", "canonical_line", "parse_json_value"]
