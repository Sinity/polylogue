"""Structural content identity for decoded provider values.

Byte equality and canonical-JSON digests both answer the wrong question about
an export member: a provider re-serializing the same conversation with
different separators, key order, or ``1`` written as ``1.0`` produces different
bytes for identical content. The identity below is computed over the *decoded*
value under the provider value contract, so it is stable across serialization
while it still separates values JSON itself distinguishes -- ``true`` from
``1``, ``"1"`` from ``1``, an absent key from a null one.
"""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from math import isfinite

from polylogue.core.text_identity import nfc

_CONTENT_IDENTITY_DOMAIN = b"polylogue:member-content:v1\0"


def _encode(value: object, out: list[bytes]) -> None:
    if value is None:
        out.append(b"z;")
        return
    # Ordered before ``int``: ``bool`` is a subclass of ``int`` and ``True``
    # must never share an identity with ``1``.
    if isinstance(value, bool):
        out.append(b"b1;" if value else b"b0;")
        return
    if isinstance(value, Decimal):
        _encode_decimal(value, out)
        return
    if isinstance(value, int):
        out.append(b"i%d;" % value)
        return
    if isinstance(value, float):
        _encode_float(value, out)
        return
    if isinstance(value, str):
        _encode_text(b"s", value, out)
        return
    if isinstance(value, (list, tuple)):
        out.append(b"a%d;" % len(value))
        for item in value:
            _encode(item, out)
        return
    if isinstance(value, dict):
        items = sorted((nfc(str(key)), item) for key, item in value.items())
        out.append(b"o%d;" % len(items))
        for key, item in items:
            _encode_text(b"k", key, out)
            _encode(item, out)
        return
    raise TypeError(f"value of type {type(value).__name__} has no structural content identity")


def _encode_text(tag: bytes, value: str, out: list[bytes]) -> None:
    encoded = nfc(value).encode("utf-8", errors="surrogatepass")
    out.append(b"%s%d:" % (tag, len(encoded)))
    out.append(encoded)
    out.append(b";")


def _encode_decimal(value: Decimal, out: list[bytes]) -> None:
    if not value.is_finite():
        raise ValueError("a non-finite number has no structural content identity")
    if value == value.to_integral_value():
        out.append(b"i%d;" % int(value))
        return
    _encode_float(float(value), out)


def _encode_float(value: float, out: list[bytes]) -> None:
    if not isfinite(value):
        raise ValueError("a non-finite number has no structural content identity")
    # The provider value contract has one numeric type. An integral float is
    # the same number as the integer it equals, so ``1.0`` and ``1`` share an
    # identity; a fractional value keeps the shortest round-trip form, which
    # is equal exactly when the two floats are equal.
    if value.is_integer():
        out.append(b"i%d;" % int(value))
        return
    out.append(b"f%s;" % repr(value).encode("ascii"))


def structural_content_identity(value: object) -> str:
    """Return the content identity digest of one decoded provider value."""
    out: list[bytes] = [_CONTENT_IDENTITY_DOMAIN]
    _encode(value, out)
    return sha256(b"".join(out)).hexdigest()


def structurally_equal(left: object, right: object) -> bool:
    """Report whether two decoded values are the same content."""
    return structural_content_identity(left) == structural_content_identity(right)


__all__ = ["structural_content_identity", "structurally_equal"]
