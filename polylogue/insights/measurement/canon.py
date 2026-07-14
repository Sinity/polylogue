"""Content-addressed canonicalization shared by measurement definitions.

Rides the same canonical-hash discipline the substrate lane's ``query:<hash>``
canonicalizer uses (rxdo.2): recursively normalize a definition payload into
a hash-stable shape and hash it with SHA-256 via
:func:`polylogue.core.hashing.hash_payload`. Two definitions with
semantically identical, order-normalized content therefore always resolve to
the same content-address ref -- the mechanism that lets ``metric:<hash>``
(rxdo.9.1) and ``ranker:<hash>`` act as a single shared identity rather than
a family of competing registries (see the rxdo.9.1 authoritative corrective
AC: "one hash/ref resolves through both query/analysis and
statistical-registry paths"). Like ``core.query_identity``'s ``_nfc``/
``_canonical_value``, every string (both mapping keys and scalar values) is
NFC-normalized before hashing -- ``hash_payload`` itself deliberately does
*not* normalize (see its docstring), so callers own that step.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping

from polylogue.core.hashing import hash_payload

JSONScalar = str | int | float | bool | None


def canonicalize(value: object) -> object:
    """Recursively normalize a definition payload into a hash-stable shape.

    - Mappings become ``dict``s with keys sorted and values canonicalized.
      Keys are NFC-normalized before sorting/emitting.
    - ``set``/``frozenset`` (order-independent by construction) become a
      list sorted by each element's canonical JSON representation, so
      insertion order never affects the hash.
    - ``list``/``tuple`` preserve declared order -- callers whose field is
      semantically order-independent (e.g. a set of filter clauses) must
      pass a ``set``/``frozenset``, not a list, to get order-invariant
      identity.
    - ``str`` scalars are NFC-normalized so visually identical strings in
      different Unicode normalization forms hash identically. Other JSON
      scalars pass through unchanged.

    Raises:
        TypeError: if ``value`` (or a nested value) is not one of the
            shapes above.
    """
    if isinstance(value, Mapping):
        return {
            _nfc(str(key)): canonicalize(inner) for key, inner in sorted(value.items(), key=lambda kv: _nfc(str(kv[0])))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(inner) for inner in value), key=_sort_key)
    if isinstance(value, (list, tuple)):
        return [canonicalize(inner) for inner in value]
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, JSONScalar):
        return value
    raise TypeError(f"cannot canonicalize value of type {type(value)!r}: {value!r}")


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_ref(kind: str, payload: Mapping[str, object]) -> str:
    """Compute a ``<kind>:<hash>`` content-address ref for a definition payload."""

    return f"{kind}:{hash_payload(canonicalize(payload))}"


__all__ = ["canonicalize", "content_ref"]
