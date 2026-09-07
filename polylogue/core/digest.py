"""Canonical JSON bytes and their SHA-256 digest under a named domain profile.

A digest is comparable only to another digest produced the same way, so the
decisions that move bytes -- Unicode normalization, ASCII escaping, which JSON
encoder formats the floats, which values are inadmissible -- belong to a named
profile rather than to each call site. Sorted object keys and compact
separators are kernel invariants: no domain has ever wanted anything else.

The profiles are not interchangeable. :data:`IDENTITY` formats numbers through
``core.json`` and the rest through stdlib ``json``; the two disagree on the
exponent form of small magnitudes (``1e-9`` against ``1e-09``, about 0.8% of
finite doubles), so moving a domain from one to the other is a versioned break
of every digest it has already written.
"""

from __future__ import annotations

import hashlib
import json as _stdlib_json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal

from polylogue.core.json import dumps_bytes

Encoder = Literal["core-json", "stdlib"]
Numbers = Literal["json", "exact-integer"]
NonFinite = Literal["encode", "reject"]

#: Largest integer a IEEE-754 double represents exactly. A profile declaring
#: ``numbers="exact-integer"`` refuses anything a JavaScript consumer would
#: silently round on the way back.
EXACT_INTEGER_LIMIT: Final = 9_007_199_254_740_991


class CanonicalizationError(ValueError):
    """A payload is inadmissible under the requested profile."""


class KeyCollisionError(CanonicalizationError):
    """Two distinct object keys normalize to one, which would drop a field."""


@dataclass(frozen=True, slots=True)
class DigestProfile:
    """The complete set of encoding decisions one domain's digests depend on."""

    name: str
    encoder: Encoder
    normalize_unicode: bool
    strict_object_keys: bool
    ensure_ascii: bool
    numbers: Numbers
    non_finite: NonFinite
    digest_prefix: str = ""


#: Material-protocol record and manifest framing: NFC-normalized, encoded
#: through ``core.json`` so checked-in fixture bytes stay reproducible.
#: Non-finite floats reach the encoder, which writes them as ``null``; JSON has
#: no non-finite literal, so no decoded provider payload can carry one.
IDENTITY: Final = DigestProfile(
    name="identity",
    encoder="core-json",
    normalize_unicode=True,
    strict_object_keys=False,
    ensure_ascii=False,
    numbers="json",
    non_finite="encode",
)

#: Durable user-tier receipts: annotation batch provenance, ontology
#: assertion ids, annotation-schema definition fingerprints. Object keys must
#: be strings and must stay distinct under NFC, so two different keys can
#: never collapse into one and silently drop a field from the receipt.
RECEIPT: Final = DigestProfile(
    name="receipt",
    encoder="stdlib",
    normalize_unicode=True,
    strict_object_keys=True,
    ensure_ascii=False,
    numbers="json",
    non_finite="reject",
)

#: Receipts whose payload is already normalized and carries opaque reference
#: tokens. Normalizing again would rewrite ref bytes and conflate canonically
#: equivalent yet distinct ids, so this profile differs from :data:`RECEIPT` in
#: exactly one declared decision.
REFERENCE: Final = DigestProfile(
    name="reference",
    encoder="stdlib",
    normalize_unicode=False,
    strict_object_keys=True,
    ensure_ascii=False,
    numbers="json",
    non_finite="reject",
)

#: Query-plan identity and the general payload digest behind result-set and
#: watch identities. Deliberately ASCII-escaped and *not* normalized here:
#: `core.query_identity` normalizes the plan itself, where it can tell a
#: field token from a literal.
QUERY: Final = DigestProfile(
    name="query",
    encoder="stdlib",
    normalize_unicode=False,
    strict_object_keys=False,
    ensure_ascii=True,
    numbers="json",
    non_finite="encode",
)

#: Browser capture-job intents and checkpoints. The digest is compared against
#: one recomputed by the extension in JavaScript, so the admissible value
#: vocabulary stops at what a double represents exactly: no floats, no integer
#: outside :data:`EXACT_INTEGER_LIMIT`.
CAPTURE: Final = DigestProfile(
    name="capture",
    encoder="stdlib",
    normalize_unicode=True,
    strict_object_keys=True,
    ensure_ascii=False,
    numbers="exact-integer",
    non_finite="reject",
    digest_prefix="sha256:",
)

PROFILES: Final[Mapping[str, DigestProfile]] = {
    profile.name: profile for profile in (IDENTITY, RECEIPT, REFERENCE, QUERY, CAPTURE)
}


def nfc(value: str) -> str:
    """Return *value* under Unicode NFC, the one normalization every profile uses."""
    return unicodedata.normalize("NFC", value)


def _prepared(value: object, profile: DigestProfile) -> object:
    if isinstance(value, str):
        return nfc(value) if profile.normalize_unicode else value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if profile.numbers == "exact-integer" and abs(value) > EXACT_INTEGER_LIMIT:
            raise CanonicalizationError(f"{profile.name} profile admits no integer beyond {EXACT_INTEGER_LIMIT}")
        return value
    if isinstance(value, float):
        if profile.numbers == "exact-integer":
            raise CanonicalizationError(f"{profile.name} profile admits no floating-point number")
        if not isfinite(value) and profile.non_finite == "reject":
            raise CanonicalizationError(f"{profile.name} profile admits no non-finite number")
        return value
    if isinstance(value, Mapping):
        return _prepared_mapping(value, profile)
    if isinstance(value, (list, tuple)):
        return [_prepared(item, profile) for item in value]
    if profile.strict_object_keys:
        raise CanonicalizationError(f"{profile.name} profile admits no value of type {type(value).__name__}")
    # A permissive profile leaves the value for the encoder, whose own
    # TypeError is the contract its callers already handle.
    return value


def _prepared_mapping(value: Mapping[str, object], profile: DigestProfile) -> dict[str, object]:
    prepared: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            if profile.strict_object_keys:
                raise CanonicalizationError(f"{profile.name} profile admits only string object keys")
            key = str(key)
        normalized = nfc(key) if profile.normalize_unicode else key
        if profile.strict_object_keys and normalized in prepared:
            raise KeyCollisionError(f"NFC-normalized JSON keys collide at {normalized!r}")
        prepared[normalized] = _prepared(item, profile)
    return prepared


def _passthrough(profile: DigestProfile) -> bool:
    """Report whether the preparation pass would leave every payload unchanged."""
    return (
        not profile.normalize_unicode
        and not profile.strict_object_keys
        and profile.numbers == "json"
        and profile.non_finite == "encode"
    )


def canonical_bytes(value: object, profile: DigestProfile) -> bytes:
    """Return the canonical UTF-8 JSON bytes of *value* under *profile*.

    Raises :class:`CanonicalizationError` for a payload the profile declares
    inadmissible. Anything the profile admits but the encoder cannot represent
    raises the encoder's own ``TypeError``/``UnicodeEncodeError``.
    """
    prepared = value if _passthrough(profile) else _prepared(value, profile)
    if profile.encoder == "core-json":
        return dumps_bytes(prepared, sort_keys=True)
    return _stdlib_json.dumps(
        prepared,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=profile.ensure_ascii,
    ).encode("utf-8")


def digest(value: object, profile: DigestProfile) -> str:
    """Return the SHA-256 hex digest of *value*'s canonical bytes under *profile*."""
    return profile.digest_prefix + hashlib.sha256(canonical_bytes(value, profile)).hexdigest()


def normalized(value: object) -> object:
    """Recursively NFC-normalize strings and object keys in a JSON-shaped value.

    Exposed for the surfaces that must hand a *normalized value* to something
    other than this module's encoder; a caller that only needs bytes or a
    digest states its profile instead.
    """
    if isinstance(value, str):
        return nfc(value)
    if isinstance(value, Mapping):
        return {nfc(str(key)): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return [normalized(item) for item in value]
    return value


def profile_for(name: str) -> DigestProfile:
    """Return the registered profile *name*, or fail closed."""
    try:
        return PROFILES[name]
    except KeyError:
        raise CanonicalizationError(f"unknown digest profile: {name!r}") from None


__all__ = [
    "CAPTURE",
    "EXACT_INTEGER_LIMIT",
    "IDENTITY",
    "PROFILES",
    "QUERY",
    "RECEIPT",
    "REFERENCE",
    "CanonicalizationError",
    "DigestProfile",
    "KeyCollisionError",
    "canonical_bytes",
    "digest",
    "nfc",
    "normalized",
    "profile_for",
]
