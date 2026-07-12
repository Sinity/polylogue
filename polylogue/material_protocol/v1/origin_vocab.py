"""Public Origin vocabulary version/digest, vendored from polylogue.core.enums.Origin.

The design explicitly requires admission to quarantine on an unknown or stale
vocabulary version: a manifest's declared (version, digest) pair must match a
frozen entry in ``KNOWN_ORIGIN_VOCABULARIES`` below. The digest for the
*current* version is intentionally a frozen literal, not computed live from
``Origin`` at import time -- if someone adds/removes an Origin member without
bumping ``CURRENT_ORIGIN_VOCABULARY_VERSION`` and adding a new frozen entry,
``current_origin_vocabulary_digest()`` (computed live) will stop matching the
frozen literal for the current version and
``tests/unit/material_protocol/v1/test_origin_vocab_pinning.py`` fails loudly
-- that mismatch *is* the latch, not a bug to silence by recomputing.
"""

from __future__ import annotations

from polylogue.core.enums import Origin
from polylogue.core.hashing import hash_bytes
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes
from polylogue.material_protocol.v1.errors import UnknownOriginVocabularyError

#: Version currently emitted by the encoder.
CURRENT_ORIGIN_VOCABULARY_VERSION = 1

#: Frozen (version -> sha256 hex digest of the sorted canonical Origin value
#: list) registry. Only versions listed here are admissible on decode/verify.
#: Computed once via:
#!     python3 -c "
#!     from polylogue.material_protocol.v1.origin_vocab import current_origin_vocabulary_digest
#!     print(current_origin_vocabulary_digest())"
KNOWN_ORIGIN_VOCABULARIES: dict[int, str] = {
    1: "2739614e35c4b934991b72d32dc2266efc328ae79332cc0dcd057bd9fc070acb",
}


def current_origin_vocabulary_digest() -> str:
    """Compute the live digest of the current ``Origin`` enum's value set."""
    values: list[JSONValue] = [origin.value for origin in sorted(Origin, key=lambda origin: origin.value)]
    return hash_bytes(canonical_bytes(values)).lower()


def resolve_current_origin_vocabulary() -> tuple[int, str]:
    """Return (version, digest) for the vocabulary this encoder emits.

    Raises AssertionError (a programmer-facing invariant, not a data-quality
    MaterialProtocolError) if the frozen registry entry for
    CURRENT_ORIGIN_VOCABULARY_VERSION has drifted from the live Origin enum --
    that means Origin changed without a matching protocol vocabulary bump.
    """
    live_digest = current_origin_vocabulary_digest()
    frozen_digest = KNOWN_ORIGIN_VOCABULARIES.get(CURRENT_ORIGIN_VOCABULARY_VERSION)
    assert frozen_digest is not None, (
        f"CURRENT_ORIGIN_VOCABULARY_VERSION={CURRENT_ORIGIN_VOCABULARY_VERSION} has no frozen registry entry"
    )
    assert live_digest == frozen_digest, (
        "polylogue.core.enums.Origin changed without bumping "
        "CURRENT_ORIGIN_VOCABULARY_VERSION and adding a new frozen "
        f"KNOWN_ORIGIN_VOCABULARIES entry (live={live_digest!r}, frozen={frozen_digest!r})"
    )
    return CURRENT_ORIGIN_VOCABULARY_VERSION, frozen_digest


def check_origin_vocabulary(version: int, digest: str) -> None:
    """Raise UnknownOriginVocabularyError unless (version, digest) is a known pair."""
    expected = KNOWN_ORIGIN_VOCABULARIES.get(version)
    if expected is None:
        raise UnknownOriginVocabularyError(f"unknown origin vocabulary version {version!r}")
    if expected != digest:
        raise UnknownOriginVocabularyError(
            f"origin vocabulary version {version!r} digest mismatch: expected {expected!r}, got {digest!r}"
        )


__all__ = [
    "CURRENT_ORIGIN_VOCABULARY_VERSION",
    "KNOWN_ORIGIN_VOCABULARIES",
    "check_origin_vocabulary",
    "current_origin_vocabulary_digest",
    "resolve_current_origin_vocabulary",
]
