"""Latch: polylogue.core.enums.Origin must not drift without a matching
material-protocol vocabulary bump.

``resolve_current_origin_vocabulary`` asserts the frozen registry entry for
``CURRENT_ORIGIN_VOCABULARY_VERSION`` still matches a live-computed digest of
the ``Origin`` enum. If someone adds/removes/renames an Origin member without
bumping the version and adding a new frozen digest, that assertion fires --
which is this test failing, not a bug in this test to silence.
"""

from __future__ import annotations

import pytest

from polylogue.material_protocol.v1.origin_vocab import (
    CURRENT_ORIGIN_VOCABULARY_VERSION,
    KNOWN_ORIGIN_VOCABULARIES,
    check_origin_vocabulary,
    current_origin_vocabulary_digest,
    resolve_current_origin_vocabulary,
)


def test_current_origin_vocabulary_matches_frozen_registry() -> None:
    version, digest = resolve_current_origin_vocabulary()
    assert version == CURRENT_ORIGIN_VOCABULARY_VERSION
    assert digest == KNOWN_ORIGIN_VOCABULARIES[CURRENT_ORIGIN_VOCABULARY_VERSION]
    assert digest == current_origin_vocabulary_digest()


def test_check_origin_vocabulary_accepts_the_current_pair() -> None:
    version, digest = resolve_current_origin_vocabulary()
    check_origin_vocabulary(version, digest)  # must not raise


def test_check_origin_vocabulary_rejects_unknown_version() -> None:
    from polylogue.material_protocol.v1.errors import UnknownOriginVocabularyError

    with pytest.raises(UnknownOriginVocabularyError):
        check_origin_vocabulary(9999, "irrelevant")


def test_check_origin_vocabulary_rejects_digest_mismatch() -> None:
    from polylogue.material_protocol.v1.errors import UnknownOriginVocabularyError

    with pytest.raises(UnknownOriginVocabularyError):
        check_origin_vocabulary(CURRENT_ORIGIN_VOCABULARY_VERSION, "0" * 64)
