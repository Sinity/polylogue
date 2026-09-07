"""The NFC fast path must be exact, not merely fast.

Anti-vacuity: make :func:`nfc` return its argument for a non-ASCII string
(the fast path applied one type-check too widely) and both
``test_non_ascii_decomposed_still_normalizes`` and
``test_nfc_matches_unicodedata_for_any_text`` go red.
"""

from __future__ import annotations

import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from polylogue.core.text_identity import nfc


@given(st.text(alphabet=st.characters(max_codepoint=127)))
def test_ascii_text_is_already_nfc(text: str) -> None:
    """Every ASCII string the fast path takes is its own NFC form."""
    assert text.isascii()
    assert unicodedata.normalize("NFC", text) == text
    assert nfc(text) == text


@given(st.text())
def test_nfc_matches_unicodedata_for_any_text(text: str) -> None:
    assert nfc(text) == unicodedata.normalize("NFC", text)


def test_non_ascii_decomposed_still_normalizes() -> None:
    decomposed = "cafe\u0301"
    composed = "caf\u00e9"
    assert not decomposed.isascii()
    assert decomposed != composed
    assert nfc(decomposed) == composed
    assert nfc(composed) == composed


def test_non_ascii_already_composed_is_unchanged() -> None:
    text = "\u4e16\u754c \U0001f30d"
    assert nfc(text) == unicodedata.normalize("NFC", text)
