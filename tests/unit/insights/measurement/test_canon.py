from __future__ import annotations

import unicodedata

import pytest

from polylogue.insights.measurement.canon import canonicalize, content_ref


def test_canonicalize_sorts_mapping_keys_regardless_of_insertion_order() -> None:
    left = canonicalize({"b": 1, "a": 2})
    right = canonicalize({"a": 2, "b": 1})

    assert left == right == {"a": 2, "b": 1}


def test_canonicalize_orders_sets_by_canonical_representation_not_insertion() -> None:
    first = canonicalize({"x", "a", "m"})
    second = canonicalize({"m", "x", "a"})

    assert first == second == ["a", "m", "x"]


def test_canonicalize_preserves_list_order_as_semantic() -> None:
    assert canonicalize([1, 2, 3]) != canonicalize([3, 2, 1])


def test_canonicalize_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError):
        canonicalize(object())


def test_content_ref_is_stable_across_key_and_set_reordering() -> None:
    first = content_ref("metric", {"construct": "cost", "confounds": {"b", "a"}})
    second = content_ref("metric", {"confounds": {"a", "b"}, "construct": "cost"})

    assert first == second
    assert first.startswith("metric:")
    _, _, digest = first.partition(":")
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_content_ref_changes_with_content() -> None:
    first = content_ref("metric", {"construct": "cost"})
    second = content_ref("metric", {"construct": "latency"})

    assert first != second


def test_canonicalize_nfc_normalizes_string_scalars() -> None:
    # A base letter + combining acute accent (NFD) vs the precomposed
    # single-codepoint form (NFC) are visually identical but byte-distinct
    # until normalized. Construct both explicitly via unicodedata rather
    # than relying on literal source text, which a tool/editor may silently
    # normalize to a single form before it ever reaches this file.
    precomposed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")

    assert decomposed != precomposed  # sanity: genuinely different bytes
    assert canonicalize(decomposed) == canonicalize(precomposed) == precomposed


def test_canonicalize_nfc_normalizes_mapping_keys() -> None:
    precomposed_key = unicodedata.normalize("NFC", "café")
    decomposed_key = unicodedata.normalize("NFD", "café")

    assert decomposed_key != precomposed_key  # sanity: genuinely different bytes
    assert canonicalize({decomposed_key: 1}) == canonicalize({precomposed_key: 1})


def test_content_ref_is_stable_across_unicode_normalization_form() -> None:
    """Two definitions differing only in NFD vs NFC spelling of the same
    construct name must resolve to the same content-address ref -- this is
    the same identity-law violation class as order-dependence, and
    hash_payload deliberately does not normalize on callers' behalf
    (see its docstring), so canon.py must own this step."""

    precomposed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")
    assert decomposed != precomposed  # sanity: genuinely different bytes

    nfd_ref = content_ref("metric", {"construct": decomposed})
    nfc_ref = content_ref("metric", {"construct": precomposed})

    assert nfd_ref == nfc_ref
