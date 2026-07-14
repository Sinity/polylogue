from __future__ import annotations

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
