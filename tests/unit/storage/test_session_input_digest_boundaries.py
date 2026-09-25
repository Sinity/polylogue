"""The publication fence must not alias distinct SQL values or row boundaries.

Anti-vacuity: the old delimiter-only encoding aliases adversarial values below before hashing. These tests do not depend on a cryptographic collision.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pytest

from polylogue.storage.derived.session.input_binding import SessionInputDigest


def _binding(rows: Sequence[Sequence[object]], relation: str | None = None) -> str:
    digest = SessionInputDigest(("s",))
    for values in rows:
        row = ("s", *values)
        if relation is None:
            digest.add_row(row)
        else:
            digest.add_related_row(relation, row)
    return digest.result()["s"]


@pytest.mark.parametrize("relation", [None, "attachments", "session_events", "provider_usage_events"])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((("alpha\x1fbeta", "gamma"),), (("alpha", "beta\x1fgamma"),)),
        (((None, "x"),), (("", "x"),)),
        (((1, "x"),), (("1", "x"),)),
        ((("a\x1eb",),), (("a",), ("b",))),
        (((),), (("",),)),
    ],
)
def test_distinct_projection_values_have_distinct_bindings(
    relation: str | None, left: Sequence[Sequence[object]], right: Sequence[Sequence[object]]
) -> None:
    assert _binding(left, relation) != _binding(right, relation)


def test_small_adversarial_domain_has_no_encoding_aliases() -> None:
    atoms: tuple[object, ...] = (None, "", 0, 1, 0.0, "0", "1", "\x00", "\x1d", "\x1e", "\x1f", "é", b"")
    seen: dict[str, tuple[object, ...]] = {}
    for width in range(3):
        for row in itertools.product(atoms, repeat=width):
            binding = _binding((row,))
            assert binding not in seen, (seen.get(binding), row)
            seen[binding] = row


def test_relation_is_part_of_the_framed_identity() -> None:
    row = (("a\x1db\x1ec\x1fd", None, ""),)
    assert _binding(row, "attachments") != _binding(row, "session_events")
    assert _binding(row) == _binding(row)
