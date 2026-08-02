"""Truthiness/bounded-access behavior of the lazy membership views in replay.py.

Regression coverage for polylogue-5svw0: ``ArtifactMemberships``,
``MembershipSamples``, ``MembershipSessionIds``, and ``MembershipObservedAts``
previously had no ``__bool__``, so ``if view:`` / ``bool(view)`` fell back to
``Sequence``/``Collection``'s default -- which for these classes means
``__len__``, an O(n) full rescan of the underlying membership source. Same
defect class as PR #3546 (``ReplayableRecordSamples.__bool__``).
``ArtifactMemberships.__getitem__`` also fully materialized via ``list(self)``
for a single bounded index/slice access.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

from polylogue.core.json import JSONValue
from polylogue.schemas.generation.models import _UnitMembership
from polylogue.schemas.generation.replay import (
    ArtifactMemberships,
    MembershipObservedAts,
    MembershipSamples,
    MembershipSessionIds,
)
from polylogue.schemas.observation import SchemaUnit


class _CountingMemberships(Sequence[_UnitMembership]):
    """A lazily-iterated membership source that records how far it was read."""

    def __init__(self, memberships: list[_UnitMembership]) -> None:
        self._memberships = memberships
        self.touched: list[int] = []

    def __iter__(self) -> Iterator[_UnitMembership]:
        for index, membership in enumerate(self._memberships):
            self.touched.append(index)
            yield membership

    def __len__(self) -> int:
        # Exercised only by the (unavoidable) __len__ path -- never by the
        # bounded __bool__ path under test.
        return sum(1 for _ in self)

    @overload
    def __getitem__(self, index: int) -> _UnitMembership: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_UnitMembership]: ...

    def __getitem__(self, index: int | slice) -> _UnitMembership | Sequence[_UnitMembership]:
        return self._memberships[index]


def _membership(
    artifact_kind: str,
    samples: list[dict[str, JSONValue]],
    *,
    session_id: str | None = None,
) -> _UnitMembership:
    unit = SchemaUnit(
        cluster_payload={},
        schema_samples=samples,
        artifact_kind=artifact_kind,
        session_id=session_id,
    )
    return _UnitMembership(unit, "family-a")


def _large_matching_source(count: int) -> _CountingMemberships:
    """``count`` memberships of kind "message", each carrying one sample."""
    memberships = [_membership("message", [{"n": index}]) for index in range(count)]
    return _CountingMemberships(memberships)


def test_artifact_memberships_bool_does_not_force_full_scan() -> None:
    source = _large_matching_source(10_000)
    view = ArtifactMemberships(source, "message")

    assert bool(view) is True
    # A bounded peek only needs to touch the first matching item.
    assert len(source.touched) <= 2, source.touched


def test_artifact_memberships_bool_false_for_no_match() -> None:
    source = _large_matching_source(50)
    view = ArtifactMemberships(source, "does-not-exist")

    assert bool(view) is False
    # Correctly reports empty only after scanning every candidate -- this is
    # the honest cost of a genuinely-empty filtered view, not a regression.
    assert len(source.touched) == 50


def test_artifact_memberships_getitem_int_does_not_force_full_scan() -> None:
    source = _large_matching_source(10_000)
    view = ArtifactMemberships(source, "message")

    first = view[0]

    assert first.unit.schema_samples == [{"n": 0}]
    assert len(source.touched) <= 2, source.touched


def test_artifact_memberships_getitem_forward_slice_does_not_force_full_scan() -> None:
    source = _large_matching_source(10_000)
    view = ArtifactMemberships(source, "message")

    head = view[:5]

    assert [m.unit.schema_samples[0]["n"] for m in head] == [0, 1, 2, 3, 4]
    assert len(source.touched) <= 6, source.touched


def test_artifact_memberships_getitem_negative_index_still_correct() -> None:
    source = _large_matching_source(20)
    view = ArtifactMemberships(source, "message")

    assert view[-1].unit.schema_samples == [{"n": 19}]


def test_membership_samples_bool_does_not_force_full_iteration() -> None:
    source = _large_matching_source(10_000)
    view = MembershipSamples(source)

    assert bool(view) is True
    assert len(source.touched) <= 2, source.touched


def test_membership_samples_bool_false_for_empty_source() -> None:
    source = _CountingMemberships([])
    view = MembershipSamples(source)

    assert bool(view) is False


def test_membership_session_ids_bool_does_not_force_full_iteration() -> None:
    memberships = [_membership("message", [{"n": index}], session_id="sess-1") for index in range(10_000)]
    source = _CountingMemberships(memberships)
    view = MembershipSessionIds(source)

    assert bool(view) is True
    assert len(source.touched) <= 2, source.touched


def test_membership_observed_ats_bool_does_not_force_full_iteration() -> None:
    memberships = [_membership("message", [{"n": index}]) for index in range(10_000)]
    source = _CountingMemberships(memberships)
    view = MembershipObservedAts(source)

    assert bool(view) is True
    assert len(source.touched) <= 2, source.touched
