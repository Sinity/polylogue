"""Replayable views over schema-unit memberships."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from polylogue.schemas.generation.models import _UnitMembership
from polylogue.schemas.generation.observation_journal import JournalMemberships
from polylogue.schemas.generation.schema_builder import SchemaInput


@dataclass(frozen=True)
class ArtifactMemberships(Sequence[_UnitMembership]):
    """Filter a replayable membership source without copying its payloads."""

    source: Sequence[_UnitMembership]
    artifact_kind: str

    def __iter__(self) -> Iterator[_UnitMembership]:
        return (item for item in self.source if item.unit.artifact_kind == self.artifact_kind)

    def __len__(self) -> int:
        return sum(1 for _item in self)

    @overload
    def __getitem__(self, index: int) -> _UnitMembership: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_UnitMembership]: ...

    def __getitem__(self, index: int | slice) -> _UnitMembership | Sequence[_UnitMembership]:
        retained = list(self)
        return retained[index]


@dataclass(frozen=True)
class MembershipSamples(Collection[SchemaInput]):
    """Flatten samples lazily across a replayable membership source."""

    memberships: Sequence[_UnitMembership]

    def __iter__(self) -> Iterator[SchemaInput]:
        for membership in self.memberships:
            yield from membership.unit.schema_samples

    def __len__(self) -> int:
        return membership_sample_count(self.memberships)

    def __contains__(self, value: object) -> bool:
        return any(sample == value for sample in self)


@dataclass(frozen=True)
class MembershipSessionIds(Collection[str | None]):
    """Replay session identity once per flattened schema sample."""

    memberships: Sequence[_UnitMembership]

    def __iter__(self) -> Iterator[str | None]:
        for membership in self.memberships:
            yield from (membership.unit.session_id for _sample in membership.unit.schema_samples)

    def __len__(self) -> int:
        return membership_sample_count(self.memberships)

    def __contains__(self, value: object) -> bool:
        return any(session_id == value for session_id in self)


def select_artifact_memberships(
    memberships: Sequence[_UnitMembership],
    artifact_kind: str,
) -> Sequence[_UnitMembership]:
    if isinstance(memberships, JournalMemberships):
        return memberships.for_artifact(artifact_kind)
    return ArtifactMemberships(memberships, artifact_kind)


def metadata_memberships(memberships: Sequence[_UnitMembership]) -> Sequence[_UnitMembership]:
    if isinstance(memberships, JournalMemberships):
        return memberships.metadata()
    return memberships


def membership_sample_count(memberships: Sequence[_UnitMembership]) -> int:
    if isinstance(memberships, JournalMemberships):
        return memberships.sample_count
    return sum(len(membership.unit.schema_samples) for membership in memberships)


__all__ = [
    "ArtifactMemberships",
    "MembershipSamples",
    "MembershipSessionIds",
    "membership_sample_count",
    "metadata_memberships",
    "select_artifact_memberships",
]
