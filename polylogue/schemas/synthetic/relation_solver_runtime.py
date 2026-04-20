"""Runtime behavior mixin for synthetic relation solving."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.schemas.synthetic.models import SchemaRecord

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import (
        ForeignKeyGraph,
        MutualExclusionGroup,
        StringLengthConstraint,
        TimeDeltaConstraint,
    )


@dataclass(frozen=True)
class _ForeignKeyAnnotation:
    source: str
    target: str


@dataclass(frozen=True)
class _TimeDeltaAnnotation:
    field_a: str
    field_b: str
    min_delta: float
    max_delta: float
    avg_delta: float


@dataclass(frozen=True)
class _MutualExclusionAnnotation:
    parent: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class _StringLengthAnnotation:
    path: str
    min_length: int
    max_length: int
    avg_length: float
    stddev: float


def _annotation_records(schema: SchemaRecord, key: str) -> tuple[SchemaRecord, ...]:
    value = schema.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _int_value(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return default


def _float_value(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _foreign_key_annotations(schema: SchemaRecord) -> tuple[_ForeignKeyAnnotation, ...]:
    annotations: list[_ForeignKeyAnnotation] = []
    for record in _annotation_records(schema, "x-polylogue-foreign-keys"):
        source = str(record.get("source", "")).strip()
        target = str(record.get("target", "")).strip()
        if source and target:
            annotations.append(_ForeignKeyAnnotation(source=source, target=target))
    return tuple(annotations)


def _time_delta_annotations(schema: SchemaRecord) -> tuple[_TimeDeltaAnnotation, ...]:
    annotations: list[_TimeDeltaAnnotation] = []
    for record in _annotation_records(schema, "x-polylogue-time-deltas"):
        annotations.append(
            _TimeDeltaAnnotation(
                field_a=str(record.get("field_a", "")).strip(),
                field_b=str(record.get("field_b", "")).strip(),
                min_delta=_float_value(record.get("min_delta"), 0.0),
                max_delta=_float_value(record.get("max_delta"), 0.0),
                avg_delta=_float_value(record.get("avg_delta"), 0.0),
            )
        )
    return tuple(annotation for annotation in annotations if annotation.field_a and annotation.field_b)


def _mutual_exclusion_annotations(schema: SchemaRecord) -> tuple[_MutualExclusionAnnotation, ...]:
    annotations: list[_MutualExclusionAnnotation] = []
    for record in _annotation_records(schema, "x-polylogue-mutually-exclusive"):
        parent = str(record.get("parent", "")).strip()
        fields = _string_tuple(record.get("fields"))
        if parent and len(fields) >= 2:
            annotations.append(_MutualExclusionAnnotation(parent=parent, fields=fields))
    return tuple(annotations)


def _string_length_annotations(schema: SchemaRecord) -> tuple[_StringLengthAnnotation, ...]:
    annotations: list[_StringLengthAnnotation] = []
    for record in _annotation_records(schema, "x-polylogue-string-lengths"):
        path = str(record.get("path", "")).strip()
        if not path:
            continue
        annotations.append(
            _StringLengthAnnotation(
                path=path,
                min_length=_int_value(record.get("min"), 0),
                max_length=_int_value(record.get("max"), 100),
                avg_length=_float_value(record.get("avg"), 50.0),
                stddev=_float_value(record.get("stddev"), 10.0),
            )
        )
    return tuple(annotations)


class RelationConstraintSolverRuntimeMixin:
    fk_graph: ForeignKeyGraph
    time_deltas: list[TimeDeltaConstraint]
    mutual_exclusions: list[MutualExclusionGroup]
    string_lengths: dict[str, StringLengthConstraint]
    _time_delta_cls: type[TimeDeltaConstraint]
    _mutual_exclusion_cls: type[MutualExclusionGroup]
    _string_length_cls: type[StringLengthConstraint]

    def _parse_foreign_keys(self, schema: SchemaRecord) -> None:
        for annotation in _foreign_key_annotations(schema):
            self.fk_graph.references[annotation.source] = annotation.target

    def _parse_time_deltas(self, schema: SchemaRecord) -> None:
        for annotation in _time_delta_annotations(schema):
            self.time_deltas.append(
                self._time_delta_cls(
                    field_a=annotation.field_a,
                    field_b=annotation.field_b,
                    min_delta=annotation.min_delta,
                    max_delta=annotation.max_delta,
                    avg_delta=annotation.avg_delta,
                )
            )

    def _parse_mutual_exclusions(self, schema: SchemaRecord) -> None:
        for annotation in _mutual_exclusion_annotations(schema):
            self.mutual_exclusions.append(
                self._mutual_exclusion_cls(
                    parent_path=annotation.parent,
                    field_names=frozenset(annotation.fields),
                )
            )

    def _parse_string_lengths(self, schema: SchemaRecord) -> None:
        for annotation in _string_length_annotations(schema):
            self.string_lengths[annotation.path] = self._string_length_cls(
                path=annotation.path,
                min_length=annotation.min_length,
                max_length=annotation.max_length,
                avg_length=annotation.avg_length,
                stddev=annotation.stddev,
            )

    def register_generated_id(self, path: str, value: str) -> None:
        self.fk_graph.register_id(path, value)

    def resolve_foreign_key(self, path: str, rng: random.Random) -> str | None:
        return self.fk_graph.resolve_reference(path, rng)

    def get_time_delta(
        self,
        field_a: str,
        field_b: str,
        rng: random.Random,
    ) -> float | None:
        for td in self.time_deltas:
            if (td.field_a == field_a and td.field_b == field_b) or (td.field_a == field_b and td.field_b == field_a):
                if td.stddev_approx > 0:
                    val = rng.gauss(td.avg_delta, td.stddev_approx)
                else:
                    val = rng.uniform(td.min_delta, td.max_delta)
                return max(td.min_delta, min(td.max_delta, val))
        return None

    def filter_mutually_exclusive(
        self,
        parent_path: str,
        field_names: set[str],
        rng: random.Random,
    ) -> set[str]:
        result = set(field_names)
        for group in self.mutual_exclusions:
            if group.parent_path != parent_path:
                continue
            overlap = result & group.field_names
            if len(overlap) > 1:
                keeper = rng.choice(sorted(overlap))
                result -= overlap
                result.add(keeper)
        return result

    def generate_string_with_length(
        self,
        path: str,
        rng: random.Random,
        base_text: str,
    ) -> str:
        constraint = self.string_lengths.get(path)
        if constraint is None:
            return base_text

        target = int(rng.gauss(constraint.avg_length, constraint.stddev))
        target = max(constraint.min_length, min(constraint.max_length, target))

        if len(base_text) == 0:
            return base_text

        if len(base_text) >= target:
            if target <= 3:
                return base_text[:target]
            truncated = base_text[:target]
            last_space = truncated.rfind(" ")
            if last_space > target // 2:
                return truncated[:last_space]
            return truncated
        repetitions = (target // len(base_text)) + 1
        extended = (base_text + " ") * repetitions
        return extended[:target].rstrip()

    def path_matches(self, schema_path: str, annotation_path: str) -> bool:
        return schema_path == annotation_path


__all__ = ["RelationConstraintSolverRuntimeMixin"]
