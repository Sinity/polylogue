"""Reduced evidence used by both sampled and source-backed schema emission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import repeat
from typing import Protocol

from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.field_stats.evidence import (
    deserialize_field_stats,
    finalize_field_stats,
    merge_field_stats_into,
    serialize_field_stats,
)
from polylogue.schemas.field_stats.stats import FieldStats, _collect_field_stats
from polylogue.schemas.generation.dynamic_keys import (
    collapse_dynamic_keys,
    dynamic_object_paths,
    merge_observed_structure_schemas,
    observed_structure_schema,
)

SCHEMA_EVIDENCE_VERSION = 1
_SHAPE_HASH_CAP = 512

SchemaInput = Mapping[str, object]
FieldStateByPath = dict[str, JSONDocument]


def _state_int(value: JSONValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class SourceObservation(Protocol):
    """The reduced collector contract supplied by source inference."""

    @property
    def logical_source_id(self) -> str: ...

    @property
    def revision_sha256(self) -> str: ...

    @property
    def subject(self) -> str: ...

    @property
    def element_kind(self) -> str: ...

    @property
    def records(self) -> Iterable[JSONValue]: ...

    @property
    def is_current(self) -> bool: ...


def _merge_structure(left: JSONDocument, right: JSONDocument) -> JSONDocument:
    return merge_observed_structure_schemas((left, right))


def _schema_digest(schema: JSONDocument) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observe_shape_hash(hashes: set[str], structure: JSONDocument) -> int:
    """Retain one shape hash and return a lower-bound increment beyond the cap."""
    digest = _schema_digest(structure)
    if digest in hashes:
        return 0
    if len(hashes) < _SHAPE_HASH_CAP:
        hashes.add(digest)
        return 0
    return 1


def _bounded_shape_evidence(structures: Iterable[JSONDocument]) -> tuple[tuple[str, ...], int]:
    """Retain shape hashes and a lower bound for observations beyond the cap."""
    hashes: set[str] = set()
    unretained_observations = sum(_observe_shape_hash(hashes, structure) for structure in structures)
    return tuple(sorted(hashes)), unretained_observations


@dataclass(frozen=True)
class SchemaEvidence:
    """A serializable sufficient-statistics summary with no source records."""

    current_structure: JSONDocument
    historical_structure: JSONDocument
    fields: FieldStateByPath
    normalization_paths: tuple[str, ...]
    current_source_count: int
    current_record_count: int
    historical_source_count: int
    historical_record_count: int
    shape_hashes: tuple[str, ...] = ()
    unretained_shape_observation_lower_bound: int = 0

    @property
    def field_stats(self) -> dict[str, FieldStats]:
        return {path: deserialize_field_stats(state) for path, state in self.fields.items()}

    @property
    def structure(self) -> JSONDocument:
        return collapse_dynamic_keys(_merge_structure(self.current_structure, self.historical_structure))

    def to_json(self) -> JSONDocument:
        """Serialize private reduced evidence without literals, records, or IDs."""
        return {
            "version": SCHEMA_EVIDENCE_VERSION,
            "current_structure": self.current_structure,
            "historical_structure": self.historical_structure,
            "fields": {path: self.fields[path] for path in sorted(self.fields)},
            "normalization_paths": list(self.normalization_paths),
            "denominators": {
                "current_sources": self.current_source_count,
                "current_records": self.current_record_count,
                "historical_sources": self.historical_source_count,
                "historical_records": self.historical_record_count,
                "distinct_shapes": len(self.shape_hashes),
                "unretained_shape_observation_lower_bound": self.unretained_shape_observation_lower_bound,
            },
            "shape_hashes": list(self.shape_hashes),
        }

    @classmethod
    def from_json(cls, payload: JSONDocument) -> SchemaEvidence:
        if payload.get("version") != SCHEMA_EVIDENCE_VERSION:
            raise ValueError("unsupported schema evidence version")
        fields = payload.get("fields")
        denominators = payload.get("denominators")
        if not isinstance(fields, dict) or not isinstance(denominators, dict):
            raise ValueError("schema evidence is missing fields or denominators")
        current_structure = payload.get("current_structure")
        historical_structure = payload.get("historical_structure")
        if not isinstance(current_structure, dict) or not isinstance(historical_structure, dict):
            raise ValueError("schema evidence is missing structure")
        normalization = payload.get("normalization_paths")
        if not isinstance(normalization, list) or not all(isinstance(path, str) for path in normalization):
            raise ValueError("schema evidence normalization paths are invalid")
        shape_hashes = payload.get("shape_hashes", [])
        if not isinstance(shape_hashes, list) or not all(isinstance(value, str) for value in shape_hashes):
            raise ValueError("schema evidence shape hashes are invalid")
        return cls(
            current_structure=json_document(current_structure),
            historical_structure=json_document(historical_structure),
            fields={path: json_document(state) for path, state in fields.items() if isinstance(state, dict)},
            normalization_paths=tuple(sorted(path for path in normalization if isinstance(path, str))),
            current_source_count=_state_int(denominators.get("current_sources", 0)),
            current_record_count=_state_int(denominators.get("current_records", 0)),
            historical_source_count=_state_int(denominators.get("historical_sources", 0)),
            historical_record_count=_state_int(denominators.get("historical_records", 0)),
            shape_hashes=tuple(sorted(value for value in shape_hashes if isinstance(value, str))[:_SHAPE_HASH_CAP]),
            unretained_shape_observation_lower_bound=_state_int(
                denominators.get(
                    "unretained_shape_observation_lower_bound",
                    denominators.get("distinct_shapes_overflow", 0),
                )
            ),
        )


def collect_source_evidence(
    observation: SourceObservation,
    *,
    dynamic_paths: Collection[str] = (),
    is_current: bool | None = None,
    include_statistics: bool = True,
) -> SchemaEvidence:
    """Consume one complete source revision once and return compact evidence.

    The caller may perform a structure-only pass first, then re-read just this
    source with a changed global normalization policy. No source record is
    retained after this function returns.
    """
    structure: JSONDocument = {}
    shape_hashes: set[str] = set()
    unretained_shape_observation_lower_bound = 0
    record_count = 0

    def records_for_stats() -> Iterable[SchemaInput]:
        nonlocal structure, record_count, unretained_shape_observation_lower_bound
        for record in observation.records:
            record_count += 1
            record_structure = observed_structure_schema(record)
            digest = _schema_digest(record_structure)
            if digest not in shape_hashes:
                structure = _merge_structure(structure, record_structure)
                if len(shape_hashes) < _SHAPE_HASH_CAP:
                    shape_hashes.add(digest)
                else:
                    unretained_shape_observation_lower_bound += 1
            if isinstance(record, Mapping):
                yield record

    stats: dict[str, FieldStats] = {}
    if include_statistics:
        stats = _collect_field_stats(
            records_for_stats(),
            session_ids=repeat(observation.logical_source_id),
            dynamic_paths=dynamic_paths,
        )
    else:
        for _record in records_for_stats():
            pass
    current = observation.is_current if is_current is None else is_current
    return SchemaEvidence(
        current_structure=structure if current else {},
        historical_structure={} if current else structure,
        fields={path: serialize_field_stats(field) for path, field in sorted(stats.items())} if current else {},
        normalization_paths=tuple(sorted(dynamic_paths)),
        current_source_count=int(current),
        current_record_count=record_count if current else 0,
        historical_source_count=int(not current),
        historical_record_count=record_count if not current else 0,
        shape_hashes=tuple(sorted(shape_hashes)),
        unretained_shape_observation_lower_bound=unretained_shape_observation_lower_bound,
    )


def collect_evidence(
    current_observations: Iterable[SourceObservation],
    *,
    historical_observations: Iterable[SourceObservation] = (),
) -> SchemaEvidence:
    """Collect current statistics and separate historical shape coverage.

    This convenience entrypoint is for repeatable inputs such as samples and
    tests. Source inventory runs should use :func:`collect_source_evidence`
    after their structure phase so one-pass streams are never buffered.
    """
    current = tuple(current_observations)
    historical = tuple(historical_observations)
    preliminary = [
        *(collect_source_evidence(item, is_current=True, include_statistics=False) for item in current),
        *(collect_source_evidence(item, is_current=False, include_statistics=False) for item in historical),
    ]
    structure = merge_observed_structure_schemas(item.structure for item in preliminary)
    paths = tuple(sorted(dynamic_object_paths(structure)))
    current_evidence = [collect_source_evidence(item, dynamic_paths=paths, is_current=True) for item in current]
    historical_evidence = [collect_source_evidence(item, dynamic_paths=paths, is_current=False) for item in historical]
    return merge_evidence((*current_evidence, *historical_evidence))


def collect_sample_evidence(
    samples: Collection[SchemaInput],
    *,
    session_ids: Collection[str | None] | None = None,
    observed_ats: Collection[str | None] | None = None,
) -> SchemaEvidence:
    """Adapt the existing sample API to the same reduced-evidence emitter."""
    raw_structure = merge_observed_structure_schemas(observed_structure_schema(sample) for sample in samples)
    structure = collapse_dynamic_keys(raw_structure)
    stats = _collect_field_stats(
        samples,
        session_ids=session_ids,
        observed_ats=observed_ats,
        dynamic_paths=dynamic_object_paths(structure),
    )
    hashes, unretained_shape_observation_lower_bound = _bounded_shape_evidence(
        observed_structure_schema(sample) for sample in samples
    )
    return SchemaEvidence(
        current_structure=raw_structure,
        historical_structure={},
        fields={path: serialize_field_stats(field) for path, field in sorted(stats.items())},
        normalization_paths=tuple(sorted(dynamic_object_paths(structure))),
        current_source_count=len(samples),
        current_record_count=len(samples),
        historical_source_count=0,
        historical_record_count=0,
        shape_hashes=hashes,
        unretained_shape_observation_lower_bound=unretained_shape_observation_lower_bound,
    )


@dataclass
class SchemaEvidenceAccumulator:
    """Fold caller-ordered summaries without retaining completed source rows.

    Field sketches and shape witnesses remain bounded. Memory grows with the
    resulting schema's field paths, rather than with its source count.
    """

    _current_structure: JSONDocument = field(default_factory=dict)
    _historical_structure: JSONDocument = field(default_factory=dict)
    _fields: dict[str, FieldStats] = field(default_factory=dict)
    _normalization_paths: tuple[str, ...] | None = None
    _current_sources: int = 0
    _current_records: int = 0
    _historical_sources: int = 0
    _historical_records: int = 0
    _shape_hashes: set[str] = field(default_factory=set)
    _unretained_shapes: int = 0

    def add(self, evidence: SchemaEvidence) -> None:
        if self._normalization_paths is None:
            self._normalization_paths = evidence.normalization_paths
        elif self._normalization_paths != evidence.normalization_paths:
            raise ValueError("cannot merge evidence collected under different dynamic-key normalization")
        self._current_structure = _merge_structure(self._current_structure, evidence.current_structure)
        self._historical_structure = _merge_structure(self._historical_structure, evidence.historical_structure)
        merge_field_stats_into(self._fields, evidence.field_stats)
        self._current_sources += evidence.current_source_count
        self._current_records += evidence.current_record_count
        self._historical_sources += evidence.historical_source_count
        self._historical_records += evidence.historical_record_count
        self._shape_hashes.update(evidence.shape_hashes)
        discarded = max(0, len(self._shape_hashes) - _SHAPE_HASH_CAP)
        if discarded:
            self._shape_hashes = set(sorted(self._shape_hashes)[:_SHAPE_HASH_CAP])
        self._unretained_shapes += evidence.unretained_shape_observation_lower_bound + discarded

    def finish(self) -> SchemaEvidence:
        finalize_field_stats(self._fields, total_samples=self._current_records)
        return SchemaEvidence(
            current_structure=self._current_structure,
            historical_structure=self._historical_structure,
            fields={path: serialize_field_stats(stats) for path, stats in sorted(self._fields.items())},
            normalization_paths=self._normalization_paths or (),
            current_source_count=self._current_sources,
            current_record_count=self._current_records,
            historical_source_count=self._historical_sources,
            historical_record_count=self._historical_records,
            shape_hashes=tuple(sorted(self._shape_hashes)),
            unretained_shape_observation_lower_bound=self._unretained_shapes,
        )


def merge_evidence(evidence: Iterable[SchemaEvidence]) -> SchemaEvidence:
    """Merge deterministic source summaries; replacement callers rebuild here."""
    ordered = tuple(
        sorted(evidence, key=lambda item: json.dumps(item.to_json(), sort_keys=True, separators=(",", ":")))
    )
    accumulator = SchemaEvidenceAccumulator()
    for item in ordered:
        accumulator.add(item)
    return accumulator.finish()


__all__ = [
    "SCHEMA_EVIDENCE_VERSION",
    "SchemaEvidence",
    "SchemaEvidenceAccumulator",
    "SourceObservation",
    "collect_evidence",
    "collect_sample_evidence",
    "collect_source_evidence",
    "merge_evidence",
]
