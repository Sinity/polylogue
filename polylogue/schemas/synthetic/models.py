"""Typed models for synthetic corpus selection and batch generation."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver
    from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
    from polylogue.schemas.synthetic.wire_formats import WireEncoding, WireFormat

SchemaScalar: TypeAlias = str | int | float | bool | None
SchemaValue: TypeAlias = SchemaScalar | list["SchemaValue"] | dict[str, "SchemaValue"]
SchemaRecord: TypeAlias = dict[str, SchemaValue]
GeneratedString: TypeAlias = str | SampledText | JoinedText
GeneratedScalar: TypeAlias = SchemaScalar | SampledText | JoinedText
SyntheticStyle: TypeAlias = Literal["default", "demo", "tool-heavy", "demo-tool-heavy", "demo-attachments"]


class GeneratedRecordOwner(Protocol):
    def record_schema(self, record: GeneratedRecord) -> SchemaRecord: ...
    def record_path(self, record: GeneratedRecord) -> str | None: ...
    def record_length(self, record: GeneratedRecord) -> int: ...
    def record_item(self, record: GeneratedRecord, key: str) -> GeneratedValue: ...
    def set_record_item(self, record: GeneratedRecord, key: str, value: GeneratedValue) -> None: ...
    def delete_record_item(self, record: GeneratedRecord, key: str) -> None: ...
    def iter_record_keys(self, record: GeneratedRecord) -> Iterator[str]: ...


class GeneratedRecord(MutableMapping[str, "GeneratedValue"]):
    """Exact generated fields and selected branch on the original native owner."""

    __slots__ = ("owner", "record_id", "generation", "_selected_schema", "__weakref__")

    def __init__(
        self, owner: GeneratedRecordOwner, record_id: int, generation: int, selected_schema: SchemaRecord | None = None
    ) -> None:
        self.owner = owner
        self.record_id = record_id
        self.generation = generation
        self._selected_schema = selected_schema

    @property
    def selected_schema(self) -> SchemaRecord:
        if self._selected_schema is None:
            self._selected_schema = self.owner.record_schema(self)
        else:
            # Cached metadata never substitutes for original owner currency.
            self.owner.record_length(self)
        return self._selected_schema

    @property
    def original_path(self) -> str | None:
        return self.owner.record_path(self)

    def __len__(self) -> int:
        return self.owner.record_length(self)

    def __iter__(self) -> Iterator[str]:
        return self.owner.iter_record_keys(self)

    def __getitem__(self, key: str) -> GeneratedValue:
        return self.owner.record_item(self, key)

    def __setitem__(self, key: str, value: GeneratedValue) -> None:
        self.owner.set_record_item(self, key, value)

    def __delitem__(self, key: str) -> None:
        self.owner.delete_record_item(self, key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        if (
            isinstance(other, GeneratedRecord)
            and self.owner is other.owner
            and self.record_id == other.record_id
            and self.generation == other.generation
        ):
            # Currency is checked even for the same retained native coordinate.
            len(self)
            return True
        from polylogue.schemas.synthetic.array_value import walk_generated_values

        for item in (self, other):
            for _ in walk_generated_values(item):
                pass

        def mapping_pairs(
            first: Mapping[object, object], second: Mapping[object, object]
        ) -> Iterator[tuple[object, object]]:
            for key in first:
                yield first[key], second[key]

        frames: list[Iterator[tuple[object, object]]] = [iter(((self, other),))]
        while frames:
            try:
                left, right = next(frames[-1])
            except StopIteration:
                frames.pop()
                continue
            if isinstance(left, Mapping) and isinstance(right, Mapping):
                if len(left) != len(right) or any(key not in right for key in left):
                    return False
                frames.append(mapping_pairs(left, right))
            elif (
                isinstance(left, Sequence)
                and isinstance(right, Sequence)
                and not isinstance(left, str | bytes | bytearray)
                and not isinstance(right, str | bytes | bytearray)
            ):
                if len(left) != len(right):
                    return False
                frames.append(iter(zip(left, right, strict=True)))
            elif left != right:
                return False
        return True


GeneratedValue: TypeAlias = (
    GeneratedScalar | GeneratedArray | GeneratedRecord | list["GeneratedValue"] | dict[str, "GeneratedValue"]
)
GeneratedObject: TypeAlias = dict[str, GeneratedValue] | GeneratedRecord


@dataclass(frozen=True)
class SyntheticSchemaSelection:
    provider: str
    package_version: str
    element_kind: str | None
    schema: SchemaRecord
    wire_format: WireFormat
    workload_profile: Mapping[str, object] | None = None
    element_schemas: Mapping[str, SchemaRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticArtifactFacts:
    """Independent facts planted by the generator before archive ingestion.

    These are intentionally a narrow provider-wire blueprint.  Consumers may
    compare normalized archive outcomes with it, but fact construction never
    calls a provider parser, query evaluator, or storage reader.
    """

    provider: str
    native_session_id: str | None
    expected_session_id: str | None
    message_count: int
    tool_use_ids: tuple[str, ...]
    tool_result_ids: tuple[str, ...]
    raw_sha256: str

    @property
    def paired_tool_ids(self) -> tuple[str, ...]:
        results = set(self.tool_result_ids)
        return tuple(tool_id for tool_id in self.tool_use_ids if tool_id in results)


@dataclass(frozen=True)
class SyntheticAdjunctChunks:
    """One original adjunct file; consume its body before requesting another."""

    name: str
    reference_kind: str
    byte_length: int
    chunks: Generator[bytes, None, None]
    directory: Literal["tool-results", "subagents"]
    artifact_kind: Literal["opaque_payload", "agent_transcript", "agent_sidecar_meta"]


@dataclass(frozen=True)
class SyntheticAdjunct:
    """Explicitly collected adjunct fixture bytes from the same driver."""

    name: str
    reference_kind: str
    raw_bytes: bytes
    directory: Literal["tool-results", "subagents"]
    artifact_kind: Literal["opaque_payload", "agent_transcript", "agent_sidecar_meta"]


@dataclass(frozen=True)
class SyntheticSessionChunks:
    """Borrow one original session until its containing iterator advances.

    Consume main chunks and every adjunct body before requesting the next
    session. Closing the containing
    iterator physically releases the original sampler owner; retaining this
    carrier does not extend that ownership or permit a later read.
    """

    index: int
    data: GeneratedValue
    message_count: int
    style: SyntheticStyle
    native_session_id: str | None
    chunks: Generator[bytes, None, None]
    inherited_tool_ids: tuple[Iterable[GeneratedString], Iterable[GeneratedString]]
    adjuncts: Generator[SyntheticAdjunctChunks, None, None]


@dataclass(frozen=True)
class SyntheticArtifact:
    raw_bytes: bytes
    message_count: int
    style: str
    facts: SyntheticArtifactFacts
    adjuncts: tuple[SyntheticAdjunct, ...]


@dataclass(frozen=True)
class SyntheticGenerationReport:
    provider: str
    package_version: str
    element_kind: str | None
    wire_encoding: WireEncoding
    requested_count: int | None
    generated_count: int
    requested_bytes: int | None
    generated_bytes: int
    style: SyntheticStyle
    seed: int | None
    workload_profile_id: str | None = None
    sampled_record_kind_counts: dict[str, int] = field(default_factory=dict)
    relationship_variant_counts: dict[str, int] = field(default_factory=dict)
    lineage_edge_count: int = 0


@dataclass(frozen=True)
class SyntheticGenerationBatch:
    artifacts: list[SyntheticArtifact]
    report: SyntheticGenerationReport

    @property
    def raw_items(self) -> list[bytes]:
        return [artifact.raw_bytes for artifact in self.artifacts]


@dataclass(frozen=True)
class SyntheticWrittenBatch:
    """Written session paths correspond to facts; files includes all adjuncts."""

    report: SyntheticGenerationReport
    facts: tuple[SyntheticArtifactFacts, ...]
    files: tuple[Path, ...]
    session_files: tuple[Path, ...]


@dataclass
class SyntheticGenerationState:
    relation_solver: RelationConstraintSolver
    semantic_generator: SemanticValueGenerator | None = None
    driver_active: bool = False


__all__ = [
    "SchemaRecord",
    "SchemaScalar",
    "SchemaValue",
    "SyntheticAdjunct",
    "SyntheticAdjunctChunks",
    "SyntheticArtifact",
    "SyntheticArtifactFacts",
    "SyntheticSessionChunks",
    "SyntheticGenerationBatch",
    "SyntheticGenerationState",
    "SyntheticGenerationReport",
    "SyntheticSchemaSelection",
    "SyntheticStyle",
    "SyntheticWrittenBatch",
]
