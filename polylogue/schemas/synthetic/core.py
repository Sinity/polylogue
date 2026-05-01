"""Schema-driven synthetic conversation generator."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic import builders as synthetic_builders
from polylogue.schemas.synthetic import runtime as synthetic_runtime
from polylogue.schemas.synthetic.models import (
    SchemaRecord,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticGenerationState,
    SyntheticSchemaSelection,
    SyntheticWrittenBatch,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.selection import available_synthetic_providers, select_synthetic_schema
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.wire_formats import WireFormat

if TYPE_CHECKING:
    from polylogue.archive.raw_payload.decode import JSONValue
    from polylogue.schemas.synthetic.showcase import ConversationTheme


class SyntheticCorpus:
    """Generate synthetic provider data from annotated schemas."""

    def __init__(
        self,
        schema: SchemaRecord,
        wire_format: WireFormat,
        provider: str,
        *,
        package_version: str = "default",
        element_kind: str | None = None,
    ) -> None:
        self.schema = schema
        self.wire_format = wire_format
        self.provider = provider
        self.package_version = package_version
        self.element_kind = element_kind
        self._generation_state = SyntheticGenerationState(
            relation_solver=RelationConstraintSolver(schema),
            semantic_generator=None,
        )

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SyntheticCorpus:
        selection = select_synthetic_schema(
            provider,
            version=version,
            element_kind=element_kind,
            registry_factory=lambda: SchemaRegistry(),
            canonical_provider_resolver=canonical_schema_provider,
        )
        return cls.from_selection(selection)

    @classmethod
    def from_selection(cls, selection: SyntheticSchemaSelection) -> SyntheticCorpus:
        return cls(
            selection.schema,
            selection.wire_format,
            selection.provider,
            package_version=selection.package_version,
            element_kind=selection.element_kind,
        )

    @classmethod
    def from_spec(cls, spec: CorpusSpec) -> SyntheticCorpus:
        return cls.for_provider(
            spec.provider,
            version=spec.package_version,
            element_kind=spec.element_kind,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        return available_synthetic_providers(registry_factory=lambda: SchemaRegistry())

    @classmethod
    def generate_batch_for_spec(cls, spec: CorpusSpec) -> SyntheticGenerationBatch:
        corpus = cls.from_spec(spec)
        return corpus.generate_batch(
            count=spec.count,
            messages_per_conversation=spec.messages_per_conversation,
            seed=spec.seed,
            style=spec.style,
        )

    @classmethod
    def generate_for_spec(cls, spec: CorpusSpec) -> list[bytes]:
        return cls.generate_batch_for_spec(spec).raw_items

    @classmethod
    def write_spec_artifacts(
        cls,
        spec: CorpusSpec,
        output_dir: Path,
        *,
        prefix: str,
        index_width: int = 2,
    ) -> SyntheticWrittenBatch:
        corpus = cls.from_spec(spec)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        batch = cls.generate_batch_for_spec(spec)
        written_files: list[Path] = []
        for idx, artifact in enumerate(batch.artifacts):
            file_path = output_dir / f"{prefix}-{idx:0{index_width}d}{ext}"
            file_path.write_bytes(artifact.raw_bytes)
            written_files.append(file_path)
        return SyntheticWrittenBatch(batch=batch, files=tuple(written_files))

    @classmethod
    def write_specs_artifacts(
        cls,
        corpus_specs: tuple[CorpusSpec, ...],
        output_root: Path,
        *,
        prefix: str,
        index_width: int = 2,
    ) -> tuple[SyntheticWrittenBatch, ...]:
        provider_counts = Counter(spec.provider for spec in corpus_specs)
        return tuple(
            cls.write_spec_artifacts(
                spec,
                output_root / spec.provider,
                prefix=prefix if provider_counts[spec.provider] == 1 else f"{prefix}-{spec.scope_label}",
                index_width=index_width,
            )
            for spec in corpus_specs
        )

    def generate_batch(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
    ) -> SyntheticGenerationBatch:
        return synthetic_builders.generate_batch(
            self,
            count=count,
            messages_per_conversation=messages_per_conversation,
            seed=seed,
            style=style,
        )

    def generate(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
    ) -> list[bytes]:
        return self.generate_batch(
            count=count,
            messages_per_conversation=messages_per_conversation,
            seed=seed,
            style=style,
        ).raw_items

    @property
    def _relation_solver(self) -> RelationConstraintSolver:
        return self._generation_state.relation_solver

    @_relation_solver.setter
    def _relation_solver(self, value: RelationConstraintSolver) -> None:
        self._generation_state.relation_solver = value

    @property
    def _semantic_gen(self) -> SemanticValueGenerator | None:
        semantic_generator = self._generation_state.semantic_generator
        return semantic_generator if isinstance(semantic_generator, SemanticValueGenerator) else None

    @_semantic_gen.setter
    def _semantic_gen(self, value: SemanticValueGenerator | None) -> None:
        self._generation_state.semantic_generator = value

    def _ensure_wire_chatgpt(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_chatgpt(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_claude_ai(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_claude_ai(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_claude_code(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_claude_code(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_codex(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_codex(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_format(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float = 1700000000.0,
        theme: ConversationTheme | None = None,
    ) -> None:
        synthetic_builders._ensure_wire_format(self, data, role, rng, index, base_ts=base_ts, theme=theme)

    def _ensure_wire_gemini(
        self,
        data: dict[str, JSONValue],
        role: str,
        rng: random.Random,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_gemini(self, data, role, rng, index=index, theme=theme)

    def _generate_array(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> list[JSONValue]:
        return synthetic_runtime._generate_array(self, schema, rng, depth=depth, max_depth=max_depth, path=path)

    def _generate_conversation(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> JSONValue:
        return synthetic_builders._generate_conversation(self, n_messages, rng, theme=theme)

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> JSONValue:
        return synthetic_runtime._generate_from_schema(
            self,
            schema,
            rng,
            skip_keys=skip_keys,
            depth=depth,
            max_depth=max_depth,
            path=path,
        )

    def _generate_jsonl_records(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> list[dict[str, JSONValue]]:
        return synthetic_builders._generate_jsonl_records(self, n_messages, rng, theme=theme)

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict[str, JSONValue]:
        return synthetic_builders._generate_linear_json(self, n_messages, rng, theme=theme)

    def _generate_number(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        is_int: bool = False,
    ) -> float | int:
        return synthetic_runtime._generate_number(self, schema, rng, is_int=is_int)

    def _generate_object(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> dict[str, JSONValue]:
        return synthetic_runtime._generate_object(
            self,
            schema,
            rng,
            skip_keys=skip_keys,
            depth=depth,
            max_depth=max_depth,
            path=path,
        )

    def _generate_string(self, schema: SchemaRecord, rng: random.Random) -> str:
        return synthetic_runtime._generate_string(self, schema, rng)

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict[str, JSONValue]:
        return synthetic_builders._generate_tree_json(self, n_messages, rng, theme=theme)

    def _role_cycle(self) -> list[str]:
        return synthetic_builders._role_cycle(self)

    def _serialize(self, data: JSONValue) -> bytes:
        return synthetic_runtime._serialize(self, data)


__all__ = [
    "SyntheticCorpus",
    "SyntheticGenerationBatch",
    "SyntheticGenerationReport",
]
