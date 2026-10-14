"""Schema-driven synthetic session generator."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic import build_batch as synthetic_batch
from polylogue.schemas.synthetic import builders as synthetic_builders
from polylogue.schemas.synthetic import runtime as synthetic_runtime
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.models import (
    GeneratedObject,
    GeneratedString,
    SchemaRecord,
    SyntheticAdjunctChunks,
    SyntheticArtifactFacts,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticGenerationState,
    SyntheticSchemaSelection,
    SyntheticSessionChunks,
    SyntheticStyle,
    SyntheticWrittenBatch,
)
from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan, RecordScopeKinds
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.selection import available_synthetic_providers, select_synthetic_schema
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText
from polylogue.schemas.synthetic.wire_formats import WireFormat
from polylogue.sources.source_layout import canonical_session_position

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.demo_themes import SessionTheme
    from polylogue.schemas.synthetic.models import GeneratedValue as JSONValue


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
        workload_profile: Mapping[str, object] | None = None,
        element_schemas: Mapping[str, SchemaRecord] | None = None,
    ) -> None:
        if workload_profile is not None:
            from polylogue.schemas.generation.workload_profiles import require_cross_record_profile

            require_cross_record_profile(workload_profile)
        self.schema = schema
        self.wire_format = wire_format
        self.provider = provider
        self.package_version = package_version
        self.element_kind = element_kind
        self.workload_profile = workload_profile
        self.element_schemas = dict(element_schemas) if element_schemas is not None else {}
        self._categorical_pools: dict[str, tuple[str, list[tuple[GeneratedString, float]]]] = {}
        self._public_record_plan: PublicRecordPlan | None = None
        self._agent_depth = 0
        self._session_header_class: SchemaRecord | None = None
        self._session_header_bound = False
        self._session_parent_id: str | None = None
        self._session_native_id: str | None = None
        self._session_record_structure: SchemaRecord | None = None
        self._public_record_counts: Counter[str] = Counter()
        self._public_scope: Generator[str, None, None] | None = None
        self._coverage_branch_choices: dict[str, int] = {}
        self._coverage_type_choices: dict[str, str] = {}
        self._coverage_null_paths: set[str] = set()
        self._coverage_witness_mode = False
        self._coverage_header_form: str | None = None
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
    def from_selection(
        cls,
        selection: SyntheticSchemaSelection,
    ) -> SyntheticCorpus:
        return cls(
            selection.schema,
            selection.wire_format,
            selection.provider,
            package_version=selection.package_version,
            element_kind=selection.element_kind,
            workload_profile=selection.workload_profile,
            element_schemas=selection.element_schemas,
        )

    def _public_record_structure(self) -> SchemaRecord | None:
        if self._session_record_structure is not None:
            return self._session_record_structure
        profile = self.workload_profile
        if profile is None:
            return None
        elements = profile.get("elements")
        if not isinstance(elements, dict) or not elements:
            raise ValueError("package record profile is missing; re-infer the package")
        element = elements.get(self.element_kind) if self.element_kind is not None else None
        if element is None and self.element_kind is None and len(elements) == 1:
            element = next(iter(elements.values()))
        if not isinstance(element, dict):
            raise ValueError("the selected schema element has no unambiguous record profile")
        structure = element.get("record_structure")
        if not isinstance(structure, dict):
            raise ValueError("package record grammar is missing; re-infer the package")
        return structure

    def _profile_message_count(self, rng: random.Random, requested: int) -> int:
        structure = self._public_record_structure()
        if structure is None:
            self._public_scope = None
            return requested
        if self.workload_profile and self.workload_profile.get("profile_kind") == "authored-fixture":
            cycle = self.workload_profile.get("authored_record_cycle")
            if (
                not isinstance(cycle, list)
                or not cycle
                or any(not isinstance(row, dict) or not isinstance(row.get("kind"), str) for row in cycle)
            ):
                raise ValueError("authored fixture has no declared original record order")
            kinds = tuple(str(row["kind"]) for row in cycle if isinstance(row, dict))
            if self.workload_profile.get("authored_session_header") is True:
                self._public_scope = (
                    kinds[0] if index == 0 else kinds[1 + (index - 1) % (len(kinds) - 1)]
                    for index in range(requested + 1)
                )
                return requested + 1
            self._public_scope = (kinds[index % len(kinds)] for index in range(requested))
            return requested
        value = synthetic_runtime._sampled_length(
            {"x-polylogue-observed-distribution": {"array_length": structure.get("scope_record_count")}},
            "array_length",
            rng,
        )
        if value is None:
            raise ValueError("package scope count is missing; re-infer the package")
        self._public_scope = RecordScopeKinds(
            structure, value, self._relation_solver.fk_graph, provider=self.provider, schema=self.schema
        ).sample(rng)
        return value

    def _start_public_record(self, rng: random.Random, schema: SchemaRecord) -> None:
        structure = self._public_record_structure()
        if structure is None:
            self._public_record_plan = None
            return
        if self._public_scope is None:
            raise RuntimeError("public record has no original scope path")
        if self.workload_profile is not None and self.workload_profile.get("profile_kind") == "authored-fixture":
            branch = schema.get("x-polylogue-authored-record-branch")
            branches = self.workload_profile.get("authored_record_structures")
            if type(branch) is not int or not isinstance(branches, list) or not 0 <= branch < len(branches):
                raise ValueError("authored record has no original conditional structure")
            selected = branches[branch]
            if not isinstance(selected, dict):
                raise ValueError("authored record conditional structure is malformed")
            structure = selected
        kind = next(self._public_scope)
        header_class = self._session_header_class if kind == "session_meta" and not self._session_header_bound else None
        plan = PublicRecordPlan(
            self.provider, structure, kind, rng, self._relation_solver.fk_graph, schema, header_class=header_class
        )
        if header_class is not None:
            plan.bind_session_header(header_class, native_id=self._session_native_id, parent_id=self._session_parent_id)
            self._session_header_bound = True
        self._public_record_plan = plan
        self._public_record_counts[plan.kind] += 1

    def _finish_public_record(self, record: dict[str, JSONValue] | GeneratedObject) -> None:
        """Settle a selected dispatcher on its exact original generated row."""
        from polylogue.core.claude_code_children import claude_dispatch_operand
        from polylogue.schemas.synthetic.models import GeneratedRecord

        plan = self._public_record_plan
        if plan is None or plan.agent_children is None:
            return
        owner = self._relation_solver.fk_graph
        if plan.owner is not owner or plan.generation != owner._generation:
            raise RuntimeError("dispatcher plan lost its original generated owner")
        status, point = plan.agent_children
        operand = claude_dispatch_operand(record)
        if (operand is None) != (status == "no_dispatch"):
            raise ValueError("selected dispatcher class conflicts with its original field draw")
        if operand is None:
            return
        result = record.get("toolUseResult")
        if not isinstance(result, Mapping) or not isinstance(result.get("agentId"), str):
            raise ValueError("dispatcher lacks its original filename field coordinate")
        agent_id = result["agentId"]
        assert isinstance(agent_id, str)
        if len(agent_id) != point[3] or len(operand[1]) != point[4]:
            raise ValueError("selected dispatcher filename joint conflicts with its original field draw")
        if not isinstance(record, GeneratedRecord) or record.owner is not owner or record.generation != plan.generation:
            raise RuntimeError("dispatcher child requires its original generated record cell")
        if point[0] or point[2]:
            connection = owner._array_owner(plan.generation)
            connection.execute("INSERT INTO generated_agent_names VALUES (?)", (operand[1],)).close()
            connection.execute(
                "INSERT INTO generated_agent_children SELECT COALESCE(max(ordinal),-1)+1,?,?,?,?,?,? FROM generated_agent_children",
                (
                    record.record_id,
                    point[0],
                    point[1],
                    point[2],
                    self._agent_depth + 1,
                    json.dumps(plan.agent_metadata, sort_keys=True, separators=(",", ":"))
                    if plan.agent_metadata is not None
                    else None,
                ),
            ).close()

    def _generate_agent_transcript(
        self, schema: SchemaRecord, structure: SchemaRecord, count: int, depth: int, rng: random.Random
    ) -> GeneratedArray:
        """Draw a selected child on the same physical sampler, never a new owner."""
        if self._public_scope is not None:
            self._public_scope.close()
            self._public_scope = None
        original = (
            self.schema,
            self.element_kind,
            self._session_record_structure,
            self._semantic_gen,
            self._agent_depth,
        )
        solver = self._relation_solver
        owner = solver.fk_graph
        connection = owner._array_owner(owner._generation)
        try:
            # These are current logical-stream indexes only. Original parent
            # records, arrays, opaque plans and retained prefix cells stay live.
            for table in (
                "generated_ids",
                "generated_tool_identities",
            ):
                connection.execute("DELETE FROM " + table).close()
            owner._counts.clear()
            owner._missing_tool_counts.clear()
            solver.select_element(schema)
            self.schema, self.element_kind = schema, "agent_transcript"
            self._session_record_structure = structure
            self._agent_depth = depth
            self._public_scope = RecordScopeKinds(
                structure, count, owner, provider=self.provider, schema=self.schema
            ).sample(rng)
            return self._generate_jsonl_records(count, rng)
        finally:
            if self._public_scope is not None:
                self._public_scope.close()
                self._public_scope = None
            self._public_record_plan = None
            self.schema, self.element_kind, self._session_record_structure, self._semantic_gen, self._agent_depth = (
                original
            )
            solver.select_element(self.schema)

    def _generate_agent_metadata(
        self, schema: SchemaRecord, metadata_class: SchemaRecord, agent_id: str, stem: str, rng: random.Random
    ) -> GeneratedObject:
        from polylogue.schemas.synthetic.models import GeneratedRecord

        original = self.schema, self._public_record_plan
        solver = self._relation_solver
        solver.select_element(schema)
        try:
            self.schema = schema
            self._public_record_plan = PublicRecordPlan(
                self.provider,
                {},
                "agent_sidecar_meta",
                rng,
                solver.fk_graph,
                schema,
                metadata_class=metadata_class,
                agent_id=agent_id,
                agent_stem=stem,
            )
            result = self._generate_from_schema(schema, rng)
            if not isinstance(result, GeneratedRecord) or result.owner is not solver.fk_graph:
                raise ValueError("original metadata element must be its generated document cell")
            return result
        finally:
            self.schema, self._public_record_plan = original
            solver.select_element(self.schema)

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
        return cls._generate_batch_for_corpus(corpus, spec)

    @classmethod
    def generate_batch_for_selection(
        cls,
        selection: SyntheticSchemaSelection,
        spec: CorpusSpec,
    ) -> SyntheticGenerationBatch:
        """Generate a spec through an already-resolved persisted schema route."""

        corpus = cls.from_selection(selection)
        return cls._generate_batch_for_corpus(corpus, spec)

    @staticmethod
    def _generate_batch_for_corpus(corpus: SyntheticCorpus, spec: CorpusSpec) -> SyntheticGenerationBatch:
        return corpus.generate_batch(
            count=spec.count,
            messages_per_session=spec.messages_per_session,
            seed=spec.seed,
            style=spec.style,
            session_native_ids=spec.session_native_ids,
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
        return cls._write_batch_artifacts(corpus, spec, output_dir, prefix=prefix, index_width=index_width)

    @classmethod
    def write_selection_artifacts(
        cls,
        selection: SyntheticSchemaSelection,
        spec: CorpusSpec,
        output_dir: Path,
        *,
        prefix: str,
        index_width: int = 2,
    ) -> SyntheticWrittenBatch:
        """Write artifacts from the exact schema selection carried by a manifest."""

        corpus = cls.from_selection(selection)
        return cls._write_batch_artifacts(corpus, spec, output_dir, prefix=prefix, index_width=index_width)

    @classmethod
    def _write_batch_artifacts(
        cls,
        corpus: SyntheticCorpus,
        spec: CorpusSpec,
        output_dir: Path,
        *,
        prefix: str,
        index_width: int,
    ) -> SyntheticWrittenBatch:
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        written_files: list[Path] = []
        session_files: list[Path] = []

        def write(
            idx: int,
            data: JSONValue,
            message_count: int,
            resolved_style: SyntheticStyle,
            native_id: str | None,
            chunks: Generator[bytes, None, None],
            inherited_tool_ids: tuple[Iterable[GeneratedString], Iterable[GeneratedString]],
            adjuncts: Generator[SyntheticAdjunctChunks, None, None],
        ) -> SyntheticArtifactFacts:
            hasher = hashlib.sha256()
            if corpus.provider == "antigravity":
                conversations_dir = output_dir / "conversations"
                conversations_dir.mkdir(parents=True, exist_ok=True)
                cascade_id = (
                    spec.session_native_ids[idx]
                    if idx < len(spec.session_native_ids)
                    else f"{prefix}-{idx:0{index_width}d}"
                )
                file_path = output_dir / canonical_session_position(corpus.provider, cascade_id, ".pb")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(b"fake-protobuf-bytes-" + cascade_id.encode("utf-8"))
                for chunk in chunks:
                    hasher.update(chunk)
            else:
                stem = f"{prefix}-{idx:0{index_width}d}"
                file_path = output_dir / canonical_session_position(corpus.provider, stem, ext)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with file_path.open("wb") as handle:
                    for chunk in chunks:
                        handle.write(chunk)
                        hasher.update(chunk)
            session_files.append(file_path)
            written_files.append(file_path)
            for adjunct in adjuncts:
                adjunct_path = file_path.parent / file_path.stem / adjunct.directory / adjunct.name
                adjunct_path.parent.mkdir(parents=True, exist_ok=True)
                with adjunct_path.open("xb") as handle:
                    for chunk in adjunct.chunks:
                        handle.write(chunk)
                written_files.append(adjunct_path)
            return synthetic_batch._planted_artifact_facts(
                corpus.provider,
                data,
                raw_sha256=hasher.hexdigest(),
                native_session_id=native_id,
                message_count=message_count,
                inherited_tool_ids=inherited_tool_ids,
            )

        facts, report = synthetic_batch._produce_artifacts(
            corpus,
            write,
            count=spec.count,
            messages_per_session=spec.messages_per_session,
            seed=spec.seed,
            style=spec.style,
            session_native_ids=spec.session_native_ids,
        )
        return SyntheticWrittenBatch(
            report=report, facts=tuple(facts), files=tuple(written_files), session_files=tuple(session_files)
        )

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

    def iter_sessions(
        self,
        *,
        count: int | None = None,
        target_bytes: int | None = None,
        messages_per_session: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
        session_native_ids: tuple[str, ...] = (),
    ) -> Generator[SyntheticSessionChunks, None, SyntheticGenerationReport]:
        """Borrow the shared driver; consume each session before advancing."""
        return synthetic_batch.iter_sessions(
            self,
            count=count,
            target_bytes=target_bytes,
            messages_per_session=messages_per_session,
            seed=seed,
            style=style,
            session_native_ids=session_native_ids,
        )

    def generate_batch(
        self,
        count: int = 5,
        messages_per_session: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
        session_native_ids: tuple[str, ...] = (),
    ) -> SyntheticGenerationBatch:
        return synthetic_builders.generate_batch(
            self,
            count=count,
            messages_per_session=messages_per_session,
            seed=seed,
            style=style,
            session_native_ids=session_native_ids,
        )

    def generate(
        self,
        count: int = 5,
        messages_per_session: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
        session_native_ids: tuple[str, ...] = (),
    ) -> list[bytes]:
        return self.generate_batch(
            count=count,
            messages_per_session=messages_per_session,
            seed=seed,
            style=style,
            session_native_ids=session_native_ids,
        ).raw_items

    @property
    def _relation_solver(self) -> RelationConstraintSolver:
        return self._generation_state.relation_solver

    @_relation_solver.setter
    def _relation_solver(self, value: RelationConstraintSolver) -> None:
        if self._public_scope is not None:
            self._public_scope.close()
            self._public_scope = None
        self._generation_state.relation_solver.close()
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
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_chatgpt(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_claude_ai(
        self,
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_claude_ai(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_claude_code(
        self,
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_claude_code(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_codex(
        self,
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_codex(self, data, role, rng, ts, index=index, theme=theme)

    def _ensure_wire_format(
        self,
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float = 1700000000.0,
        theme: SessionTheme | None = None,
    ) -> None:
        synthetic_builders._ensure_wire_format(self, data, role, rng, index, base_ts=base_ts, theme=theme)

    def _ensure_wire_gemini(
        self,
        data: GeneratedObject,
        role: str,
        rng: random.Random,
        *,
        index: int,
        theme: SessionTheme | None,
    ) -> None:
        synthetic_builders._ensure_wire_gemini(self, data, role, rng, index=index, theme=theme)

    def _generate_session(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> JSONValue:
        return synthetic_builders._generate_session(self, n_messages, rng, theme=theme)

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        path: str = "$",
    ) -> JSONValue:
        return synthetic_runtime._generate_from_schema(
            self,
            schema,
            rng,
            skip_keys=skip_keys,
            path=path,
        )

    def _generate_jsonl_records(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> GeneratedArray:
        return synthetic_builders._generate_jsonl_records(self, n_messages, rng, theme=theme)

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> GeneratedObject:
        return synthetic_builders._generate_linear_json(self, n_messages, rng, theme=theme)

    def _generate_number(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        is_int: bool = False,
    ) -> float | int:
        return synthetic_runtime._generate_number(self, schema, rng, is_int=is_int)

    def _generate_string(
        self, schema: SchemaRecord, rng: random.Random, *, path: str = "$"
    ) -> str | SampledText | JoinedText:
        return synthetic_runtime._generate_string(self, schema, rng, path=path)

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> GeneratedObject:
        return synthetic_builders._generate_tree_json(self, n_messages, rng, theme=theme)

    def _role_cycle(self) -> list[str]:
        return synthetic_builders._role_cycle(self)

    def iter_metadata_document_chunks(self, data: JSONValue) -> Generator[bytes, None, None]:
        from polylogue.schemas.synthetic.array_value import walk_generated_values
        from polylogue.schemas.synthetic.models import GeneratedRecord

        if not isinstance(data, GeneratedRecord) or data.owner is not self._relation_solver.fk_graph:
            raise ValueError("metadata serialization requires the original generated document")
        for _ in walk_generated_values(data):
            pass
        for piece in synthetic_runtime._json_chunks(data, pretty=True):
            yield piece.encode("utf-8")

    def iter_serialized_chunks(self, data: JSONValue) -> Generator[bytes, None, None]:
        return synthetic_runtime.iter_serialized_chunks(self, data)

    def _serialize(self, data: JSONValue) -> bytes:
        return synthetic_runtime._serialize(self, data)


__all__ = [
    "SyntheticCorpus",
    "SyntheticGenerationBatch",
    "SyntheticGenerationReport",
]
