"""Batch orchestration helpers for synthetic corpora."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from itertools import count as ordinal_sequence
from typing import Protocol, TypeVar, cast

from polylogue.schemas.record_structure import (
    codex_message_record,
    codex_session_header,
    iter_provider_records,
    iter_tool_records,
    record_kind,
    tool_identity_operand,
    tool_result_content_mode,
)
from polylogue.schemas.synthetic.agent_adjuncts import (
    iter_session_adjuncts,
    prepare_agent_adjuncts,
    settle_agent_adjuncts,
)
from polylogue.schemas.synthetic.array_value import GeneratedArray, walk_generated_values
from polylogue.schemas.synthetic.build_wire_formats import validate_wire_payload
from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES, SessionTheme
from polylogue.schemas.synthetic.models import (
    GeneratedRecord,
    GeneratedString,
    SchemaRecord,
    SyntheticAdjunct,
    SyntheticAdjunctChunks,
    SyntheticArtifact,
    SyntheticArtifactFacts,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticGenerationState,
    SyntheticSessionChunks,
    SyntheticStyle,
)
from polylogue.schemas.synthetic.models import GeneratedValue as JSONValue
from polylogue.schemas.synthetic.opaque_adjuncts import prepare_opaque_adjuncts
from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
from polylogue.schemas.synthetic.relations import ForeignKeyGraph, RelationConstraintSolver
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS, WireFormat


class _SyntheticBatchContext(Protocol):
    schema: SchemaRecord
    wire_format: WireFormat
    provider: str
    package_version: str
    element_kind: str | None
    workload_profile: Mapping[str, object] | None
    element_schemas: dict[str, SchemaRecord]

    def _generate_agent_transcript(
        self, schema: SchemaRecord, structure: SchemaRecord, count: int, depth: int, rng: random.Random
    ) -> GeneratedArray: ...

    def _generate_agent_metadata(
        self, schema: SchemaRecord, metadata_class: SchemaRecord, agent_id: str, stem: str, rng: random.Random
    ) -> JSONValue: ...

    _coverage_witness_mode: bool
    _coverage_header_form: str | None
    _session_header_class: SchemaRecord | None
    _session_header_bound: bool
    _session_parent_id: str | None
    _session_native_id: str | None
    _session_record_structure: SchemaRecord | None
    _public_record_plan: PublicRecordPlan | None
    _public_record_counts: Counter[str]
    _public_scope: Generator[str, None, None] | None
    _relation_solver: RelationConstraintSolver
    _semantic_gen: SemanticValueGenerator | None
    _generation_state: SyntheticGenerationState

    def _generate_jsonl_records(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> GeneratedArray: ...

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> dict[str, JSONValue] | GeneratedRecord: ...

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> dict[str, JSONValue] | GeneratedRecord: ...

    def _generate_from_schema(self, schema: SchemaRecord, rng: random.Random) -> JSONValue: ...

    def _role_cycle(self) -> list[str]: ...

    def _profile_message_count(self, rng: random.Random, requested: int) -> int: ...

    def iter_metadata_document_chunks(self, data: JSONValue) -> Generator[bytes, None, None]: ...

    def iter_serialized_chunks(self, data: JSONValue) -> Generator[bytes, None, None]: ...


def _resolve_style(style: str) -> SyntheticStyle:
    if style == "default":
        return "default"
    if style == "demo":
        return "demo"
    if style == "tool-heavy":
        return "tool-heavy"
    if style == "demo-tool-heavy":
        return "demo-tool-heavy"
    if style == "demo-attachments":
        return "demo-attachments"
    raise ValueError(f"Unknown synthetic style: {style}")


def _as_json_value(value: object) -> JSONValue:
    for current in walk_generated_values(value):
        if current is None or isinstance(current, str | SampledText | JoinedText | int | float | bool):
            continue
        if isinstance(current, list | GeneratedArray):
            continue
        if isinstance(current, dict | GeneratedRecord) and all(isinstance(key, str) for key in current):
            continue
        raise TypeError(f"Generated payload is not JSON-compatible: {type(current).__name__}")
    return cast(JSONValue, value)


def _expected_session_id(provider: str, native_session_id: str | None) -> str | None:
    if native_session_id is None:
        return None
    prefixes = {
        "chatgpt": "chatgpt-export",
        "claude-ai": "claude-ai-export",
        "claude-code": "claude-code-session",
        "codex": "codex-session",
        "gemini": "aistudio-drive",
        "gemini-cli": "gemini-cli-session",
        "hermes": "hermes-session",
        "antigravity": "antigravity-session",
    }
    prefix = prefixes.get(provider)
    return f"{prefix}:{native_session_id}" if prefix is not None else None


def _planted_artifact_facts(
    provider: str,
    data: JSONValue,
    *,
    raw_sha256: str,
    native_session_id: str | None,
    message_count: int,
    inherited_tool_ids: tuple[Iterable[GeneratedString], Iterable[GeneratedString]] = ((), ()),
) -> SyntheticArtifactFacts:
    """Record wire-level facts without consulting production normalization."""
    tool_uses = [_fact_identity(value) for value in inherited_tool_ids[0]]
    tool_results = [_fact_identity(value) for value in inherited_tool_ids[1]]

    for event, identity in _iter_planted_tool_ids(data):
        (tool_uses if event == "call" else tool_results).append(_fact_identity(identity))
    return SyntheticArtifactFacts(
        provider=provider,
        native_session_id=native_session_id,
        expected_session_id=_expected_session_id(provider, native_session_id),
        message_count=message_count,
        tool_use_ids=tuple(tool_uses),
        tool_result_ids=tuple(tool_results),
        raw_sha256=raw_sha256,
    )


def _fact_identity(value: GeneratedString) -> str:
    # Explicit fixture/fact collection opts into concrete identities, just as
    # it opts into full artifact bytes. The lazy driver retains the recipe.
    return value if isinstance(value, str) else "".join(value.pieces())


def _iter_planted_tool_ids(data: JSONValue) -> Iterator[tuple[str, GeneratedString]]:
    for value in walk_generated_values(data):
        if not isinstance(value, dict | GeneratedRecord):
            continue
        if value.get("type") == "tool_use" and isinstance(identity := value.get("id"), str | SampledText | JoinedText):
            yield "call", identity
        if value.get("type") == "tool_result" and isinstance(
            identity := value.get("tool_use_id"), str | SampledText | JoinedText
        ):
            yield "result", identity


class _RetainedToolIds:
    """Borrow prefix facts on the original native cohort and current session scope."""

    def __init__(self, owner: ForeignKeyGraph, prefix: int, messages: int, event: str) -> None:
        self.owner = owner
        self.connection = owner._live_tool_owner()
        self.generation = owner._generation
        self.prefix = prefix
        self.messages = messages
        self.event = event

    def __iter__(self) -> Iterator[GeneratedString]:
        if self.owner._connection is not self.connection or self.owner._generation != self.generation:
            raise RuntimeError("inherited tool identities lost their original prefix owner")
        identities = self.owner.iter_prefix_tools(self.prefix, self.messages, self.event)
        try:
            for identity in identities:
                if self.owner._connection is not self.connection or self.owner._generation != self.generation:
                    raise RuntimeError("inherited tool identities lost their original prefix owner")
                yield identity
        finally:
            identities.close()


def _profile_mapping(profile: SchemaRecord | None, *path: str) -> dict[str, JSONValue]:
    value: object = profile
    for name in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(name)
    return value if isinstance(value, dict) else {}


def _positive_profile_count(profile: dict[str, JSONValue], name: str) -> int:
    value = profile.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _session_native_id(provider: str, data: JSONValue) -> str | None:
    wire_format = PROVIDER_WIRE_FORMATS.get(provider)
    if (
        wire_format is not None
        and wire_format.document_session_field is not None
        and isinstance(data, dict | GeneratedRecord)
    ):
        document_native_id = data.get(wire_format.document_session_field)
        return document_native_id if isinstance(document_native_id, str) else None
    if not isinstance(data, list | GeneratedArray):
        return None
    if provider == "codex":
        for record in data:
            if not isinstance(record, dict | GeneratedRecord):
                continue
            header = codex_session_header(record)
            if header is not None:
                native_id = header.get("id")
                if isinstance(native_id, str):
                    return native_id
    if provider == "claude-code":
        for record in data:
            if isinstance(record, dict | GeneratedRecord):
                native_id = record.get("sessionId")
                if isinstance(native_id, str):
                    return native_id
    return None


_ArtifactResult = TypeVar("_ArtifactResult")


def _is_codex_message(record: JSONValue) -> bool:
    return isinstance(record, dict | GeneratedRecord) and codex_message_record(record)


def _session_chunks(
    self: _SyntheticBatchContext, data: JSONValue, prefix: tuple[int, int] | None
) -> Generator[bytes, None, None]:
    if prefix is None:
        yield from self.iter_serialized_chunks(data)
        return
    if not isinstance(data, list | GeneratedArray):
        raise ValueError("inherited Codex records require their declared JSONL carrier")
    inserted = False
    validate_wire_payload(self.provider, data)
    for record in data:
        yield from self.iter_serialized_chunks([record])
        if not inserted and isinstance(record, dict | GeneratedRecord) and codex_session_header(record) is not None:
            # The same retained bytes, after the original child header. No
            # decoder or re-sampling may change the parent's inherited prefix.
            yield from self._relation_solver.fk_graph.iter_prefix_chunks(*prefix)
            inserted = True


def _settled_session_chunks(
    self: _SyntheticBatchContext,
    data: JSONValue,
    prefix: tuple[int, int] | None,
    consumed: list[bool],
    emitted_bytes: list[int],
) -> Generator[bytes, None, None]:
    for chunk in _session_chunks(self, data, prefix):
        emitted_bytes[0] += len(chunk)
        yield chunk
    consumed[0] = True


def iter_sessions(
    self: _SyntheticBatchContext,
    *,
    count: int | None = None,
    target_bytes: int | None = None,
    messages_per_session: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
    session_native_ids: tuple[str, ...] = (),
) -> Generator[SyntheticSessionChunks, None, SyntheticGenerationReport]:
    """The sole lazy session/chunk producer for every workload consumer."""
    state = self._generation_state
    if state.driver_active:
        raise ValueError("the original session driver is already active")
    state.driver_active = True
    original_selection = (
        self.schema,
        self.wire_format,
        self.package_version,
        self.element_kind,
        self.workload_profile,
        self.element_schemas,
    )
    try:
        resolved_style = _resolve_style(style)
        if (
            resolved_style in {"tool-heavy", "demo-tool-heavy", "demo-attachments"}
            or resolved_style == "demo"
            and self.provider == "chatgpt"
        ):
            from polylogue.schemas.synthetic.authored_selection import authored_schema_selection

            selected = authored_schema_selection(
                self.provider,
                resolved_style,
                session_native_ids=session_native_ids,
                theme_index=random.Random(seed).randrange(len(_DEMO_THEMES)) if resolved_style == "demo" else 0,
            )
            self.schema = selected.schema
            self.wire_format = selected.wire_format
            self.package_version = selected.package_version
            self.element_kind = selected.element_kind
            self.workload_profile = selected.workload_profile
            self.element_schemas = dict(selected.element_schemas)
        # Selected field support is captured once for this driver lifetime.
        # Schema fragments/union overlays keep native child coordinates rather
        # than serializing the entire selected schema for every generated row.
        owner = self._relation_solver.fk_graph
        captured: dict[int, SchemaRecord] = {}
        for original in (self.schema, *self.element_schemas.values()):
            if id(original) not in captured:
                captured[id(original)] = owner.capture_schema(original)
        self.schema = captured[id(self.schema)]
        self.element_schemas = {kind: captured[id(schema)] for kind, schema in self.element_schemas.items()}
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("session count must be nonnegative")
        if target_bytes is not None and (type(target_bytes) is not int or target_bytes < 0):
            raise ValueError("byte target must be a nonnegative exact integer")
        if session_native_ids and len(session_native_ids) != count:
            raise ValueError("session_native_ids must contain exactly one id per generated session")
        rng = random.Random(seed)
        profile_id = (
            str(self.workload_profile.get("profile_id"))
            if isinstance(self.workload_profile, Mapping) and self.workload_profile.get("profile_id") is not None
            else "no-profile"
        )
        profile_rng = random.Random(f"{seed!r}\x1f{self.provider}\x1f{self.package_version}\x1f{profile_id}")
        self._public_record_counts.clear()
        selected_relationships: Counter[str] = Counter()
        inherited = self.provider == "codex" and self.workload_profile is not None
        lineage_edge_count = 0
        generated_count = 0
        emitted_bytes = [0]
        for index in ordinal_sequence():
            if self._generation_state is not state:
                raise RuntimeError("session driver lost its original generation state")
            if count is not None and index >= count or target_bytes is not None and emitted_bytes[0] >= target_bytes:
                break
            if self._public_scope is not None:
                self._public_scope.close()
                self._public_scope = None
            self._relation_solver.begin_scope(self.schema)
            self._semantic_gen = None
            self._public_record_plan = None
            self._session_record_structure = None
            self._session_header_bound = False
            header_class, active_prefix, parent_id, own_records, expected_messages = _select_session_header(
                self, profile_rng, index
            )
            self._session_header_class = header_class
            self._session_parent_id = parent_id
            native_id = session_native_ids[index] if session_native_ids else None
            self._session_native_id = native_id
            if active_prefix is not None:
                self._relation_solver.fk_graph.restore_prefix_tool_context(*active_prefix)
            if own_records is None:
                n_messages = self._profile_message_count(profile_rng, rng.choice(messages_per_session))
            else:
                from polylogue.schemas.synthetic.record_profiles import RecordScopeKinds

                # This original header/count joint already selects the scope
                # length. Do not sample and prepare a discarded marginal path.

                assert self._session_record_structure is not None
                self._public_scope = RecordScopeKinds(
                    self._session_record_structure,
                    own_records,
                    self._relation_solver.fk_graph,
                    provider=self.provider,
                    schema=self.schema,
                    message_count=expected_messages,
                ).sample(profile_rng)
                n_messages = own_records
            authored_theme = self.workload_profile.get("authored_theme_index") if self.workload_profile else None
            theme = (
                _DEMO_THEMES[authored_theme]
                if type(authored_theme) is int
                else rng.choice(_DEMO_THEMES)
                if resolved_style == "demo"
                else None
            )
            data = _generate_session(self, n_messages, rng, theme=theme)
            if header_class is not None and not self._session_header_bound:
                raise ValueError("selected first-header class was not drawn from the original schema")
            if native_id is not None and _session_native_id(self.provider, data) != native_id:
                raise ValueError("original record graph did not bind its selected session identity")
            if own_records is not None and (
                not isinstance(data, list | GeneratedArray)
                or len(data) != own_records
                or sum(_is_codex_message(record) for record in data) != expected_messages
            ):
                raise ValueError("selected header count joint conflicts with its original record grammar")
            selected_relationships.update(self._relation_solver.fk_graph.tool_relationship_counts())
            owner = self._relation_solver.fk_graph
            prepare_agent_adjuncts(self, rng)
            prepare_opaque_adjuncts(self._relation_solver, self.workload_profile, rng)
            settle_agent_adjuncts(self)
            current_id = _session_native_id(self.provider, data)
            if active_prefix is not None:
                lineage_edge_count += 1
            if inherited and current_id is not None and isinstance(data, list | GeneratedArray):
                own_messages = sum(_is_codex_message(record) for record in data)
                inherited_messages = active_prefix[1] if active_prefix is not None else 0
                prefix_id = owner.retain_session_prefix(
                    current_id,
                    parent_prefix=active_prefix[0] if active_prefix is not None else None,
                    inherited_messages=inherited_messages,
                    own_messages=own_messages,
                    record_count=len(data) + inherited_messages,
                )
                message = 0
                for record in data:
                    if not _is_codex_message(record):
                        continue
                    assert isinstance(record, dict | GeneratedRecord)
                    for ordinal, block in enumerate(iter_tool_records("codex", record)):
                        operand = tool_identity_operand(block)
                        if operand is None:
                            continue
                        _, event, field, identity = operand
                        if not isinstance(identity, str | SampledText | JoinedText):
                            continue
                        if not isinstance(block, GeneratedRecord):
                            raise ValueError("prefix tool identity lacks its original generated field schema")
                        properties = block.selected_schema.get("properties")
                        schema = properties.get(field) if isinstance(properties, dict) else None
                        if not isinstance(schema, dict):
                            raise ValueError("prefix identity field schema was not retained")
                        mode = tool_result_content_mode(block) if event == "result" else ""
                        if event == "result":
                            cursor = owner._live_tool_owner().execute(
                                "SELECT mode FROM generated_tool_reference_events WHERE record_id=?", (block.record_id,)
                            )
                            try:
                                observed = cursor.fetchone()
                            finally:
                                cursor.close()
                            if observed is not None:
                                mode = observed[0]
                        owner.retain_prefix_tool(
                            prefix_id, event, message, ordinal, identity, schema=schema, result_mode=mode
                        )
                    with_chunks = self.iter_serialized_chunks([record])
                    try:
                        for ordinal, chunk in enumerate(with_chunks):
                            owner.retain_prefix_chunk(prefix_id, message, ordinal, chunk)
                    finally:
                        with_chunks.close()
                    message += 1
                owner.complete_session_prefix(prefix_id)

            observed_message_count = (
                sum(_is_codex_message(record) for record in data) + (active_prefix[1] if active_prefix else 0)
                if self.provider == "codex" and isinstance(data, list | GeneratedArray)
                else sum(isinstance(node.get("message"), Mapping) for node in iter_provider_records("chatgpt", data))
                if self.provider == "chatgpt" and isinstance(data, Mapping)
                else sum(
                    record_kind("claude-code", record).startswith(("user_", "assistant_"))
                    for record in data
                    if isinstance(record, Mapping)
                )
                if self.provider == "claude-code" and isinstance(data, list | GeneratedArray)
                else n_messages
            )
            reported_native_id = current_id if self.provider == "codex" else native_id
            consumed = [False]
            owner = self._relation_solver.fk_graph
            connection = owner._owner()
            cursor = connection.execute(
                "SELECT 1 FROM generated_adjuncts UNION ALL SELECT 1 FROM generated_structured_adjuncts LIMIT 1"
            )
            try:
                has_adjuncts = cursor.fetchone() is not None
            finally:
                cursor.close()
            adjunct_consumed = [not has_adjuncts]
            adjuncts = iter_session_adjuncts(self._relation_solver, adjunct_consumed, emitted_bytes)
            chunks = _settled_session_chunks(self, data, active_prefix, consumed, emitted_bytes)
            try:
                yield SyntheticSessionChunks(
                    index=index,
                    data=data,
                    message_count=observed_message_count,
                    style=resolved_style,
                    native_session_id=reported_native_id,
                    chunks=chunks,
                    inherited_tool_ids=(
                        (
                            _RetainedToolIds(owner, *active_prefix, "call"),
                            _RetainedToolIds(owner, *active_prefix, "result"),
                        )
                        if active_prefix is not None
                        else ((), ())
                    ),
                    adjuncts=adjuncts,
                )
                if self._generation_state is not state:
                    raise RuntimeError("session driver lost its original generation state")
                if not consumed[0]:
                    raise ValueError("consume the original session chunks before advancing the driver")
                if not adjunct_consumed[0]:
                    raise ValueError("consume the original adjunct files before advancing the session driver")
                generated_count += 1
            finally:
                chunks.close()
                adjuncts.close()
        report = SyntheticGenerationReport(
            provider=self.provider,
            package_version=self.package_version,
            element_kind=self.element_kind,
            wire_encoding=self.wire_format.encoding,
            requested_count=count,
            generated_count=generated_count,
            requested_bytes=target_bytes,
            generated_bytes=emitted_bytes[0],
            style=resolved_style,
            seed=seed,
            workload_profile_id=None if profile_id == "no-profile" else profile_id,
            sampled_record_kind_counts=dict(sorted(self._public_record_counts.items())),
            relationship_variant_counts=dict(sorted(selected_relationships.items())),
            lineage_edge_count=lineage_edge_count,
        )
        return report
    finally:
        if self._public_scope is not None:
            self._public_scope.close()
            self._public_scope = None
        self._public_record_plan = None
        self._session_header_class = None
        self._session_record_structure = None
        self._session_native_id = None
        self._session_parent_id = None
        state.relation_solver.close()
        state.driver_active = False
        (
            self.schema,
            self.wire_format,
            self.package_version,
            self.element_kind,
            self.workload_profile,
            self.element_schemas,
        ) = original_selection


def _produce_artifacts(
    self: _SyntheticBatchContext,
    emit: Callable[
        [
            int,
            JSONValue,
            int,
            SyntheticStyle,
            str | None,
            Generator[bytes, None, None],
            tuple[Iterable[GeneratedString], Iterable[GeneratedString]],
            Generator[SyntheticAdjunctChunks, None, None],
        ],
        _ArtifactResult,
    ],
    *,
    count: int,
    messages_per_session: range,
    seed: int | None,
    style: str,
    session_native_ids: tuple[str, ...],
) -> tuple[list[_ArtifactResult], SyntheticGenerationReport]:
    """Collect explicit callback results from the same lazy original producer."""
    sessions = iter_sessions(
        self,
        count=count,
        messages_per_session=messages_per_session,
        seed=seed,
        style=style,
        session_native_ids=session_native_ids,
    )
    results: list[_ArtifactResult] = []
    try:
        while True:
            try:
                session = next(sessions)
            except StopIteration as complete:
                return results, complete.value
            results.append(
                emit(
                    session.index,
                    session.data,
                    session.message_count,
                    session.style,
                    session.native_session_id,
                    session.chunks,
                    session.inherited_tool_ids,
                    session.adjuncts,
                )
            )
    finally:
        sessions.close()


def generate_batch(
    self: _SyntheticBatchContext,
    count: int = 5,
    messages_per_session: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
    session_native_ids: tuple[str, ...] = (),
) -> SyntheticGenerationBatch:
    def collect(
        index: int,
        data: JSONValue,
        message_count: int,
        resolved_style: SyntheticStyle,
        native_id: str | None,
        chunks: Generator[bytes, None, None],
        inherited_tool_ids: tuple[Iterable[GeneratedString], Iterable[GeneratedString]],
        adjuncts: Generator[SyntheticAdjunctChunks, None, None],
    ) -> SyntheticArtifact:
        raw_bytes = b"".join(chunks)
        return SyntheticArtifact(
            raw_bytes=raw_bytes,
            adjuncts=tuple(
                SyntheticAdjunct(
                    file.name, file.reference_kind, b"".join(file.chunks), file.directory, file.artifact_kind
                )
                for file in adjuncts
            ),
            message_count=message_count,
            style=resolved_style,
            facts=_planted_artifact_facts(
                self.provider,
                data,
                raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                native_session_id=native_id,
                message_count=message_count,
                inherited_tool_ids=inherited_tool_ids,
            ),
        )

    artifacts, report = _produce_artifacts(
        self,
        collect,
        count=count,
        messages_per_session=messages_per_session,
        seed=seed,
        style=style,
        session_native_ids=session_native_ids,
    )
    return SyntheticGenerationBatch(artifacts=artifacts, report=report)


def _select_session_header(
    self: _SyntheticBatchContext, rng: random.Random, index: int
) -> tuple[SchemaRecord | None, tuple[int, int] | None, str | None, int | None, int | None]:
    """Select only original header/count joints with an exact retained parent.

    Parent candidates stay on the original native graph. Counting and an
    ordinal lookup do not turn a cohort into a Python identity population.
    """
    if self.provider != "codex" or self.workload_profile is None:
        return None, None, None, None, None
    from polylogue.schemas.generation.workload_profiles import SessionPrefixCounts

    lineage = self.workload_profile.get("session_lineage")
    classes = lineage.get("header_classes") if isinstance(lineage, dict) else None
    structures = lineage.get("record_structures") if isinstance(lineage, dict) else None
    if not isinstance(classes, list) or not classes or not isinstance(structures, dict):
        raise ValueError("first-header joint evidence is missing; re-infer the package")
    owner = self._relation_solver.fk_graph._owner()
    feasible: list[tuple[SchemaRecord, list[tuple[tuple[int, ...], float]]]] = []
    weights: list[int] = []
    selected_point: tuple[int, ...] | None = None
    authored_cycle = self.workload_profile.get("authored_session_cycle")
    if authored_cycle is not None:
        if (
            self.workload_profile.get("profile_kind") != "authored-fixture"
            or not isinstance(authored_cycle, list)
            or not authored_cycle
        ):
            raise ValueError("session declarations require an authored neutral fixture profile")
        declaration = authored_cycle[index % len(authored_cycle)]
        if not isinstance(declaration, dict):
            raise ValueError("authored first-header declaration is malformed")
        selected_class = declaration.get("class")
        if "prefix_point" in declaration:
            declared_point = declaration["prefix_point"]
            SessionPrefixCounts._validate_point(declared_point)
            assert isinstance(declared_point, list)
            selected_point = tuple(cast(int, value) for value in declared_point)
        if type(selected_class) is not int or not 0 <= selected_class < len(classes):
            raise ValueError("authored first-header declaration is malformed")
        classes = [classes[selected_class]]
    if self._coverage_witness_mode and self._coverage_header_form is not None:
        classes = [row for row in classes if isinstance(row, dict) and row.get("form") == self._coverage_header_form]
    for row in classes:
        if not isinstance(row, dict):
            raise ValueError("first-header class is malformed")
        mass_count = row.get("count")
        if type(mass_count) is not int or mass_count <= 0:
            raise ValueError("first-header class is malformed")
        outcome = row.get("outcome")
        points: list[tuple[tuple[int, ...], float]] = []
        if outcome == "matched":
            joint = SessionPrefixCounts.from_payload(row.get("prefix_joint"))
            for key, (_, mass) in joint.bins.items():
                compatible_points: list[tuple[int, ...]] = []
                for point in joint.extrema[key]:
                    if selected_point is not None and point != selected_point:
                        continue
                    cursor = owner.execute(
                        "SELECT COALESCE(MAX(selection_ordinal)+1,0) FROM generated_session_prefixes WHERE completed=1 "
                        "AND record_count=? AND inherited_messages+own_messages=?",
                        (point[1], point[3]),
                    )
                    try:
                        if cursor.fetchone()[0]:
                            compatible_points.append(point)
                    finally:
                        cursor.close()
                if compatible_points:
                    points.extend((point, mass / len(compatible_points)) for point in compatible_points)
            if not points:
                continue
        elif outcome not in {"no_explicit_parent", "spawn:missing_parent", "missing_header"}:
            raise ValueError("original first-header identity is unresolved; no generated parent is declared")
        feasible.append((row, points))
        weights.append(mass_count)
    if not feasible:
        raise ValueError("original first-header joint has no realizable retained parent")
    row, points = rng.choices(feasible, weights=weights, k=1)[0]
    carrier = row.get("carrier")
    by_form = structures.get(carrier) if isinstance(carrier, str) else None
    structure = by_form.get(row.get("form")) if isinstance(by_form, dict) else None
    if not isinstance(structure, dict):
        raise ValueError("first-header conditional record grammar is missing")
    self._session_record_structure = structure
    selected_class_row = row
    if not points:
        if row.get("outcome") == "missing_header":
            row = None
        if self.workload_profile.get("profile_kind") == "authored-fixture" and selected_point is None:
            # Explicit fixture record cycles own their declared request count;
            # they do not claim to sample an operator root-length population.
            return row, None, None, None, None
        joint = SessionPrefixCounts.from_payload(
            selected_class_row.get("prefix_joint") if row is None else row.get("prefix_joint")
        )
        if not joint.bins:
            raise ValueError("original root header count joint is missing; re-infer the package")
        eligible_bins = [key for key in joint.bins if selected_point is None or selected_point in joint.extrema[key]]
        if not eligible_bins:
            raise ValueError("authored root count declaration is absent from measured header evidence")
        key = rng.choices(eligible_bins, weights=[joint.bins[key][1] for key in eligible_bins], k=1)[0]
        point = selected_point if selected_point is not None else rng.choice(joint.extrema[key])
        if point not in joint.extrema[key]:
            raise ValueError("authored root count declaration is absent from measured header evidence")
        return row, None, None, point[0], point[2]
    point = rng.choices([item[0] for item in points], weights=[item[1] for item in points], k=1)[0]
    query = (
        " FROM generated_session_prefixes WHERE completed=1 AND record_count=? AND inherited_messages+own_messages=?"
    )
    cursor = owner.execute("SELECT COALESCE(MAX(selection_ordinal)+1,0)" + query, (point[1], point[3]))
    try:
        candidates = cursor.fetchone()[0]
    finally:
        cursor.close()
    cursor = owner.execute(
        "SELECT prefix_id,native_id" + query + " AND selection_ordinal=?",
        (point[1], point[3], rng.randrange(candidates)),
    )
    try:
        parent = cursor.fetchone()
    finally:
        cursor.close()
    return row, (parent[0], point[4]), parent[1], point[0] - point[4], point[2] - point[4]


def _generate_session(
    self: _SyntheticBatchContext,
    n_messages: int,
    rng: random.Random,
    *,
    theme: SessionTheme | None = None,
) -> JSONValue:
    wf = self.wire_format
    if wf.encoding == "jsonl":
        return _as_json_value(self._generate_jsonl_records(n_messages, rng, theme=theme))
    if wf.tree and wf.tree.container_path:
        return _as_json_value(self._generate_tree_json(n_messages, rng, theme=theme))
    if wf.messages_path:
        return _as_json_value(self._generate_linear_json(n_messages, rng, theme=theme))
    roles = self._role_cycle()
    self._semantic_gen = SemanticValueGenerator(
        rng,
        theme=theme,
        base_ts=rng.uniform(1670000000, 1760000000),
        role_cycle=roles,
    )
    return _as_json_value(self._generate_from_schema(self.schema, rng))


def _role_cycle(self: _SyntheticBatchContext) -> list[str]:
    match self.provider:
        case "chatgpt":
            return ["user", "assistant"]
        case "claude-ai":
            return ["human", "assistant"]
        case "claude-code":
            return ["user", "assistant"]
        case "codex":
            return ["user", "assistant"]
        case "gemini":
            return ["user", "model"]
        case _:
            return ["user", "assistant"]


__all__ = [
    "_generate_session",
    "_role_cycle",
    "generate_batch",
]
