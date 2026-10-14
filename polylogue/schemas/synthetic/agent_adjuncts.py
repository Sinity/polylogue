"""Original dispatcher children retained on the shared native sampler."""

from __future__ import annotations

import json
import random
from collections.abc import Generator, Mapping
from contextlib import closing
from typing import Protocol

from polylogue.core.claude_code_children import claude_dispatch_operand
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.models import GeneratedRecord, GeneratedValue, SchemaRecord, SyntheticAdjunctChunks
from polylogue.schemas.synthetic.opaque_adjuncts import iter_opaque_adjuncts
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.workload_profile_rows import WorkloadProfile


class _AgentContext(Protocol):
    provider: str
    workload_profile: Mapping[str, object] | None
    element_schemas: dict[str, SchemaRecord]
    _relation_solver: RelationConstraintSolver

    def _generate_agent_transcript(
        self, schema: SchemaRecord, structure: SchemaRecord, count: int, depth: int, rng: random.Random
    ) -> GeneratedArray: ...

    def _generate_agent_metadata(
        self,
        schema: SchemaRecord,
        metadata_class: SchemaRecord,
        agent_id: str,
        stem: str,
        rng: random.Random,
        dispatch_tool_id: str | None = None,
    ) -> GeneratedValue: ...

    def iter_metadata_document_chunks(self, data: GeneratedValue) -> Generator[bytes, None, None]: ...

    def iter_serialized_chunks(self, data: GeneratedValue) -> Generator[bytes, None, None]: ...


def prepare_agent_adjuncts(corpus: _AgentContext, rng: random.Random) -> None:
    """Complete each planned original file before the parent can be yielded."""
    if corpus.provider != "claude-code":
        return
    owner = corpus._relation_solver.fk_graph
    generation = owner._generation
    after = -1
    while True:
        connection = owner._array_owner(generation)
        with closing(
            connection.execute(
                "SELECT ordinal,record_id,transcripts,record_count,metadata,depth,metadata_json FROM generated_agent_children WHERE ordinal>? ORDER BY ordinal LIMIT 1",
                (after,),
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            return
        ordinal, record_id, transcripts, records, metadata, depth, metadata_json = row
        record = GeneratedRecord(owner, record_id, generation)
        operand = claude_dispatch_operand(record)
        if operand is None:
            raise RuntimeError("original dispatcher field lost its selected child identity")
        name = operand[1] + ".jsonl"
        if "/" in name or "\x00" in name or name in {".", ".."}:
            raise ValueError("selected original agent field cannot name its physical child")
        if transcripts not in {0, 1} or not transcripts and records:
            raise ValueError("original child association contradicts its physical transcript count")
        if transcripts:
            schema = corpus.element_schemas.get("agent_transcript")
            profile = corpus.workload_profile
            structures = profile.depth_grammars if isinstance(profile, WorkloadProfile) else None
            structure = structures.get(str(depth)) if structures is not None else None
            if schema is None or not isinstance(structure, dict):
                raise ValueError(
                    "original child schema or depth-conditioned record grammar is missing; re-infer the package"
                )
            data = corpus._generate_agent_transcript(schema, structure, records, depth, rng)
            _retain_file(corpus, name, "agent_transcript", data)
        if metadata:
            meta_schema = corpus.element_schemas.get("agent_sidecar_meta")
            meta_class = json.loads(metadata_json) if metadata_json is not None else None
            result = record.get("toolUseResult")
            if meta_schema is None or not isinstance(meta_class, dict) or not isinstance(result, GeneratedRecord):
                raise ValueError("original metadata schema/class/dispatcher cell is missing; re-infer the package")
            agent_id = result.get("agentId")
            if not isinstance(agent_id, str):
                raise RuntimeError("original metadata dispatcher identity lost its concrete coordinate")
            document = corpus._generate_agent_metadata(meta_schema, meta_class, agent_id, operand[1], rng, operand[0])
            _retain_file(corpus, operand[1] + ".meta.json", "agent_sidecar_meta", document)
        after = ordinal


def _retain_file(corpus: _AgentContext, name: str, kind: str, data: GeneratedValue) -> None:
    owner = corpus._relation_solver.fk_graph
    generation = owner._generation
    connection = owner._array_owner(generation)
    with closing(connection.execute("SELECT COALESCE(max(ordinal),-1)+1 FROM generated_structured_adjuncts")) as cursor:
        row = cursor.fetchone()
    assert row is not None
    ordinal = row[0]
    if (
        not isinstance(data, GeneratedArray | GeneratedRecord)
        or data.owner is not owner
        or data.generation != generation
    ):
        raise ValueError("original child file requires its retained native document or array")
    cell_kind = "array" if isinstance(data, GeneratedArray) else "record"
    cell_id = data.array_id if isinstance(data, GeneratedArray) else data.record_id
    connection.execute(
        "INSERT INTO generated_structured_adjuncts VALUES (?,?,?,0,?,?)", (ordinal, name, kind, cell_kind, cell_id)
    ).close()


def settle_agent_adjuncts(corpus: _AgentContext) -> None:
    """Serialize original sibling cells after the shared opaque join settles."""
    owner = corpus._relation_solver.fk_graph
    generation = owner._generation
    after = -1
    while True:
        with closing(
            owner._array_owner(generation).execute(
                "SELECT ordinal,kind,cell_kind,cell_id FROM generated_structured_adjuncts WHERE ordinal>? ORDER BY ordinal LIMIT 1",
                (after,),
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            return
        ordinal, kind, cell_kind, cell_id = row
        data = (
            GeneratedArray(owner, cell_id, generation)
            if cell_kind == "array"
            else GeneratedRecord(owner, cell_id, generation)
        )
        _retain_chunks(corpus, ordinal, kind, data)
        after = ordinal


def _retain_chunks(corpus: _AgentContext, ordinal: int, kind: str, data: GeneratedValue) -> None:
    owner = corpus._relation_solver.fk_graph
    generation = owner._generation
    chunks = (
        corpus.iter_metadata_document_chunks(data)
        if kind == "agent_sidecar_meta"
        else corpus.iter_serialized_chunks(data)
    )
    length = 0
    try:
        for chunk, payload in enumerate(chunks):
            owner._array_owner(generation).execute(
                "INSERT INTO generated_structured_adjunct_chunks VALUES (?,?,?)", (ordinal, chunk, payload)
            ).close()
            length += len(payload)
    finally:
        chunks.close()
    owner._array_owner(generation).execute(
        "UPDATE generated_structured_adjuncts SET byte_length=? WHERE ordinal=?", (length, ordinal)
    ).close()


def iter_session_adjuncts(
    solver: RelationConstraintSolver, consumed: list[bool], emitted_bytes: list[int]
) -> Generator[SyntheticAdjunctChunks, None, None]:
    opaque_settled = [False]
    opaque = iter_opaque_adjuncts(solver, opaque_settled, emitted_bytes)
    try:
        yield from opaque
    finally:
        opaque.close()
    owner = solver.fk_graph
    generation = owner._generation
    after = -1
    while True:
        connection = owner._array_owner(generation)
        with closing(
            connection.execute(
                "SELECT ordinal,name,kind,byte_length FROM generated_structured_adjuncts WHERE ordinal>? ORDER BY ordinal LIMIT 1",
                (after,),
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            consumed[0] = opaque_settled[0]
            return
        ordinal, name, kind, length = row
        if kind not in {"agent_transcript", "agent_sidecar_meta"}:
            raise RuntimeError("original structured adjunct has an undeclared kind")
        settled = [False]

        def body(
            file: int = ordinal, expected: int = length, complete: list[bool] = settled
        ) -> Generator[bytes, None, None]:
            last = -1
            count = 0
            while True:
                with closing(
                    owner._array_owner(generation).execute(
                        "SELECT chunk,data FROM generated_structured_adjunct_chunks WHERE ordinal=? AND chunk>? ORDER BY chunk LIMIT 1",
                        (file, last),
                    )
                ) as cursor:
                    cell = cursor.fetchone()
                if cell is None:
                    if count != expected:
                        raise RuntimeError("original adjunct bytes lost their completed length")
                    complete[0] = True
                    return
                last, payload = cell
                count += len(payload)
                emitted_bytes[0] += len(payload)
                yield payload

        chunks = body()
        try:
            yield SyntheticAdjunctChunks(name, "dispatch", length, chunks, "subagents", kind)
            if not settled[0]:
                raise ValueError("consume the original child body before advancing its metadata")
        finally:
            chunks.close()
        after = ordinal
