"""The original neutral spelling sequence must not restart at every fresh ID."""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
from typing import cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.generation.field_annotations import annotate_schema
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.categorical_values import _string_shape_candidates, _strings_of_shape
from polylogue.schemas.synthetic.relations import ForeignKeyGraph
from tests.infra.schema_workload_profiles import inferred_record_profile


def _neutral_calls(count: int, *, identity_lengths: tuple[int, ...] | None = None) -> SyntheticCorpus:
    header: JSONDocument = {"type": "session_meta", "payload": {"id": "neutral-session"}}
    calls: list[JSONDocument] = [
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": f"neutral-{i:0{(identity_lengths[i] if identity_lengths is not None else 29) - 8}d}",
                "name": "exec",
                "input": "printf neutral",
            },
        }
        for i in range(count)
    ]

    def shape(value: JSONValue) -> JSONDocument:
        if isinstance(value, dict):
            return {
                "type": "object",
                "required": list(value),
                "properties": {key: shape(child) for key, child in value.items()},
            }
        assert isinstance(value, str)
        return {"type": "string", "x-polylogue-values": [value]}

    branch = shape(calls[0])
    properties = cast(JSONDocument, cast(JSONDocument, branch["properties"])["payload"])["properties"]
    cast(JSONDocument, properties)["call_id"] = {"type": "string"}
    annotate_schema(branch, _collect_field_stats(calls))
    schema: JSONDocument = {"oneOf": [shape(header), branch]}
    profile = inferred_record_profile("codex", schema, [header, *calls], element="session_record_stream")
    return SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )


def test_real_driver_fresh_id_membership_probes_are_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    original_owner = ForeignKeyGraph._owner
    probes = 0

    def trace(sql: str) -> None:
        nonlocal probes
        if sql.startswith("SELECT 1 FROM generated_tool_identities WHERE value_json="):
            probes += 1

    def observed_owner(owner: ForeignKeyGraph) -> sqlite3.Connection:
        connection = original_owner(owner)
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(ForeignKeyGraph, "_owner", observed_owner)
    corpus = _neutral_calls(128)
    try:
        batch = corpus.generate_batch(count=1, seed=31, session_native_ids=("neutral-session",))
        rows = [json.loads(line) for line in batch.raw_items[0].splitlines()]
        identities = [row["payload"]["call_id"] for row in rows if row["type"] == "response_item"]
        assert identities == list(itertools.islice(_strings_of_shape(29, 0, character_class="ascii"), 128))
        assert all(identity.isascii() for identity in identities)
        assert probes <= 2 * len(identities)
    finally:
        corpus._relation_solver.close()


def test_cursor_keeps_unobserved_and_excluded_candidates_and_resets_with_inventory() -> None:
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        candidates = list(itertools.islice(_strings_of_shape(29, 0), 20))
        first = owner.fresh_shape_identity("tool", 29, 0)
        assert first == candidates[0]
        assert owner.fresh_shape_identity("tool", 29, 0) == first
        assert owner.fresh_shape_identity("tool", 29, 0, excluded=frozenset({cast(str, first)})) == candidates[1]
        assert owner.fresh_shape_identity("tool", 29, 0) == first
        for value in candidates[:10]:
            owner.observe_tool_identity(value, "call", "new", {"type": "string"}, "")
        assert owner.fresh_shape_identity("tool", 29, 0) == candidates[10]
        owner.observe_tool_identity(candidates[10], "call", "new", {"type": "string"}, "")
        owner.observe_tool_identity(candidates[10], "result", "call", {"type": "string"}, "")
        rng = random.Random(31)
        assert owner.select_tool_identity("both", rng)[0] == candidates[10]
        assert owner.fresh_shape_identity("tool", 29, 0) == candidates[11]
        owner.advance_scope()
        assert owner.fresh_shape_identity("tool", 29, 0) == first
    finally:
        owner.close()


def test_checkpoint_borrows_original_empty_prefix_and_unicode_fallback_sequences() -> None:
    for length, newlines, prefix in ((0, 0, ""), (4, 4, ""), (8, 2, "agent-"), (1, 0, "")):
        original = list(itertools.islice(_strings_of_shape(length, newlines, prefix=prefix), 300))
        coordinates = list(itertools.islice(_string_shape_candidates(length, newlines, prefix=prefix), 300))
        assert original == [value for _, _, value in coordinates]
        for phase, ordinal, value in coordinates:
            assert (
                next(_string_shape_candidates(length, newlines, prefix=prefix, phase=phase, ordinal=ordinal))[2]
                == value
            )
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        from polylogue.core.claude_code_children import claude_agent_transcript_stem

        first = owner.fresh_shape_identity("agent", 12, 0, prefix="agent-")
        assert isinstance(first, str) and first.startswith("agent-")
        assert owner.fresh_shape_identity("agent", 12, 0, prefix="agent-") == first
        owner._owner().execute(
            "INSERT INTO generated_agent_names VALUES (?)", (claude_agent_transcript_stem(first),)
        ).close()
        assert owner.fresh_shape_identity("agent", 12, 0, prefix="agent-") != first
    finally:
        owner.close()


def test_real_driver_width_projection_preserves_original_character_coverage() -> None:
    corpus = _neutral_calls(5, identity_lengths=(10, 10, 10, 20, 20))
    from polylogue.schemas.synthetic.observed_values import observed_length_support

    branches = corpus.schema["oneOf"]
    assert isinstance(branches, list)
    identifier = branches[1]
    assert isinstance(identifier, dict)
    for coordinate in (
        "properties",
        "payload",
        "properties",
        "call_id",
        "x-polylogue-observed-distribution",
        "string_length",
    ):
        selected = identifier[coordinate]
        assert isinstance(selected, dict)
        identifier = selected
    expected_widths = set(observed_length_support(identifier))
    widths: set[int] = set()
    try:
        for seed in range(16):
            batch = corpus.generate_batch(count=1, seed=seed, session_native_ids=("neutral-session",))
            rows = [json.loads(line) for line in batch.raw_items[0].splitlines()]
            identities = [row["payload"]["call_id"] for row in rows if row["type"] == "response_item"]
            assert len(identities) == 5
            assert all(value.isascii() for value in identities)
            assert all(len(value) in expected_widths for value in identities)
            widths.update(map(len, identities))
        assert widths == expected_widths
    finally:
        corpus._relation_solver.close()
