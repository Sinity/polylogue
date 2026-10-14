"""Measured paths keep a realizable original relation suffix before drawing."""

from __future__ import annotations

import random

import pytest

from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.record_profiles import _MeasuredPath
from polylogue.schemas.synthetic.relations import ForeignKeyGraph


def test_currently_eligible_fork_keeps_the_required_result_suffix() -> None:
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        schema: SchemaRecord = {"type": "string"}
        owner.observe_tool_identity("original", "call", "new", schema)
        call, result = ("call", "record"), ("result", "record")
        path = _MeasuredPath(
            {call: 1, result: 10_000},
            {result: 1},
            {call: {result: 1}, result: {result: 1}},
            2,
            owner,
            resource_effects={call: (0, 1), result: (1, -1)},
            available_resource=lambda: owner.tool_identity_count("call", "", 0, (8,)),
        ).sample(random.Random(31))
        try:
            # Either event is locally possible with the original call. Only
            # creating another call leaves the required result suffix feasible.
            assert owner.tool_identity_count("call", "", 0, (8,)) == 1
            assert next(path) == ("call", 0)
            owner.observe_tool_identity("nextcall", "call", "new", schema)
            assert next(path) == ("result", None)
            identity, captured = owner.select_tool_identity("call", random.Random(31), "", 0, (8,))
            owner.observe_tool_identity(identity, "result", "call", captured, "string:inline")
            with pytest.raises(StopIteration):
                next(path)
            assert owner.tool_identity_count("call", "", 0, (8,)) == 1
        finally:
            path.close()
    finally:
        owner.close()


def test_original_grammar_with_no_complete_resource_path_refuses_before_yield() -> None:
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        owner.observe_tool_identity("original", "call", "new", {"type": "string"})
        result = ("result", "record")
        path = _MeasuredPath(
            {result: 1},
            {result: 1},
            {result: {result: 1}},
            2,
            owner,
            resource_effects={result: (1, -1)},
            available_resource=lambda: owner.tool_identity_count("call", "", 0, (8,)),
        ).sample(random.Random(31))
        try:
            with pytest.raises(ValueError, match="no complete path"):
                next(path)
            assert owner.tool_identity_count("call", "", 0, (8,)) == 1
        finally:
            path.close()
    finally:
        owner.close()


def test_measured_main_driver_conditions_the_whole_four_record_path() -> None:
    import json

    from polylogue.core.json import JSONDocument
    from polylogue.schemas.field_stats.distributions import _bucket_index
    from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
    from tests.infra.schema_workload_profiles import inferred_record_profile

    def point(value: int) -> JSONDocument:
        return {
            "count": 1,
            "min": value,
            "max": value,
            "histogram": [[_bucket_index(float(value)), 1]],
            "log_base": 1.1,
        }

    identity: JSONDocument = {
        "type": "string",
        "x-polylogue-observed-distribution": {"string_length": point(8), "newline_count": point(0)},
    }
    call: JSONDocument = {
        "type": "object",
        "required": ["type", "id", "name", "input"],
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["tool_use"]},
            "id": identity,
            "name": {"type": "string", "x-polylogue-values": ["exec"]},
            "input": {"type": "object", "properties": {}},
        },
    }
    result: JSONDocument = {
        "type": "object",
        "required": ["type", "tool_use_id", "content"],
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
            "tool_use_id": identity,
            "content": {"type": "string", "x-polylogue-values": ["neutral"]},
        },
    }
    schema: JSONDocument = {
        "type": "object",
        "required": ["type", "message"],
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["assistant", "user"]},
            "message": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["assistant", "user"]},
                    "content": {"type": "array", "items": {"oneOf": [call, result]}},
                },
            },
        },
    }
    samples: list[JSONDocument] = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": identifier, "name": "exec", "input": {}}],
            },
        }
        for identifier in ("call0001", "call0002")
    ]
    result_samples: list[JSONDocument] = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": identifier, "content": "neutral"}],
            },
        }
        for identifier in ("call0001", "call0002")
    ]
    samples.extend(result_samples)
    from polylogue.schemas.field_stats.collection import _collect_field_stats
    from polylogue.schemas.generation.field_annotations import annotate_schema

    annotate_schema(identity, _collect_field_stats([{"id": "call0001"}, {"id": "call0002"}]), path="$.id")
    profile = inferred_record_profile("claude-code", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    for seed in range(12):
        batch = corpus.generate_batch(count=1, seed=seed)
        records = [json.loads(line) for line in batch.raw_items[0].splitlines()]
        assert len(records) == 4
        assert records[0]["type"] == "assistant" and records[-1]["type"] == "user"
        available: set[str] = set()
        observed_calls: set[str] = set()
        observed_results = 0
        for record in records:
            operand = record["message"]["content"][0]
            if record["type"] == "assistant":
                assert operand["id"] not in observed_calls
                observed_calls.add(operand["id"])
                available.add(operand["id"])
            else:
                assert operand["tool_use_id"] in available
                available.remove(operand["tool_use_id"])
                observed_results += 1
        assert batch.report.relationship_variant_counts["paired"] == observed_results
        assert corpus._public_record_counts["assistant_tool_use"] + observed_results == 4


def test_selected_system_record_preserves_incidental_opaque_message_fields() -> None:
    import json

    from polylogue.core.json import JSONDocument
    from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
    from tests.infra.schema_workload_profiles import inferred_record_profile

    schema: JSONDocument = {
        "type": "object",
        "required": ["type", "message"],
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["system"]},
            "message": {
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "label"],
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["opaque-neutral"]},
                                "label": {"type": "string", "x-polylogue-values": ["kept"]},
                            },
                        },
                    },
                },
            },
        },
    }
    original: JSONDocument = {"type": "system", "message": {"content": [{"type": "opaque-neutral", "label": "kept"}]}}
    profile = inferred_record_profile("claude-code", schema, [original], element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    batch = corpus.generate_batch(count=1, seed=31)
    assert [json.loads(line) for line in batch.raw_items[0].splitlines()] == [original]
    assert batch.artifacts[0].message_count == batch.artifacts[0].facts.message_count == 0


def test_selected_data_form_keeps_its_parent_without_a_public_subtype() -> None:
    import json

    from polylogue.core.json import JSONDocument
    from polylogue.schemas.field_stats.distributions import _bucket_index
    from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
    from tests.infra.schema_workload_profiles import inferred_record_profile

    width = 32_768
    schema: JSONDocument = {
        "type": "object",
        "x-polylogue-mutually-exclusive": [
            {"parent": "$", "fields": ["content", "data"]},
            {"parent": "$", "fields": ["message", "data"]},
        ],
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["progress"]},
            "content": {"type": "string"},
            "message": {"type": "object", "properties": {"role": {"type": "string"}}},
            "data": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "fullOutput": {
                        "type": "string",
                        "x-polylogue-observed-distribution": {
                            "string_length": {
                                "min": width,
                                "max": width,
                                "count": 1,
                                "histogram": [[_bucket_index(float(width)), 1]],
                                "log_base": 1.1,
                            }
                        },
                    },
                },
            },
        },
    }
    original: JSONDocument = {
        "type": "progress",
        "data": {"type": "neutral_unpublished_subtype", "fullOutput": "x" * width},
    }
    profile = inferred_record_profile("claude-code", schema, [original], element="session_record_stream")
    for seed in range(8):
        corpus = SyntheticCorpus(
            schema,
            WireFormat(encoding="jsonl"),
            "claude-code",
            element_kind="session_record_stream",
            workload_profile=profile,
        )
        batch = corpus.generate_batch(count=1, seed=seed)
        record = json.loads(batch.raw_items[0])
        assert record["type"] == "progress"
        assert isinstance(record["data"], dict)
        assert len(record["data"]["fullOutput"]) == width
        assert record["data"]["type"] != "neutral_unpublished_subtype"
        assert "content" not in record
        assert "message" not in record


def test_selected_message_quota_filters_a_locally_weighted_infeasible_fork() -> None:
    quiet, message = ("quiet", "record"), ("message", "record")
    owner = ForeignKeyGraph()
    try:
        for seed in range(12):
            path = _MeasuredPath(
                {quiet: 10_000, message: 1},
                {quiet: 1},
                {quiet: {quiet: 1}, message: {quiet: 1}},
                3,
                owner,
                count_effects={quiet: 0, message: 1},
                selected_count=1,
            )
            assert list(path.sample(random.Random(seed))) == [("message", None), ("quiet", None), ("quiet", None)]
    finally:
        owner.close()


def test_impossible_message_parity_refuses_during_original_plan_preparation() -> None:
    quiet, message = ("quiet", "record"), ("message", "record")
    owner = ForeignKeyGraph()
    try:
        with pytest.raises(ValueError, match="no complete original grammar path"):
            _MeasuredPath(
                {quiet: 1},
                {quiet: 1},
                {quiet: {message: 1}, message: {quiet: 1}},
                5,
                owner,
                count_effects={quiet: 0, message: 1},
                selected_count=1,
            )
        cursor = owner._live_tool_owner().execute("SELECT COUNT(*) FROM generated_records")
        try:
            assert cursor.fetchone()[0] == 0
        finally:
            cursor.close()
    finally:
        owner.close()


def test_count_plan_keeps_original_physical_owner_and_generation() -> None:
    state = ("message", "record")
    owner = ForeignKeyGraph()
    path = _MeasuredPath({state: 1}, {state: 1}, {}, 1, owner, count_effects={state: 1}, selected_count=1)
    owner.close()
    with pytest.raises(RuntimeError):
        list(path.sample(random.Random(31)))


def test_actual_codex_header_joint_conditions_messages_before_field_draws() -> None:
    import json

    from polylogue.core.json import JSONDocument
    from polylogue.schemas.record_structure import codex_message_record
    from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
    from tests.infra.schema_workload_profiles import inferred_codex_session_profile

    message: JSONDocument = {
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": []},
    }
    context: JSONDocument = {"type": "turn_context", "payload": {}}
    samples: list[JSONDocument] = [message, message, context, context]
    schema, profile = inferred_codex_session_profile(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"content": {"type": "array", "items": {"type": "object"}}},
                }
            },
        },
        samples,
        native_id="neutral-session",
    )
    for seed in range(12):
        corpus = SyntheticCorpus(
            schema,
            WireFormat(encoding="jsonl"),
            "codex",
            element_kind="session_record_stream",
            workload_profile=profile,
        )
        try:
            records = [json.loads(line) for line in corpus.generate_batch(count=1, seed=seed).raw_items[0].splitlines()]
            assert len(records) == 5
            assert sum(codex_message_record(record) for record in records) == 2
            assert records[0]["type"] == "session_meta"
            assert records[-1]["type"] == "turn_context"
        finally:
            corpus._relation_solver.close()


@pytest.mark.parametrize("lose_joint", (False, True))
def test_selected_message_quota_uses_its_own_call_resource_floor(lose_joint: bool) -> None:
    root, safe, message, quiet, result = ((name, "record") for name in ("root", "safe", "message", "quiet", "result"))
    owner = ForeignKeyGraph()
    try:
        path = _MeasuredPath(
            {root: 10_000, safe: 1},
            {quiet: 1, result: 1},
            {root: {message: 1, quiet: 1}, safe: {quiet: 1}, message: {result: 1}, quiet: {quiet: 1, result: 1}},
            3,
            owner,
            resource_effects={root: (0, 0), safe: (0, 0), message: (1, -1), quiet: (0, 0), result: (1, -1)},
            available_resource=lambda: 1,
            count_effects={root: 0, safe: 1, message: 1, quiet: 0, result: 0},
            selected_count=1,
        )
        layer = path._layer(2)
        # The other quota needs no calls; the requested quota needs two.
        root_key = next(key for key in layer if key.startswith("root"))
        assert layer[root_key][1] == 0 and layer[root_key][3][1] == 2
        if lose_joint:
            owner._live_tool_owner().execute(
                "UPDATE generated_path_layers SET count_resource=? WHERE path_id=? AND remaining=? AND state=?",
                ('{"0":0,"1":0}', path.layers.array_id, 2, root_key),
            ).close()
        chosen = path.sample(random.Random(31))
        try:
            if lose_joint:
                assert next(chosen)[0] == "root"
            else:
                assert next(chosen)[0] == "safe"
        finally:
            chosen.close()
    finally:
        owner.close()
