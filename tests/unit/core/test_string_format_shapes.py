"""Measured format conditioning keeps mixed fields and their tails intact."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONDocument
from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.field_stats.evidence import (
    deserialize_field_stats,
    merge_field_stats_into,
    serialize_field_stats,
)
from polylogue.schemas.generation.field_annotations import annotate_schema
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.observed_values import existing_value_is_supported


def _mixed_samples() -> list[JSONDocument]:
    uuids: list[JSONDocument] = [{"identifier": str(uuid.UUID(int=index + 1, version=4))} for index in range(90)]
    short: list[JSONDocument] = [{"identifier": "unformatted value x"} for _ in range(5)]
    tail: list[JSONDocument] = [{"identifier": "tail " + "x" * 57} for _ in range(5)]
    return uuids + short + tail


def _schema(samples: list[JSONDocument]) -> JSONDocument:
    return annotate_schema(
        {"type": "object", "properties": {"identifier": {"type": "string"}}},
        _collect_field_stats(samples),
    )


def test_format_conditioning_preserves_mixed_uuid_lengths_and_weights() -> None:
    schema = _schema(_mixed_samples())
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-format-control")
    values = [json.loads(raw)["identifier"] for raw in corpus.generate(count=1_000, seed=31)]
    lengths = Counter(map(len, values))

    assert set(lengths) == {19, 36, 62}
    assert 850 <= lengths[36] <= 950
    assert 20 <= lengths[19] <= 80
    assert 20 <= lengths[62] <= 80
    assert all(uuid.UUID(value).version == 4 for value in values if len(value) == 36)
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    assert all(
        existing_value_is_supported(node, next(value for value in values if len(value) == length))
        for length in (19, 36, 62)
    )
    observed = cast(JSONDocument, node["x-polylogue-observed-distribution"])
    assert cast(JSONDocument, observed["string_length"])["min"] == 19
    assert cast(JSONDocument, observed["string_length"])["max"] == 62
    assert "unformatted value" not in json.dumps(schema)
    assert str(uuid.UUID(int=1, version=4)) not in json.dumps(schema)


def test_format_shapes_survive_replay_and_incremental_merge() -> None:
    samples = _mixed_samples()
    complete = _collect_field_stats(samples)
    left = {
        path: deserialize_field_stats(serialize_field_stats(stats))
        for path, stats in _collect_field_stats(samples[:50]).items()
    }
    right = {
        path: deserialize_field_stats(serialize_field_stats(stats))
        for path, stats in _collect_field_stats(samples[50:]).items()
    }
    merge_field_stats_into(left, right)

    for fmt, expected in complete["$.identifier"].string_formats.items():
        actual = left["$.identifier"].string_formats[fmt]
        assert actual.length.count == expected.length.count
        assert actual.length.buckets == expected.length.buckets
        assert actual.length.minimum == expected.length.minimum
        assert actual.length.maximum == expected.length.maximum
        assert actual.newlines.buckets == expected.newlines.buckets
        assert actual.categorical.buckets == expected.categorical.buckets
        assert set(actual.categories) == set(expected.categories)
        for bucket, category in expected.categories.items():
            assert actual.categories[bucket].length.to_state() == category.length.to_state()
            assert actual.categories[bucket].newlines.to_state() == category.newlines.to_state()


def test_long_formatted_strings_are_measured_without_a_detection_length_cap() -> None:
    import base64

    value = base64.b64encode(b"abc" * 400).decode("ascii")
    schema = _schema([{"identifier": value} for _ in range(4)])
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-format-control")
    generated = [json.loads(raw)["identifier"] for raw in corpus.generate(count=8, seed=9)]

    assert all(len(item) == len(value) for item in generated)
    assert all(len(base64.b64decode(item, validate=True)) == 1_200 for item in generated)
    assert set(_collect_field_stats([{"identifier": value}])["$.identifier"].string_formats) == {"base64"}


@pytest.mark.parametrize(
    "corruption",
    ["missing-partition", "count-mismatch", "unknown-format", "boolean-count", "missing-category", "category-count"],
)
def test_incomplete_or_malformed_format_evidence_refuses_visibly(corruption: str) -> None:
    schema = _schema(_mixed_samples())
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    observed = cast(JSONDocument, node["x-polylogue-observed-distribution"])
    shapes = cast(JSONDocument, observed["string_formats"])
    if corruption == "missing-partition":
        del shapes[""]
    elif corruption == "count-mismatch":
        shape = cast(JSONDocument, shapes["uuid4"])
        cast(JSONDocument, shape["newline_count"])["count"] = 1
    elif corruption == "unknown-format":
        shapes["invented-format"] = shapes.pop("uuid4")
    elif corruption == "boolean-count":
        shape = cast(JSONDocument, shapes["uuid4"])
        cast(JSONDocument, shape["categorical"])["count"] = True
    else:
        shape = cast(JSONDocument, shapes[""])
        categories = cast(JSONDocument, shape["categories"])
        bucket = next(iter(categories))
        if corruption == "missing-category":
            del categories[bucket]
        else:
            cast(JSONDocument, cast(JSONDocument, categories[bucket])["string_length"])["count"] = 1

    with pytest.raises(ValueError, match="measured string format"):
        SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-format-control").generate(count=1, seed=3)


def test_persisted_output_format_keeps_original_shape_and_uses_canonical_pointer() -> None:
    from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN
    from polylogue.schemas.synthetic.classification import unsupported_schema_constructs
    from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

    samples: list[JSONDocument] = [
        {"identifier": "prefix\nFull output saved to: /private/a.txt\ntail"},
        {"identifier": "Output has been saved to /private/b.txt"},
        {"identifier": "ordinary neutral prose"},
    ]
    schema = _schema(samples)
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-control")
    values = [json.loads(raw)["identifier"] for raw in corpus.generate(count=300, seed=71)]
    from polylogue.schemas.synthetic.observed_values import observed_length_support

    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    observed = cast(JSONDocument, node["x-polylogue-observed-distribution"])
    formats = cast(JSONDocument, observed["string_formats"])
    expected_lengths = {
        length
        for shape in formats.values()
        for category in cast(JSONDocument, cast(JSONDocument, shape)["categories"]).values()
        for length in observed_length_support(cast(JSONDocument, cast(JSONDocument, category)["string_length"]))
    }
    assert set(map(len, values)) == expected_lengths
    pointer_values = [value for value in values if PERSISTED_OUTPUT_POINTER_PATTERN.search(value)]
    assert 150 < len(pointer_values) < 250
    assert {value.count("\n") for value in pointer_values} == {0, 2}
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    assert not unsupported_schema_constructs(node)
    assert all(existing_value_is_supported(node, value) for value in values)
    assert "private" not in json.dumps(schema) and "a.txt" not in json.dumps(schema)
    with ToolResultIndexAccumulator(disk_backed=True) as index:
        original = pointer_values[0]
        pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(original)
        assert pointer is not None
        index.observe(
            {"message": {"content": [{"type": "tool_result", "tool_use_id": "neutral", "content": original}]}}
        )
        match = index.reference_for_filename(Path(pointer.group(1)).name)
        assert match is not None and match.reference_kind == "persisted" and match.was_truncated


def test_persisted_output_format_streams_long_tail_without_materializing_body() -> None:
    import tracemalloc
    from contextlib import closing

    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.text_value import JoinedText

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "x-polylogue-format": "persisted-tool-output",
                "x-polylogue-observed-distribution": {
                    "pointer_shapes": [{"count": 1, "points": [[2_000_000_000, 0, 21, 0, 16, 16]]}],
                    "string_length": {
                        "count": 1,
                        "histogram": [[245, 1]],
                        "log_base": 1.1,
                        "min": 2_000_000_000,
                        "max": 2_000_000_000,
                    },
                    "newline_count": {"count": 1, "histogram": [[0, 1]], "log_base": 1.1, "min": 0, "max": 0},
                },
            }
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-tail")
    tracemalloc.start()
    try:
        with closing(corpus.iter_sessions(count=1, seed=93)) as sessions:
            session = next(sessions)
            assert isinstance(session.data, GeneratedRecord)
            text = session.data["identifier"]
            assert isinstance(text, JoinedText) and len(text) == 2_000_000_000
            first = next(session.chunks)
            assert first
            assert tracemalloc.get_traced_memory()[1] < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()
    # The same original sampler is usable again after physical settlement.
    with closing(corpus.iter_sessions(count=1, seed=93)) as sessions:
        assert next(sessions).index == 0


def test_formatted_categories_keep_measured_numeric_joints_without_private_values() -> None:
    samples: list[JSONDocument] = [
        {"identifier": "1" * 24},
        {"identifier": "2" * 100},
        {"identifier": "Full output saved to: /neutral/a.txt\nend"},
        {"identifier": "Full output saved to: /neutral/long-file.txt\nline\nend"},
    ]
    schema = _schema(samples)
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-category-joint")
    values = [json.loads(raw)["identifier"] for raw in corpus.generate(count=400, seed=41)]
    assert all(existing_value_is_supported(node, value) for value in values)
    assert {(len(value), value.count("\n")) for value in values} == {
        (len(str(row["identifier"])), str(row["identifier"]).count("\n")) for row in samples
    }
    assert all(value not in {row["identifier"] for row in samples} for value in values)


def test_original_relation_owner_finalizes_current_prior_and_ambiguous_pointer_names() -> None:
    from contextlib import closing

    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
            "tool_use_id": {"type": "string", "x-polylogue-values": ["neutraltool"]},
            "content": {
                "type": "string",
                "x-polylogue-format": "persisted-tool-output",
                "x-polylogue-observed-distribution": {
                    "pointer_shapes": [{"count": 1, "points": [[80, 2, 21, 0, 16, 16]]}],
                    "string_length": {"count": 1, "histogram": [[45, 1]], "log_base": 1.1, "min": 80, "max": 80},
                    "newline_count": {"count": 1, "histogram": [[11, 1]], "log_base": 1.1, "min": 2, "max": 2},
                },
            },
        },
        "required": ["type", "tool_use_id", "content"],
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-relation")
    with closing(corpus.iter_sessions(count=1, seed=37)) as sessions:
        session = next(sessions)
        block = session.data
        assert isinstance(block, GeneratedRecord)
        solver = corpus._relation_solver
        original_content = block["content"]
        assert isinstance(original_content, str)
        original_coordinate = (block.record_id, block.generation, block.original_path)
        original_schema = block.selected_schema
        from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN

        pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(original_content)
        assert pointer is not None
        filename = "n" * len(pointer.group(1))
        current = solver.bind_persisted_output_filename(block, filename)
        finalized = block["content"]
        assert isinstance(finalized, str)
        assert len(finalized) == len(original_content) and finalized.count("\n") == original_content.count("\n")
        assert (block.record_id, block.generation, block.original_path) == original_coordinate
        assert block.selected_schema == original_schema
        assert current.tool_use_id == "neutraltool" and current.reference_kind == "persisted" and current.was_truncated
        encoded = json.loads(b"".join(session.chunks))
        with ToolResultIndexAccumulator(disk_backed=True) as index:
            index.observe({"message": {"content": [encoded]}})
            assert index.reference_for_filename(filename) == current
            # The next original record changes inline shape, but leaves the
            # previous saved-to filename meaningful under the real join law.
            later = solver.fk_graph.new_record({"type": "object"}, path="$.later")
            later.update({"type": "tool_result", "tool_use_id": "neutraltool", "content": None})
            solver.observe_tool_result_reference(later)
            index.observe(
                {"message": {"content": [{"type": "tool_result", "tool_use_id": "neutraltool", "content": None}]}}
            )
            prior = solver.tool_result_reference_for_filename(filename)
            assert prior == index.reference_for_filename(filename)
            assert prior is not None and prior.reference_kind == "persisted" and not prior.was_truncated
            # Finalization stays forbidden for the earlier observed coordinate
            # even after a later same-ID record replaces the current index row.
            with pytest.raises(ValueError, match="precede original reference observation"):
                solver.bind_persisted_output_filename(block, filename)
            assert block["content"] == finalized
            # A direct ID takes precedence over another tool's persisted name.
            ambiguous = solver.fk_graph.new_record({"type": "object"}, path="$.ambiguous")
            ambiguous.update({"type": "tool_result", "tool_use_id": filename, "content": "inline"})
            solver.observe_tool_result_reference(ambiguous)
            index.observe(
                {"message": {"content": [{"type": "tool_result", "tool_use_id": filename, "content": "inline"}]}}
            )
            actual = solver.tool_result_reference_for_filename(filename)
            assert actual == index.reference_for_filename(filename)
            assert actual is not None and actual.reference_kind == "direct" and actual.tool_use_id == filename
            pending = solver.fk_graph.new_record(block.selected_schema, path="$.pending")
            pending.update({"type": "tool_result", "tool_use_id": "another", "content": original_content})
            with pytest.raises(ValueError, match="precedence"):
                solver.bind_persisted_output_filename(pending, filename)
            assert pending["content"] == original_content
            with pytest.raises(ValueError, match="sampled pointer shape"):
                solver.bind_persisted_output_filename(pending, filename + "x")
            assert pending["content"] == original_content
    with pytest.raises(RuntimeError, match="outlived"):
        solver.bind_persisted_output_filename(block, filename)


def test_original_pointer_filename_finalization_keeps_lazy_tail_and_currency() -> None:
    import tracemalloc
    from contextlib import closing

    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.text_value import JoinedText

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
            "tool_use_id": {"type": "string", "x-polylogue-values": ["neutraltool"]},
            "content": {
                "type": "string",
                "x-polylogue-format": "persisted-tool-output",
                "x-polylogue-observed-distribution": {
                    "pointer_shapes": [{"count": 1, "points": [[2_000_000_000, 0, 21, 0, 16, 16]]}],
                    "string_length": {
                        "count": 1,
                        "histogram": [[245, 1]],
                        "log_base": 1.1,
                        "min": 2_000_000_000,
                        "max": 2_000_000_000,
                    },
                    "newline_count": {"count": 1, "histogram": [[11, 1]], "log_base": 1.1, "min": 2, "max": 2},
                },
            },
        },
        "required": ["type", "tool_use_id", "content"],
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-relation-tail")
    tracemalloc.start()
    try:
        with closing(corpus.iter_sessions(count=1, seed=37)) as sessions:
            session = next(sessions)
            block = session.data
            assert isinstance(block, GeneratedRecord)
            before = block["content"]
            assert isinstance(before, JoinedText) and len(before) == 2_000_000_000
            name = before.parts[3]
            assert isinstance(name, str)
            corpus._relation_solver.bind_persisted_output_filename(block, "z" * len(name))
            after = block["content"]
            assert (
                isinstance(after, JoinedText)
                and after.parts[:3] == before.parts[:3]
                and after.parts[4:] == before.parts[4:]
            )
            assert len(after) == len(before)
            assert next(session.chunks)
            assert tracemalloc.get_traced_memory()[1] < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_pointer_width_joint_survives_original_evidence_replay_and_merge() -> None:
    from polylogue.core.tool_result_pointer import persisted_output_shape
    from polylogue.schemas.field_stats.evidence import deserialize_field_stats, merge_field_stats, serialize_field_stats

    samples: list[JSONDocument] = [
        {"identifier": "Full output saved to: /neutral/short.txt\nline"},
        {"identifier": "Output has been saved to:\n/another/component/a-much-longer-name.log\nline\nend"},
        {"identifier": "Full output saved to: single-name"},
    ]
    original = _collect_field_stats(samples)
    restored = merge_field_stats(
        [
            {
                path: deserialize_field_stats(serialize_field_stats(stats))
                for path, stats in _collect_field_stats([sample]).items()
            }
            for sample in samples
        ],
        total_samples=len(samples),
    )
    original_shapes = original["$.identifier"].string_formats
    restored_shapes = restored["$.identifier"].string_formats
    assert set(original_shapes) == set(restored_shapes)
    for name, shape in original_shapes.items():
        replayed = restored_shapes[name]
        assert shape.newlines.count == replayed.newlines.count
        assert shape.newlines.m2 == pytest.approx(replayed.newlines.m2)
        assert shape.length.buckets == replayed.length.buckets
        assert set(shape.categories) == set(replayed.categories)
        for category, point in shape.categories.items():
            assert point.pointer_shapes.bins == replayed.categories[category].pointer_shapes.bins
            assert point.pointer_shapes.extrema == replayed.categories[category].pointer_shapes.extrema
    schema: JSONDocument = {"type": "object", "properties": {"identifier": {"type": "string"}}}
    annotate_schema(schema, restored)
    serialized = json.dumps(schema)
    assert "short.txt" not in serialized and "a-much-longer-name.log" not in serialized
    expected = {persisted_output_shape(str(row["identifier"])) for row in samples}
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-width")
    observed = {persisted_output_shape(json.loads(raw)["identifier"]) for raw in corpus.generate(count=120, seed=41)}
    assert observed == expected
    assert corpus.generate(count=4, seed=41) == corpus.generate(count=4, seed=41)


def test_pointer_joint_retains_same_bin_tails_after_anonymous_category_collision() -> None:
    from polylogue.schemas.field_stats.distributions import categorical_bucket

    prefix = "Full output saved to: /dir/name.txt "
    short = prefix + "x" * (100 - len(prefix))
    # Deliberately collide the existing anonymous category, not a private
    # vocabulary: both lengths also occupy one original fixed-log bin.
    target = categorical_bucket(short)
    ordinal = 0
    while True:
        token = str(ordinal)
        long = prefix + "y" * (105 - len(prefix) - len(token)) + token
        if categorical_bucket(long) == target:
            break
        ordinal += 1
    schema = _schema([{"identifier": short}, {"identifier": long}])
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-pointer-tail-collision")
    values = [json.loads(raw)["identifier"] for raw in corpus.generate(count=60, seed=73)]
    assert {len(value) for value in values} == {100, 105}
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    assert all(existing_value_is_supported(node, value) for value in values)


@pytest.mark.parametrize(
    "value,fmt",
    [
        ("a" * 40 + "\n", "hex-id"),
        ("a" * 40 + "@b.c\n", "email"),
        ("https://neutral.invalid/path\n\nbody", "url"),
        ("2026-01-02T03:04:05\n\nbody", "iso8601"),
        ("00000000-0000-4000-8000-000000000001\n", "uuid4"),
        ("1700000000.25\n", "unix-epoch-str"),
        ("application/neutral\n", "mime-type"),
        ("A" * 41 + "\n", "base64"),
    ],
)
def test_format_spelling_preserves_collector_newline_contract(value: str, fmt: str) -> None:
    from polylogue.schemas.field_stats.detection import FORMAT_PATTERNS
    from polylogue.schemas.synthetic.classification import unsupported_schema_constructs

    schema = _schema([{"identifier": value} for _ in range(4)])
    assert not unsupported_schema_constructs(schema)
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-format-control")
    generated = [json.loads(raw)["identifier"] for raw in corpus.generate(count=12, seed=31)]
    pattern = dict(FORMAT_PATTERNS)[fmt]
    node = cast(JSONDocument, cast(JSONDocument, schema["properties"])["identifier"])
    assert all(len(item) == len(value) and item.count("\n") == value.count("\n") for item in generated)
    assert all(pattern.search(item) is not None for item in generated)
    assert all(existing_value_is_supported(node, item) for item in generated)


def test_anchored_format_refuses_nonterminal_multiline_support() -> None:
    from polylogue.schemas.synthetic.observed_values import format_text_shape_is_representable

    assert format_text_shape_is_representable("hex-id", 41, 1)
    assert not format_text_shape_is_representable("hex-id", 41, 2)
    assert format_text_shape_is_representable("url", 41, 2)
