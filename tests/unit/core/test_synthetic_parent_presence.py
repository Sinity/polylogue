"""Property draws use original containing-object occurrences, not root rarity."""

import random
from collections.abc import Mapping
from copy import deepcopy

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.generation.field_annotations import reannotate_property_presence
from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText


def _object(value: JSONValue) -> JSONDocument:
    assert isinstance(value, dict)
    return value


def test_rare_selected_parent_keeps_its_always_present_large_text() -> None:
    samples: list[JSONDocument] = [{} for _ in range(80)] + [{"holder": {"text": "neutral " * 512}} for _ in range(20)]
    schema = generate_schema_from_samples(samples, provider="neutral", max_stats_samples=0)
    holder = _object(_object(schema["properties"])["holder"])
    text = _object(_object(holder["properties"])["text"])
    assert holder["x-polylogue-frequency"] == pytest.approx(0.2)
    assert text.get("x-polylogue-frequency", 1) == 1
    assert _object(text["x-polylogue-observed-distribution"])["encountered_documents"] == 20
    assert _object(text["x-polylogue-observed-distribution"])["parent_object_observations"] == 20
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral")
    try:
        selected = 0
        for seed in range(300):
            record = corpus._generate_from_schema(schema, random.Random(seed))
            assert isinstance(record, Mapping)
            if "holder" not in record:
                continue
            selected += 1
            assert isinstance(record["holder"], Mapping)
            value = record["holder"]["text"]
            assert isinstance(value, str | SampledText | JoinedText)
            assert len(value) == 4096
        assert 35 <= selected <= 85
    finally:
        corpus._relation_solver.close()


def test_array_item_presence_uses_each_original_object_occurrence() -> None:
    samples: list[JSONDocument] = [{} for _ in range(8)] + [
        {"items": [{"text": "neutral"}, {"text": "neutral", "extra": 7}]} for _ in range(2)
    ]
    schema = generate_schema_from_samples(samples, provider="neutral", max_stats_samples=0)
    items = _object(_object(_object(schema["properties"])["items"])["items"])
    properties = _object(items["properties"])
    assert _object(properties["text"]).get("x-polylogue-frequency", 1) == 1
    extra = _object(properties["extra"])
    assert extra["x-polylogue-frequency"] == pytest.approx(0.5)
    assert _object(extra["x-polylogue-observed-distribution"])["parent_object_observations"] == 4
    # Root-document presence would be 0.2 for BOTH fields. The real original
    # occurrence owner distinguishes mandatory text from the half-present extra.
    assert _object(extra["x-polylogue-observed-distribution"])["encountered_documents"] == 2

    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral")
    try:
        total = extras = 0
        for seed in range(200):
            record = corpus._generate_from_schema(schema, random.Random(seed))
            assert isinstance(record, Mapping)
            if "items" not in record:
                continue
            values = record["items"]
            assert isinstance(values, list | GeneratedArray)
            for item in values:
                assert isinstance(item, Mapping)
                text = item["text"]
                assert isinstance(text, str | SampledText | JoinedText)
                assert len(text) == 7
                total += 1
                extras += "extra" in item
        assert total > 40
        assert 0.3 <= extras / total <= 0.7
    finally:
        corpus._relation_solver.close()


def test_nullable_parent_preserves_original_type_mixture_and_active_text() -> None:
    samples: list[JSONDocument] = [{} for _ in range(30)]
    samples.extend({"holder": None} for _ in range(50))
    samples.extend({"holder": {"text": "neutral " * 64}} for _ in range(20))
    schema = generate_schema_from_samples(samples, provider="neutral", max_stats_samples=0)
    holder = _object(_object(schema["properties"])["holder"])
    branches = holder["anyOf"]
    assert isinstance(branches, list)
    assert {_object(branch)["type"] for branch in branches} == {"null", "object"}
    assert all("x-polylogue-frequency" not in _object(branch) for branch in branches)
    observed = _object(holder["x-polylogue-observed-distribution"])
    assert observed["type_counts"] == {"null": 50, "object": 20}
    object_branch = next(_object(branch) for branch in branches if _object(branch)["type"] == "object")
    text = _object(_object(object_branch["properties"])["text"])
    assert _object(text["x-polylogue-observed-distribution"])["parent_object_observations"] == 20
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral")
    try:
        counts = {"absent": 0, "null": 0, "object": 0}
        for seed in range(500):
            record = corpus._generate_from_schema(schema, random.Random(seed))
            assert isinstance(record, Mapping)
            if "holder" not in record:
                counts["absent"] += 1
            elif record["holder"] is None:
                counts["null"] += 1
            else:
                counts["object"] += 1
                assert isinstance(record["holder"], Mapping)
                value = record["holder"]["text"]
                assert isinstance(value, str | SampledText | JoinedText)
                assert len(value) == 512
        assert 100 <= counts["absent"] <= 200
        assert 190 <= counts["null"] <= 310
        assert 60 <= counts["object"] <= 140
    finally:
        corpus._relation_solver.close()


def test_retained_original_counts_reannotation_matches_fresh_producer_and_preserves_evidence() -> None:
    samples: list[JSONDocument] = [{} for _ in range(8)]
    samples.extend({"holder": None} for _ in range(5))
    samples.extend({"holder": {"text": "neutral " * 32}} for _ in range(2))
    expected = generate_schema_from_samples(samples, provider="neutral", max_stats_samples=0)
    retained = deepcopy(expected)
    holder = _object(_object(retained["properties"])["holder"])
    branches = holder["anyOf"]
    assert isinstance(branches, list)
    for branch in branches:
        node = _object(branch)
        node["x-polylogue-frequency"] = 7 / 15
        if node.get("type") == "object":
            text = _object(_object(node["properties"])["text"])
            observed = _object(text["x-polylogue-observed-distribution"])
            observed.pop("parent_object_observations")
            text["x-polylogue-frequency"] = 2 / 15
    before = deepcopy(retained)
    corrected = reannotate_property_presence(retained)
    assert corrected == expected
    assert corrected != before
    assert reannotate_property_presence(deepcopy(corrected)) == expected
    # Original private package bytes are not inputs to the mutable publication
    # pass; its caller owns this detached schema and pins a new package version.
    assert (
        _object(_object(before["properties"])["holder"])["x-polylogue-observed-distribution"]
        == holder["x-polylogue-observed-distribution"]
    )


def test_retained_reannotation_refuses_missing_original_containing_object_counts() -> None:
    schema = generate_schema_from_samples([{"holder": {"text": "neutral"}}], provider="neutral", max_stats_samples=0)
    holder = _object(_object(schema["properties"])["holder"])
    _object(holder["x-polylogue-observed-distribution"]).pop("type_counts")
    with pytest.raises(ValueError):
        reannotate_property_presence(schema)
