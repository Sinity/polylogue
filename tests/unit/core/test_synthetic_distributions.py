"""Schema-driven synthesis samples every per-field distribution inference records.

Each test pins one observed-distribution kind on a small schema and checks the
generated values follow it. Removing the runtime's use of that kind turns the
test red: without it the generator falls back to shape-only values.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, categorical_values
from polylogue.schemas.synthetic.classification import classify_schema_constructs
from polylogue.schemas.synthetic.wire_formats import TreeConfig, WireFormat
from tests.infra.schema_workload_profiles import inferred_codex_session_profile

_PROVIDERS = Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"


def _point(value: int) -> JSONDocument:
    """A one-bucket histogram pinned to exactly ``value`` by its min and max."""
    return {"count": 1, "histogram": [[40, 1]], "log_base": 1.1, "min": value, "max": value}


def _corpus(schema: JSONDocument) -> SyntheticCorpus:
    return SyntheticCorpus(schema, WireFormat(encoding="json"), "test")


def _generate(schema: JSONDocument, seed: int) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], _corpus(schema)._generate_from_schema(schema, random.Random(seed)))


def _categorical_schema(length: int, *, count: int = 256, buckets: list[list[int]] | None = None) -> JSONDocument:
    return {
        "type": "string",
        "x-polylogue-observed-distribution": {
            "string_length": _point(length),
            "categorical": {
                "bucket_count": count,
                "bucket_histogram": cast(list[JSONValue], buckets or [[3, 90], [200, 10]]),
            },
        },
    }


def test_one_character_categories_keep_distinct_values_and_observed_weights() -> None:
    node = _categorical_schema(1)
    schema: JSONDocument = {"type": "object", "properties": {"long_property_name_that_used_to_erase_codes": node}}
    corpus = _corpus(schema)
    rng = random.Random(63)
    values = [
        cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng))[
            "long_property_name_that_used_to_erase_codes"
        ]
        for _ in range(2_000)
    ]
    counts = Counter(values)
    assert len(counts) == 2
    assert all(isinstance(value, str) and len(value) == 1 for value in counts)
    assert 0.87 < max(counts.values()) / len(values) < 0.93


def test_categorical_zero_length_singleton_and_impossible_multiple_values() -> None:
    singleton = _categorical_schema(0, buckets=[[3, 100]])
    assert _corpus(singleton)._generate_from_schema(singleton, random.Random(1)) == ""
    multiple = _categorical_schema(0)
    with pytest.raises(ValueError):
        _corpus(multiple)._generate_from_schema(multiple, random.Random(1))
    assert any(item.state == "unsupported" for item in classify_schema_constructs(multiple))


def test_manual_categorical_vocabulary_has_no_chosen_alphabet_cap() -> None:
    schema = _categorical_schema(1, count=1024, buckets=[[index, 2] for index in range(300)])
    corpus = _corpus(schema)
    corpus._generate_from_schema(schema, random.Random(1))
    pool = corpus._categorical_pools["$"][1]
    assert len({value for value, _ in pool}) == 300
    assert all(len(value) == 1 for value, _ in pool)


def test_mixed_zero_and_positive_lengths_allocate_without_random_refusals() -> None:
    from polylogue.schemas.synthetic.categorical_values import categorical_codebook

    for seed in range(25):
        pool = categorical_codebook([(3, 90), (200, 10)], {0: 999, 1: 1}, seed=str(seed))
        assert len({value for value, _ in pool}) == 2
        assert [count for _, count in pool] == [90, 10]
        assert all(len(value) in {0, 1} for value, _ in pool)


def test_categorical_newline_shapes_keep_distinct_values_and_support() -> None:
    pool = categorical_values.categorical_codebook([(3, 90), (200, 10)], {3: 1}, seed="multiline", newlines={2: 1})
    assert len({value for value, _ in pool}) == 2
    assert all(isinstance(value, str) and len(value) == 3 and value.count("\n") == 2 for value, _ in pool)
    with pytest.raises(ValueError):
        categorical_values.categorical_codebook([(3, 90), (200, 10)], {2: 1}, seed="all-newlines", newlines={2: 1})


def test_actual_batch_cancellation_does_not_retain_partial_category_or_record_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = categorical_values._neutral_characters

    def cancelled_characters(character_class: str | None = None) -> Generator[str, None, None]:
        for index, character in enumerate(original(character_class)):
            if index == 10:
                raise KeyboardInterrupt
            yield character

    schema: JSONDocument = {"type": "object", "properties": {"value": _categorical_schema(1)}}
    corpus = _corpus(schema)
    with monkeypatch.context() as patch:
        patch.setattr(categorical_values, "_neutral_characters", cancelled_characters)
        with pytest.raises(KeyboardInterrupt):
            corpus.generate_batch(count=1, seed=3)
    assert not corpus._categorical_pools
    assert corpus.generate_batch(count=1, seed=3).report.generated_count == 1


@pytest.mark.parametrize(
    "fmt,length",
    [
        ("uuid", 36),
        ("hex-id", 24),
        ("hex-id", 200),
        ("iso8601", 120),
        ("unix-epoch-str", 40),
        ("base64", 40),
        ("base64", 44),
        ("base64", 80),
    ],
)
def test_actual_batch_preserves_supported_format_spelling_and_observed_length(fmt: str, length: int) -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "x-polylogue-format": fmt,
                "x-polylogue-observed-distribution": {"string_length": _point(length)},
            }
        },
    }
    value = json.loads(_corpus(schema).generate_batch(count=1, seed=2).raw_items[0])["value"]
    assert len(value) == length
    if fmt == "iso8601":
        from datetime import datetime

        datetime.fromisoformat(value)
    elif fmt == "base64":
        import base64

        assert isinstance(base64.b64decode(value, validate=True), bytes)


@pytest.mark.parametrize("fmt,length", [("uuid", 1), ("iso8601", 1), ("base64", 39)])
def test_actual_batch_refuses_impossible_format_lengths_visibly(fmt: str, length: int) -> None:
    node: JSONDocument = {
        "type": "string",
        "x-polylogue-format": fmt,
        "x-polylogue-observed-distribution": {"string_length": _point(length)},
    }
    with pytest.raises(ValueError):
        _corpus(node).generate_batch(count=1, seed=2)
    assert any(item.state == "unsupported" for item in classify_schema_constructs(node))


def test_actual_batch_semantic_fields_obey_lengths_values_and_nullable_types() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "x-polylogue-semantic-role": "message_body",
                "x-polylogue-observed-distribution": {"string_length": _point(8192), "newline_count": _point(7)},
            },
            "title": {
                "type": "string",
                "x-polylogue-semantic-role": "session_title",
                "x-polylogue-values": ["declared"],
            },
            "nullable": {
                "type": ["string", "null"],
                "x-polylogue-semantic-role": "message_body",
                "x-polylogue-observed-distribution": {"type_counts": {"null": 100, "string": 0}},
            },
            "timestamp": {
                "type": "integer",
                "x-polylogue-semantic-role": "message_timestamp",
                "x-polylogue-observed-distribution": {"numeric": _point(42)},
            },
        },
    }
    payload = json.loads(_corpus(schema).generate_batch(count=1, seed=1, style="demo").raw_items[0])
    assert len(payload["body"]) == 8192 and payload["body"].count("\n") == 7
    assert payload["title"] == "declared"
    assert payload["nullable"] is None
    assert payload["timestamp"] == 42 and isinstance(payload["timestamp"], int)


@pytest.mark.parametrize(
    "provider,encoding,messages_path", [("claude-ai", "json", "chat_messages"), ("claude-code", "jsonl", None)]
)
def test_actual_linear_and_jsonl_batches_preserve_constrained_wire_fields(
    provider: str, encoding: Literal["json", "jsonl"], messages_path: str | None
) -> None:
    message: JSONDocument = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "x-polylogue-semantic-role": "message_body",
                "x-polylogue-observed-distribution": {"string_length": _point(8192)},
            },
            "observed_optional": {"type": "string", "x-polylogue-values": ["retained"]},
        },
    }
    if provider == "claude-code":
        body_schema = cast(JSONDocument, cast(JSONDocument, message["properties"])["text"])
        message = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "x-polylogue-values": ["user"]},
                "message": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "x-polylogue-values": ["user"]},
                        "content": body_schema,
                    },
                },
                "observed_optional": {"type": "string", "x-polylogue-values": ["retained"]},
            },
        }
    schema: JSONDocument = (
        message
        if messages_path is None
        else {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-polylogue-semantic-role": "session_title",
                    "x-polylogue-observed-distribution": {"string_length": _point(31)},
                },
                messages_path: {"type": "array", "items": message},
            },
        }
    )
    corpus = SyntheticCorpus(schema, WireFormat(encoding=encoding, messages_path=messages_path), provider)
    raw = corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=4, style="demo").raw_items[0]
    if messages_path is not None:
        payload = json.loads(raw)
        assert len(payload["name"]) == 31
        record = payload[messages_path][0]
    else:
        record = json.loads(raw.splitlines()[0])
        payload = [record]
    body = record["message"]["content"] if provider == "claude-code" else record["text"]
    assert len(body) == 8192
    assert record["observed_optional"] == "retained"
    from polylogue.sources.dispatch import parse_payload

    parsed = parse_payload(provider, payload, "synthetic-wire")
    assert parsed and any(message.text == body for session in parsed for message in session.messages)


def test_actual_tree_batch_preserves_constrained_title_and_body() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "x-polylogue-semantic-role": "session_title",
                "x-polylogue-observed-distribution": {"string_length": _point(43)},
            },
            "mapping": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "object",
                                    "properties": {
                                        "parts": {
                                            "type": "array",
                                            "x-polylogue-observed-distribution": {"array_length": _point(1)},
                                            "items": {
                                                "type": "string",
                                                "x-polylogue-semantic-role": "message_body",
                                                "x-polylogue-observed-distribution": {"string_length": _point(8192)},
                                            },
                                        }
                                    },
                                }
                            },
                        }
                    },
                },
            },
        },
    }
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="json", tree=TreeConfig(container_path="mapping", children_field="children")),
        "chatgpt",
    )
    payload = json.loads(corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=5).raw_items[0])
    assert len(payload["title"]) == 43
    assert all(len(node["message"]["content"]["parts"][0]) == 8192 for node in payload["mapping"].values())
    from polylogue.sources.dispatch import parse_payload

    parsed = parse_payload("chatgpt", payload, "synthetic-tree")
    assert parsed and all(len(message.text or "") == 8192 for session in parsed for message in session.messages)


def test_actual_codex_profile_preserves_generated_content_and_parser_evidence() -> None:
    from polylogue.sources.dispatch import parse_payload

    body: JSONDocument = {
        "type": "string",
        "x-polylogue-semantic-role": "message_body",
        "x-polylogue-observed-distribution": {"string_length": _point(8192)},
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "message": body,
                    "content": {
                        "type": "array",
                        "x-polylogue-observed-distribution": {"array_length": _point(1)},
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["input_text"]},
                                "text": body,
                            },
                        },
                    },
                    "observed_optional": {"type": "string", "x-polylogue-values": ["retained"]},
                },
            },
        },
    }
    samples: list[JSONDocument] = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "message": "n" * 8192,
                "content": [{"type": "input_text", "text": "n" * 8192}],
                "observed_optional": "retained",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "n" * 8192,
                "content": [{"type": "input_text", "text": "n" * 8192}],
                "observed_optional": "retained",
            },
        },
    ]
    schema, profile = inferred_codex_session_profile(schema, samples, native_id="synthetic-profile")
    corpus = SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )
    raw = corpus.generate_batch(
        count=1, messages_per_session=range(1, 2), seed=1, session_native_ids=("synthetic-profile",)
    ).raw_items[0]
    records = [json.loads(line) for line in raw.splitlines()]
    assert records[0]["type"] == "session_meta"
    records = [record for record in records if record["type"] != "session_meta"]
    assert {record["type"] for record in records} == {"response_item", "event_msg"}
    assert all(record["payload"]["observed_optional"] == "retained" for record in records)
    assert all(len(record["payload"]["message"]) == 8192 for record in records)
    assert all(len(record["payload"]["content"][0]["text"]) == 8192 for record in records)
    sessions = parse_payload("codex", records, "synthetic-profile")
    assert sessions and all(len(message.text or "") == 8192 for session in sessions for message in session.messages)


def test_declared_labels_follow_actual_producer_bucket_counts_without_invention() -> None:
    from polylogue.schemas.field_stats.distributions import CategoricalSketch

    evidence = CategoricalSketch()
    evidence.observe("Read", count=90)
    evidence.observe("Write", count=10)
    schema: JSONDocument = {
        "type": "string",
        "x-polylogue-values": ["Read", "Write"],
        "x-polylogue-observed-distribution": {"categorical": evidence.to_payload()},
    }
    corpus = _corpus(schema)
    rng = random.Random(11)
    values = [corpus._generate_from_schema(schema, rng) for _ in range(2000)]
    assert set(values) == {"Read", "Write"}
    assert 0.87 < values.count("Read") / len(values) < 0.93


def test_actual_batch_preserves_deep_optional_fields_without_a_depth_cutoff() -> None:
    from polylogue.schemas.synthetic.conservation import check_conservation, collect_planted_values

    node: JSONDocument = {
        "type": "string",
        "x-polylogue-semantic-role": "message_body",
        "x-polylogue-observed-distribution": {"string_length": _point(300)},
    }
    for _ in range(40):
        node = {"type": "object", "properties": {"optional": node}}
    payload = json.loads(_corpus(node).generate_batch(count=1, seed=2).raw_items[0])
    assert len(collect_planted_values(node, payload)) == 1
    assert not check_conservation(node, [payload], []).conserved
    for _ in range(40):
        assert "optional" in payload
        payload = payload["optional"]
    assert len(payload) == 300


def test_finite_deep_witnesses_preserve_tail_and_cyclic_schema_is_refused() -> None:
    from polylogue.schemas.synthetic.wire_formats import generate_coverage_witnesses

    node: JSONDocument = {"type": ["string", "null"], "x-polylogue-values": ["tail"]}
    for _ in range(40):
        node = {"type": "object", "properties": {"optional": node}}
    payloads = [json.loads(raw) for raw in generate_coverage_witnesses(_corpus(node), seed=4)]
    tails = []
    for payload in payloads:
        for _ in range(40):
            payload = payload["optional"]
        tails.append(payload)
    assert set(tails) == {None, "tail"}

    cyclic: JSONDocument = {"type": "object"}
    cyclic["properties"] = {"self": cyclic}
    corpus = _corpus(cyclic)
    with pytest.raises(ValueError):
        corpus.generate_batch(count=1, seed=1)
    with pytest.raises(ValueError):
        list(generate_coverage_witnesses(corpus, seed=1))


def test_string_length_and_newline_count_follow_the_observed_histograms() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "x-polylogue-observed-distribution": {"string_length": _point(300), "newline_count": _point(4)},
            }
        },
    }

    body = cast(str, _generate(schema, 1)["body"])

    assert len(body) == 300
    assert body.count("\n") == 4


def test_nullable_field_follows_the_observed_null_share() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "note": {
                "type": ["string", "null"],
                "x-polylogue-observed-distribution": {"type_counts": {"string": 5}, "null_observations": 95},
            }
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(3)

    values = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng)).get("note") for _ in range(2_000)]
    null_share = sum(value is None for value in values) / len(values)

    assert 0.9 < null_share < 0.99


def test_booleans_follow_the_observed_counts() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "flag": {
                "type": "boolean",
                "x-polylogue-observed-distribution": {"boolean_counts": {"true": 1, "false": 19}},
            }
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(5)

    flags = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng))["flag"] for _ in range(2_000)]

    assert 0.02 < sum(flag is True for flag in flags) / len(flags) < 0.09


def test_repeating_values_come_from_one_stable_pool_per_observed_bucket() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "x-polylogue-observed-distribution": {
                    "categorical": {"bucket_count": 256, "bucket_histogram": [[3, 90], [200, 10]]},
                    "string_length": _point(12),
                },
            }
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(7)

    models = Counter(
        cast(str, cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng))["model"]) for _ in range(1_000)
    )

    assert len(models) == 2
    assert {len(value) for value in models} == {12}
    assert 0.8 < max(models.values()) / 1_000 < 0.97


def test_dynamic_key_maps_follow_the_observed_fanout() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "usage_by_model": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "x-polylogue-observed-distribution": {"object_fanout": _point(5)},
            }
        },
    }

    usage = cast(dict[str, JSONValue], _generate(schema, 11)["usage_by_model"])

    assert len(usage) == 5
    assert all(isinstance(value, int) for value in usage.values())


def test_field_presence_follows_observed_co_occurrence() -> None:
    """``b`` appears whenever ``a`` does, although each alone has a 0.5 marginal."""
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "x-polylogue-frequency": 0.5,
                "x-polylogue-observed-distribution": {"encountered_documents": 50, "co_occurring_fields": {"b": 50}},
            },
            "b": {"type": "integer", "x-polylogue-frequency": 0.5},
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(13)

    records = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng)) for _ in range(500)]
    with_a = [record for record in records if "a" in record]

    assert with_a
    assert all("b" in record for record in with_a)


def test_numeric_arrays_keep_their_observed_ordering() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "offsets": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "x-polylogue-observed-distribution": {
                        "ordered_numeric_pairs": {"count": 10, "nondecreasing_count": 10, "nondecreasing_rate": 1.0}
                    },
                },
                "x-polylogue-observed-distribution": {"array_length": _point(20)},
            }
        },
    }

    offsets = cast(list[int], _generate(schema, 17)["offsets"])

    assert len(offsets) == 20
    assert offsets == sorted(offsets)


def _string_length_fields(node: object, path: str = "$") -> list[tuple[str, dict[str, object]]]:
    found: list[tuple[str, dict[str, object]]] = []
    if not isinstance(node, dict):
        return found
    observed = node.get("x-polylogue-observed-distribution")
    if isinstance(observed, dict) and isinstance(observed.get("string_length"), dict):
        found.append((path, observed["string_length"]))
    for name, child in (node.get("properties") or {}).items():
        found.extend(_string_length_fields(child, f"{path}.{name}"))
    return found


def test_fresh_packages_generate_string_lengths_inside_their_observed_range(
    qualified_schema_registry: SchemaRegistry,
) -> None:
    """Every top-level string field with an observed length histogram stays inside [min, max]."""
    checked = 0
    for provider in ("chatgpt", "claude-code", "gemini"):
        schema = qualified_schema_registry.get_element_schema(provider)
        assert isinstance(schema, dict)
        bounds = {
            path: (distribution.get("min"), distribution.get("max"))
            for path, distribution in _string_length_fields(schema)
            if path.count(".") == 1
        }
        corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), provider)
        rng = random.Random(23)
        for _ in range(30):
            record = corpus._generate_from_schema(schema, rng)
            if not isinstance(record, Mapping):
                continue
            for path, (low, high) in bounds.items():
                value = record.get(path.removeprefix("$."))
                if isinstance(value, str) and isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    assert low <= len(value) <= high, (provider, path, len(value), low, high)
                    checked += 1
    assert checked > 0


@pytest.mark.parametrize("measurement", [None, {"other": 50}, {"b": 0}, {"b": 50}])
def test_optional_presence_requires_an_explicit_cooccurrence_count(measurement: dict[str, int] | None) -> None:
    observed: JSONDocument = {"encountered_documents": 50, "numeric": _point(1)}
    if measurement is not None:
        observed["co_occurring_fields"] = cast(JSONDocument, measurement)
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "x-polylogue-observed-distribution": observed},
            "b": {"type": "integer", "x-polylogue-frequency": 0.5},
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(19)
    records = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng)) for _ in range(500)]
    share = sum("b" in record for record in records) / len(records)
    if measurement is not None and "b" in measurement:
        assert share == measurement["b"] / 50
    else:
        assert 0.35 < share < 0.65


def test_categorical_pools_follow_each_union_branch_distribution_at_the_same_path() -> None:
    schema: JSONDocument = {
        "oneOf": [
            {
                "type": "string",
                "x-polylogue-observed-distribution": {
                    "categorical": {"bucket_count": 256, "bucket_histogram": [[bucket, 100]]},
                    "string_length": _point(length),
                },
            }
            for bucket, length in ((3, 12), (200, 40))
        ]
    }
    corpus = _corpus(schema)
    rng = random.Random(23)
    values = [cast(str, corpus._generate_from_schema(schema, rng)) for _ in range(100)]
    assert {len(value) for value in values} == {12, 40}
    assert len(set(values)) == 2
    assert len(corpus._categorical_pools) == 1


def test_collected_repeated_record_cooccurrence_uses_occurrences_including_nulls() -> None:
    from polylogue.schemas.field_stats.collection import _collect_field_stats
    from polylogue.schemas.generation.field_annotations import _observed_distribution_payload

    stats = _collect_field_stats([{"records": [{"a": None, "b": 1}, {"a": None, "b": 1}, {"a": 1}, {"a": 1}]}])
    observed = _observed_distribution_payload(stats["$.records[*].a"])
    assert observed["encountered_documents"] == 1
    assert observed["type_counts"] == {"integer": 2, "null": 2}
    assert observed["null_observations"] == 2
    assert observed["co_occurring_fields"] == {"b": 2}
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": ["integer", "null"], "x-polylogue-observed-distribution": observed},
            "b": {"type": "integer", "x-polylogue-frequency": 0.5},
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(29)
    records = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng)) for _ in range(500)]
    share = sum("b" in record for record in records) / len(records)
    assert 0.35 < share < 0.65
    null_share = sum(record["a"] is None for record in records) / len(records)
    assert 0.4 < null_share < 0.6


def test_manual_nullable_cooccurrence_counts_include_separately_declared_nulls() -> None:
    observed: JSONDocument = {
        "type_counts": {"integer": 2},
        "null_observations": 2,
        "co_occurring_fields": {"b": 2},
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": ["integer", "null"], "x-polylogue-observed-distribution": observed},
            "b": {"type": "integer", "x-polylogue-frequency": 0.5},
        },
    }
    corpus = _corpus(schema)
    rng = random.Random(37)
    records = [cast(dict[str, JSONValue], corpus._generate_from_schema(schema, rng)) for _ in range(500)]
    assert 0.4 < sum(record["a"] is None for record in records) / len(records) < 0.6
    assert 0.4 < sum("b" in record for record in records) / len(records) < 0.6


def test_inconsistent_cooccurrence_count_is_refused() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "x-polylogue-observed-distribution": {"type_counts": {"integer": 2}, "co_occurring_fields": {"b": 3}},
            },
            "b": {"type": "integer", "x-polylogue-frequency": 0.5},
        },
    }
    from polylogue.schemas.synthetic.classification import classify_schema_constructs

    assert any(item.state == "unsupported" for item in classify_schema_constructs(schema))
    with pytest.raises(ValueError):
        _generate(schema, 31)


def test_collected_deep_observations_reach_annotated_batch_generation() -> None:
    from polylogue.schemas.field_stats.collection import _collect_field_stats
    from polylogue.schemas.generation.field_annotations import annotate_schema

    sample: JSONValue = "neutral tail " * 30
    schema: JSONDocument = {"type": "string", "x-polylogue-semantic-role": "message_body"}
    for _ in range(40):
        sample = {"optional": sample}
        schema = {"type": "object", "properties": {"optional": schema}}
    stats = _collect_field_stats([cast(JSONDocument, sample)])
    tail_path = "$" + ".optional" * 40
    assert stats[tail_path].max_depth_seen == 40
    assert stats[tail_path].string_length_distribution.count == 1
    annotate_schema(schema, stats)
    payload = json.loads(_corpus(schema).generate_batch(count=1, seed=2).raw_items[0])
    for _ in range(40):
        payload = payload["optional"]
    assert len(payload) == len("neutral tail " * 30)


def test_actual_codex_envelope_batch_retains_schema_payload_and_parser_content() -> None:
    from polylogue.sources.dispatch import parse_payload

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "observed_optional": {"type": "string", "x-polylogue-values": ["retained"]},
                },
            },
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex")
    raw = corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=1).raw_items[0]
    records = [json.loads(line) for line in raw.splitlines()]
    assert all(record["payload"]["observed_optional"] == "retained" for record in records)
    assert all(record["type"] == "response_item" for record in records)
    assert all(record["payload"]["content"] for record in records)
    assert parse_payload("codex", records, "synthetic-flat-payload")


def test_actual_codex_flat_batch_refuses_competing_generated_content_owners() -> None:
    def content(label: str) -> JSONDocument:
        return {
            "type": "array",
            "x-polylogue-observed-distribution": {"array_length": _point(1)},
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "x-polylogue-values": ["input_text"]},
                    "text": {"type": "string", "x-polylogue-values": [label]},
                },
            },
        }

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "content": content("first"),
            "payload": {"type": "object", "properties": {"content": content("second")}},
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex")
    with pytest.raises(ValueError):
        corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=1)


@pytest.mark.parametrize("depth", [0, 1, 40])
def test_conservation_primes_every_first_descent(depth: int) -> None:
    from polylogue.schemas.synthetic.conservation import collect_planted_values

    schema: JSONDocument = {"type": "string", "x-polylogue-semantic-role": "message_body"}
    payload: JSONValue = "neutral body"
    for _ in range(depth):
        schema = {"type": "object", "properties": {"child": schema}}
        payload = {"child": payload}
    planted = collect_planted_values(schema, payload)
    assert len(planted) == 1 and planted[0].value == "neutral body"


@pytest.mark.parametrize("outer,inner", [("anyOf", "oneOf"), ("oneOf", "anyOf")])
def test_actual_batch_retains_nested_union_null_array_items(outer: str, inner: str) -> None:
    schema: JSONDocument = {
        "type": "array",
        "x-polylogue-observed-distribution": {"array_length": _point(7)},
        "items": {outer: [{inner: [{"type": "null"}]}]},
    }
    payload = json.loads(_corpus(schema).generate_batch(count=1, seed=3).raw_items[0])
    assert payload == [None] * 7


@pytest.mark.parametrize("length", [5, 17])
def test_actual_codex_missing_optional_content_alias_is_conditioned_or_refused(length: int) -> None:
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples

    def content(size: int) -> JSONDocument:
        measured = generate_schema_from_samples([{"text": "a " * (size // 2) + "a" * (size % 2)}], max_stats_samples=0)
        text = cast(JSONDocument, cast(JSONDocument, measured["properties"])["text"])
        return {
            "type": "array",
            "x-polylogue-observed-distribution": {"array_length": _point(1)},
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "x-polylogue-values": ["input_text"]},
                    "text": {**text, "x-polylogue-semantic-role": "message_body"},
                },
            },
        }

    optional = content(length)
    optional["x-polylogue-frequency"] = 0.0
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "content": content(5),
            "payload": {"type": "object", "properties": {"content": optional}},
        },
    }
    for seed in range(12):
        corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex")
        if length != 5:
            with pytest.raises(ValueError):
                corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=seed)
        else:
            records = [
                json.loads(line)
                for line in corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=seed)
                .raw_items[0]
                .splitlines()
            ]
            assert all(record["payload"]["content"] == record["content"] for record in records)


def _skipped_label(value: str) -> JSONDocument:
    return {"type": "string", "x-polylogue-frequency": 0.0, "x-polylogue-values": [value]}


@pytest.mark.parametrize("provider", ["claude-ai", "gemini", "claude-code", "codex"])
def test_actual_batch_missing_wire_defaults_preserve_declared_labels(provider: str) -> None:
    from polylogue.sources.dispatch import parse_payload

    body: JSONDocument = {"type": "string", "x-polylogue-values": ["neutral body"]}
    if provider == "claude-ai":
        schema: JSONDocument = {
            "type": "object",
            "properties": {
                "uuid": _skipped_label("declared-message"),
                "sender": _skipped_label("human"),
                "text": body,
            },
        }
        wire = WireFormat(encoding="json", messages_path="chat_messages")
    elif provider == "gemini":
        schema = {"type": "object", "properties": {"role": _skipped_label("user"), "text": body}}
        wire = WireFormat(encoding="json", messages_path="chunkedPrompt.chunks")
    elif provider == "claude-code":
        schema = {
            "type": "object",
            "properties": {
                "type": _skipped_label("user"),
                "message": {
                    "type": "object",
                    "properties": {
                        "role": _skipped_label("user"),
                        "content": body,
                    },
                },
            },
        }
        wire = WireFormat(encoding="jsonl")
    else:
        schema = {
            "type": "object",
            "properties": {
                "id": _skipped_label("declared-message"),
                "type": _skipped_label("message"),
                "role": _skipped_label("user"),
            },
        }
        wire = WireFormat(encoding="jsonl")
    corpus = SyntheticCorpus(schema, wire, provider)
    raw = corpus.generate_batch(count=1, messages_per_session=range(2, 3), seed=2).raw_items[0]
    payload = [json.loads(line) for line in raw.splitlines()] if wire.encoding == "jsonl" else json.loads(raw)
    records = payload
    if not isinstance(records, list):
        for part in (wire.messages_path or "").split("."):
            records = records[part]
    if provider == "claude-ai":
        assert all(record["uuid"] == "declared-message" and record["sender"] == "human" for record in records)
    elif provider == "gemini":
        assert all(record["role"] == "user" for record in records)
    elif provider == "claude-code":
        assert all(record["type"] == "user" and record["message"]["role"] == "user" for record in records)
    else:
        assert all(record["id"] == "declared-message" and record["role"] == "user" for record in records)
    assert parse_payload(provider, payload, "synthetic-declared-defaults")


@pytest.mark.parametrize("provider", ["chatgpt", "claude-code"])
def test_actual_batch_refuses_selected_null_wire_carrier(provider: str) -> None:
    node: JSONDocument = {"type": "object", "properties": {"message": {"type": "null"}}}
    if provider == "chatgpt":
        schema: JSONDocument = {
            "type": "object",
            "properties": {"mapping": {"type": "object", "additionalProperties": node}},
        }
        wire = WireFormat(encoding="json", tree=TreeConfig(container_path="mapping", children_field="children"))
    else:
        schema = node
        wire = WireFormat(encoding="jsonl")
    with pytest.raises(ValueError):
        SyntheticCorpus(schema, wire, provider).generate_batch(count=1, messages_per_session=range(1, 2), seed=1)


def test_actual_chatgpt_missing_ids_roles_content_type_and_times_use_selected_schema() -> None:
    from polylogue.sources.dispatch import parse_payload

    message: JSONDocument = {
        "type": "object",
        "properties": {
            "id": _skipped_label("declared-message"),
            "author": {"type": "object", "properties": {"role": _skipped_label("user")}},
            "content": {
                "type": "object",
                "properties": {
                    "content_type": _skipped_label("text"),
                    "parts": {
                        "type": "array",
                        "x-polylogue-frequency": 0.0,
                        "x-polylogue-observed-distribution": {"array_length": _point(1)},
                        "items": {"type": "string", "x-polylogue-values": ["neutral body"]},
                    },
                },
            },
        },
    }
    timestamp: JSONDocument = {"type": "integer", "x-polylogue-frequency": 0.0, "x-polylogue-range": [42, 42]}
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "id": _skipped_label("declared-session"),
            "create_time": timestamp,
            "update_time": timestamp,
            "mapping": {
                "type": "object",
                "additionalProperties": {"type": "object", "properties": {"message": message}},
            },
        },
    }
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="json", tree=TreeConfig(container_path="mapping", children_field="children")),
        "chatgpt",
    )
    payload = json.loads(corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=1).raw_items[0])
    assert payload["id"] == "declared-session" and payload["create_time"] == payload["update_time"] == 42
    msg = next(iter(payload["mapping"].values()))["message"]
    assert msg["id"] == "declared-message" and msg["author"]["role"] == "user"
    assert msg["content"] == {"content_type": "text", "parts": ["neutral body"]}
    assert parse_payload("chatgpt", payload, "synthetic-skipped-chatgpt-fields")


def test_actual_codex_profile_missing_metadata_defaults_use_declared_labels() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "id": _skipped_label("declared-session"),
                    "cwd": _skipped_label("/workspace/declared"),
                    "model": _skipped_label("declared-model"),
                },
            }
        },
    }
    samples: list[JSONDocument] = [
        {"type": "session_meta", "payload": {"id": "declared-session"}},
        {"type": "turn_context", "payload": {"cwd": "/workspace/declared", "model": "declared-model"}},
    ]
    schema, profile = inferred_codex_session_profile(schema, samples, native_id="declared-session")
    corpus = SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )
    records = [
        json.loads(line)
        for line in corpus.generate_batch(
            count=1, messages_per_session=range(1, 2), seed=2, session_native_ids=("declared-session",)
        )
        .raw_items[0]
        .splitlines()
    ]
    assert {record["type"] for record in records} == {"session_meta", "turn_context"}
    for record in records:
        if record["type"] == "session_meta":
            assert record["payload"]["id"] == "declared-session"
        else:
            assert record["payload"]["cwd"] == "/workspace/declared"
            assert record["payload"]["model"] == "declared-model"


@pytest.mark.parametrize("fmt,length", [("uuid4", 36), ("iso8601", 20), ("base64", 44)])
def test_actual_codex_missing_alias_preserves_recognized_formats_and_nested_union(fmt: str, length: int) -> None:
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples

    values = {
        "uuid4": "11111111-1111-4111-8111-111111111111",
        "iso8601": "2025-01-01T00:00:00Z",
        "base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    }
    measured = generate_schema_from_samples([{"text": values[fmt]}], max_stats_samples=0)
    text = cast(JSONDocument, cast(JSONDocument, measured["properties"])["text"])
    text["x-polylogue-format"] = fmt
    fields: JSONDocument = {
        "type": {"type": "string", "x-polylogue-values": ["input_text"]},
        "text": text,
        "flag": {"type": "boolean", "x-polylogue-observed-distribution": {"boolean_counts": {"true": 10, "false": 0}}},
        "number": {"type": "integer", "x-polylogue-range": [42, 42]},
    }
    content: JSONDocument = {
        "type": "array",
        "x-polylogue-observed-distribution": {"array_length": _point(1)},
        "items": {"type": "object", "properties": fields},
    }
    alias_fields: JSONDocument = {**fields, "text": {"anyOf": [{"oneOf": [text, {"type": "null"}]}]}}
    alias: JSONDocument = {
        "type": "array",
        "x-polylogue-frequency": 0.0,
        "x-polylogue-observed-distribution": {"array_length": _point(1)},
        "items": {"type": "object", "properties": alias_fields},
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "content": content,
            "payload": {"type": "object", "properties": {"content": alias}},
        },
    }
    for seed in range(8):
        corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex")
        records = [
            json.loads(line)
            for line in corpus.generate_batch(count=1, messages_per_session=range(1, 2), seed=seed)
            .raw_items[0]
            .splitlines()
        ]
        assert all(record["payload"]["content"] == record["content"] for record in records)


def test_actual_batch_unconstrained_array_items_preserve_observed_length() -> None:
    schema: JSONDocument = {"type": "array", "x-polylogue-observed-distribution": {"array_length": _point(7)}}
    assert json.loads(_corpus(schema).generate_batch(count=1, seed=1).raw_items[0]) == [None] * 7


def test_actual_batch_forbidden_array_items_refuse_positive_observed_length() -> None:
    schema: JSONDocument = {
        "type": "array",
        "items": False,
        "x-polylogue-observed-distribution": {"array_length": _point(1)},
    }
    with pytest.raises(ValueError):
        _corpus(schema).generate_batch(count=1, seed=1)


def test_actual_batch_forbidden_missing_wire_default_is_visible() -> None:
    schema: JSONDocument = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["message"]},
        },
    }
    with pytest.raises(ValueError):
        SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex").generate_batch(
            count=1, messages_per_session=range(1, 2), seed=1
        )


@pytest.mark.parametrize("additional", [{}, True])
def test_actual_batch_unconstrained_dynamic_values_retain_fanout(additional: JSONValue) -> None:
    schema: JSONDocument = {
        "type": "object",
        "additionalProperties": additional,
        "x-polylogue-observed-distribution": {"object_fanout": _point(3)},
    }
    payload = json.loads(_corpus(schema).generate_batch(count=1, seed=1).raw_items[0])
    assert payload == {"key-0000": None, "key-0001": None, "key-0002": None}


def test_sampled_large_text_streams_its_exact_shape_without_output_sized_state() -> None:
    import tracemalloc
    from contextlib import closing
    from itertools import islice

    from polylogue.schemas.synthetic.runtime import _json_chunks
    from polylogue.schemas.synthetic.text_value import SampledText

    length = 2_000_000_000
    node: JSONDocument = {
        "type": "string",
        "x-polylogue-observed-distribution": {
            "string_length": _point(length),
            "newline_count": _point(length // 2),
        },
    }
    tracemalloc.start()
    try:
        value = _corpus(node)._generate_from_schema(node, random.Random(47))
        assert isinstance(value, SampledText) and len(value) == length
        with closing(_json_chunks(value, pretty=False)) as chunks:
            prefix = list(islice(chunks, 3))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert prefix[0] == '"' and len(prefix[1]) < 10_000
    assert peak < 8 * 1024 * 1024
    assert value == _corpus(node)._generate_from_schema(node, random.Random(47))

    # Explicit fixture materialization still passes through that same encoder.
    smaller = SampledText(2_000_003, value.pool, value.offset, 314_159, value.newline_phase)
    decoded = json.loads("".join(_json_chunks(smaller, pretty=False)))
    assert len(decoded) == 2_000_003 and decoded.count("\n") == 314_159


def test_selected_schema_binding_dies_with_the_generated_record() -> None:
    import gc
    import weakref

    from polylogue.schemas.synthetic.models import GeneratedRecord

    schema: JSONDocument = {"type": "object", "properties": {"value": {"type": "string"}}}
    corpus = _corpus(schema)
    value = corpus._generate_from_schema(schema, random.Random(7))
    assert isinstance(value, GeneratedRecord)
    assert value.selected_schema == schema
    # Original captured coordinates survive caller mutation without aliasing.
    cast(JSONDocument, schema["properties"])["caller_added"] = {"type": "integer"}
    assert "caller_added" not in cast(Mapping[str, object], value.selected_schema["properties"])
    original = weakref.ref(value)
    assert "selected_schema" not in value
    del value
    gc.collect()
    assert original() is None


def test_old_committed_chatgpt_shape_is_not_qualified_by_format_annotation() -> None:
    element = sorted((_PROVIDERS / "chatgpt" / "versions").glob("*/elements/*.schema.json.gz"))[-1]
    with gzip.open(element) as handle:
        schema = cast(JSONDocument, json.load(handle))
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "chatgpt")
    try:
        with pytest.raises(ValueError, match="observed shape cannot represent original wire format"):
            corpus._generate_from_schema(schema, random.Random(23))
    finally:
        corpus._relation_solver.close()
