"""Declared raw Codex subset laws condition original field draws."""

from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.classification import classify_schema_constructs
from polylogue.schemas.synthetic.models import SchemaRecord
from tests.infra.schema_workload_profiles import inferred_record_profile


def _samples() -> list[SchemaRecord]:
    return [
        {"type": "session_meta", "payload": {"id": "neutral-scope", "timestamp": "2025-06-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "neutral"}]},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "cached_input_tokens": 0,
                        "input_tokens": 1,
                        "reasoning_output_tokens": 0,
                        "output_tokens": 1,
                    }
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "cached_input_tokens": 100,
                        "input_tokens": 100,
                        "reasoning_output_tokens": 100,
                        "output_tokens": 100,
                    }
                },
            },
        },
    ]


def test_raw_codex_subset_laws_reach_the_original_lazy_record_driver() -> None:
    samples = _samples()
    schema = generate_schema_from_samples(samples, provider="codex", max_stats_samples=0)
    profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )
    checked = 0
    for raw in corpus.generate(count=40, messages_per_session=range(3, 4), seed=21):
        for line in raw.splitlines():
            record = json.loads(line)
            payload = record.get("payload", {})
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            usage = info.get("last_token_usage") if isinstance(info, dict) else None
            if not isinstance(usage, dict):
                continue
            if all(
                name in usage
                for name in ("cached_input_tokens", "input_tokens", "reasoning_output_tokens", "output_tokens")
            ):
                assert usage["cached_input_tokens"] <= usage["input_tokens"]
                assert usage["reasoning_output_tokens"] <= usage["output_tokens"]
                checked += 1
    assert checked > 0
    support = {item.construct: item.state for item in classify_schema_constructs(schema)}
    assert support["x-polylogue-numeric-order"] == "supported"
    assert schema == generate_schema_from_samples(samples, provider="codex", max_stats_samples=0)


def test_subset_declarations_are_scoped_to_raw_codex_usage_not_disjoint_input_lanes() -> None:
    for provider in (None, "claude-code", "chatgpt"):
        schema = generate_schema_from_samples(_samples(), provider=provider, max_stats_samples=0)
        assert "x-polylogue-numeric-order" not in json.dumps(schema)
    schema = generate_schema_from_samples(
        [{"usage": {"cached_input_tokens": 100, "input_tokens": 1}}], provider="codex"
    )
    assert "x-polylogue-numeric-order" not in json.dumps(schema)


def test_impossible_original_numeric_support_refuses_without_changing_either_field() -> None:
    samples = _samples()
    schema = generate_schema_from_samples(samples, provider="codex", max_stats_samples=0)

    def restrict_original_support(node: object) -> None:
        if isinstance(node, dict):
            if "x-polylogue-numeric-order" in node:
                properties = node["properties"]
                properties["cached_input_tokens"]["x-polylogue-range"] = [1000, 1000]
                properties["input_tokens"]["x-polylogue-range"] = [1, 1]
                properties["cached_input_tokens"]["x-polylogue-frequency"] = 1.0
                properties["input_tokens"]["x-polylogue-frequency"] = 1.0
            for child in node.values():
                restrict_original_support(child)
        elif isinstance(node, list):
            for child in node:
                restrict_original_support(child)

    restrict_original_support(schema)
    profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )
    original = json.dumps(schema, sort_keys=True)
    with pytest.raises(ValueError, match="numeric subset has no feasible original support"):
        corpus.generate(count=40, messages_per_session=range(3, 4), seed=21)
    assert json.dumps(schema, sort_keys=True) == original


@pytest.mark.parametrize(
    ("lower_kind", "lower_range", "upper_kind", "upper_range"),
    [
        ("number", [0.5, 0.5], "integer", [0, 1]),
        ("integer", [-1, 1], "number", [-0.5, -0.5]),
        ("integer", [-0.5, 0.5], "number", [-0.25, 0.25]),
    ],
)
def test_mixed_numeric_subset_conditions_actual_integer_truncation_support(
    lower_kind: str, lower_range: list[float], upper_kind: str, upper_range: list[float]
) -> None:
    samples = _samples()
    schema = generate_schema_from_samples(samples, provider="codex", max_stats_samples=0)

    def declare_support(node: object) -> None:
        if isinstance(node, dict):
            if "x-polylogue-numeric-order" in node:
                properties = node["properties"]
                for name, kind, support in (
                    ("cached_input_tokens", lower_kind, lower_range),
                    ("input_tokens", upper_kind, upper_range),
                ):
                    properties[name]["type"] = kind
                    properties[name]["x-polylogue-range"] = support
                    properties[name]["x-polylogue-frequency"] = 1.0
            for child in node.values():
                declare_support(child)
        elif isinstance(node, list):
            for child in node:
                declare_support(child)

    declare_support(schema)
    profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )
    checked = 0
    for raw in corpus.generate(count=20, messages_per_session=range(3, 4), seed=31):
        for line in raw.splitlines():
            payload = json.loads(line).get("payload", {})
            info = payload.get("info")
            usage = info.get("last_token_usage") if isinstance(info, dict) else None
            if not isinstance(usage, dict) or not {"cached_input_tokens", "input_tokens"} <= usage.keys():
                continue
            lower, upper = usage["cached_input_tokens"], usage["input_tokens"]
            assert lower <= upper
            assert (type(lower) is int) == (lower_kind == "integer")
            assert (type(upper) is int) == (upper_kind == "integer")
            assert lower_range[0] <= lower <= lower_range[1]
            assert upper_range[0] <= upper <= upper_range[1]
            checked += 1
    assert checked > 0


def test_numeric_subset_alias_reuse_requires_original_solver_custody() -> None:
    from polylogue.schemas.synthetic.observed_values import existing_value_is_supported

    schema = generate_schema_from_samples(_samples(), provider="codex", max_stats_samples=0)
    pending: list[object] = [schema]
    usage_schema: SchemaRecord | None = None
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if "x-polylogue-numeric-order" in node:
                usage_schema = cast(SchemaRecord, node)
                break
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    else:
        pytest.fail("original inference did not declare a numeric subset")
    payload = _samples()[2]["payload"]
    assert isinstance(payload, dict)
    info = payload["info"]
    assert isinstance(info, dict)
    usage = info["last_token_usage"]
    assert usage_schema is not None
    with pytest.raises(ValueError, match="relational solver custody"):
        existing_value_is_supported(usage_schema, usage)
