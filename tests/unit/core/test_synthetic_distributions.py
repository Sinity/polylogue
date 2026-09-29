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
from pathlib import Path
from typing import cast

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.wire_formats import WireFormat

_PROVIDERS = Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"


def _point(value: int) -> JSONDocument:
    """A one-bucket histogram pinned to exactly ``value`` by its min and max."""
    return {"count": 1, "histogram": [[40, 1]], "log_base": 1.1, "min": value, "max": value}


def _corpus(schema: JSONDocument) -> SyntheticCorpus:
    return SyntheticCorpus(schema, WireFormat(encoding="json"), "test")


def _generate(schema: JSONDocument, seed: int) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], _corpus(schema)._generate_from_schema(schema, random.Random(seed)))


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


def test_real_packages_generate_string_lengths_inside_their_observed_range() -> None:
    """Every top-level string field with an observed length histogram stays inside [min, max]."""
    checked = 0
    for provider in ("chatgpt", "claude-code", "gemini"):
        element = sorted((_PROVIDERS / provider / "versions").glob("*/elements/*.schema.json.gz"))[-1]
        schema = cast(JSONDocument, json.load(gzip.open(element)))
        bounds = {
            path: (distribution.get("min"), distribution.get("max"))
            for path, distribution in _string_length_fields(schema)
            if path.count(".") == 1
        }
        corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), provider)
        rng = random.Random(23)
        for _ in range(30):
            record = corpus._generate_from_schema(schema, rng)
            if not isinstance(record, dict):
                continue
            for path, (low, high) in bounds.items():
                value = record.get(path.removeprefix("$."))
                if isinstance(value, str) and isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    assert low <= len(value) <= high, (provider, path, len(value), low, high)
                    checked += 1
    assert checked > 0
