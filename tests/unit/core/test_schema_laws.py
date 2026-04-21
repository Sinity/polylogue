"""Law-based contracts for schema inference and validation helpers."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from polylogue.lib.json import json_document
from polylogue.schemas.schema_inference import (
    _remove_nested_required,
    _structure_fingerprint,
    collapse_dynamic_keys,
    generate_schema_from_samples,
    is_dynamic_key,
    load_samples_from_sessions,
)
from polylogue.schemas.validator import SchemaValidator
from polylogue.types import Provider
from tests.infra.schema_access import schema_node, schema_properties
from tests.infra.strategies import (
    SessionJsonlFileSpec,
    dynamic_key_strategy,
    expected_session_documents,
    json_document_strategy,
    nested_required_schema_strategy,
    record_payload_strategy,
    record_variant_signature,
    session_jsonl_tree_strategy,
    static_key_strategy,
)


def _required_fields(schema: object) -> list[str]:
    required = schema_node(schema).get("required")
    if not isinstance(required, list):
        return []
    return [value for value in required if isinstance(value, str)]


def _schema_string(schema: object, key: str) -> str | None:
    value = schema_node(schema).get(key)
    return value if isinstance(value, str) else None


def _nested_required_paths(schema: object, *, depth: int = 0, path: str = "$") -> list[str]:
    """Collect nested schema paths that still carry a required array."""
    if not isinstance(schema, dict):
        return []

    paths: list[str] = []
    if depth > 0 and "required" in schema:
        paths.append(path)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, value in properties.items():
            paths.extend(_nested_required_paths(value, depth=depth + 1, path=f"{path}.{key}"))

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        paths.extend(_nested_required_paths(additional, depth=depth + 1, path=f"{path}.*"))

    items = schema.get("items")
    if isinstance(items, dict):
        paths.extend(_nested_required_paths(items, depth=depth + 1, path=f"{path}[*]"))

    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            for index, value in enumerate(variants):
                paths.extend(_nested_required_paths(value, depth=depth + 1, path=f"{path}.{keyword}[{index}]"))

    return paths


@settings(max_examples=40)
@given(dynamic_key_strategy())
def test_dynamic_key_strategy_matches_runtime_detection(key: str) -> None:
    """Generated dynamic keys must always be recognized by the runtime heuristic."""
    assert is_dynamic_key(key)


@settings(max_examples=40)
@given(static_key_strategy())
def test_static_key_strategy_stays_outside_runtime_detection(key: str) -> None:
    """Generated static field names must remain explicit schema properties."""
    assert not is_dynamic_key(key)


@settings(max_examples=35, deadline=None)
@given(dynamic_key_strategy(), dynamic_key_strategy(), json_document_strategy())
def test_structure_fingerprint_ignores_dynamic_key_renames(
    left_key: str,
    right_key: str,
    payload: dict[str, object],
) -> None:
    """Dynamic-map renaming must not change the structural fingerprint."""
    assume(left_key != right_key)

    left = {"mapping": {left_key: payload}}
    right = {"mapping": {right_key: payload}}

    assert _structure_fingerprint(left) == _structure_fingerprint(right)


@settings(max_examples=30, deadline=None)
@given(
    st.lists(static_key_strategy(), min_size=1, max_size=3, unique=True),
    st.lists(dynamic_key_strategy(), min_size=1, max_size=3, unique=True),
    st.sampled_from(("string", "integer", "boolean")),
)
def test_collapse_dynamic_keys_preserves_static_fields_and_rehomes_dynamic_maps(
    static_keys: list[str],
    dynamic_keys: list[str],
    dynamic_type: str,
) -> None:
    """Collapsing dynamic maps must keep static fields explicit and merge dynamic ones."""
    schema: dict[str, object] = {
        "type": "object",
        "required": static_keys + dynamic_keys,
        "properties": {
            **{key: {"type": "string"} for key in static_keys},
            **{key: {"type": dynamic_type} for key in dynamic_keys},
        },
    }

    collapsed = collapse_dynamic_keys(json_document(copy.deepcopy(schema)))
    properties = schema_properties(collapsed)

    assert set(properties.keys()) == set(static_keys)
    assert all(key not in properties for key in dynamic_keys)
    assert schema_node(collapsed).get("x-polylogue-dynamic-keys") is True
    assert "additionalProperties" in schema_node(collapsed)
    assert set(_required_fields(collapsed)) == set(static_keys)


@settings(max_examples=30)
@given(st.lists(static_key_strategy(), min_size=1, max_size=4, unique=True))
def test_collapse_dynamic_keys_leaves_static_only_objects_explicit(static_keys: list[str]) -> None:
    """Static-only schemas should not gain additionalProperties noise."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in static_keys},
    }

    collapsed = collapse_dynamic_keys(json_document(copy.deepcopy(schema)))
    properties = schema_properties(collapsed)

    assert set(properties.keys()) == set(static_keys)
    assert "additionalProperties" not in schema_node(collapsed)
    assert "x-polylogue-dynamic-keys" not in schema_node(collapsed)


@settings(max_examples=35, deadline=None)
@given(nested_required_schema_strategy())
def test_remove_nested_required_strips_required_recursively_but_keeps_root(
    schema: dict[str, object],
) -> None:
    """Only the root required array should survive recursive schema cleanup."""
    root_required_value = schema.get("required", [])
    assert isinstance(root_required_value, list)
    root_required = list(root_required_value)

    result = _remove_nested_required(copy.deepcopy(schema), depth=0)

    assert result.get("required", []) == root_required
    assert _nested_required_paths(result) == []


@settings(max_examples=30, deadline=None)
@given(record_payload_strategy(min_variants=2), st.data())
def test_validation_samples_record_payloads_preserve_variant_coverage_when_capped(
    payload: list[dict[str, object]],
    data: st.DataObject,
) -> None:
    """Record sampling should retain every observed record family when the cap allows it."""
    validator = SchemaValidator(
        {
            "type": "object",
            "x-polylogue-sample-granularity": "record",
            "properties": {"type": {"type": "string"}},
        },
        provider=Provider.CODEX,
    )
    expected_signatures = {record_variant_signature(item) for item in payload}
    limit = data.draw(st.integers(min_value=len(expected_signatures), max_value=len(payload)))

    samples = validator.validation_samples(payload, max_samples=limit)

    assert len(samples) == limit
    assert {record_variant_signature(item) for item in samples} == expected_signatures


@settings(max_examples=30)
@given(record_payload_strategy())
def test_validation_samples_record_payloads_are_uncapped_by_default(payload: list[dict[str, object]]) -> None:
    """Record-oriented providers should validate every dict record unless explicitly capped."""
    validator = SchemaValidator(
        {"type": "object", "properties": {"type": {"type": "string"}}},
        provider=Provider.CODEX,
    )
    assert validator.validation_samples(payload, max_samples=None) == payload


@settings(max_examples=30, deadline=None)
@given(st.lists(json_document_strategy(), min_size=1, max_size=6), st.data())
def test_validation_samples_document_mode_matches_payload_shape(
    documents: list[dict[str, object]],
    data: st.DataObject,
) -> None:
    """Document-mode sampling should validate top-level docs and prefix-limit lists."""
    validator = SchemaValidator(
        {"type": "object", "properties": {"id": {"type": "string"}}},
        provider=Provider.CHATGPT,
    )
    limit = data.draw(st.one_of(st.none(), st.integers(min_value=1, max_value=len(documents))))
    single = data.draw(json_document_strategy())

    samples = validator.validation_samples(documents, max_samples=limit)

    assert samples == (documents if limit is None else documents[:limit])
    assert validator.validation_samples(single, max_samples=1) == [single]


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(session_jsonl_tree_strategy())
def test_load_samples_from_sessions_keeps_valid_dicts_in_recursive_mtime_order(
    file_specs: tuple[SessionJsonlFileSpec, ...],
) -> None:
    """Recursive session loading should ignore junk lines and preserve valid dict order."""
    with TemporaryDirectory() as tempdir:
        session_dir = Path(tempdir) / "sessions"
        session_dir.mkdir()
        base_mtime = 1_700_000_000

        for index, file_spec in enumerate(file_specs):
            path = session_dir / file_spec.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(file_spec.text, encoding="utf-8")
            os.utime(path, (base_mtime + index, base_mtime + index))

        assert load_samples_from_sessions(session_dir) == expected_session_documents(file_specs)


@settings(max_examples=25, deadline=None)
@given(record_payload_strategy(min_variants=2))
def test_load_samples_from_sessions_preserves_record_families_when_capped(
    payload: list[dict[str, object]],
) -> None:
    """Session sampling caps should still keep each record family represented."""
    with TemporaryDirectory() as tempdir:
        session_dir = Path(tempdir) / "sessions"
        session_dir.mkdir()
        (session_dir / "mixed.jsonl").write_text(
            "\n".join(json.dumps(record) for record in payload) + "\n",
            encoding="utf-8",
        )

        expected_signatures = {record_variant_signature(item) for item in payload}
        samples = load_samples_from_sessions(
            session_dir,
            max_samples=len(expected_signatures),
            record_type_key="type",
        )

        assert len(samples) == len(expected_signatures)
        assert {record_variant_signature(item) for item in samples} == expected_signatures


@settings(max_examples=25, deadline=None)
@given(st.lists(json_document_strategy(), min_size=1, max_size=6))
def test_generate_schema_from_samples_exposes_union_of_observed_top_level_fields(
    samples: list[dict[str, object]],
) -> None:
    """Schema generation should include every observed top-level field from the corpus."""
    pytest.importorskip("genson")
    schema = generate_schema_from_samples(samples)
    observed_keys = {key for sample in samples for key in sample}

    assert observed_keys.issubset(schema_properties(schema).keys())
    assert _schema_string(schema, "type") == "object"
    assert _schema_string(schema, "$schema") == "https://json-schema.org/draft/2020-12/schema"
