"""Promotion may only ever broaden a provider package, never narrow it.

A package records what a provider has been OBSERVED to emit. Regenerating from
the current sample window and registering that directly lets a thin window
silently replace a well-sampled package.

That happened on 2026-07-29: a promotion regenerated codex and claude-code from
a narrower window and, measured against the prior packages, narrowed 33 field
types and dropped 173 fields. Concretely, codex `timestamp` went from
`["string", "number"]` to `"string"` -- but the numeric form is real, which is
why the 151,159-sample inference emitted the union in the first place. Records
carrying it would then validate as drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.dynamic_keys import merge_observed_structure_schemas
from polylogue.schemas.registry import SchemaRegistry


def _types_by_path(schema: Any, path: str = "") -> dict[str, frozenset[str]]:
    """Every typed node in a schema, keyed by structural path."""
    found: dict[str, frozenset[str]] = {}
    if isinstance(schema, dict):
        declared = schema.get("type")
        if isinstance(declared, str):
            found[path] = frozenset({declared})
        elif isinstance(declared, list):
            found[path] = frozenset(item for item in declared if isinstance(item, str))
        for key, value in schema.items():
            if not isinstance(value, (dict, list)):
                continue
            structural = key in ("properties", "items", "additionalProperties", "anyOf", "oneOf", "allOf")
            found.update(_types_by_path(value, path if structural else f"{path}.{key}"))
    elif isinstance(schema, list):
        for entry in schema:
            found.update(_types_by_path(entry, path))
    return found


def test_merging_a_thin_window_cannot_narrow_a_union() -> None:
    """The exact codex timestamp regression: string|number must survive a string-only window."""
    promoted: JSONDocument = {
        "type": "object",
        "properties": {"timestamp": {"type": ["string", "number"]}, "id": {"type": "string"}},
    }
    thin_window: JSONDocument = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "id": {"type": "string"}},
    }

    merged = merge_observed_structure_schemas([promoted, thin_window])

    timestamp = cast("dict[str, Any]", cast("dict[str, Any]", merged["properties"])["timestamp"])
    assert set(timestamp["type"]) == {"string", "number"}


def test_merging_cannot_drop_a_field_the_package_already_carried() -> None:
    """173 fields vanished in the real incident; absence from a window is not evidence of removal."""
    promoted: JSONDocument = {
        "type": "object",
        "properties": {"kept": {"type": "string"}, "absent_from_new_window": {"type": "integer"}},
    }
    thin_window: JSONDocument = {"type": "object", "properties": {"kept": {"type": "string"}}}

    merged = merge_observed_structure_schemas([promoted, thin_window])

    properties = cast("dict[str, Any]", merged["properties"])
    assert "absent_from_new_window" in properties
    assert properties["absent_from_new_window"]["type"] == "integer"


def test_no_typed_node_narrows_under_merge_in_either_order() -> None:
    """Breadth is order-independent: merge is a union, not a last-writer-wins overwrite."""
    wide: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "number"]},
            "nested": {"type": "object", "properties": {"b": {"type": ["integer", "null"]}}},
        },
    }
    narrow: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "nested": {"type": "object", "properties": {"b": {"type": "integer"}}},
        },
    }

    for left, right in ((wide, narrow), (narrow, wide)):
        merged_types = _types_by_path(merge_observed_structure_schemas([left, right]))
        for path, wide_types in _types_by_path(wide).items():
            assert wide_types <= merged_types.get(path, frozenset()), (
                f"{path} narrowed: {sorted(wide_types)} not preserved in {sorted(merged_types.get(path, []))}"
            )


@pytest.fixture
def tmp_registry(tmp_path: Path) -> SchemaRegistry:
    return SchemaRegistry(storage_root=tmp_path / "schemas")


class TestPromoteClusterElementKindResolution:
    def test_unspecified_artifact_kind_still_merges_with_real_default_element_kind(
        self, tmp_registry: SchemaRegistry
    ) -> None:
        """`cluster_samples` without `artifact_kinds=` tags every cluster
        "unspecified" (this is what `infer_schema(cluster=True)` does today).
        A promotion built from such a cluster must still resolve to whatever
        element_kind the provider's existing default package actually uses
        (e.g. codex/claude-code's `session_record_stream`), not register a
        disconnected "unspecified"/default "session_document" element that
        orphans it.
        """
        tmp_registry.register_schema(
            "kind-prov",
            {"type": "object", "properties": {"legacy": {"type": "string"}}},
            element_kind="session_record_stream",
        )

        samples = [{"legacy": "x", "fresh": 1}]
        manifest = tmp_registry.cluster_samples("kind-prov", samples)  # no artifact_kinds -> "unspecified"
        tmp_registry.save_cluster_manifest(manifest)
        assert manifest.clusters[0].artifact_kind == "unspecified"

        new_version = tmp_registry.promote_cluster("kind-prov", manifest.clusters[0].cluster_id, samples=samples)
        schema = tmp_registry.get_schema("kind-prov", version=new_version)
        assert schema is not None
        properties = cast("dict[str, Any]", schema["properties"])
        assert "legacy" in properties
        assert "fresh" in properties
        catalog = tmp_registry.load_package_catalog("kind-prov")
        assert catalog is not None
        package = catalog.package(new_version)
        assert package is not None
        assert package.default_element_kind == "session_record_stream"

    def test_observed_artifact_count_never_decreases(self, tmp_registry: SchemaRegistry) -> None:
        tmp_registry.register_schema("count-prov", {"type": "object", "properties": {"a": {"type": "string"}}})
        manifest = tmp_registry.load_cluster_manifest("count-prov")
        assert manifest is None  # nothing clustered yet; register_schema alone doesn't create one

        samples = [{"a": "x", "b": 1}]
        manifest = tmp_registry.cluster_samples("count-prov", samples)
        # Fake a large prior observed-artifact-count directly on the existing
        # package the way a real full-corpus promotion would have left it.
        existing = tmp_registry.get_schema("count-prov")
        assert existing is not None
        existing["x-polylogue-observed-artifact-count"] = 10_000
        tmp_registry.write_schema_version("count-prov", "v1", existing)
        tmp_registry.save_cluster_manifest(manifest)

        new_version = tmp_registry.promote_cluster("count-prov", manifest.clusters[0].cluster_id, samples=samples)
        schema = tmp_registry.get_schema("count-prov", version=new_version)
        assert schema is not None
        observed_count = schema["x-polylogue-observed-artifact-count"]
        assert isinstance(observed_count, int)
        assert observed_count >= 10_000
