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
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.type_narrowing import types_by_path as _types_by_path


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


def test_types_by_path_unions_anyof_branches_at_the_same_path() -> None:
    """CodeRabbit finding on PR #3538: found.update() overwrote an earlier
    anyOf branch's types at a shared path instead of merging them, so a
    later schema keeping only the last branch's type wouldn't register as
    narrowed even though the earlier branch's type was really lost."""
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
        },
    }
    types = _types_by_path(schema)
    assert types[".value"] == frozenset({"string", "number"})


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


def _single_package_catalog(provider: str, version: str, element_kind: str) -> SchemaPackageCatalog:
    return SchemaPackageCatalog(
        provider=provider,
        packages=[
            SchemaVersionPackage(
                provider=provider,
                version=version,
                anchor_kind=element_kind,
                default_element_kind=element_kind,
                first_seen="2026-08-01T00:00:00Z",
                last_seen="2026-08-01T00:00:00Z",
                bundle_scope_count=0,
                sample_count=1,
                elements=[
                    SchemaElementManifest(
                        element_kind=element_kind,
                        schema_file=f"{element_kind}.schema.json.gz",
                        sample_count=1,
                        artifact_count=1,
                    )
                ],
            )
        ],
        default_version=version,
        latest_version=version,
        recommended_version=version,
    )


def _two_element_package_catalog(
    provider: str, version: str, anchor_kind: str, other_kind: str
) -> SchemaPackageCatalog:
    return SchemaPackageCatalog(
        provider=provider,
        packages=[
            SchemaVersionPackage(
                provider=provider,
                version=version,
                anchor_kind=anchor_kind,
                default_element_kind=anchor_kind,
                first_seen="2026-08-01T00:00:00Z",
                last_seen="2026-08-01T00:00:00Z",
                bundle_scope_count=0,
                sample_count=1,
                elements=[
                    SchemaElementManifest(
                        element_kind=anchor_kind,
                        schema_file=f"{anchor_kind}.schema.json.gz",
                        sample_count=1,
                        artifact_count=1,
                    ),
                    SchemaElementManifest(
                        element_kind=other_kind,
                        schema_file=f"{other_kind}.schema.json.gz",
                        sample_count=1,
                        artifact_count=1,
                    ),
                ],
            )
        ],
        default_version=version,
        latest_version=version,
        recommended_version=version,
    )


class TestReplaceProviderPackagesMonotonicity:
    """Guards ``SchemaRegistry.replace_provider_packages`` -- the code path every
    full-corpus ``devtools schema-generate`` run writes through
    (``persist_generated_provider_bundle`` -> ``replace_provider_packages`` in
    ``polylogue/schemas/generation/workflow.py``). Reproduces, at unit scale,
    the 2026-08-01 measured incident (polylogue-ov5r): a real full-corpus
    ``devtools schema-generate`` run against the live archive lost 722 of 944
    typed leaf paths for claude-code and narrowed codex's ``timestamp`` union
    from ``["number", "string"]`` back to ``["string"]`` -- because
    ``replace_provider_packages`` deleted the provider's entire ``versions/``
    tree and rewrote it from the fresh generation with no merge against the
    committed prior schema.

    Anti-vacuity: deleting the
    ``_merge_element_schema_with_existing``/``_existing_provider_element_schemas``
    call in ``SchemaRegistry.replace_provider_packages``
    (``polylogue/schemas/runtime_registry.py``), or reverting
    ``replace_provider_packages`` to write ``package_schemas`` directly instead
    of the merged map, makes every test below fail: the second
    ``replace_provider_packages`` call would then simply overwrite the first
    package instead of unioning into it.
    """

    def test_regen_observing_a_subset_of_known_fields_does_not_lose_them(self, tmp_registry: SchemaRegistry) -> None:
        wide_schema: JSONDocument = {
            "type": "object",
            "properties": {
                "kept": {"type": "string"},
                "dropped_in_thin_regen": {"type": "integer"},
            },
        }
        tmp_registry.replace_provider_packages(
            "regen-subset",
            _single_package_catalog("regen-subset", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": wide_schema}},
        )

        # A subsequent full-corpus regeneration observes only a subset of the
        # previously known fields -- e.g. a differently-shaped sample window,
        # exactly the scenario that dropped 722/944 claude-code leaf paths.
        thin_schema: JSONDocument = {"type": "object", "properties": {"kept": {"type": "string"}}}
        tmp_registry.replace_provider_packages(
            "regen-subset",
            _single_package_catalog("regen-subset", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": thin_schema}},
        )

        merged = tmp_registry.get_schema("regen-subset", version="v1")
        assert merged is not None
        properties = cast("dict[str, Any]", merged["properties"])
        assert "dropped_in_thin_regen" in properties
        assert properties["dropped_in_thin_regen"]["type"] == "integer"

    def test_regen_observing_genuinely_new_fields_gains_them(self, tmp_registry: SchemaRegistry) -> None:
        old_schema: JSONDocument = {"type": "object", "properties": {"kept": {"type": "string"}}}
        tmp_registry.replace_provider_packages(
            "regen-new-field",
            _single_package_catalog("regen-new-field", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": old_schema}},
        )

        new_schema: JSONDocument = {
            "type": "object",
            "properties": {"kept": {"type": "string"}, "newly_observed": {"type": "boolean"}},
        }
        tmp_registry.replace_provider_packages(
            "regen-new-field",
            _single_package_catalog("regen-new-field", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": new_schema}},
        )

        merged = tmp_registry.get_schema("regen-new-field", version="v1")
        assert merged is not None
        properties = cast("dict[str, Any]", merged["properties"])
        assert "newly_observed" in properties
        assert properties["newly_observed"]["type"] == "boolean"

    def test_regen_type_unions_only_grow_never_narrow(self, tmp_registry: SchemaRegistry) -> None:
        """The exact codex incident, reproduced against replace_provider_packages directly."""
        wide_schema: JSONDocument = {
            "type": "object",
            "properties": {"timestamp": {"type": ["string", "number"]}},
        }
        tmp_registry.replace_provider_packages(
            "regen-union",
            _single_package_catalog("regen-union", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": wide_schema}},
        )

        narrow_schema: JSONDocument = {
            "type": "object",
            "properties": {"timestamp": {"type": "string"}},
        }
        tmp_registry.replace_provider_packages(
            "regen-union",
            _single_package_catalog("regen-union", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": narrow_schema}},
        )

        merged = tmp_registry.get_schema("regen-union", version="v1")
        assert merged is not None
        timestamp = cast("dict[str, Any]", cast("dict[str, Any]", merged["properties"])["timestamp"])
        declared_type = timestamp["type"]
        observed = set(declared_type) if isinstance(declared_type, list) else {declared_type}
        assert observed == {"string", "number"}


class TestMergeElementSchemaAnnotationPreservation:
    """Guards nested ``x-polylogue-*`` annotation survival across
    ``SchemaRegistry.replace_provider_packages`` regenerations (polylogue-46kg
    P1, found by automated review on PR #3502 minutes after merge).

    ``merge_observed_structure_schemas`` (``schemas/generation/dynamic_keys.py``)
    merges only structural keywords (``type``/``properties``/``items``/
    ``additionalProperties``) -- its own docstring says "without retaining
    property history". The prior ``_merge_element_schema_with_existing``
    restored ``x-polylogue-*`` annotations only at the document root after
    merging, so every regeneration stripped freshly-computed *nested*
    annotations (semantic role, format, frequency, observed distribution)
    from property-level nodes.

    Anti-vacuity: reverting
    ``SchemaRegistry._merge_element_schema_with_existing`` to restore
    ``x-polylogue-*`` keys only at the document root (dropping the recursive
    ``_annotate_merged_schema_node`` walk into ``properties``) makes both
    tests below fail -- the nested ``user_id``/``value`` property's
    annotation is stripped by the structural merge and never restored.
    """

    def test_nested_annotation_survives_when_fresh_pass_recomputes_no_annotation(
        self, tmp_registry: SchemaRegistry
    ) -> None:
        annotated_schema: JSONDocument = {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "x-polylogue-semantic-role": "identifier",
                }
            },
        }
        tmp_registry.replace_provider_packages(
            "regen-annotation",
            _single_package_catalog("regen-annotation", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": annotated_schema}},
        )

        # A subsequent regeneration re-observes the same property's type but
        # its annotation pass didn't recompute a semantic role this time --
        # e.g. a narrower sample window that never re-derived it.
        unannotated_schema: JSONDocument = {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
        }
        tmp_registry.replace_provider_packages(
            "regen-annotation",
            _single_package_catalog("regen-annotation", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": unannotated_schema}},
        )

        merged = tmp_registry.get_schema("regen-annotation", version="v1")
        assert merged is not None
        user_id = cast("dict[str, Any]", cast("dict[str, Any]", merged["properties"])["user_id"])
        assert user_id.get("x-polylogue-semantic-role") == "identifier"

    def test_nested_annotation_fresh_value_wins_over_stale_existing(self, tmp_registry: SchemaRegistry) -> None:
        stale_schema: JSONDocument = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "x-polylogue-semantic-role": "identifier",
                }
            },
        }
        tmp_registry.replace_provider_packages(
            "regen-annotation-update",
            _single_package_catalog("regen-annotation-update", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": stale_schema}},
        )

        # This regeneration DOES carry a fresh, different annotation for the
        # same nested property -- the fresh value must win, not the stale one.
        updated_schema: JSONDocument = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "x-polylogue-semantic-role": "timestamp",
                }
            },
        }
        tmp_registry.replace_provider_packages(
            "regen-annotation-update",
            _single_package_catalog("regen-annotation-update", "v1", "session_record_stream"),
            {"v1": {"session_record_stream": updated_schema}},
        )

        merged = tmp_registry.get_schema("regen-annotation-update", version="v1")
        assert merged is not None
        value = cast("dict[str, Any]", cast("dict[str, Any]", merged["properties"])["value"])
        assert value.get("x-polylogue-semantic-role") == "timestamp"


class TestReplaceProviderPackagesCarriesForwardUnobservedElementKinds:
    """Guards ``SchemaRegistry.replace_provider_packages`` against dropping a
    whole element kind when a thinner regeneration observes zero samples for
    it (polylogue-46kg P2, found by automated review on PR #3502 minutes
    after merge).

    This is the same destructive-loss bug class ``ov5r``
    (``TestReplaceProviderPackagesMonotonicity`` above) fixed, just narrower:
    a whole element kind absent from ``element_schemas.items()`` for a
    version never enters the merge loop at all, so the destructive
    ``versions/`` delete-and-rewrite drops it entirely -- afterward
    ``get_element_schema(..., element_kind=<old kind>)`` silently returns
    ``None``.

    Anti-vacuity: removing the ``existing_elements_by_version`` carry-forward
    loop in ``replace_provider_packages`` (or the fix that persists the
    carry-forward-augmented packages via ``dataclasses.replace(catalog, ...)``
    instead of the caller's original ``catalog.packages``) makes this test
    fail -- ``"tool_result"`` vanishes from the second run's manifest/elements
    directory.
    """

    def test_kind_absent_from_fresh_observation_window_is_carried_forward(self, tmp_registry: SchemaRegistry) -> None:
        message_schema: JSONDocument = {"type": "object", "properties": {"text": {"type": "string"}}}
        tool_result_schema: JSONDocument = {"type": "object", "properties": {"exit_code": {"type": "integer"}}}
        tmp_registry.replace_provider_packages(
            "regen-kind-drop",
            _two_element_package_catalog("regen-kind-drop", "v1", "message", "tool_result"),
            {"v1": {"message": message_schema, "tool_result": tool_result_schema}},
        )
        assert tmp_registry.get_element_schema("regen-kind-drop", version="v1", element_kind="tool_result") is not None

        # A thinner regeneration window observes zero tool_result samples
        # this pass (e.g. a corpus subset with no tool calls) -- the fresh
        # catalog and package_schemas mapping for this version now mention
        # only "message".
        tmp_registry.replace_provider_packages(
            "regen-kind-drop",
            _single_package_catalog("regen-kind-drop", "v1", "message"),
            {"v1": {"message": message_schema}},
        )

        carried = tmp_registry.get_element_schema("regen-kind-drop", version="v1", element_kind="tool_result")
        assert carried is not None
        properties = cast("dict[str, Any]", carried["properties"])
        assert properties["exit_code"]["type"] == "integer"

        # "message" itself is untouched and still resolvable.
        assert tmp_registry.get_element_schema("regen-kind-drop", version="v1", element_kind="message") is not None
