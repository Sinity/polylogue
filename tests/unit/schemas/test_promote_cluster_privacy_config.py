"""polylogue-f47j: ``promote_cluster``'s samples path must honor ``privacy_config``.

``registry.promote_cluster``'s samples-based candidate generator called
``generate_schema_from_samples`` (``schemas/generation/schema_builder.py``)
without ever receiving a ``PrivacyConfig``, unlike the full ``generate``
pipeline's ``_generate_cluster_schema``, which threads one explicitly through
to ``_annotate_schema``. A caller that supplies an explicit deny rule (e.g. a
project-level privacy config denying a specific field) got silently ignored
on the samples path -- exactly what let a low-cardinality UUID-shaped field
(``current_leaf_message_uuid``) record its literal per-record values verbatim
during a 2026-07-29 promotion.

These tests exercise both layers: the generator directly, and the real
``promote_cluster`` route through ``SchemaRegistry``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.privacy_config import PrivacyConfig
from polylogue.schemas.registry import SchemaRegistry

_SAMPLES = [
    {"region_code": "us-east", "note": 1},
    {"region_code": "eu-west", "note": 2},
    {"region_code": "us-east", "note": 3},
]


def test_generate_schema_from_samples_ignores_privacy_by_default() -> None:
    """Baseline: without a privacy_config, the low-cardinality enum leaks verbatim."""
    schema = generate_schema_from_samples(_SAMPLES)

    properties = cast("dict[str, Any]", schema["properties"])
    region = cast("dict[str, Any]", properties["region_code"])
    assert region.get("x-polylogue-values") == ["us-east", "eu-west"]


def test_generate_schema_from_samples_honors_privacy_config_field_deny() -> None:
    """A field-level deny rule must suppress the annotation -- this is the plumbing fix."""
    privacy_config = PrivacyConfig(field_overrides={"$.region_code": "deny"})

    schema = generate_schema_from_samples(_SAMPLES, privacy_config=privacy_config)

    properties = cast("dict[str, Any]", schema["properties"])
    region = cast("dict[str, Any]", properties["region_code"])
    assert "x-polylogue-values" not in region


def test_promote_cluster_real_path_honors_privacy_config(tmp_path: Path) -> None:
    """The regression case: promote_cluster's samples route, not the generator alone.

    Anti-vacuity: removing the ``privacy_config=privacy_config`` forwarding in
    ``SchemaRegistry.promote_cluster`` (``schemas/tooling_registry.py``) makes
    this test fail -- the denied field's values would reappear verbatim.
    """
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    manifest = registry.cluster_samples("privprov", _SAMPLES)
    registry.save_cluster_manifest(manifest)
    cluster_id = manifest.clusters[0].cluster_id

    version = registry.promote_cluster(
        "privprov",
        cluster_id,
        samples=_SAMPLES,
        privacy_config=PrivacyConfig(field_overrides={"$.region_code": "deny"}),
    )

    schema = registry.get_schema("privprov", version=version)
    assert schema is not None
    properties = cast("dict[str, Any]", schema["properties"])
    region = cast("dict[str, Any]", properties["region_code"])
    assert "x-polylogue-values" not in region


def test_promote_cluster_real_path_without_privacy_config_still_leaks(tmp_path: Path) -> None:
    """Confirms the test above is a genuine red/green pair, not a vacuous assertion."""
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    manifest = registry.cluster_samples("privprov-baseline", _SAMPLES)
    registry.save_cluster_manifest(manifest)
    cluster_id = manifest.clusters[0].cluster_id

    version = registry.promote_cluster("privprov-baseline", cluster_id, samples=_SAMPLES)

    schema = registry.get_schema("privprov-baseline", version=version)
    assert schema is not None
    properties = cast("dict[str, Any]", schema["properties"])
    region = cast("dict[str, Any]", properties["region_code"])
    assert region.get("x-polylogue-values") == ["us-east", "eu-west"]
