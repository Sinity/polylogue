"""Infrastructure-backed contracts for schema operator workflows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.schemas.operator.inference import (
    _privacy_config,
    audit_schemas,
    compare_schema_versions,
    infer_schema,
    list_inferred_corpus_scenarios,
    list_inferred_corpus_specs,
    list_schemas,
    promote_schema_cluster,
)
from polylogue.schemas.operator.models import (
    JSONDocument,
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaInferRequest,
    SchemaListRequest,
    SchemaPromoteRequest,
)
from polylogue.schemas.registry import SchemaRegistry
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from tests.infra.storage_records import db_setup


def _seed_chatgpt_raw(workspace_env: dict[str, Path]) -> Path:
    """Store one real ChatGPT-shaped raw payload in the source tier."""
    index_db = db_setup(workspace_env)
    payload = json.dumps(
        {
            "id": "conversation-1",
            "title": "Schema inference",
            "create_time": 1_700_000_000.0,
            "update_time": 1_700_000_060.0,
            "mapping": {
                "node-1": {
                    "id": "node-1",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["infer this schema"]},
                        "create_time": 1_700_000_000.0,
                    },
                }
            },
        }
    ).encode()
    get_blob_store().write_from_bytes(payload)
    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        write_source_raw_session(
            conn,
            origin=origin_from_provider(Provider.CHATGPT),
            source_path="/fixtures/chatgpt-export.json",
            source_index=0,
            payload=payload,
            acquired_at_ms=1_700_000_000_000,
        )
    return index_db


def _seed_chatgpt_raw_with_planted_field(workspace_env: dict[str, Path], *, values: list[str]) -> Path:
    """Store several real ChatGPT-shaped raw payloads carrying a shared
    low-cardinality top-level field (``region_code``), one raw session per
    value in ``values``, all with the same structural shape so they cluster
    together for ``promote_schema_cluster``'s ``with_samples=True`` route.
    """
    index_db = db_setup(workspace_env)
    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        for index, value in enumerate(values):
            payload = json.dumps(
                {
                    "id": f"conversation-{index}",
                    "title": "Schema inference",
                    "region_code": value,
                    "create_time": 1_700_000_000.0 + index,
                    "update_time": 1_700_000_060.0 + index,
                    "mapping": {
                        "node-1": {
                            "id": "node-1",
                            "parent": None,
                            "children": [],
                            "message": {
                                "id": f"message-{index}",
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["infer this schema"]},
                                "create_time": 1_700_000_000.0 + index,
                            },
                        }
                    },
                }
            ).encode()
            get_blob_store().write_from_bytes(payload)
            write_source_raw_session(
                conn,
                origin=origin_from_provider(Provider.CHATGPT),
                source_path=f"/fixtures/chatgpt-export-{index}.json",
                source_index=0,
                payload=payload,
                acquired_at_ms=1_700_000_000_000 + index,
            )
    return index_db


def test_promote_cluster_with_samples_honors_privacy_config_through_the_real_operator_route(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-f47j, completed: verifies the fix is reachable from the
    actual production caller, not just the library function.

    ``devtools schema-promote --with-samples`` -> ``promote_schema_cluster``
    -> ``registry.promote_cluster`` -> ``generate_schema_from_samples`` is
    the real chain a live promotion runs. The library-level fix alone
    (threading ``privacy_config`` into ``generate_schema_from_samples``) was
    unreachable from here until ``SchemaPromoteRequest`` grew a
    ``privacy_config`` field and ``promote_schema_cluster`` forwarded it --
    without that, every real ``devtools schema-promote`` run stayed exactly
    as unredacted as before the library fix landed.

    Anti-vacuity: the baseline call (no ``privacy_config``) must still leak
    the planted field's values, proving this is a genuine red/green pair and
    not a heuristic that would have redacted it anyway.
    """
    index_db = _seed_chatgpt_raw_with_planted_field(workspace_env, values=["us-east", "eu-west"])
    inferred = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=True))
    assert inferred.manifest is not None
    cluster_id = inferred.manifest.clusters[0].cluster_id

    baseline = promote_schema_cluster(
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id=cluster_id,
            db_path=index_db,
            with_samples=True,
            max_samples=100,
        )
    )
    assert baseline.schema is not None
    baseline_properties = baseline.schema["properties"]
    assert isinstance(baseline_properties, dict)
    assert baseline_properties["region_code"].get("x-polylogue-values") == ["us-east", "eu-west"]

    # Reset the cluster's promotion state so it can be promoted a second
    # time in this same test -- promote_cluster refuses to re-promote an
    # already-promoted cluster, and this is the same cluster on purpose (the
    # point is to compare identical input with and without privacy_config).
    registry = SchemaRegistry()
    manifest = registry.load_cluster_manifest("chatgpt")
    assert manifest is not None
    for cluster in manifest.clusters:
        if cluster.cluster_id == cluster_id:
            cluster.promoted_package_version = None
    registry.save_cluster_manifest(manifest)

    protected = promote_schema_cluster(
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id=cluster_id,
            db_path=index_db,
            with_samples=True,
            max_samples=100,
            privacy_config={"field_overrides": {"$.region_code": "deny"}},
        )
    )
    assert protected.schema is not None
    protected_properties = protected.schema["properties"]
    assert isinstance(protected_properties, dict)
    assert "x-polylogue-values" not in protected_properties["region_code"]


def test_infer_schema_builds_schema_from_source_tier_raw(workspace_env: dict[str, Path]) -> None:
    index_db = _seed_chatgpt_raw(workspace_env)

    result = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=False))

    assert result.generation.success
    assert result.generation.sample_count == 1
    assert result.generation.schema is not None
    assert result.manifest is None
    assert len(result.corpus_specs) == 1
    assert len(result.corpus_scenarios) == 1
    assert result.corpus_specs[0].provider == "chatgpt"
    assert result.corpus_scenarios[0].provider == "chatgpt"
    assert result.corpus_scenarios[0].package_version == result.corpus_specs[0].package_version


def test_cluster_promotion_drives_real_operator_registry_views(workspace_env: dict[str, Path]) -> None:
    # This test's isolated storage root has no chatgpt catalog of its own, so
    # SchemaRegistry falls back to whatever is really committed under
    # polylogue/schemas/providers/chatgpt -- register_schema/promote_cluster
    # then append new versions on top of that real, growing version list.
    # Don't hardcode which version numbers that produces; derive them from
    # the actual pre-promotion state so this test survives future provider
    # promotions instead of re-baking today's committed version count in.
    baseline_registry = SchemaRegistry()
    baseline_versions = baseline_registry.list_versions("chatgpt")
    next_version = f"v{int(baseline_versions[-1][1:]) + 1}" if baseline_versions else "v1"
    version_after_next = f"v{int(next_version[1:]) + 1}"

    index_db = _seed_chatgpt_raw(workspace_env)
    inferred = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=True))

    assert inferred.generation.success
    assert inferred.manifest is not None
    assert inferred.manifest_path is not None and inferred.manifest_path.exists()
    cluster_id = inferred.manifest.clusters[0].cluster_id
    promoted = promote_schema_cluster(
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id=cluster_id,
            db_path=index_db,
            with_samples=False,
        )
    )

    assert promoted.package_version == next_version
    assert promoted.schema is not None
    specs = list_inferred_corpus_specs(provider="chatgpt")
    registry = SchemaRegistry()
    registry.register_schema("chatgpt", {"type": "object", "properties": {"mapping": {"type": "object"}}})
    registry.register_schema("codex", {"type": "object", "properties": {"session_id": {"type": "string"}}})
    scenarios = list_inferred_corpus_scenarios()
    comparison = compare_schema_versions(
        SchemaCompareRequest(provider="chatgpt", from_version=next_version, to_version=version_after_next)
    )
    selected = list_schemas(SchemaListRequest(provider="chatgpt"))
    listing = list_schemas(SchemaListRequest())

    assert comparison.diff.version_a == next_version
    assert comparison.diff.version_b == version_after_next
    assert selected.selected is not None
    # Only the two versions promoted into THIS workspace's isolated registry
    # are listed. baseline_versions is read from the bundled catalog purely to
    # derive non-colliding version NAMES; it is deliberately not expected in
    # the listing. Until 2026-07-30 write_schema_version() silently inherited
    # bundled-catalog versions into an isolated registry, so this assertion
    # used to see them -- that leak is fixed, and an isolated registry now
    # contains exactly what was written to it.
    assert selected.selected.versions == [next_version, version_after_next]
    assert {spec.provider for spec in specs} == {"chatgpt"}
    assert specs[0].package_version == next_version
    assert specs[0].profile.family_ids == (cluster_id,)
    assert {scenario.provider for scenario in scenarios} >= {"chatgpt", "codex"}
    assert {snapshot.provider for snapshot in listing.providers} >= {"chatgpt", "codex"}


def test_cluster_promotion_with_samples_matches_the_cluster_it_was_built_from(
    workspace_env: dict[str, Path],
) -> None:
    """Regression: promote_schema_cluster's with_samples path re-derives a
    fingerprint per candidate sample to find which ones belong to the
    requested cluster. schema_cluster_id() hashes (artifact_kind,
    structure_fingerprint) -- not the structure fingerprint alone -- and
    infer_schema(cluster=True) stamps every cluster "unspecified" (it never
    passes artifact_kinds through to cluster_samples()). Re-deriving sample
    fingerprints without threading that same artifact_kind through raised
    "No samples match cluster ..." for every real cluster ever produced by
    plain infer_schema(cluster=True) (verified live against gemini-cli's real
    exact-structure clusters before this fix).
    """
    index_db = _seed_chatgpt_raw(workspace_env)
    inferred = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=True))
    assert inferred.manifest is not None
    cluster_id = inferred.manifest.clusters[0].cluster_id
    assert inferred.manifest.clusters[0].artifact_kind == "unspecified"

    promoted = promote_schema_cluster(
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id=cluster_id,
            db_path=index_db,
            with_samples=True,
            max_samples=100,
        )
    )

    assert promoted.schema is not None


def test_infer_schema_normalizes_operator_privacy_configuration(workspace_env: dict[str, Path]) -> None:
    index_db = _seed_chatgpt_raw(workspace_env)
    privacy_payload: JSONDocument = {
        "level": "strict",
        "field_overrides": {"$.id": "drop", "invalid": 1},
        "allow_value_patterns": ["safe", 1],
        "deny_value_patterns": ["secret"],
        "safe_enum_max_length": "invalid",
        "high_entropy_min_length": 14,
        "cross_conv_min_count": 5,
        "cross_conv_proportional": True,
    }

    result = infer_schema(
        SchemaInferRequest(
            provider="chatgpt",
            db_path=index_db,
            privacy_config=privacy_payload,
        )
    )
    privacy = _privacy_config(privacy_payload)

    assert result.generation.success
    assert result.generation.schema is not None
    assert result.generation.schema["type"] == "object"
    assert privacy is not None
    assert privacy.level == "strict"
    assert privacy.safe_enum_max_length == 30
    assert privacy.high_entropy_min_length == 14
    assert privacy.cross_conv_min_count == 5
    assert privacy.cross_conv_proportional is True
    assert privacy.field_overrides == {"$.id": "drop"}
    assert privacy.allow_value_patterns == ["safe"]
    assert privacy.deny_value_patterns == ["secret"]


def test_operator_inference_reports_real_unknown_provider_and_audits_bundled_schema(
    workspace_env: dict[str, Path],
) -> None:
    index_db = db_setup(workspace_env)
    unknown = infer_schema(SchemaInferRequest(provider="not-a-provider", db_path=index_db))

    assert unknown.generation.success is False
    assert "Unknown provider" in (unknown.generation.error or "")
    with pytest.raises(ValueError, match="No cluster manifest"):
        promote_schema_cluster(
            SchemaPromoteRequest(provider="chatgpt", cluster_id="missing", db_path=index_db, with_samples=False)
        )

    report = audit_schemas(SchemaAuditRequest(provider="chatgpt"))
    assert report.provider == "chatgpt"
    assert report.checks
