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


def test_infer_schema_builds_schema_from_source_tier_raw(workspace_env: dict[str, Path]) -> None:
    index_db = _seed_chatgpt_raw(workspace_env)

    result = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=False))

    assert result.generation.success
    assert result.generation.sample_count == 1
    assert result.generation.schema is not None
    assert result.manifest is None
    assert len(result.corpus_specs) == 1
    assert len(result.corpus_scenarios) == 1


def test_cluster_promotion_drives_real_operator_registry_views(workspace_env: dict[str, Path]) -> None:
    index_db = _seed_chatgpt_raw(workspace_env)
    inferred = infer_schema(SchemaInferRequest(provider="chatgpt", db_path=index_db, cluster=True))

    assert inferred.generation.success
    assert inferred.manifest is not None
    cluster_id = inferred.manifest.clusters[0].cluster_id
    promoted = promote_schema_cluster(
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id=cluster_id,
            db_path=index_db,
            with_samples=False,
        )
    )

    assert promoted.package_version == "v2"
    assert promoted.schema is not None
    specs = list_inferred_corpus_specs(provider="chatgpt")
    registry = SchemaRegistry()
    registry.register_schema("chatgpt", {"type": "object", "properties": {"mapping": {"type": "object"}}})
    registry.register_schema("codex", {"type": "object", "properties": {"session_id": {"type": "string"}}})
    scenarios = list_inferred_corpus_scenarios()
    comparison = compare_schema_versions(SchemaCompareRequest(provider="chatgpt", from_version="v2", to_version="v3"))
    selected = list_schemas(SchemaListRequest(provider="chatgpt"))

    assert comparison.diff.version_a == "v2"
    assert comparison.diff.version_b == "v3"
    assert selected.selected is not None
    assert selected.selected.versions == ["v1", "v2", "v3"]
    assert {spec.provider for spec in specs} == {"chatgpt"}
    assert specs[0].package_version == "v2"
    assert {scenario.provider for scenario in scenarios} >= {"chatgpt", "codex"}


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
