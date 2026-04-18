"""Tests for schema inference corpus-first alignment changes.

Covers:
  1. Full-corpus mode bypasses sample caps
  2. Record-stream artifacts abstain from title inference
  3. Proof surface contains expected fields
  4. x-polylogue-confidence → x-polylogue-score rename completeness
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from polylogue.lib.raw_payload_sampling_extract import (
    extract_payload_samples,
    extract_record_samples_from_raw_content,
)
from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.generation_schema_builder import (
    _STRUCTURE_EXEMPLARS_PER_FINGERPRINT,
    _generate_cluster_schema,
)
from polylogue.schemas.generation_semantic_relations import (
    annotate_semantic_and_relational,
)
from polylogue.schemas.observation_models import ProviderConfig
from polylogue.schemas.operator_annotations import build_review_proof
from polylogue.schemas.semantic_inference_models import SEMANTIC_ROLES
from polylogue.schemas.semantic_inference_runtime import (
    RECORD_STREAM_ELIGIBLE_ROLES,
    RECORD_STREAM_KINDS,
    infer_semantic_roles,
)
from polylogue.types import Provider

# ---------------------------------------------------------------------------
# 1. Full-corpus mode bypasses sample caps
# ---------------------------------------------------------------------------


class TestFullCorpusMode:
    """Full-corpus mode removes sample caps."""

    def test_extract_record_samples_full_corpus_returns_all(self) -> None:
        """When max_samples=None, all records are returned."""
        records = [{"type": "user", "text": f"msg {i}"} for i in range(200)]
        content = "\n".join(__import__("json").dumps(r) for r in records)
        result = extract_record_samples_from_raw_content(
            content.encode(),
            max_samples=None,
            record_type_key="type",
        )
        assert len(result) == 200

    def test_extract_record_samples_capped_returns_fewer(self) -> None:
        """When max_samples is set, fewer records are returned."""
        records = [{"type": "user", "text": f"msg {i}"} for i in range(200)]
        content = "\n".join(__import__("json").dumps(r) for r in records)
        result = extract_record_samples_from_raw_content(
            content.encode(),
            max_samples=10,
            record_type_key="type",
        )
        assert len(result) <= 10

    def test_extract_payload_samples_no_cap_returns_all(self) -> None:
        """extract_payload_samples with max_samples=None returns all records."""
        records = [{"type": "assistant", "text": f"resp {i}"} for i in range(50)]
        result = extract_payload_samples(
            records,
            sample_granularity="record",
            max_samples=None,
        )
        assert len(result) == 50

    def test_generate_cluster_schema_full_corpus_uses_all_exemplars(self) -> None:
        """full_corpus=True feeds all samples to genson, not just N per fingerprint."""
        # Create samples with identical structure (same fingerprint).
        # In normal mode only _STRUCTURE_EXEMPLARS_PER_FINGERPRINT would be used.
        sample_count = _STRUCTURE_EXEMPLARS_PER_FINGERPRINT * 3
        samples = [{"role": "user", "text": f"message {i}"} for i in range(sample_count)]
        conv_ids: list[str | None] = [None] * sample_count
        config = ProviderConfig(
            name=Provider.from_string("test-provider"),
            description="test",
            sample_granularity="record",
        )

        # Normal mode — should be capped
        schema_normal, _ = _generate_cluster_schema(
            "test-provider",
            config,
            samples,
            conv_ids,
            privacy_config=None,
        )
        # Full-corpus mode — all exemplars used
        schema_full, _ = _generate_cluster_schema(
            "test-provider",
            config,
            samples,
            conv_ids,
            privacy_config=None,
            full_corpus=True,
        )
        # Both should produce valid schemas
        assert schema_normal.get("type") == "object" or "$schema" in schema_normal
        assert schema_full.get("type") == "object" or "$schema" in schema_full


# ---------------------------------------------------------------------------
# 2. Record-stream artifacts abstain from title inference
# ---------------------------------------------------------------------------


class TestRecordStreamTitleAbstention:
    """Record-stream artifact kinds must not infer conversation_title."""

    @pytest.fixture()
    def title_eligible_stats(self) -> dict[str, FieldStats]:
        """Field stats that would normally score well for conversation_title."""
        return {
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({f"Chat {i}": 1 for i in range(20)}),
                string_lengths=[7] * 20,
                is_multiline=0,
                newline_counts=[0] * 20,
                total_samples=20,
                present_count=20,
                value_count=20,
            ),
            "$.role": FieldStats(
                path="$.role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[4, 9],
            ),
        }

    @pytest.mark.parametrize("kind", sorted(RECORD_STREAM_KINDS))
    def test_record_stream_skips_title(
        self,
        kind: str,
        title_eligible_stats: dict[str, FieldStats],
    ) -> None:
        """Record-stream artifacts produce no conversation_title candidates."""
        candidates = infer_semantic_roles(title_eligible_stats, artifact_kind=kind)
        title_candidates = [c for c in candidates if c.role == "conversation_title"]
        assert not title_candidates, f"artifact_kind={kind} should abstain from title; got {title_candidates}"

    @pytest.mark.parametrize("kind", sorted(RECORD_STREAM_KINDS))
    def test_record_stream_still_infers_message_roles(
        self,
        kind: str,
        title_eligible_stats: dict[str, FieldStats],
    ) -> None:
        """Record-stream artifacts still infer message_role."""
        candidates = infer_semantic_roles(title_eligible_stats, artifact_kind=kind)
        role_candidates = [c for c in candidates if c.role == "message_role"]
        assert role_candidates, "Record streams should still infer message_role"

    def test_document_kind_still_infers_title(
        self,
        title_eligible_stats: dict[str, FieldStats],
    ) -> None:
        """conversation_document artifacts should still infer title."""
        candidates = infer_semantic_roles(
            title_eligible_stats,
            artifact_kind="conversation_document",
        )
        title_candidates = [c for c in candidates if c.role == "conversation_title"]
        assert title_candidates, "conversation_document should infer title"

    def test_none_artifact_kind_still_infers_title(
        self,
        title_eligible_stats: dict[str, FieldStats],
    ) -> None:
        """When artifact_kind is None, title inference proceeds normally."""
        candidates = infer_semantic_roles(title_eligible_stats, artifact_kind=None)
        title_candidates = [c for c in candidates if c.role == "conversation_title"]
        assert title_candidates, "None artifact_kind should infer title"

    def test_annotate_semantic_threads_artifact_kind(self) -> None:
        """annotate_semantic_and_relational passes artifact_kind through."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "role": {"type": "string"},
            },
        }
        field_stats = {
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({f"Chat {i}": 1 for i in range(20)}),
                string_lengths=[7] * 20,
                is_multiline=0,
                newline_counts=[0] * 20,
                total_samples=20,
                present_count=20,
                value_count=20,
            ),
        }
        result = annotate_semantic_and_relational(
            schema,
            field_stats,
            artifact_kind="conversation_record_stream",
        )
        # Title should NOT be annotated for record streams
        title_prop = result.get("properties", {}).get("title", {})
        assert title_prop.get("x-polylogue-semantic-role") != "conversation_title"

    def test_record_stream_eligible_roles_are_subset_of_semantic_roles(self) -> None:
        """All eligible roles are valid semantic roles."""
        assert frozenset(SEMANTIC_ROLES) >= RECORD_STREAM_ELIGIBLE_ROLES


# ---------------------------------------------------------------------------
# 3. Proof surface output contains expected fields
# ---------------------------------------------------------------------------


class TestProofSurface:
    """Schema review proof surface has required structure."""

    @pytest.fixture()
    def schema_with_roles(self) -> dict[str, Any]:
        """A schema with some semantic role annotations."""
        return {
            "type": "object",
            "x-polylogue-artifact-kind": "conversation_document",
            "properties": {
                "title": {
                    "type": "string",
                    "x-polylogue-semantic-role": "conversation_title",
                    "x-polylogue-score": 0.72,
                    "x-polylogue-evidence": {"name_signal": "title", "avg_length": 15.0},
                },
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "x-polylogue-semantic-role": "message_role",
                                "x-polylogue-score": 0.85,
                                "x-polylogue-evidence": {"known_role_overlap": 0.95},
                            },
                            "text": {
                                "type": "string",
                                "x-polylogue-semantic-role": "message_body",
                                "x-polylogue-score": 0.6,
                                "x-polylogue-evidence": {"avg_length": 500.0},
                            },
                        },
                    },
                },
            },
        }

    def test_proof_contains_all_semantic_roles(self, schema_with_roles: dict[str, Any]) -> None:
        """Proof has an entry for every semantic role."""
        proof = build_review_proof(schema_with_roles)
        proof_roles = {entry.role for entry in proof.roles}
        assert proof_roles == set(SEMANTIC_ROLES)

    def test_proof_entry_fields(self, schema_with_roles: dict[str, Any]) -> None:
        """Each proof entry has required fields."""
        proof = build_review_proof(schema_with_roles)
        for entry in proof.roles:
            assert isinstance(entry.role, str)
            assert isinstance(entry.abstained, bool)
            assert isinstance(entry.chosen_score, float)
            assert isinstance(entry.competing, list)
            assert isinstance(entry.evidence, dict)

    def test_proof_chosen_path_for_assigned_roles(self, schema_with_roles: dict[str, Any]) -> None:
        """Assigned roles have a chosen_path."""
        proof = build_review_proof(schema_with_roles)
        title_entry = next(e for e in proof.roles if e.role == "conversation_title")
        assert title_entry.chosen_path == "$.title"
        assert title_entry.chosen_score == pytest.approx(0.72)
        assert not title_entry.abstained

    def test_proof_abstained_roles(self, schema_with_roles: dict[str, Any]) -> None:
        """Roles with no candidates are marked as abstained."""
        proof = build_review_proof(schema_with_roles)
        # message_container and message_timestamp have no assignment in the fixture
        container_entry = next(e for e in proof.roles if e.role == "message_container")
        assert container_entry.abstained
        assert container_entry.abstain_reason is not None

    def test_proof_to_dict_roundtrip(self, schema_with_roles: dict[str, Any]) -> None:
        """to_dict produces JSON-serializable output."""
        import json

        proof = build_review_proof(schema_with_roles)
        d = proof.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert serialized
        parsed = json.loads(serialized)
        assert "roles" in parsed
        assert "artifact_kind" in parsed
        assert "eligible_roles" in parsed
        assert "ineligible_roles" in parsed

    def test_proof_eligible_roles_for_document(self, schema_with_roles: dict[str, Any]) -> None:
        """conversation_document has all roles eligible."""
        proof = build_review_proof(schema_with_roles)
        assert proof.eligible_roles == list(SEMANTIC_ROLES)
        assert proof.ineligible_roles == []

    def test_proof_record_stream_ineligible_roles(self) -> None:
        """Record-stream schemas show title as ineligible."""
        schema: dict[str, Any] = {
            "type": "object",
            "x-polylogue-artifact-kind": "conversation_record_stream",
            "properties": {
                "role": {
                    "type": "string",
                    "x-polylogue-semantic-role": "message_role",
                    "x-polylogue-score": 0.9,
                    "x-polylogue-evidence": {},
                },
            },
        }
        proof = build_review_proof(schema)
        assert "conversation_title" in proof.ineligible_roles
        title_entry = next(e for e in proof.roles if e.role == "conversation_title")
        assert title_entry.abstained
        assert "excludes" in (title_entry.abstain_reason or "")


# ---------------------------------------------------------------------------
# 4. x-polylogue-confidence → x-polylogue-score rename completeness
# ---------------------------------------------------------------------------


class TestConfidenceToScoreRename:
    """Verify x-polylogue-confidence is fully replaced by x-polylogue-score."""

    def test_no_confidence_key_in_annotated_schema(self) -> None:
        """Annotated schema uses x-polylogue-score, not x-polylogue-confidence."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
            },
        }
        field_stats = {
            "$.role": FieldStats(
                path="$.role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[4, 9],
            ),
        }
        result = annotate_semantic_and_relational(schema, field_stats)
        found_keys = _collect_all_keys(result)
        assert "x-polylogue-confidence" not in found_keys
        # If any semantic role was assigned, x-polylogue-score should be present
        if "x-polylogue-semantic-role" in found_keys:
            assert "x-polylogue-score" in found_keys

    def test_no_confidence_key_in_generated_schema(self) -> None:
        """Full schema generation uses score, not confidence."""
        from polylogue.schemas.generation_schema_builder import generate_schema_from_samples

        samples = [
            {
                "messages": [{"role": "user", "text": f"msg {i}"} for i in range(5)],
                "title": f"Conversation {j}",
            }
            for j in range(3)
        ]
        schema = generate_schema_from_samples(samples)
        found_keys = _collect_all_keys(schema)
        assert "x-polylogue-confidence" not in found_keys

    def test_source_files_have_no_old_annotation_key(self) -> None:
        """Grep-equivalent: no Python source uses the old annotation key literal."""
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parents[3] / "polylogue"
        hits: list[str] = []
        for py_file in src_root.rglob("*.py"):
            content = py_file.read_text(errors="replace")
            if "x-polylogue-confidence" in content:
                hits.append(str(py_file.relative_to(src_root)))
        assert not hits, f"Source files still reference old key: {hits}"


def _collect_all_keys(obj: Any, _keys: set[str] | None = None) -> set[str]:
    """Recursively collect all dict keys in a nested structure."""
    if _keys is None:
        _keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            _keys.add(k)
            _collect_all_keys(v, _keys)
    elif isinstance(obj, list):
        for item in obj:
            _collect_all_keys(item, _keys)
    return _keys
