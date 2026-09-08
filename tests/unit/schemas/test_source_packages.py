"""Package identity must not depend on changing observation statistics."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from polylogue.schemas.generation import workflow
from polylogue.schemas.generation.evidence import collect_source_evidence
from polylogue.schemas.source_inference import SchemaSourceInput, SourceInferenceResult, SourceObservation


def source_result(
    *, records: int, session_kind: str = "session_document", extra_field: bool = False
) -> SourceInferenceResult:
    elements = {}
    for kind, count in ((session_kind, records), ("agent_sidecar_meta", 1)):
        observation = SourceObservation(
            logical_source_id="synthetic-session",
            revision_sha256="a" * 64,
            subject="claude-code",
            element_kind=kind,
            records=[
                {"value": "x" * records, **({"new_field": True} if extra_field and kind == session_kind else {})}
                for _ in range(count)
            ],
        )
        elements[kind] = (collect_source_evidence(observation).to_json(),)
    return SourceInferenceResult(
        evidence_by_element=elements,
        terminal_counts={"included": 1},
        candidate_terminal_counts={"included": 1},
        terminal_reason_counts={},
        input_bytes=records * 100,
        record_count=records + 1,
        cache_hits=0,
        cache_misses=1,
        phase_timings_ms={},
        producer_version_counts={},
        producer_version_missing_sources=1,
        producer_version_conflicting_sources=0,
        producer_version_unrecognized_sources=0,
        input_manifest_digest="a" * 64,
        candidate_count=1,
        included_candidate_count=1,
        included_native_source_revision_count=1,
    )


@pytest.mark.parametrize("session_kind", ["session_document", "session_record_stream"])
def test_statistics_change_keeps_version_and_element_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_kind: str
) -> None:
    """Hashing emitted annotations reallocates v1 when only lengths/counts change."""
    current = source_result(records=2, session_kind=session_kind)
    unchanged = copy.deepcopy(current.evidence_by_element)
    monkeypatch.setattr(workflow, "infer_sources", lambda *_args, **_kwargs: current)
    inputs = (SchemaSourceInput("claude-code", tmp_path),)
    first = workflow.build_provider_bundle_from_sources(
        "claude-code",
        source_inputs=inputs,
        cache_path=None,
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert first.catalog is not None
    assert current.evidence_by_element == unchanged
    workflow.generate_provider_schema_from_sources(
        "claude-code",
        source_inputs=inputs,
        cache_path=None,
        max_workers=1,
        privacy_config=None,
    )
    assert current.evidence_by_element == unchanged
    current = source_result(records=5, session_kind=session_kind)
    second = workflow.build_provider_bundle_from_sources(
        "claude-code",
        source_inputs=inputs,
        cache_path=None,
        max_workers=1,
        privacy_config=None,
        prior_catalog=first.catalog,
    )
    assert second.catalog is not None
    assert first.result.versions == second.result.versions == ["v1"]
    package = second.catalog.packages[0]
    assert package.anchor_kind == package.default_element_kind == session_kind
    assert package.sample_count == 5
    assert package.bundle_scope_count == 1
    assert {item.element_kind: item.sample_count for item in package.elements} == {
        session_kind: 5,
        "agent_sidecar_meta": 1,
    }
    assert second.result.schema != first.result.schema
    preview = workflow.generate_provider_schema_from_sources(
        "claude-code",
        source_inputs=inputs,
        cache_path=None,
        max_workers=1,
        privacy_config=None,
    )
    assert preview.schema == second.result.schema
    assert preview.artifact_counts == second.result.artifact_counts

    current = source_result(records=5, session_kind=session_kind, extra_field=True)
    changed = workflow.build_provider_bundle_from_sources(
        "claude-code",
        source_inputs=inputs,
        cache_path=None,
        max_workers=1,
        privacy_config=None,
        prior_catalog=second.catalog,
    )
    assert changed.result.versions == ["v2"]
