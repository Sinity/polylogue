from __future__ import annotations

import copy
import json

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

from polylogue.core.json import JSONValue, json_document, require_json_value
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import ProofObligation
from polylogue.proof.runners import (
    SchemaValueGenerationObservation,
    run_provider_capability_evidence,
    run_schema_value_generation_evidence,
)
from tests.infra.strategies.schema_driven import strip_schema_extensions


def _obligation(claim_id: str, *, subject_id: str | None = None) -> ProofObligation:
    catalog = build_verification_catalog()
    for obligation in catalog.obligations:
        if obligation.claim.id != claim_id:
            continue
        if subject_id is not None and obligation.subject.id != subject_id:
            continue
        return obligation
    raise AssertionError(f"missing obligation for claim={claim_id!r} subject={subject_id!r}")


def test_provider_capability_subjects_compile_into_provider_claims() -> None:
    catalog = build_verification_catalog()
    provider_subject_ids = {subject.id for subject in catalog.subjects if subject.kind == "provider.capability"}

    assert {"provider.capability.codex", "provider.capability.claude-code"}.issubset(provider_subject_ids)
    assert len(provider_subject_ids) >= 2
    assert catalog.obligations_by_claim()["provider.capability.identity_bridge"] == len(provider_subject_ids)
    assert catalog.obligations_by_claim()["provider.capability.partial_coverage_declared"] == len(provider_subject_ids)


def test_provider_capability_identity_runner_preserves_native_and_canonical_facts() -> None:
    envelope = run_provider_capability_evidence(
        _obligation("provider.capability.identity_bridge", subject_id="provider.capability.codex")
    )

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "provider_static"
    assert "git.repository_url" in _string_values(envelope.evidence["native_identity_fields"])
    assert "Conversation.provider_meta.git.repository_url" in _string_values(
        envelope.evidence["canonical_identity_fields"]
    )
    assert envelope.counterexample is None


def test_provider_capability_partial_coverage_runner_requires_explicit_gaps() -> None:
    envelope = run_provider_capability_evidence(
        _obligation("provider.capability.partial_coverage_declared", subject_id="provider.capability.chatgpt")
    )

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "provider_static"
    assert "streaming_capability_absent" in _string_values(envelope.evidence["partial_coverage"])
    assert "sidecar_spec_absent" in _string_values(envelope.evidence["partial_coverage"])
    assert envelope.counterexample is None


@given(data=st.data())
@settings(deadline=None)
def test_value_annotation_metadata_drives_generation_and_verification(data: st.DataObject) -> None:
    obligation = _obligation("schema.values.value_closure")
    annotation_values = _json_values_attr(obligation.subject.attrs.get("values"))
    generator_raw_schema: JSONValue = {"x-polylogue-values": list(annotation_values)}
    generator_schema = json_document(strip_schema_extensions(copy.deepcopy(generator_raw_schema)))

    observed_values = tuple(
        require_json_value(data.draw(from_schema(generator_schema)), context="generated schema value") for _ in range(5)
    )
    envelope = run_schema_value_generation_evidence(
        obligation,
        SchemaValueGenerationObservation(
            observed_values=observed_values,
            schema_path=str(obligation.subject.attrs.get("schema_path", "")),
        ),
    )

    assert {_json_value_key(value) for value in observed_values}.issubset(
        {_json_value_key(value) for value in annotation_values}
    )
    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "schema_static"
    assert envelope.counterexample is None


def test_schema_value_generation_runner_reports_counterexamples() -> None:
    obligation = _obligation("schema.values.value_closure")
    annotation_values = _json_values_attr(obligation.subject.attrs.get("values"))
    unexpected_value = _outside_value(annotation_values)

    envelope = run_schema_value_generation_evidence(
        obligation,
        SchemaValueGenerationObservation(
            observed_values=(unexpected_value,),
            schema_path=str(obligation.subject.attrs.get("schema_path", "")),
        ),
    )

    assert envelope.status is OutcomeStatus.ERROR
    assert envelope.counterexample is not None
    observed_state = envelope.counterexample["observed_state"]
    assert isinstance(observed_state, dict)
    assert unexpected_value in _string_values(observed_state["unexpected_values"])


def _json_values_attr(value: object) -> tuple[JSONValue, ...]:
    if not isinstance(value, list):
        raise AssertionError("expected JSON value list")
    return tuple(require_json_value(item, context="values item") for item in value)


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AssertionError("expected string list")
    return tuple(item for item in value if isinstance(item, str))


def _outside_value(values: tuple[JSONValue, ...]) -> str:
    value_keys = {_json_value_key(value) for value in values}
    candidate = "__polylogue_outside_value__"
    while _json_value_key(candidate) in value_keys:
        candidate = f"{candidate}_x"
    return candidate


def _json_value_key(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
