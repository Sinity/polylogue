"""Executable proof evidence runners.

The runners in this module are deliberately small adapters: they evaluate one
obligation against already-selected command or semantic observations and return
an `EvidenceEnvelope`. They do not decide which obligations exist.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value
from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.models import EvidenceEnvelope, ProofObligation, SourceSpan, TrustMetadata
from polylogue.storage.sqlite.schema_ddl import SCHEMA_VERSION

_REVIEWED_AT = "2026-04-22T00:00:00+00:00"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_RUNNER_VERSION = "proof-runners.v1"


@dataclass(frozen=True, slots=True)
class SchemaValueGenerationObservation:
    """Observed generated values for a schema value-domain annotation."""

    observed_values: tuple[JSONValue, ...]
    schema_path: str
    generator: str = "hypothesis-jsonschema"


def run_schema_value_generation_evidence(
    obligation: ProofObligation,
    observation: SchemaValueGenerationObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_schema_provider_obligations.py"),
) -> EvidenceEnvelope:
    """Verify that generated schema values stay inside an annotated finite domain."""
    if obligation.claim.id != "schema.values.value_closure":
        raise ValueError(f"unsupported schema generation claim: {obligation.claim.id}")

    annotation_values = _json_value_tuple_attr(obligation.subject.attrs.get("values"))
    observed_values = tuple(
        require_json_value(value, context="schema generation observed value") for value in observation.observed_values
    )
    allowed_keys = {_json_value_key(value) for value in annotation_values}
    unexpected_values = tuple(value for value in observed_values if _json_value_key(value) not in allowed_keys)
    observed_state: JSONDocument = {
        "schema_path": observation.schema_path,
        "generator": observation.generator,
        "values": list(annotation_values),
        "observed_values": list(observed_values),
        "unexpected_values": list(unexpected_values),
    }
    expected_law = "values generated from the schema are members of the x-polylogue-values annotation set"
    evidence = _evidence_payload(
        obligation,
        runner_class="schema_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["values"] = list(annotation_values)
    evidence["observed_values"] = list(observed_values)
    evidence["schema_path"] = observation.schema_path
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if annotation_values and not unexpected_values else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.schema_static",
    )


def run_provider_capability_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_schema_provider_obligations.py"),
) -> EvidenceEnvelope:
    """Verify static provider capability metadata for a provider subject."""
    if obligation.claim.id == "provider.capability.identity_bridge":
        return _run_provider_identity_bridge_evidence(obligation, reproducer=reproducer)
    if obligation.claim.id == "provider.capability.partial_coverage_declared":
        return _run_provider_partial_coverage_evidence(obligation, reproducer=reproducer)
    raise ValueError(f"unsupported provider capability claim: {obligation.claim.id}")


def run_generated_scenario_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_generated_scenario_obligations.py"),
) -> EvidenceEnvelope:
    """Verify static generated-scenario proof subject metadata."""
    if obligation.claim.id == "generated.scenario.family_registered":
        return _run_generated_scenario_family_evidence(obligation, reproducer=reproducer)
    if obligation.claim.id == "generated.scenario.local_deterministic":
        return _run_generated_scenario_local_evidence(obligation, reproducer=reproducer)
    if obligation.claim.id == "generated.scenario.semantic_claim_mapping":
        return _run_generated_scenario_semantic_evidence(obligation, reproducer=reproducer)
    raise ValueError(f"unsupported generated scenario claim: {obligation.claim.id}")


def _run_generated_scenario_family_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...],
) -> EvidenceEnvelope:
    attrs = obligation.subject.attrs
    status = attrs.get("status")
    family_reproducer = _string_tuple_attr(attrs.get("reproducer"))
    checks: JSONDocument = {
        "name_present": bool(str(attrs.get("name", "")).strip()),
        "description_present": bool(str(attrs.get("description", "")).strip()),
        "status_valid": status in {"implemented", "migration_task"},
        "generated_world_present": bool(str(attrs.get("generated_world", "")).strip()),
        "workload_family_present": bool(str(attrs.get("workload_family", "")).strip()),
        "reproducer_present": bool(family_reproducer),
    }
    observed_state = require_json_document(
        {
            "name": attrs.get("name"),
            "status": status,
            "generated_world": attrs.get("generated_world"),
            "workload_family": attrs.get("workload_family"),
            "reproducer": list(family_reproducer),
            "checks": checks,
        },
        context="generated scenario observed state",
    )
    expected_law = "generated scenario families declare status, generated world, workload family, and reproducer"
    evidence = _evidence_payload(
        obligation,
        runner_class="generated_scenario_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["name"] = attrs.get("name")
    evidence["status"] = status
    evidence["generated_world"] = attrs.get("generated_world")
    evidence["workload_family"] = attrs.get("workload_family")
    evidence["evidence_note"] = (
        "metadata/spec completeness: checks that scenario family subject metadata "
        "is declared, not that the scenario executes"
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if all(bool(value) for value in checks.values()) else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.generated_scenario_static",
    )


def _run_generated_scenario_local_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...],
) -> EvidenceEnvelope:
    attrs = obligation.subject.attrs
    family_reproducer = _string_tuple_attr(attrs.get("reproducer"))
    checks: JSONDocument = {
        "status_implemented": attrs.get("status") == "implemented",
        "local_deterministic": attrs.get("local_deterministic") is True,
        "no_live_archive_dependency": attrs.get("live_archive_dependency") is False,
        "reproducer_present": bool(family_reproducer),
    }
    observed_state = require_json_document(
        {
            "status": attrs.get("status"),
            "local_deterministic": attrs.get("local_deterministic"),
            "live_archive_dependency": attrs.get("live_archive_dependency"),
            "reproducer": list(family_reproducer),
            "checks": checks,
        },
        context="generated scenario local deterministic observed state",
    )
    expected_law = "implemented generated scenario families are local, deterministic, and do not require live archives"
    evidence = _evidence_payload(
        obligation,
        runner_class="generated_scenario_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["local_deterministic"] = attrs.get("local_deterministic")
    evidence["live_archive_dependency"] = attrs.get("live_archive_dependency")
    evidence["reproducer"] = list(family_reproducer)
    evidence["evidence_note"] = (
        "metadata/spec completeness: checks that scenario family declares local/deterministic "
        "properties in subject metadata, not that the scenario executes"
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if all(bool(value) for value in checks.values()) else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.generated_scenario_static",
    )


def _run_generated_scenario_semantic_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...],
) -> EvidenceEnvelope:
    semantic_claims = _json_document_tuple_attr(obligation.subject.attrs.get("semantic_claims"))
    claim_states = tuple(str(claim.get("state", "")) for claim in semantic_claims)
    implemented = tuple(
        str(claim.get("family", "")) for claim in semantic_claims if claim.get("state") == "implemented"
    )
    mapped = tuple(str(claim.get("family", "")) for claim in semantic_claims if claim.get("state") == "mapped")
    mapped_missing_note = tuple(
        str(claim.get("family", ""))
        for claim in semantic_claims
        if claim.get("state") == "mapped" and not str(claim.get("mapping_note", "")).strip()
    )
    invalid_states = tuple(state for state in claim_states if state not in {"implemented", "mapped"})
    checks: JSONDocument = {
        "semantic_claims_present": bool(semantic_claims),
        "claim_states_valid": not invalid_states,
        "mapped_claims_explain_mapping": not mapped_missing_note,
    }
    observed_state: JSONDocument = {
        "semantic_claims": [dict(claim) for claim in semantic_claims],
        "implemented_claim_families": list(implemented),
        "mapped_claim_families": list(mapped),
        "invalid_states": list(invalid_states),
        "mapped_missing_note": list(mapped_missing_note),
        "checks": checks,
    }
    expected_law = "generated scenario families declare implemented or explicitly mapped semantic claim families"
    evidence = _evidence_payload(
        obligation,
        runner_class="generated_scenario_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["semantic_claims"] = [dict(claim) for claim in semantic_claims]
    evidence["implemented_claim_families"] = list(implemented)
    evidence["mapped_claim_families"] = list(mapped)
    evidence["evidence_note"] = (
        "metadata/spec completeness: checks that scenario family declares semantic claim "
        "mappings in subject metadata, not that the mapped behavior executes"
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if all(bool(value) for value in checks.values()) else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.generated_scenario_static",
    )


def _run_provider_identity_bridge_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...],
) -> EvidenceEnvelope:
    attrs = obligation.subject.attrs
    native_fields = _string_tuple_attr(attrs.get("native_identity_fields"))
    canonical_fields = _string_tuple_attr(attrs.get("canonical_identity_fields"))
    mappings = _json_document_tuple_attr(attrs.get("identity_mappings"))
    mapping_checks = tuple(
        bool(str(mapping.get("native", "")).strip() and str(mapping.get("canonical", "")).strip())
        for mapping in mappings
    )
    observed_state: JSONDocument = {
        "provider": attrs.get("provider"),
        "native_identity_fields": list(native_fields),
        "canonical_identity_fields": list(canonical_fields),
        "identity_mappings": [dict(mapping) for mapping in mappings],
        "mapping_checks": list(mapping_checks),
    }
    expected_law = "provider-native identity fields are explicitly mapped to canonical archive fields"
    evidence = _evidence_payload(
        obligation,
        runner_class="provider_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["native_identity_fields"] = list(native_fields)
    evidence["canonical_identity_fields"] = list(canonical_fields)
    evidence["identity_mappings"] = [dict(mapping) for mapping in mappings]
    ok = bool(native_fields and canonical_fields and mappings and all(mapping_checks))
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.provider_static",
    )


def _run_provider_partial_coverage_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...],
) -> EvidenceEnvelope:
    attrs = obligation.subject.attrs
    coverage_facets = require_json_document(attrs.get("coverage_facets"), context="coverage_facets")
    partial_coverage = _string_tuple_attr(attrs.get("partial_coverage"))
    required_facets = ("reasoning", "streaming", "sidecars", "tool_use", "native_identity", "timestamps")
    missing_facets = tuple(
        facet
        for facet in required_facets
        if not isinstance(coverage_facets.get(facet), str) or not str(coverage_facets.get(facet)).strip()
    )
    partial_or_absent = tuple(
        facet
        for facet in required_facets
        if isinstance(coverage_facets.get(facet), str) and coverage_facets[facet] in {"partial", "absent"}
    )
    observed_state: JSONDocument = {
        "provider": attrs.get("provider"),
        "reasoning_capability": attrs.get("reasoning_capability"),
        "streaming_capability": attrs.get("streaming_capability"),
        "sidecar_spec": attrs.get("sidecar_spec"),
        "coverage_facets": dict(coverage_facets),
        "partial_coverage": list(partial_coverage),
        "missing_facets": list(missing_facets),
        "partial_or_absent_facets": list(partial_or_absent),
    }
    expected_law = "partial or absent provider capability facets are explicit in provider metadata"
    evidence = _evidence_payload(
        obligation,
        runner_class="provider_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["reasoning_capability"] = attrs.get("reasoning_capability")
    evidence["streaming_capability"] = attrs.get("streaming_capability")
    evidence["sidecar_spec"] = attrs.get("sidecar_spec")
    evidence["coverage_facets"] = dict(coverage_facets)
    evidence["partial_coverage"] = list(partial_coverage)
    ok = not missing_facets and (not partial_or_absent or bool(partial_coverage))
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.provider_static",
    )


def _evidence_payload(
    obligation: ProofObligation,
    *,
    runner_class: str,
    expected_law: str,
    observed_state: JSONDocument,
) -> JSONDocument:
    return {
        "subject_id": obligation.subject.id,
        "claim_id": obligation.claim.id,
        "runner_id": obligation.runner.id,
        "runner_class": runner_class,
        "evidence_class": obligation.runner.evidence_class,
        "expected_law": expected_law,
        "observed_state": observed_state,
    }


def _build_envelope(
    obligation: ProofObligation,
    *,
    status: OutcomeStatus,
    evidence: JSONDocument,
    expected_law: str,
    observed_state: JSONDocument,
    reproducer: tuple[str, ...],
    provenance: SourceSpan | None,
    producer: str,
) -> EvidenceEnvelope:
    counterexample: JSONDocument | None = None
    if status is OutcomeStatus.ERROR:
        counterexample = {
            "subject_id": obligation.subject.id,
            "claim_id": obligation.claim.id,
            "runner_id": obligation.runner.id,
            "observed_state": observed_state,
            "expected_law": expected_law,
            "reproducer": list(reproducer),
        }
    environment_payload: JSONDocument = {
        "runner": obligation.runner.runner,
        "evidence_class": obligation.runner.evidence_class,
        "cost_tier": obligation.runner.cost_tier,
    }
    return EvidenceEnvelope.build(
        obligation_id=obligation.id,
        status=status,
        evidence=evidence,
        counterexample=counterexample,
        reproducer=reproducer,
        environment=environment_payload,
        provenance=provenance,
        trust=_runtime_trust_metadata(
            producer=producer,
            obligation=obligation,
            observed_state=observed_state,
            environment=environment_payload,
        ),
    )


def _runtime_trust_metadata(
    *,
    producer: str,
    obligation: ProofObligation,
    observed_state: JSONDocument,
    environment: JSONDocument,
) -> TrustMetadata:
    return TrustMetadata(
        producer=producer,
        reviewed_at=_REVIEWED_AT,
        level="generated",
        privacy="repo-local command metadata or seeded fixture ids only",
        code_revision=_git_revision(),
        dirty_state=_git_dirty_state(),
        schema_version=SCHEMA_VERSION,
        provider_schema_digest=_provider_schema_digest(obligation),
        input_fingerprint=_json_digest(observed_state),
        environment_fingerprint=_json_digest(environment),
        runner_version=_RUNTIME_RUNNER_VERSION,
        freshness="generated by proof runner against current checkout",
        origin="proof-runner",
    )


def _provider_schema_digest(obligation: ProofObligation) -> str | None:
    if obligation.subject.kind not in {"schema.annotation", "provider.capability"}:
        return None
    return _json_digest(obligation.subject.attrs)


def _git_revision() -> str | None:
    return _git_stdout(("rev-parse", "HEAD"))


def _git_dirty_state() -> bool | None:
    status = _git_stdout(("status", "--short"))
    return None if status is None else bool(status.strip())


def _git_stdout(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _json_digest(payload: JSONDocument) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _string_tuple_attr(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _json_value_tuple_attr(value: object) -> tuple[JSONValue, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(require_json_value(item, context="JSON value attribute") for item in value)


def _json_document_tuple_attr(value: object) -> tuple[JSONDocument, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        require_json_document(item, context="JSON document attribute") for item in value if isinstance(item, dict)
    )


def _json_value_key(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "SchemaValueGenerationObservation",
    "run_generated_scenario_evidence",
    "run_provider_capability_evidence",
    "run_schema_value_generation_evidence",
]
