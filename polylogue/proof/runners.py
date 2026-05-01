"""Executable proof evidence runners.

The runners in this module are deliberately small adapters: they evaluate one
obligation against already-selected command or semantic observations and return
an `EvidenceEnvelope`. They do not decide which obligations exist.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner, Result

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.models import EvidenceEnvelope, ProofObligation, SourceSpan, TrustMetadata
from polylogue.proof.traces import ObservableDiagnosticMapping, ObservableTrace, trace_signature_hash
from polylogue.storage.backends.schema_ddl import SCHEMA_VERSION

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\[.*?m")
_TRACEBACK_MARKER = "Traceback (most recent call last)"
_REVIEWED_AT = "2026-04-22T00:00:00+00:00"
_OUTPUT_SAMPLE_LIMIT = 1_000
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_RUNNER_VERSION = "proof-runners.v1"


@dataclass(frozen=True, slots=True)
class SemanticQueryObservation:
    """Observed facts for a provider-filter semantic query law."""

    provider: str
    all_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    provider_count: int
    equivalent_provider_ids: tuple[str, ...]
    query_name: str = "provider-filter"
    surface_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SchemaValueGenerationObservation:
    """Observed generated values for a schema value-domain annotation."""

    observed_values: tuple[JSONValue, ...]
    schema_path: str
    generator: str = "hypothesis-jsonschema"


@dataclass(frozen=True, slots=True)
class MaintenanceRepairObservation:
    """Observed repair/cleanup transition facts for a maintenance target."""

    target: str
    before_count: int
    preview_repaired_count: int
    after_dry_run_count: int
    repaired_count: int
    after_count: int
    second_repair_count: int
    state_effect: str
    destructive: bool
    result_success: bool = True
    failure_state: str | None = None
    operation: str = "polylogue doctor --repair"


@dataclass(frozen=True, slots=True)
class QuarantineErrorObservation:
    """Observed parser quarantine diagnostic facts with privacy witnesses."""

    provider: str
    source_path: str
    raw_id: str
    parse_error: str
    machine_payload: JSONDocument
    user_message: str
    payload_fragments: tuple[str, ...] = ()
    validation_status: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorContextObservation:
    """Observed generic machine/user error context for a proof surface."""

    error_family: str
    machine_payload: JSONDocument
    user_message: str
    required_context: tuple[str, ...] = ()
    payload_fragments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceEquivalenceObservation:
    """Observed cross-surface semantic traces for one operation."""

    traces: tuple[ObservableTrace, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticTraceMappingObservation:
    """Observed mapping from a diagnostic shape into the trace vocabulary."""

    mapping: ObservableDiagnosticMapping


_STATE_EFFECT_VALUES = {"unchanged", "changed", "rolled_back", "partially_changed"}


def run_cli_visual_evidence(
    obligation: ProofObligation,
    *,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    root_command: click.Command | None = None,
) -> EvidenceEnvelope:
    """Run a help/no-traceback/plain-mode CLI obligation."""
    if obligation.claim.id == "cli.command.help":
        return _run_cli_help_evidence(obligation, args=args, env=env, root_command=root_command)
    if obligation.claim.id == "cli.command.no_traceback":
        return _run_cli_no_traceback_evidence(obligation, args=args, env=env, root_command=root_command)
    if obligation.claim.id == "cli.command.plain_mode":
        return _run_cli_plain_mode_evidence(obligation, args=args, env=env, root_command=root_command)
    raise ValueError(f"unsupported CLI visual claim: {obligation.claim.id}")


def run_cli_json_envelope_evidence(
    obligation: ProofObligation,
    *,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    root_command: click.Command | None = None,
) -> EvidenceEnvelope:
    """Run a JSON-capable command and verify the success envelope shape."""
    command_args = tuple(args) if args is not None else _json_args_for_subject(obligation)
    result = _invoke_cli(command_args, env=env, root_command=root_command)
    parsed, parse_error = _parse_json_object(result.output)
    json_status = parsed.get("status") if parsed is not None else None
    json_result = parsed.get("result") if parsed is not None else None
    observed_state = _cli_result_state(
        result,
        command_args=command_args,
        extra={
            "json_status": json_status,
            "json_result_type": type(json_result).__name__ if json_result is not None else None,
            "parse_error": parse_error,
            "parsed_keys": sorted(parsed) if parsed is not None else [],
        },
    )
    expected_law = "command exits zero and emits a JSON object with status=ok and object result"
    ok = result.exit_code == 0 and parse_error is None and json_status == "ok" and isinstance(json_result, dict)
    evidence = _evidence_payload(
        obligation,
        runner_class="cli_json",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["json_status"] = json_status
    evidence["json_result_type"] = type(json_result).__name__ if json_result is not None else None
    evidence["parse_error"] = parse_error
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=("polylogue", *command_args),
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.cli_json",
    )


def run_semantic_query_evidence(
    obligation: ProofObligation,
    observation: SemanticQueryObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_evidence_runners.py"),
) -> EvidenceEnvelope:
    """Verify provider-filter query laws from observed semantic facts."""
    all_ids = _sorted_unique(observation.all_ids)
    provider_ids = _sorted_unique(observation.provider_ids)
    equivalent_provider_ids = _sorted_unique(observation.equivalent_provider_ids)
    law_results: JSONDocument = {
        "provider_subset": set(provider_ids).issubset(set(all_ids)),
        "provider_count_matches_list": observation.provider_count == len(provider_ids),
        "equivalent_provider_filter_constructions": equivalent_provider_ids == provider_ids,
    }
    observed_state: JSONDocument = {
        "query_name": observation.query_name,
        "provider": observation.provider,
        "all_ids": list(all_ids),
        "provider_ids": list(provider_ids),
        "provider_count": observation.provider_count,
        "equivalent_provider_ids": list(equivalent_provider_ids),
        "surface_names": list(observation.surface_names),
        "law_results": law_results,
    }
    expected_law = (
        "provider-filter ids are a subset of all ids, count(provider) equals len(list(provider)), "
        "and equivalent provider-filter constructions return identical ids"
    )
    evidence = _evidence_payload(
        obligation,
        runner_class="semantic_query",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["all_ids"] = list(all_ids)
    evidence["provider_ids"] = list(provider_ids)
    evidence["provider_count"] = observation.provider_count
    evidence["equivalent_provider_ids"] = list(equivalent_provider_ids)
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if all(law_results.values()) else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.semantic_query",
    )


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


def run_artifact_path_evidence(
    obligation: ProofObligation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
) -> EvidenceEnvelope:
    """Verify static structural closure for one runtime artifact path."""
    if obligation.claim.id != "artifact.path.dependency_closure":
        raise ValueError(f"unsupported artifact path claim: {obligation.claim.id}")

    attrs = obligation.subject.attrs
    nodes = _string_tuple_attr(attrs.get("nodes"))
    layers = require_json_document(attrs.get("layers"), context="artifact path layers")
    missing_dependencies = _string_tuple_attr(attrs.get("missing_dependencies"))
    operation_targets = _string_tuple_attr(attrs.get("operation_targets"))
    external_dependencies = _string_tuple_attr(attrs.get("external_dependencies"))
    layer_values = {value for value in layers.values() if isinstance(value, str)}
    has_durable_layer = "durable" in layer_values
    has_non_core_layer = bool({"derived", "index", "projection"} & layer_values)
    observed_state: JSONDocument = {
        "path_name": attrs.get("path_name"),
        "nodes": list(nodes),
        "layers": dict(layers),
        "missing_dependencies": list(missing_dependencies),
        "external_dependencies": list(external_dependencies),
        "operation_targets": list(operation_targets),
        "has_durable_layer": has_durable_layer,
        "has_non_core_layer": has_non_core_layer,
    }
    expected_law = (
        "runtime artifact paths resolve dependencies and include durable plus derived/index/projection semantics"
    )
    evidence = _evidence_payload(
        obligation,
        runner_class="artifact_path_static",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["path_name"] = attrs.get("path_name")
    evidence["nodes"] = list(nodes)
    evidence["layers"] = dict(layers)
    evidence["missing_dependencies"] = list(missing_dependencies)
    evidence["external_dependencies"] = list(external_dependencies)
    ok = bool(nodes) and not missing_dependencies and has_durable_layer and has_non_core_layer
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.artifact_path_static",
    )


def run_maintenance_repair_state_evidence(
    obligation: ProofObligation,
    observation: MaintenanceRepairObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
) -> EvidenceEnvelope:
    """Verify maintenance repair evidence states dry-run, execution, and failure effects."""
    if obligation.claim.id != "maintenance.repair.crash_consistency":
        raise ValueError(f"unsupported maintenance repair claim: {obligation.claim.id}")

    expected_target = str(obligation.subject.attrs.get("name", ""))
    target_matches = observation.target == expected_target
    dry_run_unchanged = observation.before_count == observation.after_dry_run_count
    preview_matches_execution = not observation.result_success or (
        observation.preview_repaired_count == observation.repaired_count
    )
    state_effect_explicit = observation.state_effect in _STATE_EFFECT_VALUES
    failure_state_explicit = observation.result_success or observation.failure_state in _STATE_EFFECT_VALUES
    converged_or_explained = observation.result_success or observation.after_count == observation.before_count
    idempotent_after_success = not observation.result_success or observation.second_repair_count == 0
    observed_state: JSONDocument = {
        "target": observation.target,
        "expected_target": expected_target,
        "destructive": observation.destructive,
        "before_count": observation.before_count,
        "preview_repaired_count": observation.preview_repaired_count,
        "after_dry_run_count": observation.after_dry_run_count,
        "repaired_count": observation.repaired_count,
        "after_count": observation.after_count,
        "second_repair_count": observation.second_repair_count,
        "state_effect": observation.state_effect,
        "result_success": observation.result_success,
        "failure_state": observation.failure_state,
        "operation": observation.operation,
        "checks": {
            "target_matches": target_matches,
            "dry_run_unchanged": dry_run_unchanged,
            "preview_matches_execution": preview_matches_execution,
            "state_effect_explicit": state_effect_explicit,
            "failure_state_explicit": failure_state_explicit,
            "converged_or_explained": converged_or_explained,
            "idempotent_after_success": idempotent_after_success,
        },
    }
    expected_law = "maintenance evidence explicitly states preview, execution, idempotence, and failure state effects"
    evidence = _evidence_payload(
        obligation,
        runner_class="maintenance_repair_state",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["target"] = observation.target
    evidence["preview_repaired_count"] = observation.preview_repaired_count
    evidence["repaired_count"] = observation.repaired_count
    evidence["state_effect"] = observation.state_effect
    evidence["failure_state"] = observation.failure_state
    ok = all(
        (
            target_matches,
            dry_run_unchanged,
            preview_matches_execution,
            state_effect_explicit,
            failure_state_explicit,
            converged_or_explained,
            idempotent_after_success,
        )
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.maintenance_repair_state",
    )


def run_quarantine_error_evidence(
    obligation: ProofObligation,
    observation: QuarantineErrorObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
) -> EvidenceEnvelope:
    """Verify parser quarantine diagnostics retain context without leaking payload text."""
    if obligation.claim.id != "parser.quarantine.context_redaction":
        raise ValueError(f"unsupported parser quarantine claim: {obligation.claim.id}")

    payload_leak_detected = _payload_leak_detected(
        observation.payload_fragments,
        observation.parse_error,
        observation.user_message,
        json.dumps(observation.machine_payload, sort_keys=True),
    )
    machine_details = _machine_details(observation.machine_payload)
    context_checks: JSONDocument = {
        "provider": bool(observation.provider) and machine_details.get("provider") == observation.provider,
        "source_path": bool(observation.source_path) and machine_details.get("source_path") == observation.source_path,
        "raw_id": bool(observation.raw_id) and machine_details.get("raw_id") == observation.raw_id,
    }
    user_context_checks: JSONDocument = {
        "provider": observation.provider in observation.user_message,
        "source_path": observation.source_path in observation.user_message,
    }
    observed_state: JSONDocument = {
        "provider": observation.provider,
        "source_path": observation.source_path,
        "raw_id": observation.raw_id,
        "parse_error": observation.parse_error,
        "validation_status": observation.validation_status,
        "machine_payload": dict(observation.machine_payload),
        "user_message": observation.user_message,
        "context_checks": context_checks,
        "user_context_checks": user_context_checks,
        "payload_leak_detected": payload_leak_detected,
    }
    expected_law = "parser quarantine diagnostics preserve source context and redact private payload fragments"
    evidence = _evidence_payload(
        obligation,
        runner_class="parser_quarantine_error",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["provider"] = observation.provider
    evidence["source_path"] = observation.source_path
    evidence["parse_error"] = observation.parse_error
    evidence["payload_leak_detected"] = payload_leak_detected
    ok = (
        observation.machine_payload.get("status") == "error"
        and all(bool(value) for value in context_checks.values())
        and all(bool(value) for value in user_context_checks.values())
        and not payload_leak_detected
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.parser_quarantine_error",
    )


def run_error_context_evidence(
    obligation: ProofObligation,
    observation: ErrorContextObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
) -> EvidenceEnvelope:
    """Verify machine/user error context for a durable error surface."""
    if obligation.claim.id != "error.machine_user_context":
        raise ValueError(f"unsupported error context claim: {obligation.claim.id}")

    subject_required_context = _string_tuple_attr(obligation.subject.attrs.get("required_context"))
    required_context = observation.required_context or subject_required_context
    machine_details = _machine_details(observation.machine_payload)
    user_context_checks: JSONDocument = {
        key: key in machine_details and str(machine_details.get(key)) in observation.user_message
        for key in required_context
    }
    machine_context_checks: JSONDocument = {
        key: key in machine_details and bool(machine_details.get(key)) for key in required_context
    }
    payload_leak_detected = _payload_leak_detected(
        observation.payload_fragments,
        observation.user_message,
        json.dumps(observation.machine_payload, sort_keys=True),
    )
    privacy_checks: JSONDocument = {
        "payload_fragments_redacted": not payload_leak_detected,
        "payload_leak_detected": payload_leak_detected,
    }
    observed_state: JSONDocument = {
        "error_family": observation.error_family,
        "machine_payload": dict(observation.machine_payload),
        "user_message": observation.user_message,
        "required_context": list(required_context),
        "machine_context_checks": machine_context_checks,
        "user_context_checks": user_context_checks,
        "privacy_checks": privacy_checks,
    }
    expected_law = "machine error payloads and user-facing text expose matching context without payload leaks"
    evidence = _evidence_payload(
        obligation,
        runner_class="error_context",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["machine_payload"] = dict(observation.machine_payload)
    evidence["user_context_checks"] = dict(user_context_checks)
    evidence["privacy_checks"] = dict(privacy_checks)
    ok = (
        observation.machine_payload.get("status") == "error"
        and bool(observation.machine_payload.get("code"))
        and bool(observation.machine_payload.get("message"))
        and all(machine_context_checks.values())
        and all(user_context_checks.values())
        and not payload_leak_detected
    )
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.error_context",
    )


def run_trace_equivalence_evidence(
    obligation: ProofObligation,
    observation: TraceEquivalenceObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_trace_evidence.py"),
) -> EvidenceEnvelope:
    """Verify that surface traces match by semantic event payloads."""
    if obligation.claim.id != "trace.operation.surface_equivalence":
        raise ValueError(f"unsupported trace-equivalence claim: {obligation.claim.id}")

    traces = observation.traces
    expected_surfaces = _string_tuple_attr(obligation.subject.attrs.get("surfaces"))
    expected_event_names = _string_tuple_attr(obligation.subject.attrs.get("event_nouns"))
    surface_names = tuple(trace.surface for trace in traces)
    signature_hashes = tuple(trace_signature_hash(trace) for trace in traces)
    first_signature = traces[0].semantic_signature() if traces else ()
    signature_match = bool(traces) and all(trace.semantic_signature() == first_signature for trace in traces)
    required_surfaces_present = set(expected_surfaces).issubset(set(surface_names))
    event_names_match = all(not expected_event_names or trace.event_names() == expected_event_names for trace in traces)
    raw_output_absent = all(
        not _json_key_present(event.payload, "raw_output") for trace in traces for event in trace.events
    )
    semantic_signature_hash = signature_hashes[0] if signature_hashes and len(set(signature_hashes)) == 1 else ""
    happens_before: list[JSONDocument] = [
        require_json_document(
            {
                "surface": trace.surface,
                "edges": [edge.to_payload() for edge in trace.happens_before],
            },
            context="trace happens-before payload",
        )
        for trace in traces
    ]
    trace_payloads: list[JSONDocument] = [trace.to_payload() for trace in traces]
    observed_state = require_json_document(
        {
            "surface_names": list(surface_names),
            "expected_surfaces": list(expected_surfaces),
            "event_names": list(expected_event_names),
            "event_names_by_surface": {trace.surface: list(trace.event_names()) for trace in traces},
            "semantic_signature_hash": semantic_signature_hash,
            "signature_hashes": list(signature_hashes),
            "trace_payloads": trace_payloads,
            "happens_before": happens_before,
            "checks": {
                "at_least_two_surfaces": len(traces) >= 2,
                "required_surfaces_present": required_surfaces_present,
                "event_names_match": event_names_match,
                "semantic_signatures_match": signature_match,
                "raw_output_absent": raw_output_absent,
            },
        },
        context="trace equivalence observed state",
    )
    expected_law = "cross-surface traces for the operation share semantic event names and payload fingerprints"
    evidence = _evidence_payload(
        obligation,
        runner_class="trace_equivalence",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["surface_names"] = list(surface_names)
    evidence["event_names"] = list(expected_event_names)
    evidence["semantic_signature_hash"] = semantic_signature_hash
    evidence["trace_payloads"] = require_json_value(trace_payloads, context="trace_payloads")
    evidence["happens_before"] = require_json_value(happens_before, context="happens_before")
    ok = len(traces) >= 2 and required_surfaces_present and event_names_match and signature_match and raw_output_absent
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.trace_equivalence",
    )


def run_diagnostic_trace_mapping_evidence(
    obligation: ProofObligation,
    observation: DiagnosticTraceMappingObservation,
    *,
    reproducer: tuple[str, ...] = ("pytest", "tests/unit/proof/test_trace_evidence.py"),
) -> EvidenceEnvelope:
    """Verify a diagnostic shape is routable through observable trace nouns."""
    if obligation.claim.id != "diagnostic.observable_trace_mapping":
        raise ValueError(f"unsupported diagnostic mapping claim: {obligation.claim.id}")

    attrs = obligation.subject.attrs
    mapping = observation.mapping
    expected_payload_contract = require_json_document(attrs.get("payload_contract"), context="payload_contract")
    payload_contract_matches = mapping.payload_contract == expected_payload_contract
    checks: JSONDocument = {
        "diagnostic_name_matches": mapping.diagnostic_name == attrs.get("diagnostic_name"),
        "source_matches": mapping.source == attrs.get("source"),
        "event_name_matches": mapping.event_name.value == attrs.get("event_noun"),
        "operation_matches": mapping.operation == attrs.get("operation"),
        "artifact_node_matches": mapping.artifact_node == attrs.get("artifact_node"),
        "payload_contract_present": bool(mapping.payload_contract),
        "payload_contract_matches": payload_contract_matches,
        "mapped_subject_id_present": bool(mapping.subject_id),
    }
    observed_state = require_json_document(
        {
            "mapping": mapping.to_payload(),
            "expected": dict(attrs),
            "checks": checks,
        },
        context="diagnostic trace mapping observed state",
    )
    expected_law = "diagnostic shape maps to an observable event noun, operation, artifact node, and payload contract"
    evidence = _evidence_payload(
        obligation,
        runner_class="diagnostic_trace_mapping",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["diagnostic_name"] = mapping.diagnostic_name
    evidence["event_name"] = mapping.event_name.value
    evidence["mapped_subject_id"] = mapping.subject_id
    evidence["operation"] = mapping.operation
    evidence["artifact_node"] = mapping.artifact_node
    ok = all(bool(value) for value in checks.values())
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=reproducer,
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.diagnostic_trace_mapping",
    )


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


def _run_cli_help_evidence(
    obligation: ProofObligation,
    *,
    args: Sequence[str] | None,
    env: Mapping[str, str] | None,
    root_command: click.Command | None,
) -> EvidenceEnvelope:
    command_args = tuple(args) if args is not None else _help_args_for_subject(obligation)
    result = _invoke_cli(command_args, env=env, root_command=root_command)
    observed_state = _cli_result_state(
        result,
        command_args=command_args,
        extra={"help_usage_banner": "Usage:" in result.output},
    )
    expected_law = "help command exits zero and renders a Usage banner"
    ok = result.exit_code == 0 and "Usage:" in result.output
    evidence = _evidence_payload(
        obligation,
        runner_class="cli_visual",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["help_exit_code"] = result.exit_code
    evidence["help_output"] = result.output[:_OUTPUT_SAMPLE_LIMIT]
    return _build_envelope(
        obligation,
        status=OutcomeStatus.OK if ok else OutcomeStatus.ERROR,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=("polylogue", *command_args),
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.cli_visual",
    )


def _run_cli_no_traceback_evidence(
    obligation: ProofObligation,
    *,
    args: Sequence[str] | None,
    env: Mapping[str, str] | None,
    root_command: click.Command | None,
) -> EvidenceEnvelope:
    command_args = tuple(args) if args is not None else _help_args_for_subject(obligation)
    result = _invoke_cli(command_args, env=env, root_command=root_command)
    traceback_present = _TRACEBACK_MARKER in result.output
    observed_state = _cli_result_state(
        result,
        command_args=command_args,
        extra={"traceback_present": traceback_present},
    )
    expected_law = "command output does not contain a Python traceback"
    evidence = _evidence_payload(
        obligation,
        runner_class="cli_visual",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["stdout"] = _stdout(result)[:_OUTPUT_SAMPLE_LIMIT]
    evidence["stderr"] = _stderr(result)[:_OUTPUT_SAMPLE_LIMIT]
    return _build_envelope(
        obligation,
        status=OutcomeStatus.ERROR if traceback_present else OutcomeStatus.OK,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=("polylogue", *command_args),
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.cli_visual",
    )


def _run_cli_plain_mode_evidence(
    obligation: ProofObligation,
    *,
    args: Sequence[str] | None,
    env: Mapping[str, str] | None,
    root_command: click.Command | None,
) -> EvidenceEnvelope:
    plain_args = tuple(args) if args is not None else _plain_help_args_for_subject(obligation)
    rich_args = _without_plain_flag(plain_args)
    runner_env = {"POLYLOGUE_FORCE_PLAIN": "1", **dict(env or {})}
    plain_result = _invoke_cli(plain_args, env=runner_env, root_command=root_command)
    rich_result = _invoke_cli(rich_args, env=env, root_command=root_command)
    ansi_present = bool(_ANSI_RE.search(plain_result.output))
    observed_state = _cli_result_state(
        plain_result,
        command_args=plain_args,
        extra={
            "ansi_present": ansi_present,
            "rich_exit_code": rich_result.exit_code,
            "rich_stdout_sample": _stdout(rich_result)[:_OUTPUT_SAMPLE_LIMIT],
        },
    )
    expected_law = "plain-mode command output does not contain ANSI escape sequences"
    evidence = _evidence_payload(
        obligation,
        runner_class="cli_visual",
        expected_law=expected_law,
        observed_state=observed_state,
    )
    evidence["plain_stdout"] = _stdout(plain_result)[:_OUTPUT_SAMPLE_LIMIT]
    evidence["rich_stdout"] = _stdout(rich_result)[:_OUTPUT_SAMPLE_LIMIT]
    return _build_envelope(
        obligation,
        status=OutcomeStatus.ERROR if ansi_present else OutcomeStatus.OK,
        evidence=evidence,
        expected_law=expected_law,
        observed_state=observed_state,
        reproducer=("polylogue", *plain_args),
        provenance=obligation.subject.source_span,
        producer="polylogue.proof.runners.cli_visual",
    )


def _invoke_cli(
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    root_command: click.Command | None,
) -> Result:
    if root_command is None:
        from polylogue.cli.click_app import cli

        root_command = cli
    runner = CliRunner(env=dict(env or {}))
    return runner.invoke(root_command, list(args), catch_exceptions=True)


def _help_args_for_subject(obligation: ProofObligation) -> tuple[str, ...]:
    command_path = _command_path_for_subject(obligation)
    return (*command_path, "--help") if command_path else ("--help",)


def _plain_help_args_for_subject(obligation: ProofObligation) -> tuple[str, ...]:
    return ("--plain", *_help_args_for_subject(obligation))


def _json_args_for_subject(obligation: ProofObligation) -> tuple[str, ...]:
    value = obligation.subject.attrs.get("json_args")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(str(item) for item in value)
    command_path = _command_path_for_subject(obligation)
    return ("--plain", *command_path, "--json")


def _command_path_for_subject(obligation: ProofObligation) -> tuple[str, ...]:
    value = obligation.subject.attrs.get("command_path")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(str(item) for item in value)
    return ()


def _without_plain_flag(args: Sequence[str]) -> tuple[str, ...]:
    filtered = tuple(arg for arg in args if arg != "--plain")
    return filtered or ("--help",)


def _parse_json_object(output: str) -> tuple[JSONDocument | None, str | None]:
    lines = output.strip().splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("{"):
            continue
        try:
            parsed = json.loads("\n".join(lines[index:]))
            return require_json_document(parsed, context="CLI JSON envelope"), None
        except (json.JSONDecodeError, TypeError) as exc:
            return None, str(exc)
    return None, "no JSON object found in output"


def _cli_result_state(
    result: Result,
    *,
    command_args: Sequence[str],
    extra: Mapping[str, object],
) -> JSONDocument:
    payload: dict[str, Any] = {
        "command_args": list(command_args),
        "exit_code": result.exit_code,
        "exception_type": type(result.exception).__name__ if result.exception is not None else None,
        "stdout_sample": _stdout(result)[:_OUTPUT_SAMPLE_LIMIT],
        "stderr_sample": _stderr(result)[:_OUTPUT_SAMPLE_LIMIT],
    }
    payload.update(extra)
    return require_json_document(payload, context="CLI observed state")


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


def _stdout(result: Result) -> str:
    value = getattr(result, "stdout", result.output)
    return value if isinstance(value, str) else result.output


def _stderr(result: Result) -> str:
    value = getattr(result, "stderr", "")
    return value if isinstance(value, str) else ""


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


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


def _machine_details(machine_payload: JSONDocument) -> JSONDocument:
    details = machine_payload.get("details")
    return require_json_document(details, context="machine error details") if isinstance(details, dict) else {}


def _payload_leak_detected(fragments: Sequence[str], *texts: str) -> bool:
    return any(fragment and any(fragment in text for text in texts) for fragment in fragments)


def _json_key_present(value: JSONValue, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_json_key_present(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_json_key_present(item, target) for item in value)
    return False


__all__ = [
    "DiagnosticTraceMappingObservation",
    "ErrorContextObservation",
    "MaintenanceRepairObservation",
    "QuarantineErrorObservation",
    "SemanticQueryObservation",
    "SchemaValueGenerationObservation",
    "TraceEquivalenceObservation",
    "run_artifact_path_evidence",
    "run_cli_json_envelope_evidence",
    "run_diagnostic_trace_mapping_evidence",
    "run_generated_scenario_evidence",
    "run_provider_capability_evidence",
    "run_error_context_evidence",
    "run_maintenance_repair_state_evidence",
    "run_quarantine_error_evidence",
    "run_schema_value_generation_evidence",
    "run_cli_visual_evidence",
    "run_semantic_query_evidence",
    "run_trace_equivalence_evidence",
]
