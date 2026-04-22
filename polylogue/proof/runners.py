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

from polylogue.lib.json import JSONDocument, JSONValue, require_json_document, require_json_value
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.models import EvidenceEnvelope, ProofObligation, SourceSpan, TrustMetadata
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


__all__ = [
    "SemanticQueryObservation",
    "SchemaValueGenerationObservation",
    "run_cli_json_envelope_evidence",
    "run_provider_capability_evidence",
    "run_schema_value_generation_evidence",
    "run_cli_visual_evidence",
    "run_semantic_query_evidence",
]
