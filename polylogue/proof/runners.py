"""Executable proof evidence runners.

The runners in this module are deliberately small adapters: they evaluate one
obligation against already-selected command or semantic observations and return
an `EvidenceEnvelope`. They do not decide which obligations exist.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import click
from click.testing import CliRunner, Result

from polylogue.lib.json import JSONDocument, require_json_document
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.models import EvidenceEnvelope, ProofObligation, SourceSpan, TrustMetadata

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\[.*?m")
_TRACEBACK_MARKER = "Traceback (most recent call last)"
_REVIEWED_AT = "2026-04-22T00:00:00+00:00"
_OUTPUT_SAMPLE_LIMIT = 1_000


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
    return EvidenceEnvelope.build(
        obligation_id=obligation.id,
        status=status,
        evidence=evidence,
        counterexample=counterexample,
        reproducer=reproducer,
        environment={
            "runner": obligation.runner.runner,
            "evidence_class": obligation.runner.evidence_class,
            "cost_tier": obligation.runner.cost_tier,
        },
        provenance=provenance,
        trust=TrustMetadata(
            producer=producer,
            reviewed_at=_REVIEWED_AT,
            level="generated",
            privacy="repo-local command metadata or seeded fixture ids only",
        ),
    )


def _stdout(result: Result) -> str:
    value = getattr(result, "stdout", result.output)
    return value if isinstance(value, str) else result.output


def _stderr(result: Result) -> str:
    value = getattr(result, "stderr", "")
    return value if isinstance(value, str) else ""


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


__all__ = [
    "SemanticQueryObservation",
    "run_cli_json_envelope_evidence",
    "run_cli_visual_evidence",
    "run_semantic_query_evidence",
]
