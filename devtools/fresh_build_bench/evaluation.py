"""Claim-specific reductions of existing receipts, without executing product work.

Inspection, measured scope completion and optimization acceptance are separate.
This evaluator has no confirmed speed-acceptance method or resource policy yet.
"""

from __future__ import annotations

import json
import math
from typing import Any, Final

from devtools.fresh_build_bench.recipe_qualification import output_equivalence

METRICS: Final = ("promotion_s", "terminal_observation_s", "cpu_to_promotion_s", "rss_sampled_peak_bytes")
_INPUT_CHECKS: Final = (
    "candidate_unchanged",
    "benchmark_unchanged",
    "corpus_unchanged",
    "compute_policy_bound",
    "candidate_import_bound",
)
_SCOPE_CHECKS: Final = (
    "promoted",
    "intake_complete",
    "raw_parse_failures_zero",
    "memberships_settled",
    "debt_zero",
    "readiness_complete",
    "readiness_import_bound",
    "clean_shutdown",
    "positive_output",
)


def _axis(failed: list[str], unknown: list[str]) -> dict[str, Any]:
    return {"state": "fail" if failed else "unknown" if unknown else "pass", "failed": failed, "unknown": unknown}


def _checks(receipt: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    checks = receipt.get("checks") or {}
    return _axis(
        [name for name in names if checks.get(name) is False],
        [name for name in names if checks.get(name) is not True and checks.get(name) is not False],
    )


def _daemon_shutdown(receipt: dict[str, Any]) -> dict[str, Any]:
    """Historical exit-only checks cannot certify captured shutdown behavior."""
    proof = receipt.get("shutdown_evidence")
    if not isinstance(proof, dict) or proof.get("format") != "polylogue.daemon-shutdown-evidence.v1":
        exit_code = receipt.get("daemon_exit_code")
        if type(exit_code) is int and exit_code != 0:
            return _axis(["daemon_exit_nonzero"], ["shutdown_evidence_unrecorded"])
        return _axis([], ["shutdown_evidence_unrecorded"])
    failed, unknown = list(proof.get("failed") or []), list(proof.get("unknown") or [])
    if proof.get("state") == "fail" and not failed:
        failed.append("daemon_shutdown_failed")
    elif proof.get("state") != "pass" and not failed and not unknown:
        unknown.append("daemon_shutdown_unproven")
    binding = proof.get("input_sha256") or {}
    capture = receipt.get("diagnostic_capture") or {}
    expected_stacks = capture.get("expected_sha256", capture.get("sha256"))
    if not (
        proof.get("bound") is True
        and binding.get("events")
        and binding.get("stacks")
        and binding.get("events") == (receipt.get("evidence_sha256") or {}).get("events")
        and binding.get("stacks") == expected_stacks
    ):
        unknown.append("shutdown_inputs_unbound")
    check = _checks(receipt, ("clean_shutdown",))
    return _axis(list(dict.fromkeys(failed + check["failed"])), list(dict.fromkeys(unknown + check["unknown"])))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _instrumentation(receipt: dict[str, Any]) -> dict[str, Any]:
    capture = receipt.get("diagnostic_capture") or {}
    declared = (receipt.get("config") or {}).get("observe_work")
    observed = ((receipt.get("effective_compute_policy") or {}).get("product_binding") or {}).get(
        "work_observation"
    ) or {}
    mode = capture.get("mode") or {}
    failed, unknown = [], []
    if not isinstance(declared, bool):
        unknown.append("declared_work_observation_mode_unrecorded")
    if observed.get("state") != "observed" or not isinstance(observed.get("enabled"), bool):
        unknown.append("actual_work_observation_mode_unrecorded")
    elif observed.get("bound") is False or isinstance(declared, bool) and observed["enabled"] != declared:
        failed.append("actual_work_observation_mode_mismatch")
    elif observed.get("bound") is not True or not observed.get("sha256"):
        unknown.append("actual_work_observation_mode_unbound")
    if capture.get("state") == "changed":
        failed.append("sampler_evidence_changed")
    elif capture.get("state") != "recorded" or not mode.get("format") or not mode.get("coverage"):
        unknown.append("actual_sampler_mode_unrecorded")
    return {
        **_axis(failed, unknown),
        "declared_work_observation": declared,
        "observed_work_observation": observed,
        "sampler": capture,
    }


def _instrumentation_mode(receipt: dict[str, Any]) -> str | None:
    proof = _instrumentation(receipt)
    if proof["state"] != "pass":
        return None
    observed = proof["observed_work_observation"]
    return json.dumps(
        {
            "work_observation": {"enabled": observed["enabled"], "sha256": observed["sha256"]},
            "sampler": proof["sampler"]["mode"],
        },
        sort_keys=True,
    )


def evaluate_run(receipt: dict[str, Any]) -> dict[str, Any]:
    """Missing evidence stays unknown, independently of the old aggregate flag."""
    validity = _checks(receipt, _INPUT_CHECKS)
    if "derived_from" in receipt or "evidence_unchanged" in (receipt.get("checks") or {}):
        seal = _checks(receipt, ("evidence_unchanged",))
        validity = _axis(validity["failed"] + seal["failed"], validity["unknown"] + seal["unknown"])
    shutdown = _daemon_shutdown(receipt)
    scope_checks = _checks(receipt, _SCOPE_CHECKS)
    scope = _axis(
        list(dict.fromkeys(scope_checks["failed"] + shutdown["failed"])),
        list(dict.fromkeys(scope_checks["unknown"] + shutdown["unknown"])),
    )
    if receipt.get("outcome") is None:
        scope["unknown"].append("terminal_outcome")
        if scope["state"] != "fail":
            scope["state"] = "unknown"
    elif receipt["outcome"] != "terminal":
        scope["failed"].append("terminal_outcome")
        scope["state"] = "fail"
    if receipt.get("measurement_scope") is None:
        scope["unknown"].append("declared_terminal_route_scope")
        if scope["state"] != "fail":
            scope["state"] = "unknown"
    elif receipt["measurement_scope"] != "ordinary-daemon-terminal-route":
        scope["failed"].append("declared_terminal_route_scope_mismatch")
        scope["state"] = "fail"
    scope.update(
        declared=receipt.get("measurement_scope"),
        coverage="sealed selected inputs only; no serving-health, installed, published or whole-install claim",
        corpus_kind=(receipt.get("corpus") or {}).get("kind"),
        selection=(receipt.get("corpus") or {}).get("parameters"),
    )
    timing, tree = receipt.get("timing_s") or {}, receipt.get("process_tree") or {}
    values = {
        "promotion_s": timing.get("promotion"),
        "terminal_observation_s": timing.get("terminal"),
        "cpu_to_promotion_s": tree.get("cpu_seconds_to_promotion"),
        "rss_sampled_peak_bytes": tree.get("rss_peak_bytes"),
    }
    required = {
        "promotion_s": ("events_lossless", "milestones_recorded", "wall_clock_steady"),
        "terminal_observation_s": ("readiness_import_bound",),
        "cpu_to_promotion_s": ("events_lossless", "milestones_recorded", "wall_clock_steady"),
        "rss_sampled_peak_bytes": (),
    }
    metrics = {}
    for name, value in values.items():
        proof = _checks(receipt, required[name])
        failed = validity["failed"] + proof["failed"]
        unknown = validity["unknown"] + proof["unknown"]
        if not _number(value):
            unknown.append("measurement_unavailable")
        if name == "terminal_observation_s":
            observation = receipt.get("terminal_measurement") or {}
            if observation.get("kind") != "readiness-observation" or not _number(observation.get("certified_s")):
                unknown.append("readiness_observation_certification_unrecorded")
        metrics[name] = {
            **_axis(failed, unknown),
            "qualified": not failed and not unknown,
            "capability": "available" if _number(value) else "unavailable",
            "value": value,
            "semantics": "observer certification with poll, scan and cache uncertainty; not product completion throughput"
            if name == "terminal_observation_s"
            else "sampled process-tree peak; child spikes between polls may be missed"
            if name == "rss_sampled_peak_bytes"
            else "event-milestone"
            if name == "promotion_s"
            else "sampled daemon process tree cut at the promotion event",
            "uncertainty": receipt.get("terminal_measurement") if name == "terminal_observation_s" else None,
        }
    profile = receipt.get("profile") or {}
    requested = bool((receipt.get("config") or {}).get("profile"))
    coverage = ((receipt.get("diagnostic_capture") or {}).get("mode") or {}).get("coverage") or {}
    return {
        "measurement_validity": validity,
        "scope_completion": scope,
        "metrics": metrics,
        "capabilities": {
            "instrumentation": _instrumentation(receipt),
            "stacks": {
                "state": "available"
                if profile.get("outcome") == "ok" and coverage.get("python") == "sampled_frames"
                else "refused"
                if profile.get("outcome") == "refused"
                else "unknown"
                if requested
                else "not-requested",
                "reason": profile.get("reason"),
                "coverage": coverage,
                "semantics": "in-daemon sampled Python frames; thread CPU accounting alone provides no native callchains",
                "required_for": "stack attribution claims only",
            },
            "cleanup": {
                "daemon_exit_code": receipt.get("daemon_exit_code"),
                "daemon_shutdown": shutdown,
                "descendant_and_external_resource_cleanup": "unknown",
                "reason": "daemon exit and log drain do not prove descendant or external resource cleanup",
            },
        },
    }


def _actual_candidate(receipt: dict[str, Any]) -> str | None:
    binding = (receipt.get("effective_compute_policy") or {}).get("product_binding") or {}
    modules = binding.get("modules")
    candidate = receipt.get("candidate")
    if binding.get("outcome") != "bound" or not modules or not candidate:
        return None
    return json.dumps(
        {"candidate": candidate, "modules": {name: row.get("sha256") for name, row in modules.items()}}, sort_keys=True
    )


def evaluate_samples(samples: list[dict[str, Any]], *, target_metric: str = "promotion_s") -> dict[str, Any]:
    """Evaluate existing paired evidence; choosing a metric cannot create a policy."""
    if target_metric not in METRICS:
        raise ValueError(f"unsupported target metric: {target_metric}")
    # Keep the established fixed-policy identity contract in its existing owner.
    from devtools.fresh_build_bench.report import comparability_problems

    runs = [evaluate_run(sample["receipt"]) for sample in samples]
    problems = (
        sorted(
            {
                problem
                for sample in samples
                for problem in comparability_problems(samples[0]["receipt"], sample["receipt"])
                if "not qualified" not in problem
            }
        )
        if samples
        else ["no recorded samples"]
    )
    validity = _axis(
        problems
        + [f"sample {i + 1}: {field}" for i, run in enumerate(runs) for field in run["measurement_validity"]["failed"]],
        [f"sample {i + 1}: {field}" for i, run in enumerate(runs) for field in run["measurement_validity"]["unknown"]],
    )
    counts = {arm: sum(sample.get("arm") == arm for sample in samples) for arm in ("A", "B")}
    scope = _axis(
        [f"sample {i + 1}: {field}" for i, run in enumerate(runs) for field in run["scope_completion"]["failed"]],
        [f"sample {i + 1}: {field}" for i, run in enumerate(runs) for field in run["scope_completion"]["unknown"]]
        + ([] if all(counts.values()) else ["both_arms_required"]),
    )
    fingerprints = [sample["receipt"].get("output_fingerprint") for sample in samples]
    missing = not fingerprints or any(not fingerprint or not fingerprint.get("digest") for fingerprint in fingerprints)
    equivalences = [] if missing else [output_equivalence(fingerprints[0], fingerprint) for fingerprint in fingerprints]
    equivalence = {
        **_axis(
            ["different_outputs"] if any(value is None for value in equivalences) else [],
            ["output_fingerprint_missing"] if missing else [],
        ),
        "kind": None
        if missing or any(value is None for value in equivalences)
        else "exact"
        if all(value == "exact" for value in equivalences)
        else "recipe-qualified-exact",
    }
    actual = [_actual_candidate(sample["receipt"]) for sample in samples]
    calibration = bool(actual) and all(value is not None and value == actual[0] for value in actual)
    overhead = {_instrumentation_mode(sample["receipt"]) for sample in samples}
    instrumentation = _axis(
        [
            f"sample {i + 1}: {field}"
            for i, run in enumerate(runs)
            for field in run["capabilities"]["instrumentation"]["failed"]
        ]
        + (["different_actual_instrumentation_modes"] if len(overhead - {None}) > 1 else []),
        [
            f"sample {i + 1}: {field}"
            for i, run in enumerate(runs)
            for field in run["capabilities"]["instrumentation"]["unknown"]
        ],
    )
    metrics = {}
    for name in METRICS:
        measurements = [run["metrics"][name] for run in runs]
        reasons = problems + instrumentation["failed"] + instrumentation["unknown"]
        qualified = (
            bool(measurements)
            and all(row["qualified"] for row in measurements)
            and not reasons
            and all(counts.values())
        )
        metrics[name] = {
            "qualified": qualified,
            "capability": "available"
            if measurements and all(row["capability"] == "available" for row in measurements)
            else "unavailable",
            "control_problems": reasons,
            "samples": measurements,
        }
    gaps = [
        "claim_policy_not_frozen_before_launch",
        "confirmation_method_not_implemented",
        "resource_policy_and_cleanup_confirmation_missing",
    ]
    if min(counts.values()) < 2:
        gaps.append("single_pair_cannot_confirm_speed_improvement")
    if calibration:
        gaps.append("identical_actual_candidate_calibration")
    if not metrics[target_metric]["qualified"]:
        gaps.append("target_metric_not_qualified")
    if validity["state"] != "pass":
        gaps.append("measurement_validity_not_confirmed")
    if scope["state"] != "pass":
        gaps.append("declared_scope_completion_not_confirmed")
    if equivalence["state"] != "pass":
        gaps.append("required_output_equivalence_not_confirmed")
    rejected = any(axis["state"] == "fail" for axis in (validity, scope, equivalence))
    return {
        "measurement_validity": validity,
        "scope_completion": scope,
        "equivalence": equivalence,
        "instrumentation_comparability": instrumentation,
        "metrics": metrics,
        "capabilities": [run["capabilities"] for run in runs],
        "sample_counts": counts,
        "experiment_kind": "A/A-calibration" if calibration else "candidate-comparison",
        "performance_decision": {
            "state": "rejected" if rejected else "inconclusive",
            "accepted": False,
            "claim": "speed-improvement",
            "target_metric": target_metric,
            "gaps": gaps,
            "reason": "descriptive samples and range separation are not optimization acceptance",
        },
    }


def evaluate_comparison(
    before: dict[str, Any], after: dict[str, Any], *, target_metric: str = "promotion_s"
) -> dict[str, Any]:
    return evaluate_samples(
        [{"arm": "A", "receipt": before}, {"arm": "B", "receipt": after}], target_metric=target_metric
    )
