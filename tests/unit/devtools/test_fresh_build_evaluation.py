"""Inspection success cannot accept an optimization or manufacture capability."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench.evaluation import evaluate_comparison, evaluate_run, evaluate_samples


def _receipt(seconds: float = 10.0, *, candidate: str = "A") -> dict[str, Any]:
    return {
        "label": candidate,
        "candidate": {"git_sha": candidate * 40, "dirty": False},
        "qualified": True,
        "outcome": "terminal",
        "measurement_scope": "ordinary-daemon-terminal-route",
        "corpus": {"digest": "sealed", "kind": "selected-real", "total_bytes": 100, "file_count": 1},
        "config": {"digest": "fixed", "profile": False, "observe_work": False},
        "environment": {},
        "effective_compute_policy": {
            "outcome": "bound",
            "states": [{"capacity_slots": 2}],
            "product_binding": {
                "outcome": "bound",
                "modules": {"polylogue": {"sha256": candidate}},
                "work_observation": {"state": "observed", "enabled": False, "bound": True, "sha256": "work-source"},
            },
        },
        "readiness_probe": {"outcome": "bound"},
        "checks": dict.fromkeys(
            (
                "candidate_unchanged",
                "benchmark_unchanged",
                "corpus_unchanged",
                "compute_policy_bound",
                "candidate_import_bound",
                "promoted",
                "intake_complete",
                "raw_parse_failures_zero",
                "memberships_settled",
                "debt_zero",
                "readiness_complete",
                "readiness_import_bound",
                "clean_shutdown",
                "positive_output",
                "events_lossless",
                "milestones_recorded",
                "wall_clock_steady",
            ),
            True,
        ),
        "timing_s": {"promotion": seconds, "terminal": seconds + 1},
        "throughput": {
            "mib_per_s_to_promotion": 100 / 2**20 / seconds,
            "mib_per_s_to_terminal": 100 / 2**20 / (seconds + 1),
        },
        "writer": {},
        "stages": {},
        "budgets": {},
        "terminal_measurement": {
            "kind": "readiness-observation",
            "certified_s": seconds + 3,
            "observation_window_s": [seconds, seconds + 1],
        },
        "process_tree": {"cpu_seconds_to_promotion": 2.0, "rss_peak_bytes": 1 << 20},
        "diagnostic_capture": {
            "state": "recorded",
            "sha256": "sampler-receipt",
            "mode": {
                "format": "polylogue.fresh-build-stack-samples.v1",
                "interval_s": 0.05,
                "coverage": {"python": "unavailable", "native": "thread_cpu_accounting_only", "off_cpu": "unavailable"},
                "profile_refusal": None,
            },
        },
        "output_fingerprint": {"digest": "same", "tables": {}},
        "daemon_exit_code": 0,
        "evidence_sha256": {"events": "events-seal"},
        "shutdown_evidence": {
            "format": "polylogue.daemon-shutdown-evidence.v1",
            "state": "pass",
            "failed": [],
            "unknown": [],
            "bound": True,
            "input_sha256": {"events": "events-seal", "stacks": "sampler-receipt"},
        },
    }


def _practical_policy() -> dict[str, Any]:
    from devtools.fresh_build_bench.acceptance import POLICY_FORMAT

    return {
        "format": POLICY_FORMAT,
        "target_metric": "promotion_s",
        "effect": {"minimum_absolute": 1.0, "minimum_relative": 0.1},
        "resources": {
            "scope": "sampled-daemon-tree",
            "cpu_to_promotion_s": {"absolute_slack": 0.2, "relative_slack": 0.1, "ceiling": 3.0},
            "rss_sampled_peak_bytes": {"absolute_slack": 100, "relative_slack": 0.1, "ceiling": 2 << 20},
        },
        "cleanup": "bound-daemon-shutdown; descendant-external-unclaimed",
        "confirmation": {
            "method": "two-abba-all-pairs-v1",
            "purpose": "held-out-confirmation",
            "held_out_basis": {
                "attested_untuned": True,
                "statement": "Reserved selection before tuning",
                "discovery": None,
            },
        },
    }


def _practical_samples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from devtools.fresh_build_bench.acceptance import LAUNCH_FORMAT, SCHEDULE, digest

    samples: list[dict[str, Any]] = [
        {"arm": arm, "receipt": _receipt(10 if arm == "A" else 5, candidate=arm)} for arm in SCHEDULE
    ]
    launch = {
        "format": LAUNCH_FORMAT,
        "policy": _practical_policy(),
        "schedule": SCHEDULE,
        "candidate": {arm: _receipt(candidate=arm)["candidate"] for arm in "AB"},
        "corpus_digest": "sealed",
        "config_digest": "fixed",
        "environment": {},
        "benchmark_implementation_sha256": "driver",
        "held_out_nonoverlap": "unavailable; operator attestation",
    }
    for index, sample in enumerate(samples):
        receipt = sample["receipt"]
        binding = {
            "path": "/synthetic/launch.json",
            "sha256": digest(launch),
            "ordinal": index + 1,
            "arm": sample["arm"],
        }
        receipt.update(acceptance_binding=binding, acceptance_launch=launch, benchmark_implementation_sha256="driver")
        receipt["checks"]["acceptance_launch_unchanged"] = True
        receipt["evidence_sha256"]["acceptance"] = "launch-copy-seal"
        receipt["effective_compute_policy"]["acceptance_binding"] = {**binding, "state": "bound-before-main"}
        receipt["effective_compute_policy"]["recipe_binding"] = {
            "imported_product_content": {"polylogue.work": sample["arm"]}
        }
    return samples, launch


def test_frozen_practical_policy_accepts_four_predetermined_pairs_without_statistical_claim() -> None:
    samples, launch = _practical_samples()
    decision = evaluate_samples(samples, launch=launch)["performance_decision"]
    assert decision["accepted"] and decision["state"] == "accepted"
    assert [row["ordinals"] for row in decision["pairs"]] == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert decision["claim"] == "selected-route-practical-repeatability"
    assert decision["method"] == "two-abba-all-pairs-v1"


@pytest.mark.parametrize(
    "change",
    [
        "absolute-effect",
        "relative-effect",
        "cpu",
        "rss",
        "missing-cpu",
        "binding",
        "posthoc",
        "outputs",
        "cleanup",
        "imports",
        "commit-only",
        "partial",
    ],
)
def test_practical_acceptance_refuses_missing_proof_and_one_losing_pair(change: str) -> None:
    samples, launch = _practical_samples()
    receipt = samples[5]["receipt"]
    if change in {"absolute-effect", "relative-effect"}:
        # The other three pairs still win. A median-only decision is wrong.
        receipt["timing_s"]["promotion"] = 9.5
        if change == "relative-effect":
            samples[4]["receipt"]["timing_s"]["promotion"] = 100.0
            receipt["timing_s"]["promotion"] = 95.0
    elif change == "cpu":
        receipt["process_tree"]["cpu_seconds_to_promotion"] = 2.3
    elif change == "rss":
        receipt["process_tree"]["rss_peak_bytes"] = 3 << 20
    elif change == "missing-cpu":
        receipt["process_tree"]["cpu_seconds_to_promotion"] = None
    elif change == "binding":
        receipt["acceptance_binding"]["ordinal"] = 1
    elif change == "posthoc":
        receipt["effective_compute_policy"]["acceptance_binding"] = None
    elif change == "outputs":
        receipt["output_fingerprint"]["digest"] = "changed"
    elif change == "cleanup":
        receipt["shutdown_evidence"]["state"] = "unknown"
    elif change == "imports":
        del receipt["effective_compute_policy"]["recipe_binding"]
    elif change == "commit-only":
        for sample in samples:
            sample["receipt"]["effective_compute_policy"]["recipe_binding"]["imported_product_content"] = {
                "polylogue.work": "same"
            }
    else:
        samples.pop()
    decision = evaluate_samples(samples, launch=launch)["performance_decision"]
    assert not decision["accepted"] and decision["gaps"]


@pytest.mark.parametrize("change", ["zero", "nan", "terminal", "hard-resource", "global-cleanup", "tuned"])
def test_practical_policy_refuses_unsupported_claims_before_launch(change: str) -> None:
    from devtools.fresh_build_bench.acceptance import validate_policy

    policy = _practical_policy()
    if change in {"zero", "nan"}:
        policy["effect"]["minimum_absolute"] = 0 if change == "zero" else float("nan")
    elif change == "terminal":
        policy["target_metric"] = "terminal_observation_s"
    elif change == "hard-resource":
        policy["resources"]["scope"] = "hard-process-tree"
    elif change == "global-cleanup":
        policy["cleanup"] = "all-descendants-and-external"
    else:
        policy["confirmation"]["held_out_basis"]["attested_untuned"] = False
    with pytest.raises(ValueError):
        validate_policy(policy)


@pytest.mark.parametrize("fingerprint,expected", [({"digest": "different", "tables": {}}, "fail"), (None, "unknown")])
def test_faster_output_difference_or_missing_fingerprint_never_accepts(fingerprint: Any, expected: str) -> None:
    before, after = _receipt(), _receipt(1.0, candidate="B")
    after["output_fingerprint"] = fingerprint
    evaluation = evaluate_comparison(before, after)
    assert evaluation["metrics"]["promotion_s"]["qualified"]
    assert evaluation["scope_completion"]["state"] == "pass"
    assert evaluation["equivalence"]["state"] == expected
    assert not evaluation["performance_decision"]["accepted"]


def test_matched_optional_profile_refusal_preserves_bound_timing_not_attribution() -> None:
    before, after = _receipt(), _receipt(9.0, candidate="B")
    for receipt in (before, after):
        receipt["qualified"] = False
        receipt["config"]["profile"] = True
        receipt["checks"]["profile_collected"] = False
        receipt["profile"] = {"outcome": "refused", "reason": "free_threaded_frame_snapshot_unavailable"}
        receipt["diagnostic_capture"]["mode"]["profile_refusal"] = "free_threaded_frame_snapshot_unavailable"
    evaluation = evaluate_comparison(before, after)
    assert evaluation["measurement_validity"]["state"] == "pass"
    assert evaluation["scope_completion"]["state"] == "pass"
    assert evaluation["metrics"]["promotion_s"]["qualified"]
    assert all(row["stacks"]["state"] == "refused" for row in evaluation["capabilities"])
    assert not evaluation["performance_decision"]["accepted"]


@pytest.mark.parametrize("change", ["intake_complete", "readiness_complete", "clean_shutdown", "wrong_scope"])
def test_true_incomplete_or_wrong_scope_keeps_valid_promotion_separate(change: str) -> None:
    before, after = _receipt(), _receipt(1.0, candidate="B")
    if change == "wrong_scope":
        for receipt in (before, after):
            receipt["measurement_scope"] = "component-only"
    else:
        after["checks"][change] = False
    evaluation = evaluate_comparison(before, after)
    assert evaluation["scope_completion"]["state"] == "fail"
    assert evaluation["equivalence"]["state"] == "pass"
    assert evaluation["performance_decision"]["state"] == "rejected"
    assert evaluate_run(after)["metrics"]["promotion_s"]["qualified"]


def test_missing_confirming_evidence_does_not_inherit_aggregate_qualification() -> None:
    receipt = _receipt()
    del receipt["checks"]["candidate_import_bound"]
    evaluation = evaluate_run(receipt)
    assert receipt["qualified"]
    assert evaluation["measurement_validity"]["state"] == "unknown"
    assert not evaluation["metrics"]["promotion_s"]["qualified"]


@pytest.mark.parametrize("evidence", ["historical", "failed", "unbound", "clean"])
def test_shutdown_capability_keeps_terminal_readiness_independent(evidence: str) -> None:
    receipt = _receipt()
    if evidence == "historical":
        del receipt["shutdown_evidence"]
    elif evidence == "failed":
        receipt["shutdown_evidence"].update(state="fail", failed=["daemon.shutdown.services_failed"])
        receipt["checks"]["clean_shutdown"] = False
    elif evidence == "unbound":
        receipt["shutdown_evidence"]["input_sha256"]["events"] = "changed"
    evaluation = evaluate_run(receipt)
    shutdown = evaluation["capabilities"]["cleanup"]["daemon_shutdown"]
    assert (
        shutdown["state"]
        == {"historical": "unknown", "failed": "fail", "unbound": "unknown", "clean": "pass"}[evidence]
    )
    assert evaluation["capabilities"]["cleanup"]["descendant_and_external_resource_cleanup"] == "unknown"
    assert evaluation["metrics"]["terminal_observation_s"]["qualified"] is True
    assert evaluation["metrics"]["terminal_observation_s"]["value"] == receipt["timing_s"]["terminal"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_unavailable_target_metric_is_not_an_observed_zero(value: Any) -> None:
    before, after = _receipt(), _receipt(candidate="B")
    after["process_tree"]["cpu_seconds_to_promotion"] = value
    evaluation = evaluate_comparison(before, after, target_metric="cpu_to_promotion_s")
    assert evaluation["metrics"]["promotion_s"]["qualified"]
    assert not evaluation["metrics"]["cpu_to_promotion_s"]["qualified"]
    assert evaluation["metrics"]["cpu_to_promotion_s"]["capability"] == "unavailable"
    assert "target_metric_not_qualified" in evaluation["performance_decision"]["gaps"]


@pytest.mark.parametrize("mode", ["work", "profile", "unrecorded"])
def test_actual_instrumentation_modes_must_match_for_metric_comparison(mode: str) -> None:
    before, after = _receipt(), _receipt(candidate="B")
    if mode == "work":
        after["config"]["observe_work"] = True
        after["effective_compute_policy"]["product_binding"]["work_observation"]["enabled"] = True
    elif mode == "profile":
        after["diagnostic_capture"]["mode"]["coverage"]["python"] = "sampled_frames"
    else:
        del after["diagnostic_capture"]
    evaluation = evaluate_comparison(before, after)
    assert all(row["capability"] == "available" for row in evaluation["metrics"].values())
    assert not evaluation["metrics"]["promotion_s"]["qualified"]
    assert not evaluation["performance_decision"]["accepted"]


def test_a_a_and_separated_ranges_cannot_create_a_launch_policy_or_speed_confirmation() -> None:
    before, after = _receipt(), _receipt(1.0, candidate="B")
    single = evaluate_comparison(before, after)
    assert "single_pair_cannot_confirm_speed_improvement" in single["performance_decision"]["gaps"]
    repeated = evaluate_samples(
        [
            {"arm": "A", "receipt": before},
            {"arm": "B", "receipt": after},
            {"arm": "B", "receipt": after},
            {"arm": "A", "receipt": before},
        ]
    )
    assert repeated["metrics"]["promotion_s"]["qualified"]
    assert repeated["performance_decision"]["state"] == "inconclusive"
    assert "claim_policy_not_frozen_before_launch" in repeated["performance_decision"]["gaps"]
    calibration = evaluate_comparison(before, copy.deepcopy(before))
    assert calibration["experiment_kind"] == "A/A-calibration"
    assert "identical_actual_candidate_calibration" in calibration["performance_decision"]["gaps"]
    assert not calibration["performance_decision"]["accepted"]


def test_exit_zero_for_inspection_never_means_acceptance_even_with_post_hoc_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools.fresh_build_bench.cli import main

    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    receipts = [_receipt(), _receipt(1.0, candidate="B")]
    receipts[1]["budgets"] = {"promotion_s": {"limit": 2.0, "pass": True}}
    for path, receipt in zip(paths, receipts, strict=True):
        path.write_text(json.dumps(receipt))
    assert main(["compare", *map(str, paths), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["inspection_admissible"] and not inspected["performance_decision"]["accepted"]
    assert main(["compare", *map(str, paths), "--json", "--accept", "--metric", "rss_sampled_peak_bytes"]) == 1
    accepted = json.loads(capsys.readouterr().out)
    assert not accepted["performance_decision"]["accepted"]
    assert "claim_policy_not_frozen_before_launch" in accepted["performance_decision"]["gaps"]


def test_daemon_exit_zero_cannot_prove_resource_cleanup_or_hard_child_peak_bound() -> None:
    evaluation = evaluate_run(_receipt())
    cleanup = evaluation["capabilities"]["cleanup"]
    assert cleanup["daemon_shutdown"]["state"] == "pass"
    assert cleanup["descendant_and_external_resource_cleanup"] == "unknown"
    assert "child spikes" in evaluation["metrics"]["rss_sampled_peak_bytes"]["semantics"]


@pytest.mark.parametrize("command", ["run", "experiment"])
def test_work_observation_flag_is_available_on_both_existing_routes(command: str) -> None:
    from devtools.fresh_build_bench.cli import _parser

    arguments = [command, "--corpus", "/synthetic/corpus", "--work", "/synthetic/work", "--observe-work"]
    if command == "experiment":
        arguments += ["--before", "/synthetic/A", "--after", "/synthetic/B"]
    assert _parser().parse_args(arguments).observe_work


@pytest.mark.parametrize(
    "change,expected", [("missing", "unknown"), ("unavailable", "unknown"), ("unbound", "fail"), ("enabled", "fail")]
)
def test_actual_work_observation_handshake_cannot_inherit_declared_mode(change: str, expected: str) -> None:
    before, after = _receipt(), _receipt(candidate="B")
    binding = after["effective_compute_policy"]["product_binding"]
    if change == "missing":
        del binding["work_observation"]
    elif change == "unavailable":
        binding["work_observation"] = {"state": "unavailable", "enabled": None}
    elif change == "unbound":
        binding["work_observation"]["bound"] = False
    else:
        binding["work_observation"]["enabled"] = True
    evaluation = evaluate_comparison(before, after)
    assert evaluation["measurement_validity"]["state"] == "pass"
    assert evaluation["instrumentation_comparability"]["state"] == expected
    assert evaluation["metrics"]["promotion_s"]["capability"] == "available"
    assert not evaluation["metrics"]["promotion_s"]["qualified"]
    assert not evaluation["performance_decision"]["accepted"]
