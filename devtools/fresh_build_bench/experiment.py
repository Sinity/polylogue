"""Paired ordinary-daemon experiments with immutable constituent receipts."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, cast

from devtools.fresh_build_bench.corpus import load_manifest, verify_manifest
from devtools.fresh_build_bench.evaluation import evaluate_run, evaluate_samples
from devtools.fresh_build_bench.recipe_qualification import output_equivalence
from devtools.fresh_build_bench.report import comparability_problems
from devtools.fresh_build_bench.run import RunConfig, run_build


class _ArmMetric(TypedDict):
    raw_s: list[float | None]
    median_s: float | None
    mad_s: float | None
    range_s: list[float] | None


def reduce_experiment(samples: list[dict[str, Any]], *, launch: dict[str, Any] | None = None) -> dict[str, Any]:
    """Descriptive arm differences; neither range separation nor A/A proves speed."""
    problems: list[str] = []
    reference = samples[0]["receipt"]
    fingerprints = set()
    equivalences = []
    policies: dict[str, set[str]] = {"A": set(), "B": set()}
    candidates: dict[str, set[str]] = {"A": set(), "B": set()}
    for sample in samples:
        receipt = sample["receipt"]
        candidates[sample["arm"]].add(json.dumps(receipt.get("candidate"), sort_keys=True))
        problems.extend(comparability_problems(reference, receipt))
        fingerprint = receipt.get("output_fingerprint")
        if not fingerprint or not fingerprint.get("digest"):
            problems.append("missing exact output fingerprint")
        else:
            fingerprints.add(fingerprint["digest"])
            equivalences.append(output_equivalence(reference.get("output_fingerprint") or {}, fingerprint))
        policy = receipt.get("effective_compute_policy")
        if not policy or policy.get("outcome") != "bound":
            problems.append("unbound effective compute policy")
        else:
            policies[sample["arm"]].add(json.dumps(policy["states"], sort_keys=True))
    if any(value is None for value in equivalences):
        problems.append("different exact outputs without validated parser/lowering recipe equivalence")
    for arm, values in policies.items():
        if len(values) != 1:
            problems.append(f"arm {arm} effective compute policy changed or unavailable")
    for arm, values in candidates.items():
        if len(values) != 1:
            problems.append(f"arm {arm} candidate changed between runs")

    def actual_candidate(receipt: dict[str, Any]) -> str | None:
        binding = (receipt.get("effective_compute_policy") or {}).get("product_binding") or {}
        candidate = receipt.get("candidate")
        modules = binding.get("modules")
        if not candidate or not modules or binding.get("outcome") != "bound":
            return None
        return json.dumps(
            {"candidate": candidate, "modules": {name: row.get("sha256") for name, row in modules.items()}},
            sort_keys=True,
        )

    actual = [actual_candidate(sample["receipt"]) for sample in samples]
    calibration = bool(actual) and all(value is not None and value == actual[0] for value in actual)
    metrics = {}
    for name in ("promotion", "terminal"):
        arms: dict[str, _ArmMetric] = {}
        for arm in ("A", "B"):
            raw: list[float | None] = [
                sample["receipt"]["timing_s"].get(name) for sample in samples if sample["arm"] == arm
            ]
            if any(value is None for value in raw):
                problems.append(f"missing {name} measurement for {arm}")
                arms[arm] = {"raw_s": raw, "median_s": None, "mad_s": None, "range_s": None}
                continue
            measured = cast(list[float], raw)
            median = statistics.median(measured)
            arms[arm] = {
                "raw_s": raw,
                "median_s": median,
                "mad_s": statistics.median(abs(value - median) for value in measured),
                "range_s": [min(measured), max(measured)],
            }
        a, b = arms["A"], arms["B"]
        delta = None if a["median_s"] is None or b["median_s"] is None else b["median_s"] - a["median_s"]
        # Ranges overlap: this small experiment does not separate variation
        # from improvement. No significance or confidence interval is claimed.
        distribution = len(a["raw_s"]) > 1 and len(b["raw_s"]) > 1
        separated = (
            distribution
            and a["range_s"] is not None
            and b["range_s"] is not None
            and (b["range_s"][1] < a["range_s"][0] or a["range_s"][1] < b["range_s"][0])
        )
        assessment = "single-sample" if not distribution else "within-observed-noise"
        if separated and delta is not None:
            assessment = "observed-arm-difference"
        metrics[name] = {
            "arms": arms,
            "delta_median_s": delta,
            "assessment": assessment,
            "sample_kind": "repeated-samples" if distribution else "single-sample-per-arm",
            "inference": "descriptive-only; observed ranges are not confidence intervals; sample insufficiency for speed proof",
            "measurement": "event-milestone"
            if name == "promotion"
            else "readiness-observation-with-poll-scan-and-cache-delay",
            "measurement_uncertainty": [
                sample["receipt"].get("terminal_measurement", {"kind": "unrecorded-observer-resolution"})
                for sample in samples
            ]
            if name == "terminal"
            else [],
        }
    return {
        "evaluation": evaluate_samples(samples, launch=launch),
        "qualified": not problems,
        "problems": sorted(set(problems)),
        "exact_output_equivalent": len(fingerprints) == 1 and not any("fingerprint" in p for p in problems),
        "metrics": metrics,
        "experiment_kind": "A/A-calibration" if calibration else "candidate-comparison",
        "code_improvement": "not-applicable-identical-actual-candidate" if calibration else "not-established",
        "output_equivalence": "exact"
        if len(fingerprints) == 1
        else "recipe-qualified-exact"
        if equivalences and all(equivalences)
        else None,
        "scope": "ordinary-daemon-terminal-route",
        "cache_state": "fresh archive and bytecode directory per run; host filesystem cache uncontrolled",
    }


def run_experiment(
    config: RunConfig,
    *,
    before: Path,
    after: Path,
    order: str = "ABBA",
    repetitions: int = 1,
    progress: Callable[[str], None] = print,
    acceptance_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if repetitions < 1 or not order or set(order) != {"A", "B"} or order.count("A") != order.count("B"):
        raise ValueError("order must contain equal positive counts of A and B; repetitions must be positive")
    if not config.fingerprint:
        raise ValueError("paired experiments require exact output fingerprints")
    manifest = load_manifest(config.corpus)
    verify_manifest(config.corpus, manifest)
    if not manifest["file_count"]:
        raise ValueError("the corpus holds no files")
    if config.work.exists() and any(config.work.iterdir()):
        raise ValueError("experiment work directory must be absent or empty")
    config.work.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.work.chmod(0o700)
    samples = []
    candidates = {"A": before.resolve(), "B": after.resolve()}
    launch = None
    if acceptance_policy is not None:
        from devtools.fresh_build_bench.acceptance import SCHEDULE, freeze_launch

        if order != "ABBA" or repetitions not in {1, 2}:
            raise ValueError("acceptance policy owns two ABBA blocks; conflicting order/repetitions refused")
        launch = freeze_launch(acceptance_policy, manifest, config, candidates)
        order, repetitions = SCHEDULE[:4], 2
    for index, arm in enumerate(order * repetitions):
        work = config.work / f"{index + 1:03d}-{arm}"
        progress(f"experiment {index + 1}/{len(order) * repetitions} arm {arm}")
        binding = None
        if launch is not None:
            from devtools.fresh_build_bench.acceptance import launch_binding

            binding = launch_binding(config.work / "launch.json", launch, index + 1, arm)
        receipt = run_build(
            replace(config, candidate=candidates[arm], work=work, label=arm, acceptance_binding=binding),
            progress=progress,
        )
        path = work / "receipt.json"
        samples.append(
            {
                "arm": arm,
                "ordinal": index + 1,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "receipt": receipt,
            }
        )
        # An invalid run is useful evidence, but cannot justify more paired
        # work. Preserve its receipt and emit a partial experiment document.
        evaluation = evaluate_run(receipt)
        if any(evaluation[axis]["state"] != "pass" for axis in ("measurement_validity", "scope_completion")):
            break
    complete = len(samples) == len(order) * repetitions
    # A partial sequence may lack one arm; avoid inventing its statistics.
    reduction = (
        reduce_experiment(samples, launch=launch)
        if {sample["arm"] for sample in samples} == {"A", "B"}
        else {
            "qualified": False,
            "problems": ["paired sequence incomplete"],
            "metrics": {},
            "evaluation": evaluate_samples(samples, launch=launch),
        }
    )
    document = {
        "format": "polylogue.fresh-build-experiment.v1",
        "order": order,
        "repetitions": repetitions,
        "complete": complete,
        "acceptance_launch": launch,
        **reduction,
        "samples": [{key: value for key, value in sample.items() if key != "receipt"} for sample in samples],
    }
    document["qualified"] = document["qualified"] and complete
    _sequence_completion(document)
    (config.work / "experiment.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document


def refresh_experiment(path: Path) -> dict[str, Any]:
    """Write a derived comparison; preserve experiment and constituent receipts."""
    from devtools.fresh_build_bench.report import benchmark_implementation_sha256, refresh

    original = path.read_bytes()
    document: dict[str, Any] = json.loads(original)
    if document.get("acceptance_launch") is not None:
        from devtools.fresh_build_bench.acceptance import digest

        recorded_launch = json.loads(path.with_name("launch.json").read_text())
        if digest(recorded_launch) != digest(document["acceptance_launch"]):
            raise ValueError("experiment frozen launch changed")
    samples = []
    for sample in document["samples"]:
        recording = Path(sample["path"])
        if hashlib.sha256(recording.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("experiment constituent receipt changed")
        samples.append({**sample, "receipt": refresh(recording)})
    reduction = (
        reduce_experiment(samples, launch=document.get("acceptance_launch"))
        if {sample["arm"] for sample in samples} == {"A", "B"}
        else {
            "qualified": False,
            "problems": ["paired sequence incomplete"],
            "metrics": {},
            "evaluation": evaluate_samples(samples, launch=document.get("acceptance_launch")),
        }
    )
    document.update(reduction)
    document["qualified"] = reduction["qualified"] and document["complete"]
    _sequence_completion(document)
    document["derived_from"] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(original).hexdigest(),
        "reducer_sha256": benchmark_implementation_sha256(),
    }
    payload = json.dumps(document, indent=1, sort_keys=True) + "\n"
    view = path.with_name(f"{path.stem}.view-{hashlib.sha256(payload.encode()).hexdigest()}.json")
    view.write_text(payload)
    return document


def _sequence_completion(document: dict[str, Any]) -> None:
    """Completing a run does not complete the manifest's paired sequence."""
    evaluation = document["evaluation"]
    evaluation["scope_completion"]["paired_sequence_complete"] = document["complete"]
    if not document["complete"]:
        evaluation["scope_completion"]["state"] = "fail"
        evaluation["scope_completion"]["failed"].append("paired_sequence_incomplete")
        evaluation["performance_decision"]["state"] = "rejected"
        evaluation["performance_decision"]["accepted"] = False
        if "declared_scope_completion_not_confirmed" not in evaluation["performance_decision"]["gaps"]:
            evaluation["performance_decision"]["gaps"].append("declared_scope_completion_not_confirmed")
