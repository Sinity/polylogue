"""Paired ordinary-daemon experiments with immutable constituent receipts."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from devtools.fresh_build_bench.corpus import load_manifest, verify_manifest
from devtools.fresh_build_bench.report import comparability_problems
from devtools.fresh_build_bench.run import RunConfig, run_build


def reduce_experiment(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose raw samples, median and spread; claim speed only beyond noise."""
    problems: list[str] = []
    reference = samples[0]["receipt"]
    fingerprints = set()
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
        policy = receipt.get("effective_compute_policy")
        if not policy or policy.get("outcome") != "bound":
            problems.append("unbound effective compute policy")
        else:
            policies[sample["arm"]].add(json.dumps(policy["states"], sort_keys=True))
    if len(fingerprints) > 1:
        problems.append("different exact outputs")
    for arm, values in policies.items():
        if len(values) != 1:
            problems.append(f"arm {arm} effective compute policy changed or unavailable")
    for arm, values in candidates.items():
        if len(values) != 1:
            problems.append(f"arm {arm} candidate changed between runs")
    metrics = {}
    for name in ("promotion", "terminal"):
        arms = {}
        for arm in ("A", "B"):
            raw = [sample["receipt"]["timing_s"].get(name) for sample in samples if sample["arm"] == arm]
            if any(value is None for value in raw):
                problems.append(f"missing {name} measurement for {arm}")
                arms[arm] = {"raw_s": raw, "median_s": None, "mad_s": None, "range_s": None}
                continue
            median = statistics.median(raw)
            arms[arm] = {
                "raw_s": raw,
                "median_s": median,
                "mad_s": statistics.median(abs(value - median) for value in raw),
                "range_s": [min(raw), max(raw)],
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
            assessment = "B-faster-than-observed-spread" if delta < 0 else "B-slower-than-observed-spread"
        metrics[name] = {"arms": arms, "delta_median_s": delta, "assessment": assessment}
    return {
        "qualified": not problems,
        "problems": sorted(set(problems)),
        "exact_output_equivalent": len(fingerprints) == 1 and not any("fingerprint" in p for p in problems),
        "metrics": metrics,
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
    for index, arm in enumerate(order * repetitions):
        work = config.work / f"{index + 1:03d}-{arm}"
        progress(f"experiment {index + 1}/{len(order) * repetitions} arm {arm}")
        receipt = run_build(replace(config, candidate=candidates[arm], work=work, label=arm), progress=progress)
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
        if not receipt.get("qualified"):
            break
    complete = len(samples) == len(order) * repetitions
    # A partial sequence may lack one arm; avoid inventing its statistics.
    reduction = (
        reduce_experiment(samples)
        if {sample["arm"] for sample in samples} == {"A", "B"}
        else {"qualified": False, "problems": ["paired sequence incomplete"], "metrics": {}}
    )
    document = {
        "format": "polylogue.fresh-build-experiment.v1",
        "order": order,
        "repetitions": repetitions,
        "complete": complete,
        **reduction,
        "samples": [{key: value for key, value in sample.items() if key != "receipt"} for sample in samples],
    }
    document["qualified"] = document["qualified"] and complete
    (config.work / "experiment.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document
