"""Launch-frozen operational repeatability policy and bootstrap binding.

Decisions belong to evaluation.py. This module validates and retains inputs;
it provides no statistical inference or global resource-cleanup proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

POLICY_FORMAT = "polylogue.fresh-build-practical-policy.v1"
LAUNCH_FORMAT = "polylogue.fresh-build-practical-launch.v1"
SCHEDULE = "ABBAABBA"
RESOURCE_METRICS = ("cpu_to_promotion_s", "rss_sampled_peak_bytes")


def controlled_environment(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row
        for key, row in value.items()
        if key not in {"load_average_at_start", "host_mem_available_kib_at_start"}
    }


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fields(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} requires exactly {sorted(fields)}")
    return value


def _finite(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def validate_policy(policy: Any) -> dict[str, Any]:
    validated = _fields(policy, {"format", "target_metric", "effect", "resources", "cleanup", "confirmation"}, "policy")
    if validated["format"] != POLICY_FORMAT or validated["target_metric"] != "promotion_s":
        raise ValueError("unsupported policy: only promotion_s practical repeatability is implemented")
    effect = _fields(validated["effect"], {"minimum_absolute", "minimum_relative"}, "effect")
    if not all(_finite(value, positive=True) for value in effect.values()):
        raise ValueError("effect thresholds must be positive finite numbers")
    resources = _fields(validated["resources"], {"scope", *RESOURCE_METRICS}, "resources")
    if resources["scope"] != "sampled-daemon-tree":
        raise ValueError("unsupported resource scope: hard physical bounds are unavailable")
    for metric in RESOURCE_METRICS:
        limits = _fields(resources[metric], {"absolute_slack", "relative_slack", "ceiling"}, metric)
        if not all(_finite(value) for value in limits.values()) or not _finite(limits["ceiling"], positive=True):
            raise ValueError("resource slack must be finite nonnegative and ceilings finite positive")
    if validated["cleanup"] != "bound-daemon-shutdown; descendant-external-unclaimed":
        raise ValueError("unsupported cleanup policy: descendant and external cleanup is unknown")
    confirmation = _fields(validated["confirmation"], {"method", "purpose", "held_out_basis"}, "confirmation")
    if confirmation["method"] != "two-abba-all-pairs-v1" or confirmation["purpose"] != "held-out-confirmation":
        raise ValueError("unsupported confirmation method or purpose")
    basis = _fields(confirmation["held_out_basis"], {"attested_untuned", "statement", "discovery"}, "held_out_basis")
    if (
        basis["attested_untuned"] is not True
        or not isinstance(basis["statement"], str)
        or not basis["statement"].strip()
    ):
        raise ValueError("held-out confirmation requires an explicit untuned-selection attestation")
    if basis["discovery"] is not None:
        reference = _fields(basis["discovery"], {"path", "sha256"}, "discovery reference")
        if not all(isinstance(value, str) and value for value in reference.values()):
            raise ValueError("discovery reference requires path and sha256")
    return validated


def _units(parameters: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(row["origin"], row["source_root"], row["unit"]) for row in parameters.get("provenance", [])}


def selected_unit_hashes(parameters: dict[str, Any]) -> list[str]:
    return sorted(digest(unit) for unit in _units(parameters))


def freeze_launch(
    policy: dict[str, Any], manifest: dict[str, Any], config: Any, candidates: dict[str, Path]
) -> dict[str, Any]:
    from devtools.fresh_build_bench.report import benchmark_implementation_sha256, config_digest
    from devtools.fresh_build_bench.run import candidate_identity, environment

    validate_policy(policy)
    policy = json.loads(json.dumps(policy))
    if manifest["kind"] != "selected-real":
        raise ValueError("practical acceptance requires a selected-real confirmation corpus")
    discovery = policy["confirmation"]["held_out_basis"]["discovery"]
    nonoverlap = "unavailable; untuned held-out basis is an operator attestation"
    if discovery:
        data = Path(discovery["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != discovery["sha256"]:
            raise ValueError("discovery reference changed")
        document = json.loads(data)
        if document.get("format") != "polylogue.fresh-build-experiment.v1" or not document.get("samples"):
            raise ValueError("discovery reference must be a recorded experiment")
        current_units = set(selected_unit_hashes(manifest.get("parameters") or {}))
        observed_units: set[str] = set()
        for sample in document["samples"]:
            recording = Path(sample["path"]).read_bytes()
            if hashlib.sha256(recording).hexdigest() != sample["sha256"]:
                raise ValueError("discovery constituent changed")
            corpus = json.loads(recording)["corpus"]
            if corpus["digest"] == manifest["digest"]:
                raise ValueError("confirmation selection was used for discovery")
            observed_units.update(
                corpus.get("selected_unit_sha256") or selected_unit_hashes(corpus.get("parameters") or {})
            )
        if current_units and observed_units:
            if current_units & observed_units:
                raise ValueError("discovery and confirmation session units overlap")
            nonoverlap = "recorded selected session units disjoint; shared dependencies permitted"
    launch = {
        "format": LAUNCH_FORMAT,
        "policy": policy,
        "schedule": SCHEDULE,
        "candidate": {arm: candidate_identity(path) for arm, path in candidates.items()},
        "corpus_digest": manifest["digest"],
        "config_digest": config_digest(config),
        "environment": controlled_environment(environment(config)),
        "benchmark_implementation_sha256": benchmark_implementation_sha256(),
        "held_out_nonoverlap": nonoverlap,
    }
    path = config.work / "launch.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(launch, sort_keys=True, indent=1) + "\n")
    path.chmod(0o400)
    return launch


def launch_binding(path: Path, launch: dict[str, Any], ordinal: int, arm: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(launch), "ordinal": ordinal, "arm": arm}


def read_bound_launch(binding: dict[str, Any]) -> dict[str, Any]:
    launch: dict[str, Any] = json.loads(Path(binding["path"]).read_text())
    if launch.get("format") != LAUNCH_FORMAT or digest(launch) != binding["sha256"]:
        raise ValueError("frozen launch changed")
    validate_policy(launch["policy"])
    ordinal = binding["ordinal"]
    if type(ordinal) is not int or not 1 <= ordinal <= len(SCHEDULE) or launch["schedule"] != SCHEDULE:
        raise ValueError("invalid planned confirmation ordinal")
    if launch["schedule"][ordinal - 1] != binding["arm"]:
        raise ValueError("confirmation arm differs from frozen schedule")
    return launch


def bootstrap_binding() -> dict[str, Any] | None:
    """Called in the actual daemon before its main entry point."""
    raw = os.environ.get("POLYLOGUE_BENCH_ACCEPTANCE")
    if raw is None:
        return None
    binding = json.loads(raw)
    read_bound_launch(binding)
    return {**binding, "state": "bound-before-main"}
