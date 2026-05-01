"""Generate semantic-axis performance evidence from synthetic benchmark tiers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDocument, require_json_value
from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.models import EvidenceEnvelope, TrustMetadata
from polylogue.scenarios import CorpusSourceKind
from polylogue.storage.backends.schema_ddl import SCHEMA_VERSION

from .benchmark_campaigns import SYNTHETIC_CAMPAIGNS, run_synthetic_benchmark_campaign
from .large_archive_generator import ScaleLevel, generate_archive, get_default_spec

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / ".local" / "performance-evidence"
RUNNER_VERSION = "semantic-axis-evidence.v1"
SCALE_ORDER = ("small", "medium", "large", "stretch")
AXIS_DB_STATS: dict[str, str] = {
    "messages": "messages_count",
    "conversations": "conversations_count",
    "content-blocks": "content_blocks_count",
    "raw-artifacts": "raw_conversations_count",
    "action-events": "action_events_count",
}


@dataclass(frozen=True, slots=True)
class ScaleObservation:
    """One campaign measurement at one semantic scale tier."""

    scale: str
    axis_value: float
    metric_value: float
    metrics: Mapping[str, float]
    db_stats: Mapping[str, int]
    generation_wall_s: float | None = None

    def to_payload(self) -> JSONDocument:
        return {
            "scale": self.scale,
            "axis_value": self.axis_value,
            "metric_value": self.metric_value,
            "metrics": dict(self.metrics),
            "db_stats": dict(self.db_stats),
            "generation_wall_s": self.generation_wall_s,
        }


def build_scale_observation(
    *,
    scale: str,
    metric: str,
    axis_stat_key: str,
    metrics: Mapping[str, float],
    db_stats: Mapping[str, int],
    generation_wall_s: float | None = None,
) -> ScaleObservation:
    """Extract the selected semantic axis and metric from one campaign result."""
    if metric not in metrics:
        raise ValueError(f"campaign result does not include metric {metric!r}")
    if axis_stat_key not in db_stats:
        raise ValueError(f"campaign result does not include axis stat {axis_stat_key!r}")

    axis_value = float(db_stats[axis_stat_key])
    metric_value = float(metrics[metric])
    if axis_value <= 0:
        raise ValueError(f"axis stat {axis_stat_key!r} must be positive, got {axis_value}")
    if metric_value < 0:
        raise ValueError(f"metric {metric!r} must be non-negative, got {metric_value}")

    return ScaleObservation(
        scale=scale,
        axis_value=axis_value,
        metric_value=metric_value,
        metrics=dict(metrics),
        db_stats=dict(db_stats),
        generation_wall_s=generation_wall_s,
    )


def build_semantic_axis_evidence(
    *,
    campaign: str,
    semantic_axis: str,
    axis_stat_key: str,
    metric: str,
    observations: Sequence[ScaleObservation],
    reproducer: Sequence[str],
    environment: JSONDocument | None = None,
    baseline_payload: JSONDocument | None = None,
    reviewed_at: str | None = None,
    artifacts: Sequence[str] = (),
) -> EvidenceEnvelope:
    """Build a performance EvidenceEnvelope from scale-tier observations."""
    ordered = _ordered_observations(observations)
    adjacent_growth = _adjacent_growth(ordered)
    growth_shape = _classify_growth_shape(adjacent_growth)
    growth_summary = _growth_summary(adjacent_growth, growth_shape)
    baseline_comparison = _baseline_comparison(growth_shape, baseline_payload)
    counterexample: JSONDocument | None = None
    if baseline_comparison.get("changed_growth_behavior") is True:
        counterexample = {
            "reason": "growth-shape changed against comparable baseline",
            "baseline_comparison": baseline_comparison,
        }

    status = (
        OutcomeStatus.ERROR if counterexample is not None or growth_shape == "insufficient-tiers" else OutcomeStatus.OK
    )
    environment_payload = environment or _environment_payload(campaign=campaign)
    scale_tiers = [observation.to_payload() for observation in ordered]
    evidence: JSONDocument = {
        "runner_class": "semantic_axis_performance",
        "campaign": campaign,
        "semantic_axis": semantic_axis,
        "axis_db_stat": axis_stat_key,
        "metric": metric,
        "growth_shape": growth_shape,
        "growth_summary": growth_summary,
        "scale_tiers": require_json_value(scale_tiers),
        "adjacent_growth": require_json_value(adjacent_growth),
        "baseline_comparison": baseline_comparison,
        "regression_expression": (
            "regressions are expressed as changed growth_shape or candidate-vs-baseline tier metric drift"
        ),
    }
    trust = _trust_metadata(
        producer="devtools.semantic_axis_evidence",
        reviewed_at=reviewed_at,
        input_payload={
            "campaign": campaign,
            "semantic_axis": semantic_axis,
            "axis_stat_key": axis_stat_key,
            "metric": metric,
            "observations": require_json_value(scale_tiers),
            "baseline_comparison": baseline_comparison,
        },
        environment=environment_payload,
    )
    return EvidenceEnvelope.build(
        obligation_id=f"performance.semantic_axis.{campaign}.{semantic_axis}",
        status=status,
        evidence=evidence,
        counterexample=counterexample,
        reproducer=tuple(reproducer),
        artifacts=tuple(artifacts),
        environment=environment_payload,
        trust=trust,
    )


async def run_semantic_axis_evidence(
    *,
    campaign: str,
    semantic_axis: str,
    metric: str,
    scales: Sequence[str],
    output_dir: Path,
    corpus_source: CorpusSourceKind,
    seed: int,
    baseline_payload: JSONDocument | None = None,
) -> EvidenceEnvelope:
    """Run a benchmark campaign across semantic scale tiers and return evidence."""
    axis_stat_key = AXIS_DB_STATS[semantic_axis]
    observations: list[ScaleObservation] = []
    archive_root = output_dir / "archives" / campaign
    archive_root.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        level = ScaleLevel(scale)
        spec = get_default_spec(level)
        if seed != spec.seed:
            from dataclasses import replace

            spec = replace(spec, seed=seed)
        archive_dir = archive_root / scale
        generation_started = time.monotonic()
        await generate_archive(spec, archive_dir, corpus_source=corpus_source)
        generation_wall_s = time.monotonic() - generation_started
        result = await run_synthetic_benchmark_campaign(campaign, archive_dir / "benchmark.db")
        result.scale_level = scale
        observations.append(
            build_scale_observation(
                scale=scale,
                metric=metric,
                axis_stat_key=axis_stat_key,
                metrics=result.metrics,
                db_stats=result.db_stats,
                generation_wall_s=round(generation_wall_s, 3),
            )
        )

    return build_semantic_axis_evidence(
        campaign=campaign,
        semantic_axis=semantic_axis,
        axis_stat_key=axis_stat_key,
        metric=metric,
        observations=observations,
        reproducer=_reproducer_command(
            campaign=campaign,
            semantic_axis=semantic_axis,
            metric=metric,
            scales=scales,
            corpus_source=corpus_source,
            seed=seed,
        ),
        environment=_environment_payload(campaign=campaign),
        baseline_payload=baseline_payload,
    )


def _ordered_observations(observations: Sequence[ScaleObservation]) -> list[ScaleObservation]:
    if len(observations) < 2:
        return list(observations)
    scale_rank = {scale: index for index, scale in enumerate(SCALE_ORDER)}
    return sorted(observations, key=lambda item: (scale_rank.get(item.scale, len(SCALE_ORDER)), item.axis_value))


def _adjacent_growth(observations: Sequence[ScaleObservation]) -> list[JSONDocument]:
    growth: list[JSONDocument] = []
    for previous, current in zip(observations, observations[1:], strict=False):
        axis_ratio = current.axis_value / previous.axis_value if previous.axis_value > 0 else None
        metric_ratio = current.metric_value / previous.metric_value if previous.metric_value > 0 else None
        elasticity = None
        if axis_ratio is not None and metric_ratio is not None and axis_ratio > 1.0 and metric_ratio > 0.0:
            elasticity = math.log(metric_ratio) / math.log(axis_ratio)
        growth.append(
            {
                "from_scale": previous.scale,
                "to_scale": current.scale,
                "axis_ratio": axis_ratio,
                "metric_ratio": metric_ratio,
                "elasticity": elasticity,
                "from_metric_per_axis": previous.metric_value / previous.axis_value,
                "to_metric_per_axis": current.metric_value / current.axis_value,
            }
        )
    return growth


def _classify_growth_shape(adjacent_growth: Sequence[Mapping[str, Any]]) -> str:
    elasticities = [
        float(item["elasticity"])
        for item in adjacent_growth
        if isinstance(item.get("elasticity"), int | float) and not math.isnan(float(item["elasticity"]))
    ]
    metric_ratios = [
        float(item["metric_ratio"])
        for item in adjacent_growth
        if isinstance(item.get("metric_ratio"), int | float) and not math.isnan(float(item["metric_ratio"]))
    ]
    if not elasticities:
        return "insufficient-tiers"
    if metric_ratios and max(metric_ratios) <= 1.15:
        return "near-constant"
    max_elasticity = max(elasticities)
    if max_elasticity <= 0.75:
        return "sublinear"
    if max_elasticity <= 1.35:
        return "approximately-linear"
    return "superlinear"


def _growth_summary(adjacent_growth: Sequence[Mapping[str, Any]], growth_shape: str) -> JSONDocument:
    elasticities = [
        float(item["elasticity"])
        for item in adjacent_growth
        if isinstance(item.get("elasticity"), int | float) and not math.isnan(float(item["elasticity"]))
    ]
    metric_ratios = [
        float(item["metric_ratio"])
        for item in adjacent_growth
        if isinstance(item.get("metric_ratio"), int | float) and not math.isnan(float(item["metric_ratio"]))
    ]
    axis_ratios = [
        float(item["axis_ratio"])
        for item in adjacent_growth
        if isinstance(item.get("axis_ratio"), int | float) and not math.isnan(float(item["axis_ratio"]))
    ]
    return {
        "shape": growth_shape,
        "max_elasticity": max(elasticities) if elasticities else None,
        "mean_elasticity": (sum(elasticities) / len(elasticities)) if elasticities else None,
        "max_metric_ratio": max(metric_ratios) if metric_ratios else None,
        "max_axis_ratio": max(axis_ratios) if axis_ratios else None,
        "adjacent_pair_count": len(adjacent_growth),
    }


def _baseline_comparison(growth_shape: str, baseline_payload: JSONDocument | None) -> JSONDocument:
    if baseline_payload is None:
        return {
            "compared": False,
            "changed_growth_behavior": False,
            "note": "no baseline supplied; evidence uses multiple generated scale tiers",
        }
    baseline_evidence = baseline_payload.get("evidence", {})
    baseline_shape = None
    if isinstance(baseline_evidence, dict):
        baseline_shape = baseline_evidence.get("growth_shape")
    changed = baseline_shape is not None and str(baseline_shape) != growth_shape
    return {
        "compared": True,
        "baseline_growth_shape": baseline_shape,
        "candidate_growth_shape": growth_shape,
        "changed_growth_behavior": changed,
    }


def _environment_payload(*, campaign: str) -> JSONDocument:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "campaign": campaign,
    }


def _trust_metadata(
    *,
    producer: str,
    input_payload: JSONDocument,
    environment: JSONDocument,
    reviewed_at: str | None = None,
) -> TrustMetadata:
    return TrustMetadata(
        producer=producer,
        reviewed_at=reviewed_at or datetime.now(UTC).isoformat(),
        level="generated",
        privacy="repo-local synthetic benchmark metadata only",
        code_revision=_git_stdout("rev-parse", "HEAD"),
        dirty_state=_git_dirty_state(),
        schema_version=SCHEMA_VERSION,
        input_fingerprint=_json_digest(input_payload),
        environment_fingerprint=_json_digest(environment),
        runner_version=RUNNER_VERSION,
        freshness="generated from current checkout synthetic benchmark tiers",
        origin="semantic-axis-evidence",
    )


def _json_digest(payload: JSONDocument) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_stdout(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_dirty_state() -> bool | None:
    status = _git_stdout("status", "--short")
    return None if status is None else bool(status.strip())


def _reproducer_command(
    *,
    campaign: str,
    semantic_axis: str,
    metric: str,
    scales: Sequence[str],
    corpus_source: CorpusSourceKind,
    seed: int,
) -> tuple[str, ...]:
    return (
        "devtools semantic-axis-evidence "
        f"--campaign {campaign} "
        f"--axis {semantic_axis} "
        f"--metric {metric} "
        f"--corpus-source {corpus_source.value} "
        f"--seed {seed} "
        f"--scales {' '.join(scales)}",
    )


def _default_metric(campaign: str) -> str:
    entry = SYNTHETIC_CAMPAIGNS[campaign]
    if not entry.summary_metric:
        raise ValueError(f"campaign {campaign!r} does not declare a summary metric")
    return entry.summary_metric


def _load_baseline(path: Path | None) -> JSONDocument | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"baseline evidence must be a JSON object: {path}")
    return payload


def _write_evidence(output_dir: Path, envelope: EvidenceEnvelope) -> tuple[Path, JSONDocument]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    campaign = str(envelope.evidence["campaign"])
    axis = str(envelope.evidence["semantic_axis"])
    path = output_dir / f"{stamp}-{campaign}-{axis}.evidence.json"
    payload = envelope.to_payload()
    payload["artifacts"] = [str(path)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=sorted(SYNTHETIC_CAMPAIGNS), default="fts-rebuild")
    parser.add_argument("--axis", choices=sorted(AXIS_DB_STATS), default="messages")
    parser.add_argument(
        "--metric", help="Metric key from the campaign result. Defaults to the campaign summary metric."
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=SCALE_ORDER,
        default=("small", "medium"),
        help="Synthetic scale tiers to sweep; at least two are needed for growth shape evidence.",
    )
    parser.add_argument(
        "--corpus-source",
        choices=[kind.value for kind in CorpusSourceKind],
        default=CorpusSourceKind.DEFAULT.value,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline", type=Path, help="Optional prior EvidenceEnvelope JSON for growth-shape comparison."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true", help="Print JSON only; do not write an evidence artifact.")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    metric = args.metric or _default_metric(args.campaign)
    baseline = _load_baseline(args.baseline)
    envelope = await run_semantic_axis_evidence(
        campaign=args.campaign,
        semantic_axis=args.axis,
        metric=metric,
        scales=tuple(args.scales),
        output_dir=args.output,
        corpus_source=CorpusSourceKind(args.corpus_source),
        seed=args.seed,
        baseline_payload=baseline,
    )
    payload = envelope.to_payload()
    if not args.no_save:
        _, payload = _write_evidence(args.output, envelope)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if envelope.status is OutcomeStatus.OK else 3


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
