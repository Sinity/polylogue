"""Stratified rebuild-cost benchmark (polylogue-623q follow-up).

The only way to know whether a rebuild-path change helped used to be running
a full multi-hour rebuild against the live archive. This module builds a
cheap substitute: stratify the raw population by (origin x byte-size decile),
synthesize a small representative sample per stratum, drive it through the
REAL rebuild engine (``polylogue.maintenance.rebuild_index.
rebuild_index_from_source_sync`` -- the same function the offline CLI and
daemon bulk-rebuild call, no reimplementation), measure seconds-per-raw at
each stratum's characteristic size, and extrapolate to the full population
using the population's real per-stratum counts.

Two cost regimes exist in this archive's raw population and neither should be
assumed to dominate a priori:

- byte-bound: a few thousand huge raws (multi-MB Codex rollouts) where cost
  scales with payload bytes (parse/decode + writer throughput).
- count-bound: tens of thousands of small raws where FIXED per-raw overhead
  (transaction bookkeeping, blob open, census receipt, index inserts) matters
  more than their few KB of payload.

Sampling at each stratum's OWN characteristic mean size and then scaling by
seconds-per-raw naturally blends both regimes without needing to fit which
one applies -- a stratum's measured seconds-per-raw already reflects
whatever mix of fixed and byte-proportional cost governs raws of that shape.

``PopulationSnapshot`` below is a captured, aggregate-only (counts and byte
sums, never content) description of the live archive's raw population,
recorded because the live archive is not available to CI/cloud runs and
carries private content this public repo must never expose. Point
``collect_population_strata`` at a real ``source.db`` (read-only) to refresh
it from a live archive -- the snapshot embedded here does not need to be
re-derived to run the model; it is a fixed reference point analogous to
``PopulationSnapshot.captured_at``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

# ---------------------------------------------------------------------------
# Population strata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stratum:
    """One (origin, byte-size decile) population bucket."""

    label: str
    provider: Provider
    count: int
    total_bytes: int

    @property
    def mean_bytes(self) -> int:
        return max(1, round(self.total_bytes / self.count)) if self.count else 0


def collect_population_strata(
    source_db: Path, *, deciles_by_origin: Sequence[str] = ("codex-session", "claude-code-session")
) -> list[Stratum]:
    """Read-only: stratify a live archive's raw_sessions by origin x byte-weighted decile.

    Never reads payload content -- only ``origin``/``blob_size`` aggregate
    columns. Origins listed in ``deciles_by_origin`` (the byte-dominant ones)
    are split into 10 byte-weighted deciles each; every other origin is
    pooled into one "long tail" stratum per origin group, since they are
    small enough in aggregate bytes that within-origin size variance does not
    materially change the wall-clock projection.
    """
    strata: list[Stratum] = []
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        origins = [str(r[0]) for r in conn.execute("SELECT DISTINCT origin FROM raw_sessions")]
        tail_count = 0
        tail_bytes = 0
        for origin in origins:
            provider = _provider_for_origin(origin)
            sizes = [int(r[0]) for r in conn.execute("SELECT blob_size FROM raw_sessions WHERE origin = ?", (origin,))]
            if not sizes:
                continue
            if origin not in deciles_by_origin:
                tail_count += len(sizes)
                tail_bytes += sum(sizes)
                continue
            sizes.sort(reverse=True)
            total = sum(sizes)
            bucket_target = total / 10 if total else 0
            bucket_counts = [0] * 10
            bucket_bytes = [0] * 10
            acc = 0
            bi = 0
            for size in sizes:
                acc += size
                bucket_counts[bi] += 1
                bucket_bytes[bi] += size
                if bucket_target and acc >= bucket_target * (bi + 1) and bi < 9:
                    bi += 1
            for decile, (count, nbytes) in enumerate(zip(bucket_counts, bucket_bytes, strict=True)):
                if count == 0:
                    continue
                strata.append(Stratum(label=f"{origin}/d{decile}", provider=provider, count=count, total_bytes=nbytes))
        if tail_count:
            strata.append(
                Stratum(
                    label="long-tail/other-origins", provider=Provider.CODEX, count=tail_count, total_bytes=tail_bytes
                )
            )
    return strata


def _provider_for_origin(origin: str) -> Provider:
    # Synthesis only supports origins with a validated realistic payload
    # generator below (Codex, Claude Code). Every other origin is folded into
    # the long-tail Codex-shaped stratum by collect_population_strata's
    # deciles_by_origin default -- see module docstring's "documented
    # simplification".
    if origin == "claude-code-session":
        return Provider.CLAUDE_CODE
    return Provider.CODEX


#: Aggregate-only snapshot of the live archive's raw population, captured
#: 2026-07-30 via a read-only ``mode=ro`` connection (counts/bytes only, no
#: content -- see ``.claude/CLAUDE.md`` "repo is PUBLIC" constraint). Refresh
#: with ``collect_population_strata`` against a live ``source.db`` when the
#: archive shape has materially changed.
POPULATION_SNAPSHOT_CAPTURED_AT = datetime(2026, 7, 30, tzinfo=timezone.utc)
POPULATION_SNAPSHOT: tuple[Stratum, ...] = (
    Stratum("codex-session/d0", Provider.CODEX, 16, 7_201_200_000),
    Stratum("codex-session/d1", Provider.CODEX, 19, 7_344_300_000),
    Stratum("codex-session/d2", Provider.CODEX, 24, 6_929_000_000),
    Stratum("codex-session/d3", Provider.CODEX, 38, 7_139_300_000),
    Stratum("codex-session/d4", Provider.CODEX, 49, 7_178_300_000),
    Stratum("codex-session/d5", Provider.CODEX, 111, 7_060_600_000),
    Stratum("codex-session/d6", Provider.CODEX, 234, 7_128_400_000),
    Stratum("codex-session/d7", Provider.CODEX, 428, 7_146_500_000),
    Stratum("codex-session/d8", Provider.CODEX, 715, 7_131_400_000),
    Stratum("codex-session/d9", Provider.CODEX, 7508, 7_136_700_000),
    Stratum("claude-code-session/d0", Provider.CLAUDE_CODE, 5, 2_266_100_000),
    Stratum("claude-code-session/d1", Provider.CLAUDE_CODE, 32, 2_073_200_000),
    Stratum("claude-code-session/d2", Provider.CLAUDE_CODE, 51, 2_111_600_000),
    Stratum("claude-code-session/d3", Provider.CLAUDE_CODE, 93, 2_155_900_000),
    Stratum("claude-code-session/d4", Provider.CLAUDE_CODE, 139, 2_136_200_000),
    Stratum("claude-code-session/d5", Provider.CLAUDE_CODE, 205, 2_150_000_000),
    Stratum("claude-code-session/d6", Provider.CLAUDE_CODE, 339, 2_147_700_000),
    Stratum("claude-code-session/d7", Provider.CLAUDE_CODE, 745, 2_145_500_000),
    Stratum("claude-code-session/d8", Provider.CLAUDE_CODE, 2142, 2_146_600_000),
    Stratum("claude-code-session/d9", Provider.CLAUDE_CODE, 15954, 2_147_700_000),
    Stratum("long-tail/other-origins", Provider.CODEX, 12725, 6_439_959_524),
)

#: The one real, measured full-corpus rebuild wall-clock -- the acceptance
#: criterion. A model that cannot reproduce this within a stated margin is
#: not trustworthy for evaluating future changes (operator directive).
CALIBRATION_WALL_S = 4 * 3600 + 20 * 60  # 4h20m
CALIBRATION_RAW_COUNT = 41_363
CALIBRATION_TOTAL_BYTES = round(92.4 * 1024**3)


# ---------------------------------------------------------------------------
# Synthetic per-stratum corpus
# ---------------------------------------------------------------------------

_ENVELOPE_OVERHEAD_BYTES = 220


def _codex_payload(index: int, *, target_bytes: int) -> bytes:
    session_meta = (
        json.dumps(
            {"type": "session_meta", "payload": {"id": f"cost-model-{index:06d}", "timestamp": "2026-06-01T00:00:00Z"}},
            separators=(",", ":"),
        )
        + "\n"
    )
    text_len = max(1, target_bytes - len(session_meta) - _ENVELOPE_OVERHEAD_BYTES)
    text = f"cost-model-payload-{index:06d}-" + ("x" * text_len)
    response_item = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return (session_meta + response_item).encode()


def _claude_code_payload(index: int, *, target_bytes: int) -> bytes:
    session_id = str(uuid.UUID(int=index, version=4))
    base_ts = 1_700_000_000.0 + index * 60
    pad_len = max(1, target_bytes - 2 * _ENVELOPE_OVERHEAD_BYTES)
    lines: list[str] = []
    for turn, role in enumerate(("user", "assistant")):
        record = {
            "type": role,
            "uuid": str(uuid.UUID(int=index * 2 + turn + 1, version=4)),
            "parentUuid": str(uuid.UUID(int=index * 2 + turn, version=4)) if (index or turn) else None,
            "sessionId": session_id,
            "message": {
                "role": role,
                "content": [{"type": "text", "text": ("x" * pad_len) if turn == 0 else "ack"}],
            },
            "timestamp": datetime.fromtimestamp(base_ts + turn, tz=timezone.utc).isoformat(),
        }
        lines.append(json.dumps(record, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def _payload_for_provider(provider: Provider, index: int, *, target_bytes: int) -> bytes:
    if provider is Provider.CLAUDE_CODE:
        return _claude_code_payload(index, target_bytes=target_bytes)
    return _codex_payload(index, target_bytes=target_bytes)


def build_stratum_sample_corpus(archive_root: Path, stratum: Stratum, *, sample_n: int) -> list[str]:
    """Write ``sample_n`` synthetic raws sized at ``stratum.mean_bytes`` each.

    Fresh archive root; the caller owns cleanup.
    """
    initialize_active_archive_root(archive_root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(sample_n):
            payload = _payload_for_provider(stratum.provider, index, target_bytes=stratum.mean_bytes)
            raw_id = archive.write_raw_payload(
                provider=stratum.provider,
                payload=payload,
                source_path=f"cost-model/{stratum.label}/{index:06d}.jsonl",
                acquired_at_ms=index + 1,
            )
            raw_ids.append(raw_id)
    return raw_ids


def default_sample_n(stratum: Stratum, *, target_sample_bytes: int = 2_000_000, min_n: int = 3, max_n: int = 40) -> int:
    """More samples for small/count-bound strata, fewer for byte-bound whales."""
    if stratum.mean_bytes <= 0:
        return min(stratum.count, min_n)
    raw_target = max(min_n, round(target_sample_bytes / stratum.mean_bytes))
    return max(1, min(stratum.count, min(max_n, raw_target)))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StratumMeasurement:
    """Two-point regression result for one stratum.

    A single-sample measurement conflates two different costs: the FIXED
    one-time overhead ``rebuild_index_from_source_sync`` pays per PASS
    (generation bootstrap, embeddings/FTS bulk-repopulate, promote,
    readiness checks -- paid once no matter how many raws the pass
    replays) and the MARGINAL cost per raw actually replayed. Dividing a
    single sample's wall-clock by its (small) sample_n and multiplying by
    the population count multiplies the fixed cost too, which the real
    full rebuild pays only ONCE across the whole population -- this was
    the source of a measured 1.78x over-prediction in the first version of
    this model (see polylogue-623q notes).

    Fitting two points (``n1`` < ``n2`` raws, same stratum, fresh archive
    each) separates the two: ``marginal_s_per_raw`` is the slope
    ``(wall_s2 - wall_s1) / (n2 - n1)``; ``fixed_s`` is the intercept.
    Population extrapolation then multiplies ONLY the marginal term
    (``predicted_wall_s``) -- ``fixed_s`` is reported for transparency but
    deliberately not added into any per-stratum or population total: the
    real full rebuild runs as ONE pass, so its one-time fixed cost should
    be counted once across the whole run, not once per stratum sample (and
    this model does not attempt that separate accounting -- see the
    module-level report footer).
    """

    stratum: Stratum
    n1: int
    n2: int
    wall_s1: float
    wall_s2: float
    fixed_s: float
    marginal_s_per_raw: float
    sample_bytes: int
    parse_s: float
    apply_s: float
    regression_valid: bool

    @property
    def predicted_wall_s(self) -> float:
        """Population extrapolation using ONLY the marginal per-raw term."""
        return max(0.0, self.marginal_s_per_raw) * self.stratum.count


def _run_one_rebuild_pass(archive_root: Path, stratum: Stratum, n: int) -> tuple[float, dict[str, object]]:
    build_stratum_sample_corpus(archive_root, stratum, sample_n=n)
    prior_env = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        started = time.perf_counter()
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=archive_root, promote=True, raw_batch_size=max(n, 1))
        )
        wall_s = time.perf_counter() - started
    finally:
        if prior_env is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = prior_env
    assert receipt.status == "replayed", f"stratum {stratum.label}: unexpected receipt status {receipt.status!r}"
    return wall_s, receipt.replay


def two_point_sample_sizes(stratum: Stratum) -> tuple[int, int]:
    """(n1, n2) sample sizes for the regression -- n2 > n1 when population allows."""
    n1 = default_sample_n(stratum)
    n2 = min(stratum.count, max(n1 + 3, n1 * 4))
    return n1, n2


def measure_stratum(
    archive_root: Path, stratum: Stratum, *, sample_sizes: tuple[int, int] | None = None
) -> StratumMeasurement:
    """Two-point regression: replay the same stratum at two sample sizes to
    separate fixed per-pass overhead from marginal per-raw cost.

    Each sample gets a fresh scratch archive (``archive_root / "n1"`` /
    ``"n2"``) so neither pass's index/generation state leaks into the other.
    """
    n1, n2 = sample_sizes if sample_sizes is not None else two_point_sample_sizes(stratum)
    wall_s1, replay1 = _run_one_rebuild_pass(archive_root / "n1", stratum, n1)
    regression_valid = n2 > n1
    if regression_valid:
        wall_s2, replay2 = _run_one_rebuild_pass(archive_root / "n2", stratum, n2)
        marginal_s_per_raw = (wall_s2 - wall_s1) / (n2 - n1)
        fixed_s = wall_s1 - marginal_s_per_raw * n1
        replay = replay2
    else:
        # Population too small to split (n2 == n1): fall back to treating the
        # whole single-sample wall-clock as marginal cost, matching the prior
        # (known-biased) behavior for this stratum only. These strata are a
        # tiny fraction of the population by construction (see
        # POPULATION_SNAPSHOT) so the bias this reintroduces is bounded.
        wall_s2 = wall_s1
        marginal_s_per_raw = wall_s1 / n1 if n1 else 0.0
        fixed_s = 0.0
        replay = replay1

    parse_s = float(cast("float", replay.get("parse_s", 0.0)))
    apply_s = float(cast("float", replay.get("apply_s", 0.0)))
    return StratumMeasurement(
        stratum=stratum,
        n1=n1,
        n2=n2,
        wall_s1=wall_s1,
        wall_s2=wall_s2,
        fixed_s=fixed_s,
        marginal_s_per_raw=marginal_s_per_raw,
        sample_bytes=n2 * stratum.mean_bytes,
        parse_s=parse_s,
        apply_s=apply_s,
        regression_valid=regression_valid,
    )


@dataclass(slots=True)
class PredictedRun:
    measurements: list[StratumMeasurement] = field(default_factory=list)

    @property
    def total_predicted_wall_s(self) -> float:
        return sum(m.predicted_wall_s for m in self.measurements)

    @property
    def total_raws(self) -> int:
        return sum(m.stratum.count for m in self.measurements)

    @property
    def total_bytes(self) -> int:
        return sum(m.stratum.total_bytes for m in self.measurements)

    def calibration_ratio(self, calibration_wall_s: float = CALIBRATION_WALL_S) -> float:
        """predicted / actual -- 1.0 is a perfect match, >1 over-predicts."""
        return self.total_predicted_wall_s / calibration_wall_s if calibration_wall_s else float("nan")

    @property
    def total_fixed_s_measured(self) -> list[float]:
        """Every stratum's measured one-time pass overhead (not summed into
        any total -- the real full rebuild pays this ONCE, not once per
        stratum; see StratumMeasurement's docstring)."""
        return [m.fixed_s for m in self.measurements if m.regression_valid]

    def to_report(self) -> str:
        lines = [
            f"{'stratum':28s} {'n_pop':>8s} {'n1':>4s} {'n2':>4s} {'fixed_s':>8s} {'marg_s/raw':>11s} {'pred_min':>9s}",
        ]
        for m in sorted(self.measurements, key=lambda m: -m.predicted_wall_s):
            flag = "" if m.regression_valid else "*"
            lines.append(
                f"{m.stratum.label:28s} {m.stratum.count:8d} {m.n1:4d} {m.n2:4d} "
                f"{m.fixed_s:8.3f} {m.marginal_s_per_raw:11.4f} {m.predicted_wall_s / 60:9.2f}{flag}"
            )
        total_min = self.total_predicted_wall_s / 60
        lines.append(
            f"\npredicted total (marginal term only): {total_min:.1f} min over "
            f"{self.total_raws} raws, {self.total_bytes / 1024**3:.2f} GiB"
        )
        fixed_values = self.total_fixed_s_measured
        if fixed_values:
            lines.append(
                f"measured one-time per-pass fixed overhead across strata: "
                f"min={min(fixed_values):.2f}s max={max(fixed_values):.2f}s "
                f"median={sorted(fixed_values)[len(fixed_values) // 2]:.2f}s "
                "-- NOT added into the predicted total (the real run pays this once, "
                "not once per stratum sample; see module docstring)."
            )
        starred = [m.stratum.label for m in self.measurements if not m.regression_valid]
        if starred:
            lines.append(f"* regression not possible (population too small to split): {', '.join(starred)}")
        lines.append(
            f"calibration: actual={CALIBRATION_WALL_S / 60:.1f} min "
            f"({CALIBRATION_RAW_COUNT} raws, {CALIBRATION_TOTAL_BYTES / 1024**3:.2f} GiB) "
            f"-> ratio(predicted/actual)={self.calibration_ratio():.2f}"
        )
        return "\n".join(lines)


def run_cost_model(
    workdir: Path,
    strata: Sequence[Stratum] = POPULATION_SNAPSHOT,
    *,
    sample_sizes_override: tuple[int, int] | None = None,
) -> PredictedRun:
    """Measure every stratum (each gets a fresh scratch archive) and extrapolate."""
    predicted = PredictedRun()
    for i, stratum in enumerate(strata):
        stratum_root = workdir / f"stratum-{i:03d}"
        try:
            measurement = measure_stratum(stratum_root, stratum, sample_sizes=sample_sizes_override)
            predicted.measurements.append(measurement)
        finally:
            shutil.rmtree(stratum_root, ignore_errors=True)
    return predicted


__all__ = [
    "CALIBRATION_RAW_COUNT",
    "CALIBRATION_TOTAL_BYTES",
    "CALIBRATION_WALL_S",
    "POPULATION_SNAPSHOT",
    "POPULATION_SNAPSHOT_CAPTURED_AT",
    "PredictedRun",
    "Stratum",
    "StratumMeasurement",
    "build_stratum_sample_corpus",
    "collect_population_strata",
    "default_sample_n",
    "measure_stratum",
    "run_cost_model",
    "two_point_sample_sizes",
]
