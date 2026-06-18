"""Deterministic per-tier ingest write-amplification probe (#1851).

The targeted-FTS-repair fix for daemon live-ingest write amplification has
already landed.  This probe is the *measurement* artifact the issue still
wanted: a repeatable, host-independent way to attribute the bytes a series of
small append batches writes into each archive tier, so future changes can be
compared against a captured baseline.

It is additive tooling.  It does **not** touch production ingest logic — it
drives the existing public batch-ingest path (``parse_sources_archive``) over a
synthetic, deterministic fixture corpus built under a temporary directory, and
attributes bytes per tier using ``PRAGMA page_count * PRAGMA page_size`` plus the
on-disk WAL file size for each tier database (``source.db`` / ``index.db`` /
``embeddings.db`` / ``user.db`` / ``ops.db``).

Operators run::

    devtools bench ingest-amplification --json > baseline.json
    devtools bench ingest-amplification --batches 8 --seed 1851

The emitted report carries per-batch, per-tier byte deltas plus a summary with
the bytes-written-per-payload-byte amplification ratio.  No wallclock thresholds
are involved: every measured number is a byte count or a derived ratio, so the
report is reproducible for a given SQLite build and fixture seed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

# Bumped when the JSON shape gains/changes a top-level key or field type.
REPORT_VERSION = 1

_DEFAULT_PROVIDER = "chatgpt"
_DEFAULT_BATCHES = 5
_DEFAULT_SEED = 1851
_DEFAULT_MESSAGES_MIN = 4
_DEFAULT_MESSAGES_MAX = 8

# Stable tier ordering for the report, sourced from the canonical tier specs.
_TIER_ORDER: tuple[str, ...] = tuple(tier.value for tier in ARCHIVE_TIER_SPECS)
# Attributed write surfaces: the five tier DB files plus the content-addressed
# blob store. Raw payloads land in the blob store (filesystem), not source.db,
# so excluding it would understate write amplification.
_BLOB_COMPONENT = "blob_store"
_COMPONENT_ORDER: tuple[str, ...] = (*_TIER_ORDER, _BLOB_COMPONENT)


def _tier_paths(archive_root: Path) -> dict[str, Path]:
    return {tier.value: archive_root / spec.filename for tier, spec in ARCHIVE_TIER_SPECS.items()}


def _blob_store_bytes() -> int:
    """Sum the on-disk size of the content-addressed blob store."""
    from polylogue.paths import blob_store_root

    root = blob_store_root()
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            with suppress(OSError):
                total += path.stat().st_size
    return total


def _measure_tier(path: Path) -> dict[str, int]:
    """Measure one tier file's allocated page bytes plus its WAL byte footprint."""
    if not path.exists():
        return {"allocated_bytes": 0, "wal_bytes": 0, "total_bytes": 0}
    wal_path = path.with_name(path.name + "-wal")
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    # A plain read-write connection reads the committed logical size without
    # forcing a checkpoint, so allocated_bytes reflects the main-file pages and
    # wal_bytes reflects the still-resident WAL churn separately.
    conn = sqlite3.connect(path)
    try:
        page_size = int((conn.execute("PRAGMA page_size").fetchone() or (0,))[0] or 0)
        page_count = int((conn.execute("PRAGMA page_count").fetchone() or (0,))[0] or 0)
    finally:
        conn.close()
    allocated_bytes = page_size * page_count
    return {
        "allocated_bytes": allocated_bytes,
        "wal_bytes": wal_bytes,
        "total_bytes": allocated_bytes + wal_bytes,
    }


def _measure_all_components(archive_root: Path) -> dict[str, dict[str, int]]:
    measured = {name: _measure_tier(path) for name, path in _tier_paths(archive_root).items()}
    blob_bytes = _blob_store_bytes()
    measured[_BLOB_COMPONENT] = {
        "allocated_bytes": blob_bytes,
        "wal_bytes": 0,
        "total_bytes": blob_bytes,
    }
    return measured


def _component_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for component in _COMPONENT_ORDER:
        b = before.get(component, {})
        a = after.get(component, {})
        delta[component] = {
            key: int(a.get(key, 0)) - int(b.get(key, 0)) for key in ("allocated_bytes", "wal_bytes", "total_bytes")
        }
    return delta


def _build_fixture_files(
    workdir: Path,
    *,
    provider: str,
    batches: int,
    seed: int,
    messages_min: int,
    messages_max: int,
) -> list[Path]:
    """Write ``batches`` single-session synthetic source files deterministically."""
    from polylogue.scenarios import build_default_corpus_specs
    from polylogue.schemas.synthetic import SyntheticCorpus

    available = set(SyntheticCorpus.available_providers())
    if provider not in available:
        raise ValueError(f"provider {provider!r} not available; choose from {sorted(available)}")

    (spec,) = build_default_corpus_specs(
        providers=(provider,),
        count=batches,
        messages_min=messages_min,
        messages_max=messages_max,
        seed=seed,
    )
    corpus_dir = workdir / "corpus" / provider
    written = SyntheticCorpus.write_spec_artifacts(spec, corpus_dir, prefix="batch", index_width=3)
    # One artifact == one session == one append batch.
    return list(written.files)


async def _ingest_batch(archive_root: Path, provider: str, source_file: Path) -> dict[str, int]:
    from polylogue.config import Source
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive

    result = await parse_sources_archive(archive_root, [Source(name=provider, path=source_file)])
    return {
        "sessions": int(result.counts.get("sessions", 0)),
        "messages": int(result.counts.get("messages", 0)),
    }


def measure_ingest_amplification(
    *,
    provider: str = _DEFAULT_PROVIDER,
    batches: int = _DEFAULT_BATCHES,
    seed: int = _DEFAULT_SEED,
    messages_min: int = _DEFAULT_MESSAGES_MIN,
    messages_max: int = _DEFAULT_MESSAGES_MAX,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Run ``batches`` single-session ingest batches and attribute bytes per tier.

    Returns a stable, JSON-serializable report.  When ``workdir`` is omitted a
    private temporary directory is created and torn down automatically; the
    archive and its blob store stay entirely inside that directory.
    """
    if batches < 1:
        raise ValueError("batches must be >= 1")

    owns_workdir = workdir is None
    base = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="plg-ingest-amp-"))
    base.mkdir(parents=True, exist_ok=True)

    # Isolate every global path (archive root + blob store) inside the workdir so
    # the probe never reads or writes real user data.
    archive_root = base / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    saved_env = {key: os.environ.get(key) for key in ("XDG_DATA_HOME", "POLYLOGUE_ARCHIVE_ROOT")}
    os.environ["XDG_DATA_HOME"] = str(base / "xdg-data")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)

    try:
        from polylogue.storage.blob_store import reset_blob_store
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        reset_blob_store()

        # Bootstrap the five-tier archive file set up front so per-batch deltas
        # exclude the one-time schema-DDL cost.
        ArchiveStore.open_existing(archive_root, read_only=False).close()

        source_files = _build_fixture_files(
            base,
            provider=provider,
            batches=batches,
            seed=seed,
            messages_min=messages_min,
            messages_max=messages_max,
        )
        if len(source_files) < batches:
            batches = len(source_files)

        baseline = _measure_all_components(archive_root)
        previous = baseline
        batch_reports: list[dict[str, Any]] = []
        total_payload_bytes = 0
        total_bytes_written = 0
        per_component_total: dict[str, int] = dict.fromkeys(_COMPONENT_ORDER, 0)

        for index in range(batches):
            source_file = source_files[index]
            payload_bytes = source_file.stat().st_size
            ingested = asyncio.run(_ingest_batch(archive_root, provider, source_file))
            after = _measure_all_components(archive_root)
            delta = _component_delta(previous, after)
            batch_total = sum(delta[component]["total_bytes"] for component in _COMPONENT_ORDER)
            for component in _COMPONENT_ORDER:
                per_component_total[component] += delta[component]["total_bytes"]
            total_payload_bytes += payload_bytes
            total_bytes_written += batch_total
            batch_reports.append(
                {
                    "batch_index": index,
                    "payload_bytes": payload_bytes,
                    "sessions_ingested": ingested["sessions"],
                    "messages_ingested": ingested["messages"],
                    "component_bytes": after,
                    "component_byte_delta": delta,
                    "total_bytes_written": batch_total,
                    "amplification_ratio": round(batch_total / payload_bytes, 4) if payload_bytes else 0.0,
                }
            )
            previous = after

        ratios = [b["amplification_ratio"] for b in batch_reports]
        steady = batch_reports[1:]
        steady_payload = sum(b["payload_bytes"] for b in steady)
        steady_written = sum(b["total_bytes_written"] for b in steady)
        summary = {
            "batch_count": len(batch_reports),
            "total_payload_bytes": total_payload_bytes,
            "total_bytes_written": total_bytes_written,
            "overall_amplification_ratio": (
                round(total_bytes_written / total_payload_bytes, 4) if total_payload_bytes else 0.0
            ),
            "mean_batch_amplification_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
            "steady_state_amplification_ratio": (round(steady_written / steady_payload, 4) if steady_payload else None),
            "per_component_total_bytes": dict(per_component_total),
            "per_component_share": {
                component: round(per_component_total[component] / total_bytes_written, 4)
                if total_bytes_written
                else 0.0
                for component in _COMPONENT_ORDER
            },
        }

        return {
            "ok": True,
            "report_version": REPORT_VERSION,
            "tool": "bench ingest-amplification",
            "fixture": {
                "provider": provider,
                "batch_count": len(batch_reports),
                "seed": seed,
                "messages_min": messages_min,
                "messages_max": messages_max,
            },
            "tiers": list(_TIER_ORDER),
            "components": list(_COMPONENT_ORDER),
            "baseline_component_bytes": baseline,
            "batches": batch_reports,
            "summary": summary,
        }
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with suppress(Exception):
            from polylogue.storage.blob_store import reset_blob_store as _reset

            _reset()
        if owns_workdir:
            import shutil

            with suppress(OSError):
                shutil.rmtree(base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--provider", default=_DEFAULT_PROVIDER, help="Synthetic provider to ingest")
    parser.add_argument("--batches", type=int, default=_DEFAULT_BATCHES, help="Number of single-session append batches")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Deterministic corpus seed")
    parser.add_argument("--messages-min", type=int, default=_DEFAULT_MESSAGES_MIN, help="Min messages per session")
    parser.add_argument("--messages-max", type=int, default=_DEFAULT_MESSAGES_MAX, help="Max messages per session")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Reuse a directory for the fixture/archive instead of a private temp dir",
    )
    return parser


def _format_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    fixture = report.get("fixture") or {}
    lines.append("Ingest write-amplification probe")
    lines.append(
        f"  fixture: provider={fixture.get('provider')} batches={fixture.get('batch_count')} seed={fixture.get('seed')}"
    )
    lines.append("")
    lines.append("  per-batch (payload -> bytes written, amplification):")
    for batch in report.get("batches") or []:
        lines.append(
            f"    batch {batch['batch_index']}: "
            f"{batch['payload_bytes']}B payload, "
            f"{batch['messages_ingested']} msgs -> "
            f"{batch['total_bytes_written']}B written ({batch['amplification_ratio']:.2f}x)"
        )
    summary = report.get("summary") or {}
    lines.append("")
    lines.append(
        f"  totals: {summary.get('total_payload_bytes', 0)}B payload -> "
        f"{summary.get('total_bytes_written', 0)}B written"
    )
    lines.append(f"  overall amplification: {summary.get('overall_amplification_ratio', 0.0):.2f}x")
    steady = summary.get("steady_state_amplification_ratio")
    if steady is not None:
        lines.append(f"  steady-state amplification (excl. batch 0): {steady:.2f}x")
    lines.append("  per-component share of bytes written:")
    for component in report.get("components") or []:
        share = (summary.get("per_component_share") or {}).get(component, 0.0)
        component_bytes = (summary.get("per_component_total_bytes") or {}).get(component, 0)
        lines.append(f"    {component}: {component_bytes}B ({share * 100:.1f}%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = measure_ingest_amplification(
            provider=args.provider,
            batches=max(1, args.batches),
            seed=args.seed,
            messages_min=args.messages_min,
            messages_max=args.messages_max,
            workdir=args.workdir,
        )
    except ValueError as exc:
        print(f"ingest-amplification-probe failed: {exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
