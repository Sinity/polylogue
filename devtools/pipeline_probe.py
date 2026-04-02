"""Exercise the real pipeline on bounded synthetic or archive-subset corpora and emit JSON metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from polylogue.config import Config, Source
from polylogue.paths import blob_store_root
from polylogue.pipeline.runner import RUN_STAGE_CHOICES, run_sources
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.backends import create_backend
from polylogue.storage.backends.connection import (
    _build_provider_scope_filter,
    _build_source_scope_filter,
    default_db_path,
    open_connection,
)
from polylogue.storage.backends.queries.raw_state import EFFECTIVE_RAW_PROVIDER_SQL
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord
from polylogue.types import Provider

_EXT_MAP = {
    "chatgpt": ".json",
    "claude-ai": ".json",
    "gemini": ".json",
    "claude-code": ".jsonl",
    "codex": ".jsonl",
}
_INPUT_MODES = ("synthetic", "archive-subset")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real pipeline against a bounded synthetic or archive-subset corpus and emit a JSON summary.",
    )
    parser.add_argument(
        "--input-mode",
        choices=_INPUT_MODES,
        default="synthetic",
        help="Probe input mode: synthetic fixture generation or archive-subset replay (default: synthetic)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help=(
            "Provider selector. In synthetic mode this must resolve to exactly one provider "
            f"({', '.join(sorted(SyntheticCorpus.available_providers()))}). "
            "In archive-subset mode it filters the sampled providers and may be repeated."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Synthetic source files to generate (default: 5)",
    )
    parser.add_argument(
        "--messages-min",
        type=int,
        default=4,
        help="Minimum messages per conversation (default: 4)",
    )
    parser.add_argument(
        "--messages-max",
        type=int,
        default=12,
        help="Maximum messages per conversation (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synthetic corpus seed (default: 42)",
    )
    parser.add_argument(
        "--sample-per-provider",
        type=int,
        default=50,
        help="Archive-subset sample size per provider (default: 50)",
    )
    parser.add_argument(
        "--source",
        dest="source_filters",
        action="append",
        default=None,
        help="Archive-subset source-name filter. Repeatable.",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        help="Archive-subset source database path (default: current archive database)",
    )
    parser.add_argument(
        "--source-blob-root",
        type=Path,
        help="Archive-subset blob-store root (default: current blob store root)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional path for the archive-subset selection manifest.",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Replay a previously persisted archive-subset manifest instead of resampling.",
    )
    parser.add_argument(
        "--stage",
        choices=RUN_STAGE_CHOICES,
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help=(
            "Probe workspace root. If omitted, a temporary workspace is created and removed after the run. "
            "Pass an explicit path when you want to keep the run/database artifacts."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the JSON summary.",
    )
    parser.add_argument(
        "--max-total-ms",
        type=float,
        default=None,
        help="Fail if total pipeline runtime exceeds this budget in milliseconds.",
    )
    parser.add_argument(
        "--max-peak-rss-mb",
        type=float,
        default=None,
        help="Fail if peak RSS exceeds this budget in MiB.",
    )
    return parser.parse_args(argv)


@contextmanager
def _isolated_env(workdir: Path) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in (
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CONFIG_HOME",
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_RENDER_ROOT",
    )}
    env_updates = {
        "XDG_DATA_HOME": str(workdir / "xdg-data"),
        "XDG_STATE_HOME": str(workdir / "xdg-state"),
        "XDG_CONFIG_HOME": str(workdir / "xdg-config"),
        "POLYLOGUE_ARCHIVE_ROOT": str(workdir / "archive"),
        "POLYLOGUE_RENDER_ROOT": str(workdir / "render"),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    reset_blob_store()
    try:
        yield
    finally:
        reset_blob_store()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _db_row_counts(db_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("raw_conversations", "conversations", "messages", "content_blocks"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[f"{table}_count"] = int(row[0]) if row else 0
    return stats


def _write_probe_sources(
    *,
    provider: str,
    count: int,
    messages_min: int,
    messages_max: int,
    seed: int,
    source_root: Path,
) -> tuple[list[Path], int]:
    corpus = SyntheticCorpus.for_provider(provider)
    source_root.mkdir(parents=True, exist_ok=True)
    raw_items = corpus.generate(
        count=count,
        messages_per_conversation=range(messages_min, messages_max + 1),
        seed=seed,
    )
    total_bytes = 0
    files: list[Path] = []
    extension = _EXT_MAP.get(provider, ".json")
    for index, raw_bytes in enumerate(raw_items):
        file_path = source_root / f"{provider}-{index:03d}{extension}"
        file_path.write_bytes(raw_bytes)
        files.append(file_path)
        total_bytes += len(raw_bytes)
    return files, total_bytes


def _names(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _effective_provider_name(record: RawConversationRecord) -> str:
    payload_provider = record.payload_provider
    if isinstance(payload_provider, Provider):
        return payload_provider.value
    if payload_provider is not None:
        return str(payload_provider)
    return record.provider_name


def _source_bucket_name(record: RawConversationRecord) -> str:
    return record.source_name or "<unknown>"


def _normalize_record_for_replay(record: RawConversationRecord) -> RawConversationRecord:
    """Reset parse/validation state so a copied raw row behaves like post-acquire input."""
    return record.model_copy(
        update={
            "parsed_at": None,
            "parse_error": None,
            "validated_at": None,
            "validation_status": None,
            "validation_error": None,
            "validation_drift_count": None,
            "validation_provider": None,
            "validation_mode": None,
        }
    )


def _fetch_archive_candidates(
    *,
    db_path: Path,
    provider_filters: list[str],
    source_filters: list[str],
) -> list[RawConversationRecord]:
    where_clauses: list[str] = []
    params: list[str] = []

    provider_predicate, provider_params = _build_provider_scope_filter(
        provider_filters or None,
        provider_column=EFFECTIVE_RAW_PROVIDER_SQL,
    )
    if provider_predicate:
        where_clauses.append(provider_predicate)
        params.extend(provider_params)

    source_predicate, source_params = _build_source_scope_filter(
        source_filters or None,
        source_column="source_name",
    )
    if source_predicate:
        where_clauses.append(source_predicate)
        params.extend(source_params)

    sql = "SELECT * FROM raw_conversations"
    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"
    sql += " ORDER BY acquired_at DESC, raw_id ASC"

    with open_connection(db_path) as conn:
        cursor = conn.execute(sql, tuple(params))
        return [
            RawConversationRecord.model_validate(dict(row))
            for row in cursor.fetchall()
        ]


def _raw_conversation_count(db_path: Path) -> int:
    with open_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM raw_conversations").fetchone()
    return int(row[0]) if row else 0


def _empty_archive_manifest_error(
    *,
    source_db: Path,
    source_blob_root: Path,
    provider_filters: list[str],
    source_filters: list[str],
    candidate_count: int,
    missing_blob_count: int,
) -> str:
    qualifier: list[str] = []
    if provider_filters:
        qualifier.append(f"providers={provider_filters}")
    if source_filters:
        qualifier.append(f"sources={source_filters}")
    qualifier_suffix = f" with {' '.join(qualifier)}" if qualifier else ""

    if candidate_count == 0:
        message = f"archive-subset probe found no raw conversations in {source_db}{qualifier_suffix}"
    else:
        message = (
            f"archive-subset probe found {candidate_count} candidate raw conversations in {source_db}{qualifier_suffix}, "
            f"but all {missing_blob_count} matching blobs were missing under {source_blob_root}"
        )

    sibling_db = source_db.with_name(f"{source_db.name}_")
    if candidate_count == 0 and sibling_db.exists():
        sibling_count = _raw_conversation_count(sibling_db)
        if sibling_count > 0:
            message += (
                f"; sibling archive {sibling_db} contains {sibling_count} raw conversations. "
                f"Pass --source-db {sibling_db} if that is the intended source archive."
            )
    return message


def _sample_provider_records(
    *,
    records: list[RawConversationRecord],
    sample_size: int,
    rng: random.Random,
) -> list[RawConversationRecord]:
    source_buckets: dict[str, list[RawConversationRecord]] = {}
    for record in records:
        source_buckets.setdefault(_source_bucket_name(record), []).append(record)

    for bucket in source_buckets.values():
        rng.shuffle(bucket)

    selected: list[RawConversationRecord] = []
    active_sources = list(source_buckets)
    while active_sources and len(selected) < sample_size:
        rng.shuffle(active_sources)
        next_active_sources: list[str] = []
        for source_name in active_sources:
            bucket = source_buckets[source_name]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_active_sources.append(source_name)
            if len(selected) >= sample_size:
                break
        active_sources = next_active_sources
    return selected


def _provider_counts(records: list[RawConversationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        provider = _effective_provider_name(record)
        counts[provider] = counts.get(provider, 0) + 1
    return dict(sorted(counts.items()))


def _source_counts(records: list[RawConversationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        source_name = _source_bucket_name(record)
        counts[source_name] = counts.get(source_name, 0) + 1
    return dict(sorted(counts.items()))


def _persist_manifest(manifest: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_archive_manifest(
    *,
    source_db: Path,
    source_blob_root: Path,
    provider_filters: list[str],
    source_filters: list[str],
    sample_per_provider: int,
    seed: int,
) -> dict[str, Any]:
    source_blob_store = BlobStore(source_blob_root)
    candidate_records = _fetch_archive_candidates(
        db_path=source_db,
        provider_filters=provider_filters,
        source_filters=source_filters,
    )
    candidates_with_blobs = [
        record for record in candidate_records if source_blob_store.exists(record.raw_id)
    ]
    missing_blob_count = len(candidate_records) - len(candidates_with_blobs)

    by_provider: dict[str, list[RawConversationRecord]] = {}
    for record in candidates_with_blobs:
        by_provider.setdefault(_effective_provider_name(record), []).append(record)

    rng = random.Random(seed)
    sampled_records: list[RawConversationRecord] = []
    sampled_by_provider: dict[str, int] = {}
    available_by_provider: dict[str, int] = {}
    for provider_name in sorted(by_provider):
        provider_records = by_provider[provider_name]
        available_by_provider[provider_name] = len(provider_records)
        sampled = _sample_provider_records(
            records=provider_records,
            sample_size=sample_per_provider,
            rng=rng,
        )
        sampled_by_provider[provider_name] = len(sampled)
        sampled_records.extend(sampled)

    if not sampled_records:
        raise ValueError(
            _empty_archive_manifest_error(
                source_db=source_db,
                source_blob_root=source_blob_root,
                provider_filters=provider_filters,
                source_filters=source_filters,
                candidate_count=len(candidate_records),
                missing_blob_count=missing_blob_count,
            )
        )

    sampled_records.sort(key=lambda record: (
        _effective_provider_name(record),
        _source_bucket_name(record),
        record.acquired_at,
        record.raw_id,
    ))

    return {
        "input_mode": "archive-subset",
        "source_db": str(source_db),
        "source_blob_root": str(source_blob_root),
        "seed": seed,
        "sample_per_provider": sample_per_provider,
        "provider_filters": provider_filters,
        "source_filters": source_filters,
        "candidate_count": len(candidate_records),
        "missing_blob_count": missing_blob_count,
        "available_by_provider": available_by_provider,
        "sampled_by_provider": sampled_by_provider,
        "records": [record.model_dump(mode="json") for record in sampled_records],
    }


def _resolve_archive_manifest(
    *,
    manifest_in: Path | None,
    source_db: Path | None,
    source_blob_root: Path | None,
    provider_filters: list[str],
    source_filters: list[str],
    sample_per_provider: int,
    seed: int,
) -> dict[str, Any]:
    if manifest_in is not None:
        manifest = _load_manifest(manifest_in)
        if source_blob_root is not None:
            manifest["source_blob_root"] = str(source_blob_root.resolve())
        if source_db is not None:
            manifest["source_db"] = str(source_db.resolve())
        return manifest

    resolved_source_db = (source_db or default_db_path()).resolve()
    resolved_source_blob_root = (source_blob_root or blob_store_root()).resolve()
    return _build_archive_manifest(
        source_db=resolved_source_db,
        source_blob_root=resolved_source_blob_root,
        provider_filters=provider_filters,
        source_filters=source_filters,
        sample_per_provider=sample_per_provider,
        seed=seed,
    )


async def _seed_archive_subset(
    *,
    manifest: dict[str, Any],
    repository: ConversationRepository,
    target_blob_store: BlobStore,
) -> dict[str, Any]:
    records = [
        RawConversationRecord.model_validate(record_payload)
        for record_payload in manifest.get("records", [])
    ]
    source_blob_root = Path(str(manifest["source_blob_root"])).resolve()
    source_blob_store = BlobStore(source_blob_root)
    copied_records = 0
    copied_blob_bytes = 0

    for record in records:
        source_blob_path = source_blob_store.blob_path(record.raw_id)
        if not source_blob_path.exists():
            raise FileNotFoundError(
                f"archive-subset probe is missing blob {record.raw_id} under {source_blob_root}"
            )
        destination_blob_path = target_blob_store.blob_path(record.raw_id)
        if not destination_blob_path.exists():
            destination_blob_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_blob_path, destination_blob_path)
            copied_blob_bytes += destination_blob_path.stat().st_size
        await repository.save_raw_conversation(_normalize_record_for_replay(record))
        copied_records += 1

    return {
        "selected_count": len(records),
        "copied_records": copied_records,
        "copied_blob_bytes": copied_blob_bytes,
        "provider_counts": _provider_counts(records),
        "source_counts": _source_counts(records),
        "sample_per_provider": manifest.get("sample_per_provider"),
        "candidate_count": manifest.get("candidate_count"),
        "missing_blob_count": manifest.get("missing_blob_count", 0),
        "available_by_provider": manifest.get("available_by_provider", {}),
        "sampled_by_provider": manifest.get("sampled_by_provider", {}),
    }


def _resolve_synthetic_provider(args: argparse.Namespace) -> str:
    provider_names = _names(getattr(args, "provider", None))
    provider_name = provider_names[0] if provider_names else "chatgpt"
    available = set(SyntheticCorpus.available_providers())
    if provider_name not in available:
        raise ValueError(
            f"--provider must be one of {sorted(available)} in synthetic mode"
        )
    if len(provider_names) > 1:
        raise ValueError("synthetic mode accepts exactly one --provider")
    return provider_name


def _probe_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "input_mode", "synthetic"))


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    probe_mode = _probe_mode(args)
    if probe_mode not in _INPUT_MODES:
        raise ValueError(f"--input-mode must be one of {_INPUT_MODES}")

    workdir = args.workdir.resolve()
    archive_root = workdir / "archive"
    render_root = workdir / "render"
    db_path: Path | None = None
    summary: dict[str, Any]

    if probe_mode == "synthetic":
        if args.count <= 0:
            raise ValueError("--count must be positive")
        if args.messages_min <= 0 or args.messages_max < args.messages_min:
            raise ValueError("--messages-min/--messages-max must define a positive inclusive range")
        provider_name = _resolve_synthetic_provider(args)
        source_root = workdir / "sources" / provider_name

        with _isolated_env(workdir):
            files, total_bytes = _write_probe_sources(
                provider=provider_name,
                count=args.count,
                messages_min=args.messages_min,
                messages_max=args.messages_max,
                seed=args.seed,
                source_root=source_root,
            )
            config = Config(
                sources=[Source(name=provider_name, path=source_root)],
                archive_root=archive_root,
                render_root=render_root,
            )
            db_path = config.db_path
            result = await run_sources(
                config=config,
                stage=args.stage,
                source_names=[provider_name],
            )
            run_payload: dict[str, Any] = {}
            if result.run_path:
                run_payload = json.loads(Path(result.run_path).read_text(encoding="utf-8"))

        summary = {
            "probe": {
                "input_mode": "synthetic",
                "provider": provider_name,
                "stage": args.stage,
                "count": args.count,
                "messages_min": args.messages_min,
                "messages_max": args.messages_max,
                "seed": args.seed,
            },
            "paths": {
                "workdir": str(workdir),
                "source_root": str(source_root),
                "archive_root": str(archive_root),
                "render_root": str(render_root),
                "db_path": str(db_path),
                "run_path": result.run_path,
            },
            "source_files": {
                "count": len(files),
                "total_bytes": total_bytes,
            },
            "result": result.model_dump(),
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path) if db_path is not None else {},
        }
    else:
        if args.sample_per_provider <= 0:
            raise ValueError("--sample-per-provider must be positive")
        provider_filters = _names(getattr(args, "provider", None))
        source_filters = _names(getattr(args, "source_filters", None))
        manifest = _resolve_archive_manifest(
            manifest_in=getattr(args, "manifest_in", None),
            source_db=getattr(args, "source_db", None),
            source_blob_root=getattr(args, "source_blob_root", None),
            provider_filters=provider_filters,
            source_filters=source_filters,
            sample_per_provider=args.sample_per_provider,
            seed=args.seed,
        )
        manifest_path = workdir / "archive-subset-manifest.json"
        _persist_manifest(manifest, manifest_path)
        if args.manifest_out is not None:
            _persist_manifest(manifest, args.manifest_out)

        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=render_root,
        )
        with _isolated_env(workdir):
            db_path = config.db_path
            backend = create_backend(db_path=db_path)
            repository = ConversationRepository(backend=backend)
            try:
                target_blob_store = BlobStore(blob_store_root())
                sample_summary = await _seed_archive_subset(
                    manifest=manifest,
                    repository=repository,
                    target_blob_store=target_blob_store,
                )
                result = await run_sources(
                    config=config,
                    stage=args.stage,
                    source_names=None,
                    backend=backend,
                    repository=repository,
                )
                run_payload = {}
                if result.run_path:
                    run_payload = json.loads(Path(result.run_path).read_text(encoding="utf-8"))
            finally:
                await repository.close()

        summary = {
            "probe": {
                "input_mode": "archive-subset",
                "stage": args.stage,
                "seed": args.seed,
                "sample_per_provider": args.sample_per_provider,
                "provider_filters": provider_filters,
                "source_filters": source_filters,
            },
            "paths": {
                "workdir": str(workdir),
                "archive_root": str(archive_root),
                "render_root": str(render_root),
                "db_path": str(db_path),
                "run_path": result.run_path,
                "manifest_path": str(manifest_path),
                "source_db": str(manifest["source_db"]),
                "source_blob_root": str(manifest["source_blob_root"]),
            },
            "sample": sample_summary,
            "result": result.model_dump(),
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path) if db_path is not None else {},
        }

    return summary


def _build_budget_report(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.max_total_ms is None and args.max_peak_rss_mb is None:
        return None

    metrics = summary.get("run_payload", {}).get("metrics", {})
    observed_total_ms = metrics.get("total_duration_ms", summary.get("result", {}).get("duration_ms"))
    observed_peak_rss_mb = metrics.get("peak_rss_mb")
    violations: list[str] = []

    if args.max_total_ms is not None:
        if observed_total_ms is None:
            violations.append("missing total runtime metric")
        elif float(observed_total_ms) > args.max_total_ms:
            violations.append(
                f"total runtime {float(observed_total_ms):.1f} ms exceeded budget {args.max_total_ms:.1f} ms"
            )

    if args.max_peak_rss_mb is not None:
        if observed_peak_rss_mb is None:
            violations.append("missing peak RSS metric")
        elif float(observed_peak_rss_mb) > args.max_peak_rss_mb:
            violations.append(
                f"peak RSS {float(observed_peak_rss_mb):.1f} MiB exceeded budget {args.max_peak_rss_mb:.1f} MiB"
            )

    return {
        "ok": not violations,
        "max_total_ms": args.max_total_ms,
        "observed_total_ms": observed_total_ms,
        "max_peak_rss_mb": args.max_peak_rss_mb,
        "observed_peak_rss_mb": observed_peak_rss_mb,
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="polylogue-pipeline-probe-") as tempdir:
            args.workdir = Path(tempdir)
            with redirect_stdout(sys.stderr):
                summary = asyncio.run(run_probe(args))
    else:
        with redirect_stdout(sys.stderr):
            summary = asyncio.run(run_probe(args))
    budget_report = _build_budget_report(summary, args)
    if budget_report is not None:
        summary["budgets"] = budget_report
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if budget_report is not None and not budget_report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
