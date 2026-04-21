"""Exercise the real pipeline on bounded synthetic, archive-subset, or staged source corpora."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import NotRequired, Protocol

from typing_extensions import TypedDict

from devtools.regression_cases import DEFAULT_REGRESSION_CASE_DIR, RegressionCase, RegressionCaseStore
from polylogue.config import Config, Source
from polylogue.lib.json import JSONDocument, JSONValue, is_json_document, json_document, loads, require_json_document
from polylogue.paths import blob_store_root, db_path
from polylogue.pipeline.runner import RUN_STAGE_CHOICES, run_sources
from polylogue.scenarios import (
    CorpusRequest,
    CorpusSourceKind,
    PipelineProbeInputMode,
    PipelineProbeRequest,
)
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.showcase.workspace import VerificationWorkspace, create_verification_workspace
from polylogue.storage.backends import SQLiteBackend, create_backend
from polylogue.storage.backends.connection import (
    _build_provider_scope_filter,
    _build_source_scope_filter,
    open_connection,
)
from polylogue.storage.backends.queries.raw_state import EFFECTIVE_RAW_PROVIDER_SQL
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.run_state import RunResult
from polylogue.storage.store import RawConversationRecord
from polylogue.types import Provider

_EXT_MAP = {
    "chatgpt": ".json",
    "claude-ai": ".json",
    "gemini": ".json",
    "claude-code": ".jsonl",
    "codex": ".jsonl",
}
_INPUT_MODES = tuple(mode.value for mode in PipelineProbeInputMode)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_BACKED_PROBE_STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "acquire": ("acquire",),
    "schema": ("acquire", "schema"),
    "parse": ("acquire", "parse"),
    "materialize": ("acquire", "parse", "materialize"),
    "render": ("acquire", "parse", "render"),
    "index": ("acquire", "parse", "index"),
    "reprocess": ("acquire", "parse", "materialize", "render", "index"),
    "all": ("acquire", "parse", "materialize", "render", "index"),
}


class _RawConversationStore(Protocol):
    async def save_raw_conversation(self, record: RawConversationRecord) -> bool: ...


class RawFanoutEntry(TypedDict):
    raw_id: str
    payload_provider: str | None
    source_name: str | None
    blob_size_bytes: int
    conversation_count: int
    message_count: int
    parse_error: str | None


class PathFingerprint(TypedDict):
    path: str
    kind: str
    sha256: str
    file_count: int
    total_bytes: int


class StagedSourceEntry(TypedDict):
    input_path: str
    staged_path: str
    kind: str
    file_count: int
    bytes: int


class SourceInputsSummary(TypedDict):
    input_count: int
    staged_entry_count: int
    staged_file_count: int
    total_bytes: int
    entries: list[StagedSourceEntry]


class ProbeProvenance(TypedDict):
    git_commit: str | None
    worktree_dirty: bool | None
    manifest_sha256: NotRequired[str]
    source_input_fingerprints: NotRequired[list[PathFingerprint]]
    source_inputs_sha256: NotRequired[str]


class ArchiveManifest(TypedDict):
    input_mode: str
    source_db: str
    source_blob_root: str
    seed: int
    sample_per_provider: int
    provider_filters: list[str]
    source_filters: list[str]
    candidate_count: int
    missing_blob_count: int
    available_by_provider: dict[str, int]
    sampled_by_provider: dict[str, int]
    records: list[JSONDocument]


class ArchiveSubsetSampleSummary(TypedDict):
    selected_count: int
    copied_records: int
    copied_blob_bytes: int
    provider_counts: dict[str, int]
    source_counts: dict[str, int]
    sample_per_provider: int
    candidate_count: int
    missing_blob_count: int
    available_by_provider: dict[str, int]
    sampled_by_provider: dict[str, int]


class BudgetReport(TypedDict):
    ok: bool
    max_total_ms: float | None
    observed_total_ms: JSONValue
    max_peak_rss_mb: float | None
    observed_peak_rss_mb: JSONValue
    violations: list[str]


class RegressionCaseSummary(TypedDict):
    case_id: str
    name: str
    path: str
    tags: list[str]


class ProbeSummary(TypedDict):
    probe: JSONDocument
    paths: JSONDocument
    provenance: ProbeProvenance
    result: JSONDocument
    run_payload: JSONDocument
    db_stats: dict[str, int]
    raw_fanout: list[RawFanoutEntry]
    source_files: NotRequired[JSONDocument]
    source_inputs: NotRequired[SourceInputsSummary]
    sample: NotRequired[ArchiveSubsetSampleSummary]
    budgets: NotRequired[BudgetReport]
    regression_case: NotRequired[RegressionCaseSummary]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real pipeline against a bounded synthetic, archive-subset, "
            "or staged real-source corpus and emit a JSON summary."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=_INPUT_MODES,
        default="synthetic",
        help=(
            "Probe input mode: synthetic fixture generation, archive-subset replay, "
            "or staged real-source inputs (default: synthetic)"
        ),
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
        "--corpus-source",
        choices=[kind.value for kind in CorpusSourceKind],
        default=CorpusSourceKind.DEFAULT.value,
        help="Synthetic corpus source to use in synthetic mode (default: default)",
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
        "--style",
        default="default",
        help="Synthetic corpus style (default: default)",
    )
    parser.add_argument(
        "--package-version",
        default="default",
        help="Synthetic package version selector (default: default)",
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
        "--source-path",
        dest="source_paths",
        action="append",
        type=Path,
        default=None,
        help=(
            "Real-source input path for source-subset mode. Files or directories are copied "
            "into the isolated probe workspace. Repeatable."
        ),
    )
    parser.add_argument(
        "--source-name",
        default="inbox",
        help="Source name assigned to staged source-subset inputs (default: inbox)",
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
        "--raw-batch-size",
        type=int,
        default=None,
        help="Use this raw-record batch size for the probe.",
    )
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=None,
        help="Use this worker count for the probe ingest stage.",
    )
    parser.add_argument(
        "--measure-ingest-result-size",
        action="store_true",
        help="Measure serialized IngestRecordResult sizes for this probe.",
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
        "--capture-regression",
        metavar="NAME",
        help=(
            "Capture the emitted probe summary as a durable regression case under "
            ".local/regression-cases or --regression-output-dir."
        ),
    )
    parser.add_argument(
        "--regression-output-dir",
        type=Path,
        default=DEFAULT_REGRESSION_CASE_DIR,
        help=f"Output directory for --capture-regression (default: {DEFAULT_REGRESSION_CASE_DIR}).",
    )
    parser.add_argument(
        "--regression-tag",
        action="append",
        default=[],
        help="Tag to attach to the captured regression case. Repeatable.",
    )
    parser.add_argument(
        "--regression-note",
        action="append",
        default=[],
        help="Note to attach to the captured regression case. Repeatable.",
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


def _request_from_args(args: argparse.Namespace) -> PipelineProbeRequest:
    """Project argparse input onto the authored pipeline-probe request contract."""
    provider_names = tuple(_names(getattr(args, "provider", None)))
    corpus_request: CorpusRequest | None = None
    input_mode = PipelineProbeInputMode(_probe_mode(args))
    if input_mode is PipelineProbeInputMode.SYNTHETIC:
        corpus_request = CorpusRequest(
            providers=provider_names or ("chatgpt",),
            source=CorpusSourceKind(getattr(args, "corpus_source", CorpusSourceKind.DEFAULT.value)),
            count=args.count,
            messages_min=args.messages_min,
            messages_max=args.messages_max,
            seed=args.seed,
            style=getattr(args, "style", "default"),
            package_version=getattr(args, "package_version", "default"),
        )
    return PipelineProbeRequest(
        stage=args.stage,
        input_mode=input_mode,
        corpus_request=corpus_request,
        sample_per_provider=args.sample_per_provider,
        source_filters=tuple(_names(getattr(args, "source_filters", None))),
        source_paths=tuple(str(path) for path in _paths(getattr(args, "source_paths", None))),
        source_name=str(getattr(args, "source_name", "inbox") or "inbox"),
        source_db=str(args.source_db) if args.source_db is not None else None,
        source_blob_root=str(args.source_blob_root) if args.source_blob_root is not None else None,
        manifest_out=str(args.manifest_out) if args.manifest_out is not None else None,
        manifest_in=str(args.manifest_in) if args.manifest_in is not None else None,
        raw_batch_size=args.raw_batch_size,
        ingest_workers=args.ingest_workers,
        measure_ingest_result_size=bool(args.measure_ingest_result_size),
        workdir=str(args.workdir) if args.workdir is not None else None,
        json_out=str(args.json_out) if args.json_out is not None else None,
        max_total_ms=args.max_total_ms,
        max_peak_rss_mb=args.max_peak_rss_mb,
    )


@contextmanager
def _isolated_env(workspace: VerificationWorkspace) -> Iterator[None]:
    previous = {
        key: os.environ.get(key)
        for key in (
            "HOME",
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "XDG_CONFIG_HOME",
            "POLYLOGUE_ARCHIVE_ROOT",
        )
    }
    for key, value in workspace.env_vars.items():
        os.environ[key] = value
    reset_blob_store()
    try:
        yield
    finally:
        reset_blob_store()
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _workspace_for_request(request: PipelineProbeRequest) -> VerificationWorkspace:
    workdir = Path(request.workdir).resolve() if request.workdir is not None else None
    return create_verification_workspace(workdir, prefix="polylogue-pipeline-probe-")


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


def _db_raw_fanout(db_path: Path) -> list[RawFanoutEntry]:
    if not db_path.exists():
        return []
    with open_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                r.raw_id,
                COALESCE(r.payload_provider, r.provider_name) AS payload_provider,
                r.source_name,
                r.blob_size,
                r.parse_error,
                COUNT(DISTINCT c.conversation_id) AS conversation_count,
                COUNT(m.message_id) AS message_count
            FROM raw_conversations r
            LEFT JOIN conversations c ON c.raw_id = r.raw_id
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            GROUP BY
                r.raw_id,
                COALESCE(r.payload_provider, r.provider_name),
                r.source_name,
                r.blob_size,
                r.parse_error
            ORDER BY r.blob_size DESC, r.raw_id ASC
            """
        ).fetchall()
    return [
        {
            "raw_id": str(row["raw_id"]),
            "payload_provider": row["payload_provider"],
            "source_name": row["source_name"],
            "blob_size_bytes": int(row["blob_size"]),
            "conversation_count": int(row["conversation_count"]),
            "message_count": int(row["message_count"]),
            "parse_error": row["parse_error"],
        }
        for row in rows
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_path(path: Path) -> PathFingerprint:
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": _sha256_file(path),
            "file_count": 1,
            "total_bytes": path.stat().st_size,
        }

    if path.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        total_bytes = 0
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = file_path.relative_to(path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total_bytes += len(chunk)
            digest.update(b"\0")
            file_count += 1
        return {
            "path": str(path),
            "kind": "dir",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "total_bytes": total_bytes,
        }

    raise FileNotFoundError(f"Cannot fingerprint missing path: {path}")


def _safe_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def _safe_git_worktree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=_REPO_ROOT,
            text=True,
        )
    except Exception:
        return None
    return any(line[3:].strip() for line in status.splitlines())


def _build_probe_provenance(
    *,
    manifest_path: Path | None = None,
    source_inputs: SourceInputsSummary | None = None,
) -> ProbeProvenance:
    provenance: ProbeProvenance = {
        "git_commit": _safe_git_commit(),
        "worktree_dirty": _safe_git_worktree_dirty(),
    }
    if manifest_path is not None and manifest_path.exists():
        provenance["manifest_sha256"] = _sha256_file(manifest_path)
    if source_inputs is not None:
        fingerprints = [_fingerprint_path(Path(entry["staged_path"])) for entry in source_inputs["entries"]]
        provenance["source_input_fingerprints"] = fingerprints
        digest = hashlib.sha256()
        for entry in fingerprints:
            digest.update(entry["kind"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(entry["path"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(entry["sha256"]).encode("utf-8"))
            digest.update(b"\0")
        provenance["source_inputs_sha256"] = digest.hexdigest()
    return provenance


def _write_probe_sources(
    *,
    request: CorpusRequest,
    source_root: Path,
) -> tuple[list[Path], int]:
    scenarios = request.resolve_scenarios(
        origin="compiled.pipeline-probe",
        tags=("synthetic", "probe", "scenario"),
    )
    corpus_specs = tuple(spec for scenario in scenarios for spec in scenario.corpus_specs)
    source_root.mkdir(parents=True, exist_ok=True)
    provider = request.providers[0] if request.providers else "probe"
    written_batches = SyntheticCorpus.write_specs_artifacts(corpus_specs, source_root, prefix=provider, index_width=3)
    files = [path for written in written_batches for path in written.files]
    total_bytes = sum(len(artifact.raw_bytes) for written in written_batches for artifact in written.batch.artifacts)
    return files, total_bytes


def _names(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _paths(value: object | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        return [Path(value)]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, Path) else Path(str(item)) for item in value]
    return [Path(str(value))]


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
        return [RawConversationRecord.model_validate(dict(row)) for row in cursor.fetchall()]


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


def _persist_manifest(manifest: ArchiveManifest, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _load_manifest(manifest_path: Path) -> ArchiveManifest:
    payload = require_json_document(loads(manifest_path.read_text(encoding="utf-8")), context="archive subset manifest")
    return _archive_manifest_from_payload(payload)


def _string_value(document: JSONDocument, key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise TypeError(f"manifest field {key!r} must be a string")
    return value


def _int_value(document: JSONDocument, key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"manifest field {key!r} must be an integer")
    return value


def _string_list_value(document: JSONDocument, key: str) -> list[str]:
    value = document.get(key)
    if not isinstance(value, list):
        raise TypeError(f"manifest field {key!r} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"manifest field {key!r} must be a list of strings")
        result.append(item)
    return result


def _int_map_value(document: JSONDocument, key: str) -> dict[str, int]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"manifest field {key!r} must be an object")
    result: dict[str, int] = {}
    for item_key, item_value in value.items():
        if isinstance(item_value, bool) or not isinstance(item_value, int):
            raise TypeError(f"manifest field {key!r}.{item_key} must be an integer")
        result[str(item_key)] = item_value
    return result


def _document_list_value(document: JSONDocument, key: str) -> list[JSONDocument]:
    value = document.get(key)
    if not isinstance(value, list):
        raise TypeError(f"manifest field {key!r} must be a list of JSON objects")
    result: list[JSONDocument] = []
    for item in value:
        if not is_json_document(item):
            raise TypeError(f"manifest field {key!r} must be a list of JSON objects")
        result.append(item)
    return result


def _archive_manifest_from_payload(payload: JSONDocument) -> ArchiveManifest:
    return {
        "input_mode": _string_value(payload, "input_mode"),
        "source_db": _string_value(payload, "source_db"),
        "source_blob_root": _string_value(payload, "source_blob_root"),
        "seed": _int_value(payload, "seed"),
        "sample_per_provider": _int_value(payload, "sample_per_provider"),
        "provider_filters": _string_list_value(payload, "provider_filters"),
        "source_filters": _string_list_value(payload, "source_filters"),
        "candidate_count": _int_value(payload, "candidate_count"),
        "missing_blob_count": _int_value(payload, "missing_blob_count"),
        "available_by_provider": _int_map_value(payload, "available_by_provider"),
        "sampled_by_provider": _int_map_value(payload, "sampled_by_provider"),
        "records": _document_list_value(payload, "records"),
    }


def _build_archive_manifest(
    *,
    source_db: Path,
    source_blob_root: Path,
    provider_filters: list[str],
    source_filters: list[str],
    sample_per_provider: int,
    seed: int,
) -> ArchiveManifest:
    source_blob_store = BlobStore(source_blob_root)
    candidate_records = _fetch_archive_candidates(
        db_path=source_db,
        provider_filters=provider_filters,
        source_filters=source_filters,
    )
    candidates_with_blobs = [record for record in candidate_records if source_blob_store.exists(record.raw_id)]
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

    sampled_records.sort(
        key=lambda record: (
            _effective_provider_name(record),
            _source_bucket_name(record),
            record.acquired_at,
            record.raw_id,
        )
    )

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
        "records": [_raw_record_payload(record) for record in sampled_records],
    }


def _raw_record_payload(record: RawConversationRecord) -> JSONDocument:
    return require_json_document(record.model_dump(mode="json"), context="archive subset raw record")


def _resolve_archive_manifest(
    *,
    manifest_in: Path | None,
    source_db: Path | None,
    source_blob_root: Path | None,
    provider_filters: list[str],
    source_filters: list[str],
    sample_per_provider: int,
    seed: int,
) -> ArchiveManifest:
    if manifest_in is not None:
        manifest = _load_manifest(manifest_in)
        if source_blob_root is not None:
            manifest["source_blob_root"] = str(source_blob_root.resolve())
        if source_db is not None:
            manifest["source_db"] = str(source_db.resolve())
        return manifest

    resolved_source_db = (source_db or db_path()).resolve()
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
    manifest: ArchiveManifest,
    repository: _RawConversationStore,
    target_blob_store: BlobStore,
) -> ArchiveSubsetSampleSummary:
    records = [RawConversationRecord.model_validate(record_payload) for record_payload in manifest["records"]]
    source_blob_root = Path(str(manifest["source_blob_root"])).resolve()
    source_blob_store = BlobStore(source_blob_root)
    copied_records = 0
    copied_blob_bytes = 0

    for record in records:
        source_blob_path = source_blob_store.blob_path(record.raw_id)
        if not source_blob_path.exists():
            raise FileNotFoundError(f"archive-subset probe is missing blob {record.raw_id} under {source_blob_root}")
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
        "sample_per_provider": manifest["sample_per_provider"],
        "candidate_count": manifest["candidate_count"],
        "missing_blob_count": manifest["missing_blob_count"],
        "available_by_provider": manifest["available_by_provider"],
        "sampled_by_provider": manifest["sampled_by_provider"],
    }


def _resolved_corpus_request(request: PipelineProbeRequest) -> CorpusRequest:
    if request.corpus_request is not None:
        return request.corpus_request
    return CorpusRequest(
        providers=("chatgpt",),
        source=CorpusSourceKind.DEFAULT,
        count=5,
        messages_min=4,
        messages_max=12,
        seed=42,
        style="default",
        package_version="default",
    )


def _resolve_synthetic_provider(request: PipelineProbeRequest) -> str:
    provider_names = tuple(_resolved_corpus_request(request).providers or ("chatgpt",))
    provider_name = provider_names[0] if provider_names else "chatgpt"
    available = set(SyntheticCorpus.available_providers())
    if provider_name not in available:
        raise ValueError(f"--provider must be one of {sorted(available)} in synthetic mode")
    if len(provider_names) > 1:
        raise ValueError("synthetic mode accepts exactly one --provider")
    return provider_name


def _probe_mode(args: argparse.Namespace | PipelineProbeRequest) -> str:
    if isinstance(args, PipelineProbeRequest):
        return args.input_mode_kind.value
    return str(getattr(args, "input_mode", "synthetic"))


def _load_run_payload(run_path: str | None) -> JSONDocument:
    if not run_path:
        return {}
    return require_json_document(loads(Path(run_path).read_text(encoding="utf-8")), context="pipeline run payload")


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def _copy_source_subset_entry(*, source_path: Path, destination_path: Path) -> tuple[str, int, int]:
    if source_path.is_file():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return "file", 1, destination_path.stat().st_size
    if source_path.is_dir():
        shutil.copytree(source_path, destination_path)
        staged_file_count = sum(1 for child in destination_path.rglob("*") if child.is_file())
        return "directory", staged_file_count, _path_size_bytes(destination_path)
    raise FileNotFoundError(f"source-subset probe input does not exist: {source_path}")


def _stage_source_subset(
    *,
    source_paths: list[Path],
    source_root: Path,
) -> SourceInputsSummary:
    source_root.mkdir(parents=True, exist_ok=True)
    entries: list[StagedSourceEntry] = []
    staged_file_count = 0
    total_bytes = 0

    for index, raw_source_path in enumerate(source_paths):
        source_path = raw_source_path.expanduser().resolve()
        destination_name = f"{index:03d}-{source_path.name or 'source'}"
        destination_path = source_root / destination_name
        entry_kind, entry_file_count, entry_bytes = _copy_source_subset_entry(
            source_path=source_path,
            destination_path=destination_path,
        )
        staged_file_count += entry_file_count
        total_bytes += entry_bytes
        entries.append(
            {
                "input_path": str(source_path),
                "staged_path": str(destination_path),
                "kind": entry_kind,
                "file_count": entry_file_count,
                "bytes": entry_bytes,
            }
        )

    return {
        "input_count": len(source_paths),
        "staged_entry_count": len(entries),
        "staged_file_count": staged_file_count,
        "total_bytes": total_bytes,
        "entries": entries,
    }


async def _run_probe_pipeline(
    *,
    config: Config,
    request: PipelineProbeRequest,
    stage_sequence: list[str] | None,
    source_names: list[str] | None,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
) -> tuple[RunResult, JSONDocument]:
    result = await run_sources(
        config=config,
        stage=request.stage,
        stage_sequence=stage_sequence,
        source_names=source_names,
        raw_batch_size=request.raw_batch_size or 50,
        ingest_workers=request.ingest_workers,
        measure_ingest_result_size=request.measure_ingest_result_size,
        backend=backend,
        repository=repository,
    )
    return result, _load_run_payload(result.run_path)


def _probe_stage_sequence(probe_mode: str, stage: str) -> list[str] | None:
    if probe_mode not in {"synthetic", "source-subset"}:
        return None
    return list(_SOURCE_BACKED_PROBE_STAGE_SEQUENCES[stage])


def _json_object_or_empty(value: object | None) -> JSONDocument:
    return json_document(value)


def _json_float_or_none(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _run_result_payload(result: RunResult) -> JSONDocument:
    return require_json_document(result.model_dump(mode="json"), context="pipeline probe result")


def _json_string_sequence(value: list[str] | None) -> list[JSONValue] | None:
    if value is None:
        return None
    result: list[JSONValue] = []
    result.extend(value)
    return result


async def run_probe(request: PipelineProbeRequest) -> ProbeSummary:
    probe_mode = _probe_mode(request)
    if probe_mode not in _INPUT_MODES:
        raise ValueError(f"--input-mode must be one of {_INPUT_MODES}")

    workspace = _workspace_for_request(request)
    workdir = workspace.root
    raw_batch_size = request.raw_batch_size
    if raw_batch_size is not None and raw_batch_size <= 0:
        raise ValueError("--raw-batch-size must be positive")
    ingest_workers = request.ingest_workers
    if ingest_workers is not None and ingest_workers <= 0:
        raise ValueError("--ingest-workers must be positive")
    archive_root = workspace.archive_root
    render_root = workspace.render_root
    db_path: Path | None = None
    summary: ProbeSummary

    if probe_mode == PipelineProbeInputMode.SYNTHETIC.value:
        corpus_request = _resolved_corpus_request(request)
        if corpus_request.count <= 0:
            raise ValueError("--count must be positive")
        if corpus_request.messages_min <= 0 or corpus_request.messages_max < corpus_request.messages_min:
            raise ValueError("--messages-min/--messages-max must define a positive inclusive range")
        provider_name = _resolve_synthetic_provider(request)
        source_root = workdir / "sources" / provider_name

        stage_sequence = _probe_stage_sequence(probe_mode, request.stage)
        with _isolated_env(workspace):
            files, total_bytes = _write_probe_sources(
                request=corpus_request,
                source_root=source_root,
            )
            config = Config(
                sources=[Source(name=provider_name, path=source_root)],
                archive_root=archive_root,
                render_root=render_root,
            )
            db_path = config.db_path
            result, run_payload = await _run_probe_pipeline(
                config=config,
                request=request,
                stage_sequence=stage_sequence,
                source_names=[provider_name],
            )
        result_payload = _run_result_payload(result)

        summary = {
            "probe": {
                "input_mode": "synthetic",
                "provider": provider_name,
                "corpus_source": corpus_request.source_kind.value,
                "stage": request.stage,
                "stage_sequence": _json_string_sequence(stage_sequence),
                "count": corpus_request.count,
                "messages_min": corpus_request.messages_min,
                "messages_max": corpus_request.messages_max,
                "seed": corpus_request.seed,
                "style": corpus_request.style,
                "package_version": corpus_request.package_version,
                "raw_batch_size": request.raw_batch_size,
                "ingest_workers": request.ingest_workers,
                "measure_ingest_result_size": request.measure_ingest_result_size,
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
            "provenance": _build_probe_provenance(),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path) if db_path is not None else {},
            "raw_fanout": _db_raw_fanout(db_path) if db_path is not None else [],
        }
    elif probe_mode == PipelineProbeInputMode.SOURCE_SUBSET.value:
        source_paths = _paths(request.source_paths)
        if not source_paths:
            raise ValueError("--source-path is required in source-subset mode")

        source_name = request.source_name.strip() or "inbox"
        source_root = workdir / "sources" / source_name
        stage_sequence = _probe_stage_sequence(probe_mode, request.stage)

        with _isolated_env(workspace):
            source_inputs = _stage_source_subset(
                source_paths=source_paths,
                source_root=source_root,
            )
            config = Config(
                sources=[Source(name=source_name, path=source_root)],
                archive_root=archive_root,
                render_root=render_root,
            )
            db_path = config.db_path
            result, run_payload = await _run_probe_pipeline(
                config=config,
                request=request,
                stage_sequence=stage_sequence,
                source_names=[source_name],
            )
        result_payload = _run_result_payload(result)

        summary = {
            "probe": {
                "input_mode": "source-subset",
                "source_name": source_name,
                "stage": request.stage,
                "stage_sequence": _json_string_sequence(stage_sequence),
                "raw_batch_size": request.raw_batch_size,
                "ingest_workers": request.ingest_workers,
                "measure_ingest_result_size": request.measure_ingest_result_size,
            },
            "paths": {
                "workdir": str(workdir),
                "source_root": str(source_root),
                "archive_root": str(archive_root),
                "render_root": str(render_root),
                "db_path": str(db_path),
                "run_path": result.run_path,
            },
            "source_inputs": source_inputs,
            "provenance": _build_probe_provenance(source_inputs=source_inputs),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path) if db_path is not None else {},
            "raw_fanout": _db_raw_fanout(db_path) if db_path is not None else [],
        }
    else:
        sample_per_provider = request.sample_per_provider or 50
        if sample_per_provider <= 0:
            raise ValueError("--sample-per-provider must be positive")
        provider_filters = list(_resolved_corpus_request(request).providers or ())
        source_filters = list(request.source_filters)
        resolved_corpus_request = _resolved_corpus_request(request)
        manifest = _resolve_archive_manifest(
            manifest_in=Path(request.manifest_in) if request.manifest_in is not None else None,
            source_db=Path(request.source_db) if request.source_db is not None else None,
            source_blob_root=Path(request.source_blob_root) if request.source_blob_root is not None else None,
            provider_filters=provider_filters,
            source_filters=source_filters,
            sample_per_provider=sample_per_provider,
            seed=resolved_corpus_request.seed or 42,
        )
        manifest_path = workdir / "archive-subset-manifest.json"
        _persist_manifest(manifest, manifest_path)
        if request.manifest_out is not None:
            _persist_manifest(manifest, Path(request.manifest_out))

        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=render_root,
        )
        with _isolated_env(workspace):
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
                result, run_payload = await _run_probe_pipeline(
                    config=config,
                    request=request,
                    stage_sequence=None,
                    source_names=None,
                    backend=backend,
                    repository=repository,
                )
            finally:
                await repository.close()
        result_payload = _run_result_payload(result)

        summary = {
            "probe": {
                "input_mode": "archive-subset",
                "stage": request.stage,
                "seed": resolved_corpus_request.seed,
                "sample_per_provider": sample_per_provider,
                "provider_filters": _json_string_sequence(provider_filters),
                "source_filters": _json_string_sequence(source_filters),
                "raw_batch_size": request.raw_batch_size,
                "ingest_workers": request.ingest_workers,
                "measure_ingest_result_size": request.measure_ingest_result_size,
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
            "provenance": _build_probe_provenance(manifest_path=manifest_path),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path) if db_path is not None else {},
            "raw_fanout": _db_raw_fanout(db_path) if db_path is not None else [],
        }

    return summary


def _build_budget_report(summary: ProbeSummary, request: PipelineProbeRequest) -> BudgetReport | None:
    if request.max_total_ms is None and request.max_peak_rss_mb is None:
        return None

    run_payload = _json_object_or_empty(summary.get("run_payload"))
    metrics = _json_object_or_empty(run_payload.get("metrics"))
    result_payload = _json_object_or_empty(summary.get("result"))
    observed_total_ms = metrics.get("total_duration_ms", result_payload.get("duration_ms"))
    observed_peak_rss_mb = metrics.get("peak_rss_self_mb")
    violations: list[str] = []

    if request.max_total_ms is not None:
        if observed_total_ms is None:
            violations.append("missing total runtime metric")
        elif (observed_total_ms_value := _json_float_or_none(observed_total_ms)) is None:
            violations.append("non-numeric total runtime metric")
        elif observed_total_ms_value > request.max_total_ms:
            violations.append(
                f"total runtime {observed_total_ms_value:.1f} ms exceeded budget {request.max_total_ms:.1f} ms"
            )

    if request.max_peak_rss_mb is not None:
        if observed_peak_rss_mb is None:
            violations.append("missing peak RSS metric")
        elif (observed_peak_rss_mb_value := _json_float_or_none(observed_peak_rss_mb)) is None:
            violations.append("non-numeric peak RSS metric")
        elif observed_peak_rss_mb_value > request.max_peak_rss_mb:
            violations.append(
                f"peak RSS {observed_peak_rss_mb_value:.1f} MiB exceeded budget {request.max_peak_rss_mb:.1f} MiB"
            )

    return {
        "ok": not violations,
        "max_total_ms": request.max_total_ms,
        "observed_total_ms": observed_total_ms,
        "max_peak_rss_mb": request.max_peak_rss_mb,
        "observed_peak_rss_mb": observed_peak_rss_mb,
        "violations": violations,
    }


def _capture_regression_case(summary: ProbeSummary, args: argparse.Namespace) -> RegressionCaseSummary | None:
    name = getattr(args, "capture_regression", None)
    if not isinstance(name, str) or not name.strip():
        return None
    output_dir = getattr(args, "regression_output_dir", DEFAULT_REGRESSION_CASE_DIR)
    if not isinstance(output_dir, Path):
        output_dir = Path(str(output_dir))
    tags = tuple(str(tag) for tag in getattr(args, "regression_tag", []) or [])
    notes = tuple(str(note) for note in getattr(args, "regression_note", []) or [])
    case = RegressionCase.from_probe_summary(
        name=name,
        summary=json_document(summary),
        tags=tags,
        notes=notes,
    )
    path = RegressionCaseStore(output_dir).write(case)
    return {
        "case_id": case.case_id,
        "name": case.name,
        "path": str(path),
        "tags": list(case.tags),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    request = _request_from_args(args)
    active_request = request
    if request.workdir is None:
        with tempfile.TemporaryDirectory(prefix="polylogue-pipeline-probe-") as tempdir:
            active_request = replace(request, workdir=str(Path(tempdir)))
            with redirect_stdout(sys.stderr):
                summary = asyncio.run(run_probe(active_request))
    else:
        with redirect_stdout(sys.stderr):
            summary = asyncio.run(run_probe(active_request))
    budget_report = _build_budget_report(summary, active_request)
    if budget_report is not None:
        summary["budgets"] = budget_report
    regression_case = _capture_regression_case(summary, args)
    if regression_case is not None:
        summary["regression_case"] = regression_case
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if active_request.json_out is not None:
        json_out = Path(active_request.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if budget_report is not None and not budget_report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
