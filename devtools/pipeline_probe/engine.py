"""Pipeline probe execution: synthetic, archive-subset, and source-subset modes."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from devtools.pipeline_probe.request import (
    _INPUT_MODES,
    _REPO_ROOT,
    _SOURCE_BACKED_PROBE_STAGE_SEQUENCES,
    ArchiveManifest,
    ArchiveSubsetSampleSummary,
    ProbeProvenance,
    ProbeSummary,
    SourceInputsSummary,
    _load_run_payload,
    _paths,
    _probe_mode,
    _resolve_synthetic_provider,
    _resolved_corpus_request,
)
from devtools.pipeline_probe.result import (
    _db_raw_fanout,
    _db_row_counts,
    _json_string_sequence,
    _run_result_payload,
)
from devtools.pipeline_probe.staging import (
    _fingerprint_path,
    _sha256_file,
    _stage_source_subset,
    _write_probe_sources,
)
from polylogue.config import Config, Source
from polylogue.core.json import JSONDocument, is_json_document, loads, require_json_document
from polylogue.paths import blob_store_root, db_path
from polylogue.pipeline.runner import run_sources
from polylogue.scenarios import (
    PipelineProbeInputMode,
    PipelineProbeRequest,
)
from polylogue.showcase.workspace import VerificationWorkspace, create_verification_workspace
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.run_state import RunResult
from polylogue.storage.runtime import RawConversationRecord
from polylogue.storage.sqlite import SQLiteBackend, create_backend
from polylogue.storage.sqlite.connection import (
    _build_provider_scope_filter,
    _build_source_scope_filter,
    open_connection,
)
from polylogue.storage.sqlite.queries.raw_state import EFFECTIVE_RAW_PROVIDER_SQL
from polylogue.types import Provider


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


def _raw_record_payload(record: RawConversationRecord) -> JSONDocument:
    return require_json_document(record.model_dump(mode="json"), context="archive subset raw record")


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


class _RawConversationStore(Protocol):
    async def save_raw_conversation(self, record: RawConversationRecord) -> bool: ...


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
    db_path_val: Path | None = None
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
            db_path_val = config.db_path
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
                "db_path": str(db_path_val),
                "run_path": result.run_path,
            },
            "source_files": {
                "count": len(files),
                "total_bytes": total_bytes,
            },
            "provenance": _build_probe_provenance(),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path_val) if db_path_val is not None else {},
            "raw_fanout": _db_raw_fanout(db_path_val) if db_path_val is not None else [],
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
            db_path_val = config.db_path
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
                "db_path": str(db_path_val),
                "run_path": result.run_path,
            },
            "source_inputs": source_inputs,
            "provenance": _build_probe_provenance(source_inputs=source_inputs),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path_val) if db_path_val is not None else {},
            "raw_fanout": _db_raw_fanout(db_path_val) if db_path_val is not None else [],
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
            db_path_val = config.db_path
            backend = create_backend(db_path=db_path_val)
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
                "db_path": str(db_path_val),
                "run_path": result.run_path,
                "manifest_path": str(manifest_path),
                "source_db": str(manifest["source_db"]),
                "source_blob_root": str(manifest["source_blob_root"]),
            },
            "sample": sample_summary,
            "provenance": _build_probe_provenance(manifest_path=manifest_path),
            "result": result_payload,
            "run_payload": run_payload,
            "db_stats": _db_row_counts(db_path_val) if db_path_val is not None else {},
            "raw_fanout": _db_raw_fanout(db_path_val) if db_path_val is not None else {},
        }

    return summary


__all__ = [
    "_build_probe_provenance",
    "_empty_archive_manifest_error",
    "_fetch_archive_candidates",
    "_isolated_env",
    "_normalize_record_for_replay",
    "_probe_stage_sequence",
    "_provider_counts",
    "_raw_conversation_count",
    "_run_probe_pipeline",
    "_safe_git_commit",
    "_safe_git_worktree_dirty",
    "_sample_provider_records",
    "_seed_archive_subset",
    "_source_bucket_name",
    "_source_counts",
    "_workspace_for_request",
    "run_probe",
]
