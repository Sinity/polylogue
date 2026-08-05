"""source parsing helpers."""

from __future__ import annotations

import json
import zipfile
from concurrent.futures import as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.logging import get_logger
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.pipeline.services.process_pool import (
    process_pool_executor,
    resolve_archive_ingest_dispatch,
)
from polylogue.sources.decoder_zip import (
    ZipBombError,
    ZipEntryValidator,
    open_bounded_zip_entry,
    zip_entry_session_artifact,
)
from polylogue.sources.parsers.base import ParsedSession, RawSessionData
from polylogue.sources.source_parsing import (
    has_decoded_session_evidence,
    iter_antigravity_language_server_sessions,
    iter_source_sessions_with_raw,
    parse_one_source_path,
)
from polylogue.sources.source_walk import _setup_source_walk
from polylogue.sources.sqlite_snapshot import hermes_profile_raw_id
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError, deterministic_raw_session_id
from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers
from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_archive_wals

logger = get_logger(__name__)

# Work-based commit batching (#dogfood ingest-commit-batching). Re-ingest is
# I/O-wait-bound index writes benefit from committing once ~8000 accumulated
# messages (validated ~1.37x throughput, ~4x fewer bytes, peak WAL ~14 MB <<
# 40 MB autocheckpoint). Durable source references commit per raw artifact so
# parallel workers can establish pre-publication reservations without blocking
# behind a long source transaction. Session-count batching was rejected (uneven
# index transaction size -> larger WAL); one-shot was rejected (slower + large
# WAL). Override with POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES; a value <= 0
# restores per-session index commit (escape hatch).
COMMIT_BATCH_MESSAGE_THRESHOLD = 8000
POST_COMMIT_UPKEEP_REASON = "archive_ingest_commit"


def _commit_batch_message_threshold() -> int:
    from polylogue.config import load_polylogue_config

    try:
        return load_polylogue_config().ingest_commit_batch_messages
    except ValueError:
        return COMMIT_BATCH_MESSAGE_THRESHOLD


def _parse_source_path_worker(
    path_str: str,
    file_mtime: str | None,
    source_name: str,
    sidecar_data: Any,
    capture_raw: bool,
    blob_root_str: str,
    source_db_path_str: str,
) -> list[tuple[RawSessionData | None, ParsedSession]]:
    """ProcessPool worker: parse one file and return materialized tuples.

    Materializing into a list is required so the worker can pickle results back
    to the main process. Errors propagate via the future and are caught by the
    driver. ``cursor_state`` is intentionally ``None``: re-ingest does not use
    cursor state, and it could not cross the process boundary anyway.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    publisher = ArchiveBlobPublisher(Path(source_db_path_str), Path(blob_root_str))
    try:
        return list(
            parse_one_source_path(
                path_str,
                file_mtime=file_mtime,
                source_name=source_name,
                sidecar_data=sidecar_data,
                capture_raw=capture_raw,
                cursor_state=None,
                blob_root=Path(blob_root_str),
                blob_store=publisher,
            )
        )
    finally:
        publisher.discard_pending()


async def parse_sources_archive(
    archive_root: Path, sources: list[Source], *, parse_workers: int | None = None
) -> ParseResult:
    """Parse configured sources directly into archive source/index tiers.

    Source files are parsed across a process pool (``POLYLOGUE_INGEST_PARSE_WORKERS``)
    while the main process remains the single SQLite writer. Write order does
    not matter: archive writes are idempotent by content hash and session links
    resolve out-of-order. Blob writes from workers are
    content-addressed and atomic, so concurrent worker writes are process-safe.

    ``parse_workers`` overrides the ambient/env-resolved worker count for this
    call only (used by the demo seeder to force sequential parsing -- see
    ``polylogue/demo/seed.py``). ``None`` preserves the normal
    ``POLYLOGUE_INGEST_PARSE_WORKERS``/cpu-count resolution.
    """
    result = ParseResult()
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    threshold = _commit_batch_message_threshold()
    batched = threshold > 0
    workers = resolve_archive_ingest_dispatch(parse_workers=parse_workers).worker_count
    blob_root = archive_root / "blob"
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    parse_blob_publisher = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
    counters = {"raw_rows": 0, "index_rows": 0, "pending_messages": 0}
    # A grouped JSONL file (Claude Code/Codex resume-fork chains, Gemini/Drive
    # bundles) can parse into MULTIPLE sessions that all share the identical
    # captured raw bytes (`_SessionEmitter._emit_grouped` yields the SAME
    # `raw_data` for every session in the group). Without this cache, each
    # session's write independently derives its raw_id from
    # `deterministic_raw_session_id(..., native_id=session.provider_session_id)`
    # (see write_source_raw_session), so byte-identical content produces a
    # DIFFERENT raw_sessions row per split session -- and a second, unrelated
    # raw row for the same bytes reappears on every re-ingest. The live daemon
    # watcher instead writes ONE raw per file (`write_raw_payload`, no
    # native_id) and defers session identity to membership-census
    # classification; this cache makes the one-shot importer converge on that
    # SAME single-raw-per-bytes model, keyed by the raw's own (origin,
    # source_path, source_index, blob_hash) so a later re-ingest of identical
    # bytes resolves to the SAME raw_id every time. Ref polylogue-sjf6.
    shared_raw_ids: dict[tuple[str, str, int, str], str] = {}
    shared_raw_memberships: dict[tuple[str, str, int, str], dict[str, ParsedSession]] = {}

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        counters["raw_rows"] += _admit_non_session_origin_artifacts(
            archive,
            sources,
            acquired_at_ms=acquired_at_ms,
        )

        async def write_pair(
            source: Source,
            raw_data: RawSessionData | None,
            session: ParsedSession,
        ) -> None:
            payload = _archive_raw_payload(raw_data, session, blob_root=blob_root)
            source_path = _archive_raw_source_path(raw_data, source)
            source_index = _archive_raw_source_index(raw_data)
            raw_id = None
            blob_hash = getattr(raw_data, "blob_hash", None)
            blob_hash_str = blob_hash if isinstance(blob_hash, str) and blob_hash else None
            if session.source_name is Provider.HERMES and blob_hash_str is not None:
                raw_id = hermes_profile_raw_id(source_path, source_index, blob_hash_str)
            shared_key: tuple[str, str, int, str] | None = None
            if raw_id is None and blob_hash_str is not None:
                # Keyed by origin (not provider) to match deterministic_raw_session_id
                # below -- origin_from_provider is non-injective (GEMINI and DRIVE
                # both collapse to AISTUDIO_DRIVE), so two sessions with different
                # `source_name` but the same origin must still share one raw_id.
                shared_key = (
                    origin_from_provider(session.source_name).value,
                    source_path,
                    source_index,
                    blob_hash_str,
                )
            existing_raw_id = shared_raw_ids.get(shared_key) if shared_key is not None else None
            try:
                if existing_raw_id is not None:
                    # A prior session parsed from this exact raw already
                    # committed the raw row this batch; index this session
                    # against that SAME raw_id instead of writing a duplicate.
                    retained_result = archive.write_parsed_for_retained_raw_result(
                        session,
                        raw_id=existing_raw_id,
                        source_path=source_path,
                        acquired_at_ms=acquired_at_ms,
                        source_index=source_index,
                        stage_timings_s=result.stage_timings_s,
                        manage_transaction=not batched,
                    )
                    write_result = retained_result
                else:
                    if shared_key is not None and blob_hash_str is not None:
                        raw_id = deterministic_raw_session_id(
                            origin_from_provider(session.source_name),
                            source_path,
                            source_index,
                            bytes.fromhex(blob_hash_str),
                            None,
                        )
                    write_result = archive.admit_raw_and_parsed_result(
                        session,
                        payload=payload,
                        source_path=source_path,
                        acquired_at_ms=acquired_at_ms,
                        source_index=source_index,
                        raw_id=raw_id,
                        shared_raw=shared_key is not None,
                        logical_source_key=(
                            f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
                        ),
                        stage_timings_s=result.stage_timings_s,
                        manage_transaction=not batched,
                        blob_publication_receipt_id=(
                            raw_data.blob_publication_receipt_id if raw_data is not None else None
                        ),
                    )
                if shared_key is not None:
                    shared_raw_ids[shared_key] = write_result.raw_id
                    memberships = shared_raw_memberships.setdefault(shared_key, {})
                    memberships[session.provider_session_id] = session
                    archive.replace_raw_membership_census(
                        write_result.raw_id,
                        list(memberships.values()),
                        parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                        censused_at_ms=acquired_at_ms,
                        # This mutates durable source.db state. Its transaction
                        # must close before the next grouped raw publisher can
                        # reserve through its separate source connection;
                        # index message batching applies only to index.db.
                        manage_transaction=True,
                    )
            except ContentExcisedError as exc:
                # The archive can forget on purpose (polylogue-27m): this raw
                # payload's content hash is durably excised, so acquire
                # refuses to re-store it. This is deliberate, not a failure --
                # skip only this one file and continue the batch (unlike the
                # broad except below, do NOT roll back prior sessions already
                # staged in this batch).
                result.excised_skips += 1
                logger.info("Skipping durably excised content: %s", exc)
                return
            except Exception:
                # Discard the in-flight uncommitted batch so a failed write
                # never leaves prior sessions in this batch half-applied.
                # Re-ingest is restartable from durable source evidence.
                if batched:
                    archive.rollback()
                raise
            counters["raw_rows"] += 1
            index_changed = (
                write_result.counts.get("sessions", 0)
                + write_result.counts.get("messages", 0)
                + write_result.counts.get("attachments", 0)
                + write_result.counts.get("session_events", 0)
                + write_result.counts.get("raw_links", 0)
            ) > 0
            counters["index_rows"] += int(index_changed)
            await result.merge_result(
                write_result.session_id,
                write_result.counts,
                content_changed=write_result.content_changed,
            )
            if batched:
                counters["pending_messages"] += len(session.messages)
                if counters["pending_messages"] >= threshold:
                    archive.commit()
                    _record_post_commit_upkeep(archive_root, result, reason=POST_COMMIT_UPKEEP_REASON)
                    counters["pending_messages"] = 0
            else:
                _record_post_commit_upkeep(archive_root, result, reason=POST_COMMIT_UPKEEP_REASON)

        if workers <= 1:
            # Escape hatch: exact sequential behavior, no pool.
            for source in sources:
                for raw_data, session in iter_source_sessions_with_raw(
                    source,
                    capture_raw=True,
                    blob_root=blob_root,
                    blob_store=parse_blob_publisher,
                ):
                    await write_pair(source, raw_data, session)
        else:
            # Antigravity language-server export stays sequential (it drives a
            # local loopback subprocess); only the file-walk parallelizes.
            # `capture_raw=True` is required here (polylogue-3m3de): without
            # it every exported session's `raw_data` is `None`, so
            # `_archive_raw_payload`/`_archive_raw_source_path` fall back to
            # a JSON dump of the parsed session and the shared source root
            # path -- collapsing every conversation's raw provenance onto one
            # non-unique source_path instead of its actual `.pb` file and
            # real bytes.
            for source in sources:
                for raw_data, session in iter_antigravity_language_server_sessions(
                    source,
                    capture_raw=True,
                    blob_root=blob_root,
                    blob_store=parse_blob_publisher,
                ):
                    await write_pair(source, raw_data, session)

            failed = 0
            total_paths = 0
            with process_pool_executor(max_workers=workers) as pool:
                future_to_source: dict[Any, tuple[Source, Path]] = {}
                for source in sources:
                    walk = _setup_source_walk(
                        source,
                        cursor_state=None,
                        include_mtime=True,
                        known_mtimes=None,
                        discover_sidecars=True,
                        blob_store=parse_blob_publisher,
                    )
                    if walk is None:
                        continue
                    for path, file_mtime in walk.paths_to_process:
                        future = pool.submit(
                            _parse_source_path_worker,
                            str(path),
                            file_mtime,
                            source.name,
                            walk.sidecar_data,
                            True,
                            str(blob_root),
                            str(archive_root / "source.db"),
                        )
                        future_to_source[future] = (source, path)
                        total_paths += 1

                for future in as_completed(future_to_source):
                    source, path = future_to_source[future]
                    try:
                        pairs = future.result()
                    except Exception as exc:
                        # Worker error isolation: one bad file must not kill the
                        # run. Mirror the sequential iterator's failure handling.
                        failed += 1
                        result.parse_failures += 1
                        logger.error("Failed to parse %s in worker: %s", path, exc)
                        continue
                    for raw_data, session in pairs:
                        await write_pair(source, raw_data, session)

            if failed > 0:
                logger.warning(
                    "Skipped %d of %d files due to parse/read errors during parallel ingest.",
                    failed,
                    total_paths,
                )

        if batched and counters["pending_messages"] > 0:
            archive.commit()
            _record_post_commit_upkeep(archive_root, result, reason=POST_COMMIT_UPKEEP_REASON)
        parse_blob_publisher.discard_pending()

    raw_rows_written = counters["raw_rows"]
    index_rows_written = counters["index_rows"]
    if any(Provider.from_string(source.name) is Provider.CLAUDE_CODE for source in sources):
        try:
            from polylogue.insights.claude_workflow_materializer import materialize_claude_workflow_archive

            workflow_summary = materialize_claude_workflow_archive(archive_root)
            result.batch_observations.append(
                {
                    "claude_workflow_materialization": True,
                    "claude_workflow_summary": workflow_summary.as_dict(),
                }
            )
        except Exception as exc:
            logger.error("Claude Workflow materialization failed after direct archive ingest", exc_info=True)
            result.batch_observations.append(
                {
                    "claude_workflow_materialization": True,
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result.batch_observations.append(
        {
            "primary_ingest_store": "archive_file_set",
            "archive_primary_write": True,
            "archive_write_mode": "archive",
            "archive_root": str(archive_root),
            "archive_write_targets": ["source.db", "index.db"],
            "archive_source_rows": raw_rows_written,
            "archive_index_rows": index_rows_written,
            "records": len(result.processed_ids),
            "sessions": result.counts["sessions"],
            "messages": result.counts["messages"],
            "changed_sessions": result.changed_counts["sessions"],
            "excised_skips": result.excised_skips,
        }
    )
    return result


def _admit_non_session_origin_artifacts(
    archive: ArchiveStore,
    sources: list[Source],
    *,
    acquired_at_ms: int,
) -> int:
    """Retain configured OriginSpec fact artifacts skipped by session parsing."""

    admitted = 0
    for source in sources:
        if Provider.from_string(source.name) is not Provider.CLAUDE_CODE:
            continue
        walk = _setup_source_walk(
            source,
            cursor_state=None,
            include_mtime=False,
            known_mtimes=None,
            discover_sidecars=True,
        )
        if walk is None:
            continue
        provider = Provider.from_string(source.name)
        for candidate, _mtime in walk.paths_to_process:
            if candidate.suffix.lower() == ".zip":
                admitted += _admit_non_session_zip_artifacts(
                    archive,
                    candidate,
                    provider=provider,
                    acquired_at_ms=acquired_at_ms,
                )
                continue
            classification = classify_artifact_path(candidate, provider=source.name)
            if (
                classification is None
                or classification.parse_as_session
                or has_decoded_session_evidence(candidate, provider=provider)
            ):
                continue
            try:
                # polylogue-1fijp arm 4: route through the raw-admission
                # chokepoint so this configured fact artifact gets its
                # raw_artifacts classification row immediately, instead of
                # only a bare raw_sessions row waiting on the next offline
                # materialize_artifact_observations sweep.
                archive.admit_raw_artifact_payload(
                    provider=Provider.CLAUDE_CODE,
                    payload=Path(candidate).read_bytes(),
                    source_path=str(candidate),
                    source_index=0,
                    acquired_at_ms=acquired_at_ms,
                    classification=classification,
                )
                admitted += 1
            except Exception:
                logger.error("Failed to admit configured Claude fact artifact %s", candidate, exc_info=True)
    return admitted


def _admit_non_session_zip_artifacts(
    archive: ArchiveStore,
    zip_path: Path,
    *,
    provider: Provider,
    acquired_at_ms: int,
) -> int:
    """Retain ZIP member artifacts only after decoded JSONL evidence is absent."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    admitted = 0
    publisher = ArchiveBlobPublisher(archive.source_db_path, archive.archive_root / "blob")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            validator = ZipEntryValidator(provider, cursor_state=None, zip_path=zip_path)
            for info in validator.filter_entries(zf.infolist()):
                classification = classify_artifact_path(info.filename, provider=provider)
                if (
                    classification is None
                    or classification.parse_as_session
                    or zip_entry_session_artifact(zf, info, provider=provider) is not None
                ):
                    continue
                with open_bounded_zip_entry(zf, info.filename) as payload:
                    blob_hash, blob_size = publisher.write_from_fileobj(payload)
                receipt_id = publisher.receipt_id(blob_hash)
                publisher.flush()
                archive.admit_raw_artifact_blob_ref(
                    provider=provider,
                    blob_hash_hex=blob_hash,
                    blob_size=blob_size,
                    source_path=f"{zip_path}:{info.filename}",
                    source_index=0,
                    acquired_at_ms=acquired_at_ms,
                    classification=classification,
                    blob_publication_receipt_id=receipt_id,
                )
                admitted += 1
    except (OSError, ZipBombError, zipfile.BadZipFile):
        logger.error("Failed to admit configured ZIP artifacts from %s", zip_path, exc_info=True)
    finally:
        publisher.discard_pending()
    return admitted


def _record_post_commit_upkeep(archive_root: Path, result: ParseResult, *, reason: str) -> None:
    """Run bounded archive-tier upkeep after a direct archive ingest commit.

    Direct re-ingest writes through ``ArchiveStore`` instead of the daemon's
    ingest-batch core.  The upkeep still belongs on the ingest path: WAL
    checkpointing and planner statistics are rebuildable archive invariants, not
    operator maintenance chores that only happen when the daemon has been up for
    a full periodic cycle.
    """

    wal_observations = maybe_checkpoint_archive_wals(archive_root, reason=reason, allow_truncate=False)
    optimize_observations = maybe_optimize_archive_tiers(archive_root, reason=reason)
    result.batch_observations.append(
        {
            "archive_post_commit_upkeep": True,
            "reason": reason,
            "archive_root": str(archive_root),
            "wal_checkpoint_modes": [observation.mode for observation in wal_observations if observation.ran],
            "wal_checkpoint_errors": [observation.error for observation in wal_observations if observation.error],
            "wal_checkpoint_blocked_count": sum(
                1 for observation in wal_observations if observation.busy_pages > 0 or observation.blocking_processes
            ),
            "sqlite_optimize_ran": sum(1 for observation in optimize_observations if observation.ran),
            "sqlite_optimize_errors": [observation.error for observation in optimize_observations if observation.error],
        }
    )


def _archive_raw_payload(raw_data: object, session: Any, *, blob_root: Path) -> bytes:
    from polylogue.storage.blob_store import BlobStore

    raw_bytes = getattr(raw_data, "raw_bytes", None)
    if isinstance(raw_bytes, bytes) and raw_bytes:
        return raw_bytes
    blob_hash = getattr(raw_data, "blob_hash", None)
    if isinstance(blob_hash, str) and blob_hash:
        return BlobStore(blob_root).read_all(blob_hash)
    if callable(getattr(session, "model_dump_json", None)):
        return str(session.model_dump_json()).encode("utf-8")
    return json.dumps(str(session), sort_keys=True).encode("utf-8")


def _archive_raw_source_path(raw_data: object, source: Source) -> str:
    source_path = getattr(raw_data, "source_path", None)
    if source_path is not None:
        return str(source_path)
    return str(source.path)


def _archive_raw_source_index(raw_data: object) -> int:
    source_index = getattr(raw_data, "source_index", None)
    return int(source_index) if source_index is not None else 0


__all__ = ["parse_sources_archive"]
