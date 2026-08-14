"""Source parsing iterators over traversal and parser-emission helpers."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.archive.raw_payload.decode import jsonl_session_artifact
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads
from polylogue.logging import get_logger
from polylogue.sources.assembly import SidecarData
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _ParseContext, _record_cursor_failure
from .decoders import _process_zip
from .dispatch import GROUP_PROVIDERS as _GROUP_PROVIDERS
from .dispatch import is_jsonl_source_path
from .emitter import _SessionEmitter
from .parsers import antigravity, hermes_state, hermes_verification
from .parsers.base import ParsedSession, RawSessionData
from .source_walk import _setup_source_walk
from .sqlite_snapshot import is_sqlite_path, original_sqlite_source_path, snapshot_sqlite_to_blob

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger


def has_decoded_session_evidence(path: Path, *, provider: Provider) -> bool:
    """Return whether decoded JSON content outranks a non-session path rule."""
    if is_jsonl_source_path(str(path)):
        return jsonl_session_artifact(path, provider=provider) is not None

    if path.suffix.lower() != ".json":
        return False
    try:
        document = json_loads(path.read_bytes())
    except (JSONDecodeError, OSError):
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_session


def iter_antigravity_language_server_sessions(
    source: Source,
    *,
    capture_raw: bool = False,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    only_cascade_ids: frozenset[str] | None = None,
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Yield Antigravity language-server export sessions for a source.

    Primary acquisition route for the real ``.pb`` conversation trajectories
    under ``conversations/`` (polylogue-eo81): each is converted via the
    local Antigravity language server's ``ConvertTrajectoryToMarkdown`` RPC,
    which understands the protobuf wire format so this code never has to.
    Kept sequential (it drives a local HTTP loopback subprocess) and shared
    between the sequential iterator and the parallel ingest driver so the
    file-walk parallelization never touches this path.

    When ``capture_raw`` is set, the underlying ``.pb`` bytes are also
    snapshotted into the blob store per exported session (mirrors
    ``hermes_state``'s sqlite snapshot pattern above), so ``source.db`` holds
    real acquired bytes with a stable content hash for idempotent re-ingest,
    not just an ephemeral export that re-runs unconditionally every pass.

    ``only_cascade_ids``, when given, restricts export to that subset of
    cascade ids (see ``antigravity.iter_language_server_exports``) instead of
    the whole ``conversations/`` corpus -- used by the daemon's periodic
    reconciliation loop to convert only not-yet-acquired cascades.

    Brain ``*.md.metadata.json`` files are artifacts, never a fallback
    transcript source. If the exporter or its ``conversations/`` directory is
    unavailable, this yields no sessions and logs the resulting coverage gap.
    A mid-export *abort* (some sessions obtained, the rest lost) is likewise
    surfaced loudly rather than being silently backfilled with degraded
    per-artifact fragments.
    """
    if not source.path:
        return
    provider_hint = Provider.from_string(source.name)
    if provider_hint is not Provider.ANTIGRAVITY:
        return

    conversations_dir = source.path / "conversations"
    if not conversations_dir.is_dir():
        logger.warning(
            "antigravity_coverage_gap",
            source_name="antigravity",
            source_path=str(source.path),
            reason="conversations_directory_missing",
            action="restore the conversations/ directory or configure the language-server export",
        )
        return

    try:
        for session in antigravity.iter_language_server_exports(source.path, only_cascade_ids=only_cascade_ids):
            raw_data = _antigravity_raw_snapshot(
                source.path,
                session,
                capture_raw=capture_raw,
                blob_root=blob_root,
                blob_store=blob_store,
            )
            yield (raw_data, session)
    except antigravity.AntigravityBinaryUnavailableError as exc:
        logger.warning(
            "antigravity_coverage_gap",
            source_name="antigravity",
            source_path=str(source.path),
            reason="language_server_unavailable",
            action="install or configure the language-server export and rerun acquisition",
            detail=str(exc),
        )
    except antigravity.AntigravityPartialExportError as exc:
        # Mid-export failure: some sessions were obtained before the abort.
        # Surface obtained-vs-expected loudly instead of replacing the lost
        # conversations with unrelated per-artifact fragments.
        logger.error(
            "antigravity_coverage_gap",
            source_name="antigravity",
            source_path=str(source.path),
            reason="partial_language_server_export",
            action="resolve the export failure and rerun acquisition for the uncovered conversations",
            obtained=exc.obtained,
            expected=exc.expected,
            uncovered=max(exc.expected - exc.obtained, 0),
            detail=str(exc),
        )
    except antigravity.AntigravityExportError as exc:
        logger.warning(
            "antigravity_coverage_gap",
            source_name="antigravity",
            source_path=str(source.path),
            reason="language_server_export_failed",
            action="repair the language-server export and rerun acquisition",
            detail=str(exc),
        )


def _antigravity_raw_snapshot(
    root: Path,
    session: ParsedSession,
    *,
    capture_raw: bool,
    blob_root: Path | None,
    blob_store: BlobStore | None,
) -> RawSessionData | None:
    """Snapshot the raw ``.pb`` bytes backing an exported session, if requested."""
    if not capture_raw:
        return None
    pb_path = root / "conversations" / f"{session.provider_session_id}.pb"
    if not pb_path.is_file():
        return None
    resolved_blob_root = blob_root
    if resolved_blob_root is None:
        from polylogue.paths import blob_store_root

        resolved_blob_root = blob_store_root()
    resolved_store = blob_store or BlobStore(resolved_blob_root)
    blob_hash, blob_size = resolved_store.write_from_path(pb_path)
    from polylogue.storage.blob_publication import flush_blob_publications, publication_receipt_id

    receipt_id = publication_receipt_id(resolved_store, blob_hash)
    flush_blob_publications(resolved_store)
    return RawSessionData(
        raw_bytes=b"",
        source_path=str(pb_path),
        source_index=None,
        file_mtime=_cursor._get_file_mtime(pb_path),
        provider_hint=Provider.ANTIGRAVITY,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=receipt_id,
    )


def parse_one_source_path(
    path_str: str,
    *,
    file_mtime: str | None,
    source_name: str,
    sidecar_data: SidecarData,
    capture_raw: bool,
    cursor_state: CursorStatePayload | None = None,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Parse a single source file into ``(raw, session)`` tuples.

    Module-level and picklable-by-argument so it can run inside a
    ``ProcessPoolExecutor`` worker: all parameters are picklable (str, str,
    str, the dataclass-backed ``SidecarData`` mapping, bool) and the yielded
    ``RawSessionData``/``ParsedSession`` pydantic models pickle cheaply (pickle
    round-trip is ~6x cheaper than parsing). Blob writes are content-addressed
    and atomic (tempfile + ``os.replace`` in ``blob_store.write_from_path``), so
    concurrent worker blob writes are process-safe.

    Errors (parse/decode/missing-file) propagate to the caller; the sequential
    iterator records them against ``cursor_state`` and the parallel driver
    catches per-future and increments ``parse_failures``.
    """
    path = Path(path_str)
    provider_hint = Provider.from_string(source_name)
    path_classification = classify_artifact_path(path, provider=source_name)
    if (
        path_classification is not None
        and not path_classification.parse_as_session
        and not has_decoded_session_evidence(path, provider=provider_hint)
    ):
        return
    should_group = provider_hint in _GROUP_PROVIDERS

    if path.suffix.lower() == ".zip":
        yield from _process_zip(
            path,
            provider_hint=provider_hint,
            should_group=should_group,
            file_mtime=file_mtime,
            capture_raw=capture_raw,
            cursor_state=cursor_state,
            blob_root=blob_root,
            blob_store=blob_store,
            sidecar_data=sidecar_data,
        )
        return

    original_source_path = original_sqlite_source_path(path) if is_sqlite_path(path) else None
    if (provider_hint is Provider.HERMES or original_source_path is not None) and hermes_state.looks_like_state_db_path(
        path
    ):
        if blob_root is None:
            from polylogue.paths import blob_store_root

            blob_root = blob_store_root()
        resolved_store = blob_store or BlobStore(blob_root)
        snapshot = snapshot_sqlite_to_blob(path, resolved_store)
        from polylogue.storage.blob_publication import flush_blob_publications

        flush_blob_publications(resolved_store)
        retained_path = resolved_store.blob_path(snapshot.blob_hash)
        raw_data = None
        if capture_raw:
            raw_data = RawSessionData(
                raw_bytes=b"",
                source_path=str(original_source_path or path),
                source_index=None,
                file_mtime=file_mtime,
                provider_hint=provider_hint,
                blob_hash=snapshot.blob_hash,
                blob_size=snapshot.blob_size,
                blob_publication_receipt_id=snapshot.blob_publication_receipt_id,
            )
        for session in hermes_state.parse_state_db(
            retained_path,
            fallback_id=path.stem,
            profile_root=(original_source_path or path).parent,
            immutable=True,
        ):
            yield (raw_data, session)
        return

    if (
        provider_hint is Provider.HERMES or original_source_path is not None
    ) and hermes_verification.looks_like_verification_evidence_db_path(path):
        if blob_root is None:
            from polylogue.paths import blob_store_root

            blob_root = blob_store_root()
        resolved_store = blob_store or BlobStore(blob_root)
        snapshot = snapshot_sqlite_to_blob(path, resolved_store)
        from polylogue.storage.blob_publication import flush_blob_publications

        flush_blob_publications(resolved_store)
        retained_path = resolved_store.blob_path(snapshot.blob_hash)
        raw_data = None
        if capture_raw:
            raw_data = RawSessionData(
                raw_bytes=b"",
                source_path=str(original_source_path or path),
                source_index=None,
                file_mtime=file_mtime,
                provider_hint=provider_hint,
                blob_hash=snapshot.blob_hash,
                blob_size=snapshot.blob_size,
                blob_publication_receipt_id=snapshot.blob_publication_receipt_id,
            )
        for session in hermes_verification.parse_verification_evidence_db(
            retained_path,
            fallback_id=path.stem,
            profile_root=(original_source_path or path).parent,
            immutable=True,
        ):
            yield (raw_data, session)
        return

    ctx = _ParseContext(
        provider_hint=provider_hint,
        should_group=should_group,
        source_path_str=str(path),
        fallback_id=path.stem,
        file_mtime=file_mtime,
        capture_raw=capture_raw,
        sidecar_data=sidecar_data,
    )
    emitter = _SessionEmitter(ctx)

    if capture_raw and should_group:
        if blob_root is None:
            from polylogue.paths import blob_store_root

            blob_root = blob_store_root()
        resolved_store = blob_store or BlobStore(blob_root)
        blob_hash, blob_size = resolved_store.write_from_path(path)
        from polylogue.storage.blob_publication import flush_blob_publications, publication_receipt_id

        receipt_id = publication_receipt_id(resolved_store, blob_hash)
        flush_blob_publications(resolved_store)
        raw_data = RawSessionData(
            raw_bytes=b"",
            source_path=str(path),
            source_index=None,
            file_mtime=file_mtime,
            provider_hint=provider_hint,
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_publication_receipt_id=receipt_id,
        )
        with path.open("rb") as handle:
            yield from emitter.emit(handle, path.name, precomputed_raw=raw_data)
    else:
        with path.open("rb") as handle:
            yield from emitter.emit(handle, path.name)


def iter_source_sessions(
    source: Source,
    *,
    cursor_state: CursorStatePayload | None = None,
) -> Iterable[ParsedSession]:
    """Iterate parsed sessions from one configured source."""
    for _raw, session in iter_source_sessions_with_raw(
        source,
        cursor_state=cursor_state,
        capture_raw=False,
    ):
        yield session


def iter_source_sessions_with_raw(
    source: Source,
    *,
    cursor_state: CursorStatePayload | None = None,
    capture_raw: bool = True,
    known_mtimes: dict[str, str] | None = None,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Iterate parsed sessions with optional raw byte capture."""
    if not source.path:
        return

    yield from iter_antigravity_language_server_sessions(
        source,
        capture_raw=capture_raw,
        blob_root=blob_root,
        blob_store=blob_store,
    )

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=capture_raw,
        known_mtimes=known_mtimes,
        discover_sidecars=True,
        blob_store=blob_store,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            yield from parse_one_source_path(
                str(path),
                file_mtime=file_mtime,
                source_name=source.name,
                sidecar_data=walk.sidecar_data,
                capture_raw=capture_raw,
                cursor_state=cursor_state,
                blob_root=blob_root,
                blob_store=blob_store,
            )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except (JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            failed_count += 1
            logger.warning("Failed to parse %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error processing %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="parse/read",
    )


__all__ = [
    "iter_antigravity_language_server_sessions",
    "iter_source_sessions",
    "iter_source_sessions_with_raw",
    "has_decoded_session_evidence",
    "parse_one_source_path",
]
