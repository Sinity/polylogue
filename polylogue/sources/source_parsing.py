"""Source parsing iterators over traversal and parser-emission helpers."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.assembly import SidecarData
from polylogue.storage.blob_store import BlobStore, get_blob_store
from polylogue.storage.cursor_state import CursorStatePayload

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _ParseContext, _record_cursor_failure
from .decoders import _process_zip
from .dispatch import GROUP_PROVIDERS as _GROUP_PROVIDERS
from .emitter import _SessionEmitter
from .parsers import antigravity
from .parsers.base import ParsedSession, RawSessionData
from .source_walk import _setup_source_walk

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger


def iter_antigravity_language_server_sessions(
    source: Source,
) -> Iterable[tuple[None, ParsedSession]]:
    """Yield Antigravity language-server export sessions for a source.

    No-op unless the source is Antigravity and exposes a ``sessions/``
    directory. Kept sequential (it drives a local HTTP loopback subprocess) and
    shared between the sequential iterator and the parallel ingest driver so the
    file-walk parallelization never touches this path.
    """
    if not source.path:
        return
    provider_hint = Provider.from_string(source.name)
    if provider_hint is not Provider.ANTIGRAVITY or not (source.path / "sessions").is_dir():
        return
    try:
        for session in antigravity.iter_language_server_exports(source.path):
            yield (None, session)
    except antigravity.AntigravityBinaryUnavailableError as exc:
        # Benign: Antigravity is simply not installed. Fall back to the
        # brain-artifact walk at INFO — this is not data loss.
        logger.info(
            "Antigravity language server unavailable for %s; using parseable artifacts: %s",
            source.path,
            exc,
        )
    except antigravity.AntigravityPartialExportError as exc:
        # Mid-export failure: some sessions were obtained before the
        # abort and the remainder is dropped. Surface obtained-vs-expected
        # loudly instead of conflating it with a benign fallback.
        logger.error(
            "Antigravity language-server export of %s truncated mid-iteration: "
            "obtained %d of %d sessions; %d lost before fallback: %s",
            source.path,
            exc.obtained,
            exc.expected,
            max(exc.expected - exc.obtained, 0),
            exc,
        )
    except antigravity.AntigravityExportError as exc:
        # Connection/protocol failure before any session was obtained.
        logger.warning(
            "Antigravity language-server export failed for %s; falling back to parseable artifacts: %s",
            source.path,
            exc,
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
    if path_classification is not None and not path_classification.parse_as_session:
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
        )
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
        blob_store = BlobStore(blob_root) if blob_root is not None else get_blob_store()
        blob_hash, blob_size = blob_store.write_from_path(path)
        raw_data = RawSessionData(
            raw_bytes=b"",
            source_path=str(path),
            source_index=None,
            file_mtime=file_mtime,
            provider_hint=provider_hint,
            blob_hash=blob_hash,
            blob_size=blob_size,
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
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Iterate parsed sessions with optional raw byte capture."""
    if not source.path:
        return

    yield from iter_antigravity_language_server_sessions(source)

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=capture_raw,
        known_mtimes=known_mtimes,
        discover_sidecars=True,
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
            )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
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
    "parse_one_source_path",
]
