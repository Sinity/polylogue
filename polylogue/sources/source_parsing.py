"""Source parsing iterators over traversal and parser-emission helpers."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.config import Source
from polylogue.logging import get_logger
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.types import Provider

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
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Iterate parsed sessions with optional raw byte capture."""
    if not source.path:
        return

    provider_hint = Provider.from_string(source.name)
    if provider_hint is Provider.ANTIGRAVITY and (source.path / "sessions").is_dir():
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
            path_classification = classify_artifact_path(path, provider=source.name)
            if path_classification is not None and not path_classification.parse_as_session:
                continue
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
            else:
                ctx = _ParseContext(
                    provider_hint=provider_hint,
                    should_group=should_group,
                    source_path_str=str(path),
                    fallback_id=path.stem,
                    file_mtime=file_mtime,
                    capture_raw=capture_raw,
                    sidecar_data=walk.sidecar_data,
                )
                emitter = _SessionEmitter(ctx)

                if capture_raw and should_group:
                    blob_hash, blob_size = get_blob_store().write_from_path(path)
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
                        yield from emitter.emit(
                            handle,
                            path.name,
                            precomputed_raw=raw_data,
                        )
                else:
                    with path.open("rb") as handle:
                        yield from emitter.emit(handle, path.name)
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
    "iter_source_sessions",
    "iter_source_sessions_with_raw",
]
