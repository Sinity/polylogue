"""Raw source acquisition iterators over traversal and provider detection helpers."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from typing import Any

from polylogue.config import Source
from polylogue.logging import get_logger
from polylogue.types import Provider

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _record_cursor_failure
from .decoders import _zip_entry_provider_hint, _ZipEntryValidator
from .dispatch import _detect_provider_from_raw_bytes
from .parsers.base import RawConversationData
from .source_walk import _setup_source_walk

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[RawConversationData]:
    """Iterate raw source payloads without parsing provider payload semantics."""
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=True,
        known_mtimes=known_mtimes,
        build_session_indices=False,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            provider_hint = Provider.from_string(source.name)

            if path.suffix.lower() == ".zip":
                validator = _ZipEntryValidator(
                    provider_hint,
                    cursor_state=cursor_state,
                    zip_path=path,
                    conversation_only=False,
                )
                with zipfile.ZipFile(path) as zf:
                    for info in validator.filter_entries(zf.infolist()):
                        entry_path = f"{path}:{info.filename}"
                        entry_provider_hint = _zip_entry_provider_hint(info.filename, provider_hint)
                        with zf.open(info.filename) as handle:
                            raw_bytes = handle.read()
                        entry_provider_hint = _detect_provider_from_raw_bytes(
                            raw_bytes,
                            info.filename,
                            entry_provider_hint,
                        )
                        yield RawConversationData(
                            raw_bytes=raw_bytes,
                            source_path=entry_path,
                            source_index=None,
                            file_mtime=file_mtime,
                            provider_hint=entry_provider_hint,
                        )
            else:
                raw_bytes = path.read_bytes()
                detected_provider = _detect_provider_from_raw_bytes(
                    raw_bytes,
                    path.name,
                    provider_hint,
                )
                yield RawConversationData(
                    raw_bytes=raw_bytes,
                    source_path=str(path),
                    source_index=None,
                    file_mtime=file_mtime,
                    provider_hint=detected_provider,
                )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except (UnicodeDecodeError, zipfile.BadZipFile, OSError) as exc:
            failed_count += 1
            logger.warning("Failed to read %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error reading %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="read",
    )


__all__ = ["iter_source_raw_data"]
